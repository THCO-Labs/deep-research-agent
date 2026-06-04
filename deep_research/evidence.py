from __future__ import annotations

import re

from deep_research.schemas import EvidenceCard, ResearchBranch, SourceRecordV2
from deep_research.source_validation import anchor_groups_for_branch, branch_terms, content_terms, validate_source_content

SENTENCE_SPLIT_RE = re.compile(r"\n\s*\n|(?<=[.!?。！？])\s*|[;；]\s*")


def build_evidence_cards(
    *,
    branches: list[ResearchBranch],
    sources: list[SourceRecordV2],
    source_texts: dict[int, str],
    question: str = "",
    max_cards_per_source: int = 3,
) -> list[EvidenceCard]:
    branch_by_id = {branch.id: branch for branch in branches}
    question_terms = content_terms(question)
    cards: list[EvidenceCard] = []
    next_id = 1
    for source in sources:
        branch = branch_by_id.get(source.branch_id)
        if branch is None:
            continue
        source_text = source_texts.get(source.id, "")
        alignment = validate_source_content(
            title=source.title,
            content=source_text,
            branch=branch,
            min_words=0,
            min_relevant_chunks=0,
            question=question,
        )
        if not alignment.usable:
            continue
        terms = branch_terms(branch)
        candidates = _rank_sentences(source_text, terms, question_terms, anchor_groups_for_branch(branch))
        for sentence in candidates[:max_cards_per_source]:
            claim = _clean_claim(sentence)
            if len(claim) < 50:
                continue
            cards.append(
                EvidenceCard(
                    id=next_id,
                    source_id=source.id,
                    branch_id=source.branch_id,
                    claim=claim,
                    supporting_excerpt=sentence,
                    source_url=source.url,
                    source_title=source.title,
                    quality_score=source.quality_score,
                    relevance_score=source.relevance_score,
                    confidence=_confidence(source.quality_score, source.relevance_score, claim),
                    limitations=[] if source.quality_score >= 0.7 else ["Source is not primary or high-authority."],
                )
            )
            next_id += 1
    return cards


def coverage_for_branch(branch: ResearchBranch, cards: list[EvidenceCard]) -> tuple[list[str], list[str]]:
    branch_cards = [card for card in cards if card.branch_id == branch.id]
    corpus = " ".join(
        card.claim + " " + card.supporting_excerpt + " " + " ".join(card.semantic_notes)
        for card in branch_cards
    )
    normalized_corpus = corpus.lower().replace("-", " ")
    required = branch.required_terms or sorted(branch_terms(branch))[:8]
    covered = [term for term in required if term.lower().replace("-", " ") in normalized_corpus]
    missing = [term for term in required if term not in covered]
    return covered, missing


def _rank_sentences(
    text: str,
    terms: set[str],
    question_terms: set[str],
    anchor_groups: list[frozenset[str]],
) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_RE.split(text)
        if 60 <= len(sentence.strip()) <= 600
    ]
    minimum_question_hits = _minimum_question_hits(question_terms)
    ranked = sorted(
        ((index, sentence, content_terms(sentence)) for index, sentence in enumerate(sentences)),
        key=lambda item: (
            -len(item[2] & question_terms),
            -len(item[2] & terms),
            -len(item[2]),
            item[0],
        ),
    )
    return [
        sentence
        for _, sentence, sentence_terms in ranked
        if _sentence_matches_branch(
            sentence_terms,
            branch_terms=terms,
            question_terms=question_terms,
            anchor_groups=anchor_groups,
            minimum_question_hits=minimum_question_hits,
        )
    ]


def _minimum_question_hits(question_terms: set[str]) -> int:
    if not question_terms:
        return 0
    return min(2, len(question_terms))


def _matches_anchor_group(sentence_terms: set[str], anchor_groups: list[frozenset[str]]) -> bool:
    for group in anchor_groups:
        if group <= sentence_terms:
            return True
    return False


def _sentence_matches_branch(
    sentence_terms: set[str],
    *,
    branch_terms: set[str],
    question_terms: set[str],
    anchor_groups: list[frozenset[str]],
    minimum_question_hits: int,
) -> bool:
    branch_hits = len(sentence_terms & branch_terms)
    if branch_hits <= 0:
        return False
    question_hits = len(sentence_terms & question_terms)
    if question_hits < minimum_question_hits:
        return False
    if not anchor_groups or _matches_anchor_group(sentence_terms, anchor_groups):
        return True
    return branch_hits >= _strong_branch_overlap_threshold(branch_terms)


def _strong_branch_overlap_threshold(branch_terms: set[str]) -> int:
    return max(3, min(6, len(branch_terms) // 8))


def _clean_claim(sentence: str) -> str:
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    cleaned = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[[0-9][0-9,\s;:\-]*\]", "", cleaned)
    cleaned = cleaned.strip(" -")
    return cleaned


def _confidence(quality_score: float, relevance_score: float, claim: str) -> float:
    length_bonus = 0.05 if len(claim) >= 120 else 0.0
    return round(min(1.0, (quality_score * 0.45) + (relevance_score * 0.5) + length_bonus), 4)

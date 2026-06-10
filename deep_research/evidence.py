from __future__ import annotations

import re
from typing import Callable

from deep_research.schemas import EvidenceCard, ResearchBranch, SourceRecordV2
from deep_research.source_validation import (
    anchor_groups_for_branch,
    branch_terms,
    content_terms,
    uses_translated_branch_context,
    validate_source_content,
)

SENTENCE_SPLIT_RE = re.compile(r"\n\s*\n|(?<=[.!?。！？])\s*|[;；]\s*")


def build_evidence_cards(
    *,
    branches: list[ResearchBranch],
    sources: list[SourceRecordV2],
    source_texts: dict[int, str],
    question: str = "",
    max_cards_per_source: int = 3,
    existing_cards: list[EvidenceCard] | None = None,
    source_checkpoint_callback: Callable[[list[EvidenceCard]], None] | None = None,
) -> list[EvidenceCard]:
    """Build evidence cards from sources.

    existing_cards: cards already built in a previous (partial) run — sources
      whose IDs already appear in existing_cards are skipped so we don't redo work.
    source_checkpoint_callback: called after each source is processed so progress
      is saved mid-node; never raises (caller must guard).
    """
    branch_by_id = {branch.id: branch for branch in branches}
    question_terms = content_terms(question)

    # Re-use cards from a previous partial run and skip those source IDs.
    cards: list[EvidenceCard] = list(existing_cards or [])
    processed_source_ids: set[int] = {card.source_id for card in cards}
    next_id = max((card.id for card in cards), default=0) + 1

    for source in sources:
        if source.id in processed_source_ids:
            continue
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
        source_question_terms = (
            set()
            if uses_translated_branch_context(
                question=question,
                branch=branch,
                title=source.title,
                content=source_text,
            )
            else question_terms
        )
        candidates = _rank_sentences(source_text, terms, source_question_terms, anchor_groups_for_branch(branch))
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

        # Mid-node checkpoint: save progress after each source so a crash+resume
        # doesn't redo the full extraction from scratch.
        if source_checkpoint_callback is not None:
            try:
                source_checkpoint_callback(cards)
            except Exception:
                pass

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
    sentences = _candidate_passages(text)
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


def _candidate_passages(text: str) -> list[str]:
    passages: list[str] = []
    for segment in SENTENCE_SPLIT_RE.split(text):
        cleaned = _clean_passage(segment)
        if _passage_length_ok(cleaned):
            passages.append(cleaned)

    lines = [_clean_passage(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    for line in lines:
        if _passage_length_ok(line, min_length=35):
            passages.append(line)

    for window_size in (2, 3, 4):
        for index in range(0, max(0, len(lines) - window_size + 1)):
            window = _clean_passage(" ".join(lines[index : index + window_size]))
            if _passage_length_ok(window):
                passages.append(window)
    return _dedupe_passages(passages)


def _clean_passage(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().strip(" -|")


def _passage_length_ok(text: str, *, min_length: int = 50, max_length: int = 800) -> bool:
    return min_length <= len(text) <= max_length


def _dedupe_passages(passages: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for passage in passages:
        key = re.sub(r"\s+", " ", passage.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(passage)
    return result


def _minimum_question_hits(question_terms: set[str]) -> int:
    if not question_terms:
        return 0
    return min(2, len(question_terms))


def _matches_anchor_group(sentence_terms: set[str], anchor_groups: list[frozenset[str]]) -> bool:
    for group in anchor_groups:
        if group and all(_term_matches(term, sentence_terms) for term in group):
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
    if not question_terms and branch_hits >= _translated_branch_overlap_threshold(branch_terms):
        return True
    return branch_hits >= _strong_branch_overlap_threshold(branch_terms)


def _translated_branch_overlap_threshold(branch_terms: set[str]) -> int:
    return max(2, min(4, len(branch_terms) // 14))


def _strong_branch_overlap_threshold(branch_terms: set[str]) -> int:
    return max(3, min(6, len(branch_terms) // 8))


def _term_matches(term: str, observed_terms: set[str]) -> bool:
    if term in observed_terms:
        return True
    return any(variant in observed_terms for variant in _term_variants(term))


def _term_variants(term: str) -> set[str]:
    variants = {term}
    if len(term) > 4 and term.endswith("ies"):
        variants.add(term[:-3] + "y")
    if len(term) > 4 and term.endswith("es"):
        variants.add(term[:-2])
    if len(term) > 3 and term.endswith("s"):
        variants.add(term[:-1])
    if len(term) > 3:
        variants.add(term + "s")
    return variants


def _clean_claim(sentence: str) -> str:
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    cleaned = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[[0-9][0-9,\s;:\-]*\]", "", cleaned)
    cleaned = cleaned.strip(" -")
    return cleaned


def _confidence(quality_score: float, relevance_score: float, claim: str) -> float:
    length_bonus = 0.05 if len(claim) >= 120 else 0.0
    return round(min(1.0, (quality_score * 0.45) + (relevance_score * 0.5) + length_bonus), 4)

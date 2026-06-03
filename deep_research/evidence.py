from __future__ import annotations

import re

from deep_research.schemas import EvidenceCard, ResearchBranch, SourceRecordV2
from deep_research.source_validation import branch_terms, content_terms


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
        terms = branch_terms(branch)
        candidates = _rank_sentences(source_texts.get(source.id, ""), terms, question_terms)
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


def _rank_sentences(text: str, terms: set[str], question_terms: set[str]) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", text)
        if 60 <= len(sentence.strip()) <= 600
    ]
    minimum_question_hits = _minimum_question_hits(question_terms)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -len(content_terms(item[1]) & question_terms),
            -len(content_terms(item[1]) & terms),
            -len(content_terms(item[1])),
            item[0],
        ),
    )
    return [
        sentence
        for _, sentence in ranked
        if len(content_terms(sentence) & terms) > 0
        and len(content_terms(sentence) & question_terms) >= minimum_question_hits
    ]


def _minimum_question_hits(question_terms: set[str]) -> int:
    if not question_terms:
        return 0
    return min(2, len(question_terms))


def _clean_claim(sentence: str) -> str:
    cleaned = re.sub(r"\s+", " ", sentence).strip()
    cleaned = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"\[[0-9][0-9,\s;:\-]*\]", "", cleaned)
    cleaned = cleaned.strip(" -")
    return cleaned


def _confidence(quality_score: float, relevance_score: float, claim: str) -> float:
    length_bonus = 0.05 if len(claim) >= 120 else 0.0
    return round(min(1.0, (quality_score * 0.45) + (relevance_score * 0.5) + length_bonus), 4)

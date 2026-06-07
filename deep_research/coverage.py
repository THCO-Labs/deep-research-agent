from __future__ import annotations

from dataclasses import dataclass

from deep_research.schemas import BranchCoverage, CoverageMatrix, EvidenceCard, ResearchBranch, SourceRecordV2
from deep_research.source_validation import content_terms


@dataclass(frozen=True)
class _TermCoverage:
    required: list[str]
    covered: list[str]
    missing: list[str]
    complete: bool
    score: float


@dataclass(frozen=True)
class _SemanticCoverage:
    required: list[str]
    covered: list[str]
    missing: list[str]
    complete: bool
    score: float


def build_coverage_matrix(
    *,
    branches: list[ResearchBranch],
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
) -> CoverageMatrix:
    coverage_rows: list[BranchCoverage] = []
    for branch in branches:
        branch_cards = [card for card in evidence_cards if card.branch_id == branch.id]
        source_count = len([source for source in sources if source.branch_id == branch.id])
        required = [f"usable sources >= {branch.min_sources}", "branch evidence cards"]
        covered = []
        missing = []
        if source_count >= branch.min_sources:
            covered.append(required[0])
        else:
            missing.append(required[0])
        if branch_cards:
            covered.append(required[1])
        else:
            missing.append(required[1])
        term_coverage = _required_term_coverage(branch, branch_cards)
        required.extend(term_coverage.required)
        covered.extend(term_coverage.covered)
        missing.extend(term_coverage.missing)
        semantic_coverage = _semantic_branch_coverage(branch, branch_cards, source_count=source_count, term_score=term_coverage.score)
        required.extend(semantic_coverage.required)
        covered.extend(semantic_coverage.covered)
        missing.extend(semantic_coverage.missing)
        complete = (
            source_count >= branch.min_sources
            and bool(branch_cards)
            and (term_coverage.complete or semantic_coverage.complete)
        )
        coverage_rows.append(
            BranchCoverage(
                branch_id=branch.id,
                required_points=required,
                covered_points=covered,
                missing_points=missing,
                source_count=source_count,
                complete=complete,
            )
        )
    completed = sum(1 for row in coverage_rows if row.complete)
    branch_score = completed / max(len(coverage_rows), 1)
    required_source_total = sum(max(branch.min_sources, 1) for branch in branches)
    source_score = min(1.0, len(sources) / max(required_source_total, 1))
    coverage_score = round((branch_score + source_score) / 2, 4)
    missing_branches = [row.branch_id for row in coverage_rows if not row.complete]
    return CoverageMatrix(
        branches=coverage_rows,
        complete=not missing_branches,
        coverage_score=coverage_score,
        missing_branches=missing_branches,
    )


def _required_term_coverage(
    branch: ResearchBranch,
    branch_cards: list[EvidenceCard],
) -> _TermCoverage:
    if not branch.required_terms:
        return _TermCoverage(required=[], covered=[], missing=[], complete=True, score=1.0)
    corpus = " ".join(
        card.claim + " " + card.supporting_excerpt + " " + " ".join(card.semantic_notes)
        for card in branch_cards
    )
    normalized_corpus = corpus.lower().replace("-", " ")
    corpus_terms = content_terms(corpus)
    term_rows: list[tuple[str, bool]] = []
    for term in branch.required_terms:
        label = f"required term: {term}"
        term_terms = content_terms(term)
        if not term_terms:
            continue
        normalized_term = term.lower().replace("-", " ")
        is_covered = normalized_term in normalized_corpus or term_terms <= corpus_terms
        if not is_covered and len(term_terms) > 2:
            is_covered = len(term_terms & corpus_terms) / len(term_terms) >= 0.60
        term_rows.append((label, is_covered))
    if not term_rows:
        return _TermCoverage(required=[], covered=[], missing=[], complete=True, score=1.0)

    covered_terms = [label for label, is_covered in term_rows if is_covered]
    missing_terms = [label for label, is_covered in term_rows if not is_covered]
    score = len(covered_terms) / len(term_rows)
    threshold = _required_term_threshold(term_rows)
    threshold_label = f"required term coverage >= {round(threshold * 100)}%"
    required = [threshold_label, *[label for label, _is_covered in term_rows]]
    if score >= threshold:
        return _TermCoverage(
            required=required,
            covered=[threshold_label, *covered_terms],
            missing=[],
            complete=True,
            score=round(score, 4),
        )
    return _TermCoverage(
        required=required,
        covered=covered_terms,
        missing=[f"{threshold_label} (actual {round(score * 100)}%)", *missing_terms],
        complete=False,
        score=round(score, 4),
    )


def _semantic_branch_coverage(
    branch: ResearchBranch,
    branch_cards: list[EvidenceCard],
    *,
    source_count: int,
    term_score: float,
) -> _SemanticCoverage:
    if not branch_cards:
        return _SemanticCoverage(
            required=["semantic evidence sufficiency"],
            covered=[],
            missing=["semantic evidence sufficiency: no branch evidence cards"],
            complete=False,
            score=0.0,
        )

    expected_evidence_items = max(2, min(branch.min_sources, 4))
    unique_source_count = len({card.source_id for card in branch_cards})
    strong_cards = [
        card
        for card in branch_cards
        if _card_semantic_strength(card) >= 0.62
    ]
    score = round(
        (
            min(1.0, unique_source_count / max(expected_evidence_items, 1)) * 0.45
            + min(1.0, len(strong_cards) / max(expected_evidence_items, 1)) * 0.45
            + _average_card_strength(branch_cards) * 0.10
        ),
        4,
    )
    label = "semantic evidence sufficiency >= 70%"
    covered: list[str] = []
    missing: list[str] = []
    if score >= 0.70:
        covered.append(label)
    else:
        missing.append(f"{label} (actual {round(score * 100)}%)")

    diversity_label = f"semantic evidence from >= {expected_evidence_items} source(s)"
    if unique_source_count >= expected_evidence_items:
        covered.append(diversity_label)
    else:
        missing.append(f"{diversity_label} (actual {unique_source_count})")

    strong_label = f"strong semantic evidence cards >= {expected_evidence_items}"
    if len(strong_cards) >= expected_evidence_items:
        covered.append(strong_label)
    else:
        missing.append(f"{strong_label} (actual {len(strong_cards)})")

    synthesis_label = "evidence-limited synthesis readiness"
    synthesis_ready = _evidence_limited_synthesis_ready(
        branch=branch,
        branch_cards=branch_cards,
        source_count=source_count,
        unique_source_count=unique_source_count,
        expected_evidence_items=expected_evidence_items,
        strong_card_count=len(strong_cards),
        average_strength=_average_card_strength(branch_cards),
        term_score=term_score,
    )
    if synthesis_ready:
        covered.append(synthesis_label)
    else:
        missing.append(synthesis_label)

    complete = (
        score >= 0.70
        and unique_source_count >= expected_evidence_items
        and len(strong_cards) >= expected_evidence_items
    ) or synthesis_ready
    return _SemanticCoverage(
        required=[label, diversity_label, strong_label, synthesis_label],
        covered=covered,
        missing=[] if complete else missing,
        complete=complete,
        score=score,
    )


def _evidence_limited_synthesis_ready(
    *,
    branch: ResearchBranch,
    branch_cards: list[EvidenceCard],
    source_count: int,
    unique_source_count: int,
    expected_evidence_items: int,
    strong_card_count: int,
    average_strength: float,
    term_score: float,
) -> bool:
    if len(branch_cards) < expected_evidence_items:
        return False
    if unique_source_count < expected_evidence_items:
        return False
    if source_count < max(branch.min_sources, expected_evidence_items):
        return False
    evidence_rich = len(branch_cards) >= expected_evidence_items * 3 and unique_source_count >= expected_evidence_items
    minimum_strength = 0.42 if evidence_rich else 0.52
    if average_strength < minimum_strength:
        return False
    if term_score < 0.25 and average_strength < 0.58 and not evidence_rich:
        return False
    if strong_card_count == 0 and not evidence_rich and source_count < expected_evidence_items * 3:
        return False
    return True


def _card_semantic_strength(card: EvidenceCard) -> float:
    semantic = card.semantic_score if card.semantic_score is not None else None
    if semantic is None:
        semantic = (float(card.relevance_score) + float(card.confidence)) / 2
    return max(
        0.0,
        min(
            1.0,
            (
                float(semantic) * 0.50
                + float(card.relevance_score) * 0.25
                + float(card.confidence) * 0.25
            ),
        ),
    )


def _average_card_strength(cards: list[EvidenceCard]) -> float:
    if not cards:
        return 0.0
    return sum(_card_semantic_strength(card) for card in cards) / len(cards)


def _required_term_threshold(term_rows: list[tuple[str, bool]]) -> float:
    count = len(term_rows)
    if count <= 2:
        return 1.0
    if count <= 4:
        return 0.75
    return 0.55

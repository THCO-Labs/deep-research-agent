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
        complete = source_count >= branch.min_sources and bool(branch_cards) and term_coverage.complete
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


def _required_term_threshold(term_rows: list[tuple[str, bool]]) -> float:
    count = len(term_rows)
    if count <= 2:
        return 1.0
    if count <= 4:
        return 0.75
    return 0.55

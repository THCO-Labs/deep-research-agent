from __future__ import annotations

from deep_research.schemas import BranchCoverage, CoverageMatrix, EvidenceCard, ResearchBranch, SourceRecordV2
from deep_research.source_validation import content_terms


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
        term_required, term_covered, term_missing = _required_term_coverage(branch, branch_cards)
        required.extend(term_required)
        covered.extend(term_covered)
        missing.extend(term_missing)
        complete = not missing
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
) -> tuple[list[str], list[str], list[str]]:
    if not branch.required_terms:
        return [], [], []
    corpus = " ".join(
        card.claim + " " + card.supporting_excerpt + " " + " ".join(card.semantic_notes)
        for card in branch_cards
    )
    normalized_corpus = corpus.lower().replace("-", " ")
    corpus_terms = content_terms(corpus)
    required: list[str] = []
    covered: list[str] = []
    missing: list[str] = []
    for term in branch.required_terms:
        label = f"required term: {term}"
        term_terms = content_terms(term)
        if not term_terms:
            continue
        required.append(label)
        normalized_term = term.lower().replace("-", " ")
        is_covered = normalized_term in normalized_corpus or term_terms <= corpus_terms
        if is_covered:
            covered.append(label)
        else:
            missing.append(label)
    return required, covered, missing

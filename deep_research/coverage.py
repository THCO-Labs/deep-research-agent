from __future__ import annotations

from deep_research.schemas import BranchCoverage, CoverageMatrix, EvidenceCard, ResearchBranch, SourceRecordV2


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

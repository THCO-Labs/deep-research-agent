from __future__ import annotations

from typing import Any

from deep_research.core.schemas import EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.core.settings import Settings

_LOCAL_FAILURE_TERMS = (
    "citation",
    "cited",
    "source support",
    "weakly supported",
    "unsupported claim",
    "uncited",
)


def classify_repair_failures(failures: list[str]) -> tuple[bool, list[str]]:
    """Split local citation failures from structural report failures."""
    structural: list[str] = []
    for failure in failures:
        text = str(failure).lower()
        if not any(term in text for term in _LOCAL_FAILURE_TERMS):
            structural.append(str(failure))
    return not structural, structural


def apply_targeted_citation_repair(
    *,
    report: str,
    weakly_supported_claims: list[dict[str, Any]],
    unsupported_claims: list[str],
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    plan: ResearchPlan,
    settings: Settings,
) -> tuple[str, int]:
    """Conservative targeted repair placeholder.

    Returning zero patches deliberately hands control back to the graph's full
    repair path, which has the complete synthesis context and verifier loop.
    """
    return report, 0

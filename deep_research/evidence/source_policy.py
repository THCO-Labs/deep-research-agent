from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from deep_research.core.schemas import ResearchPlan, SourceRecordV2


@dataclass(frozen=True)
class SourcePolicy:
    schema_version: int
    task_type: str
    label: str
    preferred_source_types: list[str]
    acceptable_source_types: list[str]
    low_trust_source_types: list[str]
    min_independent_sources: int
    rationale: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_source_policy(plan: ResearchPlan) -> SourcePolicy:
    text = _plan_text(plan)
    task_type, reasons = _task_type(text)
    preferred, acceptable, low_trust, min_sources = _policy_types(task_type)
    return SourcePolicy(
        schema_version=1,
        task_type=task_type,
        label=f"{task_type}_source_policy",
        preferred_source_types=preferred,
        acceptable_source_types=acceptable,
        low_trust_source_types=low_trust,
        min_independent_sources=min_sources,
        rationale=reasons or ["general research task; prefer diverse source types"],
    )


def score_sources_against_policy(policy: SourcePolicy, sources: list[SourceRecordV2]) -> dict[str, Any]:
    preferred = set(policy.preferred_source_types)
    acceptable = set(policy.acceptable_source_types)
    low_trust = set(policy.low_trust_source_types)
    usable = [source for source in sources if source.usable]
    preferred_count = sum(1 for source in usable if source.quality_type in preferred)
    acceptable_count = sum(1 for source in usable if source.quality_type in acceptable or source.quality_type in preferred)
    low_trust_count = sum(1 for source in usable if source.quality_type in low_trust)
    independent_domains = {_domain_key(source.canonical_url or source.url) for source in usable}
    source_count = len(usable)
    breadth_score = min(1.0, source_count / max(policy.min_independent_sources, 1))
    preferred_score = min(1.0, preferred_count / max(policy.min_independent_sources, 1))
    independence_score = min(1.0, len(independent_domains) / max(policy.min_independent_sources, 1))
    penalty = min(0.35, low_trust_count * 0.08)
    score = round(max(0.0, min(1.0, 0.35 * breadth_score + 0.35 * preferred_score + 0.30 * independence_score - penalty)), 4)
    return {
        "schema_version": 1,
        "policy": policy.to_dict(),
        "source_count": source_count,
        "preferred_source_count": preferred_count,
        "acceptable_source_count": acceptable_count,
        "low_trust_source_count": low_trust_count,
        "independent_domain_count": len(independent_domains),
        "score": score,
    }


def _task_type(text: str) -> tuple[str, list[str]]:
    signals = [
        (
            "medical_legal_financial",
            r"\b(?:clinical|patient|therapy|therapeutic|medical|legal|law|regulation|financial|investment|securities|tax)\b",
            "high-stakes medical/legal/financial language",
        ),
        (
            "comparative_benchmark",
            r"\b(?:best|strongest|rank|ranking|leaderboard|benchmark|eval(?:uation)?|score|performance)\b",
            "comparative or benchmark-based decision language",
        ),
        (
            "technical_procurement",
            r"\b(?:choose|procurement|vendor|machine|hardware|software|datasheet|manual|specs?|integration|compatibility|cost)\b",
            "technical comparison or procurement language",
        ),
        (
            "academic",
            r"\b(?:literature|paper|study|studies|journal|meta-analysis|systematic review|research evidence|peer[-\s]?reviewed)\b",
            "academic or literature-review language",
        ),
        (
            "market_research",
            r"\b(?:market|industry|competitor|company filing|revenue|pricing|forecast|customer|growth|strategy)\b",
            "market or strategic research language",
        ),
        (
            "current_events",
            r"\b(?:latest|current|today|recent|news|announced|this year|202[4-9])\b",
            "freshness-sensitive current information",
        ),
    ]
    matches = [(task, reason) for task, pattern, reason in signals if re.search(pattern, text, flags=re.I)]
    if matches:
        return matches[0][0], [reason for _, reason in matches]
    return "general", []


def _policy_types(task_type: str) -> tuple[list[str], list[str], list[str], int]:
    low_trust = ["user_content", "reference", "general_web"]
    if task_type == "technical_procurement":
        return (
            ["product_page", "vendor_page", "spec_sheet", "datasheet", "manual_pdf", "brochure_pdf", "official_docs", "standards_or_government"],
            ["academic", "news", "software_repository", "government"],
            low_trust,
            4,
        )
    if task_type == "comparative_benchmark":
        return (
            ["academic", "official_docs", "software_repository", "standards_or_government"],
            ["news", "reference", "government", "general_web"],
            ["user_content"],
            4,
        )
    if task_type == "medical_legal_financial":
        return (
            ["government", "standards_or_government", "academic", "official_docs"],
            ["news", "reference"],
            ["user_content", "general_web", "vendor_page", "product_page"],
            5,
        )
    if task_type == "academic":
        return (
            ["academic", "government", "standards_or_government"],
            ["official_docs", "news", "reference"],
            ["user_content", "general_web"],
            4,
        )
    if task_type == "market_research":
        return (
            ["government", "standards_or_government", "news", "official_docs"],
            ["academic", "reference", "general_web"],
            ["user_content"],
            5,
        )
    if task_type == "current_events":
        return (
            ["news", "government", "official_docs"],
            ["academic", "standards_or_government", "reference"],
            ["user_content", "general_web"],
            4,
        )
    return (
        ["academic", "government", "official_docs", "standards_or_government", "news"],
        ["reference", "general_web", "product_page", "vendor_page", "spec_sheet", "datasheet"],
        ["user_content"],
        3,
    )


def _plan_text(plan: ResearchPlan) -> str:
    return " ".join(
        [
            plan.question,
            " ".join(plan.report_outline),
            " ".join(plan.acceptance_criteria),
            " ".join(f"{branch.title} {branch.objective} {' '.join(branch.queries)}" for branch in plan.branches),
        ]
    ).lower()


def _domain_key(url: str) -> str:
    match = re.search(r"https?://([^/]+)", url.lower())
    host = match.group(1) if match else url.lower()
    return host.removeprefix("www.")

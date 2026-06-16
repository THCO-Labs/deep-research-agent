from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Literal

from deep_research.lexical_expansion import term_matches
from deep_research.schemas import CoverageMatrix, ResearchPlan


ReasoningAction = Literal[
    "search_more",
    "scrape_more",
    "contradiction_search",
    "analyze_with_python",
    "synthesize",
    "repair",
    "stop",
]


@dataclass(frozen=True)
class SubQuestion:
    id: str
    branch_id: str
    question: str
    status: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Finding:
    id: str
    branch_id: str
    claim: str
    source_ids: list[int]
    evidence_card_ids: list[int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Hypothesis:
    id: str
    branch_id: str
    statement: str
    confidence: float
    evidence_claim_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Unknown:
    id: str
    branch_id: str
    description: str
    reason: str
    severity: str
    focus_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeakClaim:
    claim_id: str
    branch_id: str
    claim: str
    source_ids: list[int]
    evidence_card_ids: list[int]
    confidence: float
    reasons: list[str]
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Contradiction:
    id: str
    claim_id: str
    branch_id: str
    description: str
    source_ids: list[int]
    confidence: float
    needs_caveat: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NextAction:
    action: ReasoningAction
    rationale: str
    branch_ids: list[str] = field(default_factory=list)
    focus_terms: list[str] = field(default_factory=list)
    priority: str = "medium"
    deferred: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SectionConfidence:
    section_id: str
    branch_id: str
    score: float
    evidence_count: int
    source_count: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchStateArtifact:
    schema_version: int
    original_query: str
    task_type: str
    sub_questions: list[SubQuestion]
    findings: list[Finding]
    hypotheses: list[Hypothesis]
    unknowns: list[Unknown]
    weak_claims: list[WeakClaim]
    contradictions: list[Contradiction]
    missing_evidence: list[Unknown]
    confidence_by_section: list[SectionConfidence]
    next_recommended_actions: list[NextAction]
    readiness_status: str
    source_policy_label: str
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "original_query": self.original_query,
            "task_type": self.task_type,
            "sub_questions": [item.to_dict() for item in self.sub_questions],
            "findings": [item.to_dict() for item in self.findings],
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "unknowns": [item.to_dict() for item in self.unknowns],
            "weak_claims": [item.to_dict() for item in self.weak_claims],
            "contradictions": [item.to_dict() for item in self.contradictions],
            "missing_evidence": [item.to_dict() for item in self.missing_evidence],
            "confidence_by_section": [item.to_dict() for item in self.confidence_by_section],
            "next_recommended_actions": [item.to_dict() for item in self.next_recommended_actions],
            "readiness_status": self.readiness_status,
            "source_policy_label": self.source_policy_label,
            "summary": self.summary,
        }


def build_research_state_artifact(
    *,
    plan: ResearchPlan,
    evidence_graph: dict[str, Any],
    coverage: CoverageMatrix,
    source_policy: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> ResearchStateArtifact:
    claims = [row for row in evidence_graph.get("claims", []) if isinstance(row, dict)]
    weak_claims = [_weak_claim(row) for row in claims if row.get("weak")]
    contradictions = [_contradiction(row) for row in evidence_graph.get("contradiction_edges", []) if isinstance(row, dict)]
    confidence = _section_confidence(plan, claims, coverage)
    unknowns = _unknowns_from_coverage(plan, coverage, confidence, claims, source_policy)
    deferred_actions = _deferred_actions(claims)
    readiness = _readiness_status(coverage=coverage, weak_claims=weak_claims, unknowns=unknowns)
    action = decide_next_action(
        reasoning_state={
            "weak_claims": [item.to_dict() for item in weak_claims],
            "unknowns": [item.to_dict() for item in unknowns],
            "contradictions": [item.to_dict() for item in contradictions],
            "readiness_status": readiness,
        },
        evidence_graph=evidence_graph,
        coverage=coverage.to_dict(),
        metrics=metrics or {},
    )
    actions = [NextAction(**action)] + deferred_actions
    return ResearchStateArtifact(
        schema_version=1,
        original_query=plan.question,
        task_type=str(source_policy.get("task_type") or source_policy.get("policy", {}).get("task_type") or "general"),
        sub_questions=[
            SubQuestion(
                id=f"subq_{index}",
                branch_id=branch.id,
                question=f"{branch.title}: {branch.objective}",
                status="complete" if branch.id not in coverage.missing_branches else "missing_evidence",
                confidence=_confidence_for_branch(confidence, branch.id),
            )
            for index, branch in enumerate(plan.branches, start=1)
        ],
        findings=[_finding(row) for row in claims if not row.get("weak")][:80],
        hypotheses=[_hypothesis(row) for row in claims if row.get("high_impact")][:30],
        unknowns=unknowns,
        weak_claims=weak_claims,
        contradictions=contradictions,
        missing_evidence=unknowns,
        confidence_by_section=confidence,
        next_recommended_actions=actions,
        readiness_status=readiness,
        source_policy_label=str(source_policy.get("label") or source_policy.get("policy", {}).get("label") or "general_source_policy"),
        summary={
            "coverage_score": coverage.coverage_score,
            "coverage_complete": coverage.complete,
            "claim_count": len(claims),
            "weak_claim_count": len(weak_claims),
            "unknown_count": len(unknowns),
            "contradiction_count": len(contradictions),
            "primary_action": action["action"],
        },
    )


def decide_next_action(
    *,
    reasoning_state: dict[str, Any],
    evidence_graph: dict[str, Any],
    coverage: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    weak_claims = [row for row in reasoning_state.get("weak_claims", []) if isinstance(row, dict)]
    unknowns = [row for row in reasoning_state.get("unknowns", []) if isinstance(row, dict)]
    claims = [row for row in evidence_graph.get("claims", []) if isinstance(row, dict)]
    evidence_count = int(metrics.get("evidence_card_count", 0) or evidence_graph.get("metrics", {}).get("support_edge_count", 0) or 0)
    coverage_complete = bool(coverage.get("complete"))
    coverage_score = _float(coverage.get("coverage_score"), 0.0)
    reasoning_iterations = int(metrics.get("reasoning_iteration_count", 0) or 0)
    max_reasoning_iterations = int(metrics.get("max_reasoning_iterations", 5) or 5)
    contradiction_iterations = int(metrics.get("contradiction_search_iterations", 0) or 0)
    model_action = _model_recommended_action(
        reasoning_state.get("model_recommended_action"),
        reasoning_iterations=reasoning_iterations,
        max_reasoning_iterations=max_reasoning_iterations,
        contradiction_iterations=contradiction_iterations,
        coverage_complete=coverage_complete,
        coverage_score=coverage_score,
    )
    if model_action is not None:
        return model_action
    if evidence_count <= 0 and _acquisition_exhausted_or_plateaued(metrics):
        return _action("stop", "No usable evidence remains after exhausted or plateaued acquisition.", priority="high")
    if unknowns and reasoning_iterations < max_reasoning_iterations:
        return _action(
            "search_more",
            "One or more branches still have missing or weak evidence.",
            branch_ids=_branch_ids(unknowns),
            focus_terms=_focus_terms(unknowns),
            priority="high",
        )
    if _important_single_source_claims(claims) and reasoning_iterations < max_reasoning_iterations:
        return _action(
            "search_more",
            "High-impact claims need independent support before synthesis.",
            branch_ids=_branch_ids(_important_single_source_claims(claims)),
            focus_terms=[str(row.get("claim") or "") for row in _important_single_source_claims(claims)[:8]],
            priority="high",
        )
    if _claims_need_contradiction_check(claims) and contradiction_iterations < 1 and reasoning_iterations < max_reasoning_iterations:
        return _action(
            "contradiction_search",
            "High-impact supported claims have not been checked against opposing or newer evidence.",
            branch_ids=_branch_ids(claims),
            focus_terms=[str(row.get("claim") or "") for row in claims if row.get("high_impact")][:8],
            priority="medium",
        )
    if _has_numeric_or_table_claim(claims) and not unknowns and not _important_single_source_claims(claims):
        return _action(
            "analyze_with_python",
            "Numeric, table, benchmark, price, or unit-heavy evidence should be checked with Python; deferred in Phase 1.",
            deferred=True,
        )
    if not unknowns and (coverage_complete or (coverage_score >= 0.75 and len(weak_claims) <= max(2, len(claims) // 8))):
        return _action("synthesize", "Coverage and reasoning confidence are sufficient for synthesis.", priority="medium")
    if not _acquisition_exhausted_or_plateaued(metrics) and reasoning_iterations < max_reasoning_iterations:
        return _action(
            "search_more",
            "Coverage is not complete and acquisition budget remains.",
            branch_ids=list(coverage.get("missing_branches", []) or []),
            priority="high",
        )
    if reasoning_iterations >= max_reasoning_iterations:
        return _action("synthesize", "Reasoning search loop budget is exhausted; synthesize with caveats from the reasoning state.", priority="medium")
    return _action("synthesize", "Acquisition is exhausted; synthesize with caveats from the reasoning state.", priority="medium")


def _model_recommended_action(
    value: Any,
    *,
    reasoning_iterations: int,
    max_reasoning_iterations: int,
    contradiction_iterations: int,
    coverage_complete: bool,
    coverage_score: float,
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or value.get("deferred"):
        return None
    action = str(value.get("action") or "")
    if action in {"search_more", "scrape_more"} and reasoning_iterations < max_reasoning_iterations:
        return _action(
            "search_more",
            str(value.get("rationale") or "Model refinement recommended more evidence."),
            branch_ids=[str(item) for item in value.get("branch_ids", []) if str(item).strip()],
            focus_terms=[str(item) for item in value.get("focus_terms", []) if str(item).strip()],
            priority=str(value.get("priority") or "medium"),
        )
    if action == "contradiction_search" and contradiction_iterations < 1 and reasoning_iterations < max_reasoning_iterations:
        return _action(
            "contradiction_search",
            str(value.get("rationale") or "Model refinement recommended contradiction checking."),
            branch_ids=[str(item) for item in value.get("branch_ids", []) if str(item).strip()],
            focus_terms=[str(item) for item in value.get("focus_terms", []) if str(item).strip()],
            priority=str(value.get("priority") or "medium"),
        )
    if action == "synthesize" and (coverage_complete or coverage_score >= 0.75):
        return _action("synthesize", str(value.get("rationale") or "Model refinement judged synthesis ready."))
    if action == "stop":
        return _action("stop", str(value.get("rationale") or "Model refinement recommended stopping."))
    return None


def _action(
    action: ReasoningAction,
    rationale: str,
    *,
    branch_ids: list[str] | None = None,
    focus_terms: list[str] | None = None,
    priority: str = "medium",
    deferred: bool = False,
) -> dict[str, Any]:
    return NextAction(
        action=action,
        rationale=rationale,
        branch_ids=_dedupe(branch_ids or []),
        focus_terms=_dedupe([term for term in (focus_terms or []) if term.strip()])[:12],
        priority=priority,
        deferred=deferred,
    ).to_dict()


def _weak_claim(row: dict[str, Any]) -> WeakClaim:
    return WeakClaim(
        claim_id=str(row.get("id") or ""),
        branch_id=str(row.get("branch_id") or ""),
        claim=str(row.get("claim") or ""),
        source_ids=[int(value) for value in row.get("source_ids", []) if isinstance(value, int)],
        evidence_card_ids=[int(value) for value in row.get("evidence_card_ids", []) if isinstance(value, int)],
        confidence=_float(row.get("average_confidence"), 0.0),
        reasons=[str(value) for value in row.get("weakness_reasons", [])],
        recommended_action="search_more",
    )


def _contradiction(row: dict[str, Any]) -> Contradiction:
    return Contradiction(
        id=str(row.get("id") or ""),
        claim_id=str(row.get("claim_id") or ""),
        branch_id=str(row.get("branch_id") or ""),
        description=str(row.get("description") or ""),
        source_ids=[int(value) for value in row.get("source_ids", []) if isinstance(value, int)],
        confidence=_float(row.get("confidence"), 0.0),
        needs_caveat=bool(row.get("needs_caveat", True)),
    )


def _finding(row: dict[str, Any]) -> Finding:
    return Finding(
        id=str(row.get("id") or ""),
        branch_id=str(row.get("branch_id") or ""),
        claim=str(row.get("claim") or ""),
        source_ids=[int(value) for value in row.get("source_ids", []) if isinstance(value, int)],
        evidence_card_ids=[int(value) for value in row.get("evidence_card_ids", []) if isinstance(value, int)],
        confidence=_float(row.get("average_confidence"), 0.0),
    )


def _hypothesis(row: dict[str, Any]) -> Hypothesis:
    return Hypothesis(
        id=f"hypothesis_{row.get('id')}",
        branch_id=str(row.get("branch_id") or ""),
        statement=str(row.get("claim") or ""),
        confidence=_float(row.get("average_confidence"), 0.0),
        evidence_claim_ids=[str(row.get("id") or "")],
    )


def _section_confidence(plan: ResearchPlan, claims: list[dict[str, Any]], coverage: CoverageMatrix) -> list[SectionConfidence]:
    claims_by_branch: dict[str, list[dict[str, Any]]] = {}
    for claim in claims:
        claims_by_branch.setdefault(str(claim.get("branch_id") or ""), []).append(claim)
    coverage_by_branch = {row.branch_id: row for row in coverage.branches}
    rows: list[SectionConfidence] = []
    for branch in plan.branches:
        branch_claims = claims_by_branch.get(branch.id, [])
        source_ids = {int(value) for claim in branch_claims for value in claim.get("source_ids", []) if isinstance(value, int)}
        weak_count = sum(1 for claim in branch_claims if claim.get("weak"))
        avg_conf = sum(_float(claim.get("average_confidence"), 0.0) for claim in branch_claims) / max(len(branch_claims), 1)
        coverage_row = coverage_by_branch.get(branch.id)
        coverage_bonus = 0.2 if coverage_row and coverage_row.complete else 0.0
        score = round(max(0.0, min(1.0, avg_conf * 0.65 + min(len(source_ids), 4) * 0.05 + coverage_bonus - weak_count * 0.04)), 4)
        reasons = []
        if not branch_claims:
            reasons.append("no evidence claims")
        if weak_count:
            reasons.append(f"{weak_count} weak claim(s)")
        if coverage_row and not coverage_row.complete:
            reasons.append("coverage incomplete")
        rows.append(
            SectionConfidence(
                section_id=branch.id,
                branch_id=branch.id,
                score=score,
                evidence_count=len(branch_claims),
                source_count=len(source_ids),
                reasons=reasons,
            )
        )
    return rows


def _unknowns_from_coverage(
    plan: ResearchPlan,
    coverage: CoverageMatrix,
    confidence_rows: list[SectionConfidence],
    claims: list[dict[str, Any]],
    source_policy: dict[str, Any],
) -> list[Unknown]:
    confidence_by_branch = {row.branch_id: row for row in confidence_rows}
    unknowns: list[Unknown] = []
    coverage_by_branch = {row.branch_id: row for row in coverage.branches}
    for branch in plan.branches:
        coverage_row = coverage_by_branch.get(branch.id)
        confidence = confidence_by_branch.get(branch.id)
        if coverage_row and coverage_row.complete and confidence and confidence.score >= 0.55:
            continue
        missing_points = list(coverage_row.missing_points if coverage_row else [])
        reasons = list(confidence.reasons if confidence else [])
        if not missing_points and not reasons:
            continue
        unknowns.append(
            Unknown(
                id=f"unknown_{branch.id}",
                branch_id=branch.id,
                description=f"Evidence is incomplete for {branch.title}",
                reason="; ".join(missing_points[:4] + reasons[:3]) or "low reasoning confidence",
                severity="high" if not coverage_row or not coverage_row.complete else "medium",
                focus_terms=_dedupe(branch.required_terms + missing_points + [branch.title])[:10],
            )
        )
    unknowns.extend(_comparative_benchmark_unknowns(plan, claims, source_policy))
    return unknowns


def _readiness_status(*, coverage: CoverageMatrix, weak_claims: list[WeakClaim], unknowns: list[Unknown]) -> str:
    if not coverage.complete and unknowns:
        return "needs_more_research"
    if weak_claims:
        return "needs_caveats_or_more_support"
    return "ready_for_synthesis"


def _deferred_actions(claims: list[dict[str, Any]]) -> list[NextAction]:
    if not _has_numeric_or_table_claim(claims):
        return []
    return [
        NextAction(
            action="analyze_with_python",
            rationale="Numeric, table, benchmark, price, or unit-heavy claims were detected; Python analysis is deferred in Phase 1.",
            priority="medium",
            deferred=True,
        )
    ]


def _comparative_benchmark_unknowns(
    plan: ResearchPlan,
    claims: list[dict[str, Any]],
    source_policy: dict[str, Any],
) -> list[Unknown]:
    policy = source_policy.get("policy", source_policy) if isinstance(source_policy, dict) else {}
    task_type = str(policy.get("task_type") or source_policy.get("task_type") or "")
    plan_text = " ".join(
        [
            plan.question,
            " ".join(plan.report_outline),
            " ".join(plan.acceptance_criteria),
            " ".join(f"{branch.title} {branch.objective} {' '.join(branch.required_terms)}" for branch in plan.branches),
        ]
    )
    if task_type != "comparative_benchmark" and not (
        term_matches(plan_text, {"compare", "rank", "benchmark", "evaluate"})
        and term_matches(plan_text, {"best", "strong", "performance", "score"})
    ):
        return []
    claim_text = " ".join(str(claim.get("claim") or "") for claim in claims)
    has_evaluation_basis = term_matches(
        claim_text,
        {"benchmark", "evaluation", "score", "metric", "leaderboard", "measurement", "test"},
    )
    has_comparison_dimension = term_matches(
        claim_text,
        {"cost", "latency", "memory", "context", "accuracy", "reasoning", "deployment", "performance"},
    )
    if has_evaluation_basis and has_comparison_dimension:
        return []
    branch_id = plan.branches[0].id if plan.branches else "comparison"
    return [
        Unknown(
            id="unknown_comparative_benchmark_basis",
            branch_id=branch_id,
            description="Comparative decision lacks enough like-for-like benchmark or evaluation evidence.",
            reason="The evidence should identify comparable criteria, measurements, or benchmark results before ranking options.",
            severity="high",
            focus_terms=[
                "benchmark comparison",
                "evaluation metrics",
                "leaderboard results",
                "cost latency context reasoning deployment",
            ],
        )
    ]


def _confidence_for_branch(rows: list[SectionConfidence], branch_id: str) -> float:
    for row in rows:
        if row.branch_id == branch_id:
            return row.score
    return 0.0


def _important_single_source_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        claim
        for claim in claims
        if claim.get("high_impact") and int(claim.get("support_count", 0) or 0) < 2 and claim.get("weak")
    ]


def _claims_need_contradiction_check(claims: list[dict[str, Any]]) -> bool:
    return any(claim.get("high_impact") and not claim.get("weak") for claim in claims)


def _has_numeric_or_table_claim(claims: list[dict[str, Any]]) -> bool:
    return any(
        re.search(
            r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|usd|eur|gbp|\$|kw|hp|rpm|mm|kg|score|rate|price|cost)\b",
            str(claim.get("claim") or ""),
            flags=re.I,
        )
        for claim in claims
    )


def _acquisition_exhausted_or_plateaued(metrics: dict[str, Any]) -> bool:
    candidate_count = int(metrics.get("candidate_count_total", metrics.get("candidate_count", 0)) or 0)
    candidate_budget = int(metrics.get("max_candidates", 0) or 0)
    search_count = int(metrics.get("search_count", 0) or 0)
    search_budget = int(metrics.get("max_search_queries", 0) or 0)
    return bool(
        metrics.get("acquisition_time_budget_exhausted")
        or (search_budget > 0 and search_count >= search_budget)
        or (candidate_budget > 0 and candidate_count >= candidate_budget)
        or int(metrics.get("last_acquire_added_sources", 1) or 0) <= 0
        and int(metrics.get("last_acquire_searches", 1) or 0) <= 0
    )


def _branch_ids(rows: list[dict[str, Any]]) -> list[str]:
    return _dedupe([str(row.get("branch_id") or "") for row in rows if str(row.get("branch_id") or "").strip()])


def _focus_terms(rows: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for row in rows:
        terms.extend(str(value) for value in row.get("focus_terms", []) if str(value).strip())
        if row.get("claim"):
            terms.append(str(row.get("claim")))
    return _dedupe(terms)


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result

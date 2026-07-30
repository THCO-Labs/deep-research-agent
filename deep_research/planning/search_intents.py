from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from deep_research.models.model_router import model_for_role
from deep_research.core.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2
from deep_research.core.settings import Settings
from deep_research.synthesis.synthesis_formatting import json_dumps
from deep_research.synthesis.synthesis_runtime import _invoke_with_synthesis_budget, _synthesis_model_spec

MAX_INTENTS_TOTAL = 16
MAX_INTENTS_PER_BRANCH = 4
MAX_PLAN_ADDED_BRANCHES = 1
GENERIC_QUERY_TOKENS = {
    "research",
    "evidence",
    "source",
    "sources",
    "information",
    "overview",
    "background",
    "analysis",
    "report",
}


@dataclass(frozen=True)
class SearchIntent:
    id: str
    branch_id: str
    gap: str
    query: str
    expected_evidence: str
    success_criteria: str
    source_preference: str
    priority: str
    origin: str
    rationale: str
    claim_ids: list[str] | None = None
    source_ids: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["claim_ids"] = list(self.claim_ids or [])
        payload["source_ids"] = list(self.source_ids or [])
        return payload


@dataclass(frozen=True)
class SearchIntentResult:
    intent_id: str
    branch_id: str
    query: str
    status: str
    accepted_source_ids: list[int]
    evidence_card_ids: list[int]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_search_intents_with_model(
    *,
    plan: ResearchPlan,
    coverage: CoverageMatrix,
    reasoning_state: dict[str, Any],
    evidence_graph: dict[str, Any],
    source_policy: dict[str, Any],
    settings: Settings,
    writing_guidance: str = "",
) -> tuple[list[SearchIntent], dict[str, Any]]:
    fallback = fallback_search_intents(
        plan=plan,
        coverage=coverage,
        reasoning_state=reasoning_state,
        source_policy=source_policy,
    )
    if not settings.llm_synthesis:
        return fallback, {"applied": False, "reason": "llm_disabled"}
    model_spec = _synthesis_model_spec(settings, plan, writing_guidance)
    model = model_for_role(settings, "orchestrator", model_spec)
    if not isinstance(model, BaseChatModel):
        return fallback, {"applied": False, "reason": f"orchestrator role is not chat model: {model!r}"}
    prompt = _search_intent_prompt(
        plan=plan,
        coverage=coverage,
        reasoning_state=reasoning_state,
        evidence_graph=evidence_graph,
        source_policy=source_policy,
    )
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        payload, repaired = _loads_json_object_with_repair(
            str(response.content),
            model=model,
            settings=settings,
            model_spec=model_spec,
            schema_name="search intents",
        )
    except Exception as exc:
        return fallback, {"applied": False, "reason": f"{type(exc).__name__}: {exc}", "fallback_count": len(fallback)}
    intents = _validate_and_dedupe_intents(
        payload.get("search_intents", []),
        plan=plan,
        evidence_graph=evidence_graph,
        origin="llm",
    )
    if not intents:
        return fallback, {
            "applied": False,
            "reason": "llm_produced_no_valid_intents",
            "model_spec": model_spec,
            "json_repair_applied": repaired,
            "fallback_count": len(fallback),
        }
    return intents, {
        "applied": True,
        "reason": "llm_search_intents_clamped",
        "model_spec": model_spec,
        "json_repair_applied": repaired,
        "proposed_count": _list_len(payload.get("search_intents")),
        "accepted_count": len(intents),
    }


def fallback_search_intents(
    *,
    plan: ResearchPlan,
    coverage: CoverageMatrix,
    reasoning_state: dict[str, Any],
    source_policy: dict[str, Any],
) -> list[SearchIntent]:
    rows: list[dict[str, Any]] = []
    rows.extend(row for row in reasoning_state.get("unknowns", []) if isinstance(row, dict))
    rows.extend(row for row in reasoning_state.get("weak_claims", []) if isinstance(row, dict))
    if not rows:
        coverage_by_branch = {row.branch_id: row for row in coverage.branches}
        for branch in plan.branches:
            coverage_row = coverage_by_branch.get(branch.id)
            if coverage_row and coverage_row.complete:
                continue
            rows.append(
                {
                    "branch_id": branch.id,
                    "description": f"Missing evidence for {branch.title}",
                    "reason": "; ".join((coverage_row.missing_points if coverage_row else [])[:3]),
                    "focus_terms": branch.required_terms[:6] + branch.queries[:2],
                }
            )
    return _validate_and_dedupe_intents(
        [_fallback_intent_row(row, plan=plan, source_policy=source_policy, index=index) for index, row in enumerate(rows, start=1)],
        plan=plan,
        evidence_graph={},
        origin="deterministic_fallback",
    )


def evaluate_search_intent_results_with_model(
    *,
    intents: list[SearchIntent],
    sources: list[SourceRecordV2],
    evidence_cards: list[EvidenceCard],
    settings: Settings,
    plan: ResearchPlan,
    writing_guidance: str = "",
) -> tuple[list[SearchIntentResult], dict[str, Any]]:
    fallback = fallback_search_intent_results(intents=intents, sources=sources, evidence_cards=evidence_cards)
    if not intents or not settings.llm_synthesis:
        return fallback, {"applied": False, "reason": "llm_disabled_or_no_intents"}
    model_spec = _synthesis_model_spec(settings, plan, writing_guidance)
    model = model_for_role(settings, "orchestrator", model_spec)
    if not isinstance(model, BaseChatModel):
        return fallback, {"applied": False, "reason": f"orchestrator role is not chat model: {model!r}"}
    prompt = _intent_results_prompt(intents=intents, sources=sources, evidence_cards=evidence_cards)
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        payload, repaired = _loads_json_object_with_repair(
            str(response.content),
            model=model,
            settings=settings,
            model_spec=model_spec,
            schema_name="search intent results",
        )
    except Exception as exc:
        return fallback, {"applied": False, "reason": f"{type(exc).__name__}: {exc}", "fallback_count": len(fallback)}
    results = _validate_intent_results(
        payload.get("intent_results", []),
        intents=intents,
        sources=sources,
        evidence_cards=evidence_cards,
        fallback=fallback,
    )
    return results, {
        "applied": True,
        "reason": "llm_search_intent_result_evaluation_clamped",
        "model_spec": model_spec,
        "json_repair_applied": repaired,
        "proposed_count": _list_len(payload.get("intent_results")),
        "accepted_count": len(results),
    }


def fallback_search_intent_results(
    *,
    intents: list[SearchIntent],
    sources: list[SourceRecordV2],
    evidence_cards: list[EvidenceCard],
) -> list[SearchIntentResult]:
    sources_by_intent: dict[str, list[int]] = {}
    for source in sources:
        intent_id = str(source.metadata.get("search_intent_id") or "")
        if intent_id:
            sources_by_intent.setdefault(intent_id, []).append(source.id)
    cards_by_source: dict[int, list[int]] = {}
    for card in evidence_cards:
        cards_by_source.setdefault(card.source_id, []).append(card.id)
    rows: list[SearchIntentResult] = []
    for intent in intents:
        source_ids = sorted(set(sources_by_intent.get(intent.id, [])))
        card_ids = sorted({card_id for source_id in source_ids for card_id in cards_by_source.get(source_id, [])})
        status = "satisfied" if card_ids else ("partially_satisfied" if source_ids else "unsatisfied")
        rows.append(
            SearchIntentResult(
                intent_id=intent.id,
                branch_id=intent.branch_id,
                query=intent.query,
                status=status,
                accepted_source_ids=source_ids,
                evidence_card_ids=card_ids,
                rationale=(
                    "Accepted evidence cards exist for this intent."
                    if card_ids
                    else "Accepted sources exist but no evidence card yet."
                    if source_ids
                    else "No accepted source was tied to this intent."
                ),
            )
        )
    return rows


def apply_plan_revisions_with_model(
    *,
    plan: ResearchPlan,
    reasoning_state: dict[str, Any],
    search_intent_results: list[dict[str, Any]],
    settings: Settings,
    writing_guidance: str = "",
    iteration_count: int = 0,
) -> tuple[ResearchPlan, dict[str, Any]]:
    if iteration_count > 0:
        return plan, {"applied": False, "reason": "replan_iteration_budget_exhausted"}
    unsatisfied = [row for row in search_intent_results if row.get("status") in {"unsatisfied", "partially_satisfied"}]
    if not unsatisfied and not reasoning_state.get("unknowns"):
        return plan, {"applied": False, "reason": "no_unsatisfied_intents_or_unknowns"}
    if not settings.llm_synthesis:
        return plan, {"applied": False, "reason": "llm_disabled"}
    model_spec = _synthesis_model_spec(settings, plan, writing_guidance)
    model = model_for_role(settings, "orchestrator", model_spec)
    if not isinstance(model, BaseChatModel):
        return plan, {"applied": False, "reason": f"orchestrator role is not chat model: {model!r}"}
    prompt = _plan_revision_prompt(
        plan=plan,
        reasoning_state=reasoning_state,
        search_intent_results=search_intent_results,
    )
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        payload, repaired = _loads_json_object_with_repair(
            str(response.content),
            model=model,
            settings=settings,
            model_spec=model_spec,
            schema_name="plan revisions",
        )
    except Exception as exc:
        return plan, {"applied": False, "reason": f"{type(exc).__name__}: {exc}"}
    revised, applied = _apply_plan_revisions(plan, payload)
    metadata = {
        "applied": bool(applied),
        "reason": "llm_plan_revisions_clamped" if applied else "no_valid_plan_revisions",
        "model_spec": model_spec,
        "json_repair_applied": repaired,
        "applied_revisions": applied,
        "proposed": payload,
    }
    return revised, metadata


def _search_intent_prompt(
    *,
    plan: ResearchPlan,
    coverage: CoverageMatrix,
    reasoning_state: dict[str, Any],
    evidence_graph: dict[str, Any],
    source_policy: dict[str, Any],
) -> str:
    return f"""You are choosing targeted follow-up searches for a deep research agent.

Return only JSON. Do not reveal hidden chain-of-thought.

Goal:
- Create specific search intents that directly address evidence gaps.
- Do not repeat the user's broad question.
- Each intent must say the exact evidence it expects to find.
- Use only existing branch IDs, claim IDs, and source IDs shown here.
- Do not invent facts, branch IDs, claim IDs, source IDs, or sources.

Question:
{plan.question}

Branches:
{_branch_lines(plan)}

Coverage:
{json_dumps(coverage.to_dict())}

Reasoning state:
{json_dumps(_compact_reasoning(reasoning_state))}

Source policy:
{json_dumps(_compact_source_policy(source_policy))}

Evidence graph:
{json_dumps(_compact_evidence_graph(evidence_graph))}

Return this schema:
{{
  "search_intents": [
    {{
      "branch_id": "existing branch id",
      "gap": "specific missing or weak evidence",
      "query": "specific web search query",
      "expected_evidence": "exact evidence expected, such as official spec table, benchmark, trial result, cost data, contrary source",
      "success_criteria": "how the agent will know the gap is satisfied",
      "source_preference": "preferred source type",
      "priority": "high|medium|low",
      "claim_ids": ["optional existing claim id"],
      "source_ids": [1],
      "rationale": "short visible reason"
    }}
  ]
}}
"""


def _intent_results_prompt(
    *,
    intents: list[SearchIntent],
    sources: list[SourceRecordV2],
    evidence_cards: list[EvidenceCard],
) -> str:
    return f"""Evaluate whether targeted search intents were satisfied.

Return only JSON. Do not reveal hidden chain-of-thought.
Use only the source IDs and evidence card IDs shown here. Do not invent IDs.
Mark an intent satisfied only when accepted evidence cards directly address its expected evidence.

Search intents:
{json_dumps({"search_intents": [intent.to_dict() for intent in intents]})}

Accepted sources:
{json_dumps({"sources": [_compact_source(source) for source in sources]})}

Evidence cards:
{json_dumps({"evidence_cards": [_compact_card(card) for card in evidence_cards]})}

Return this schema:
{{
  "intent_results": [
    {{
      "intent_id": "existing intent id",
      "status": "satisfied|partially_satisfied|unsatisfied",
      "accepted_source_ids": [1],
      "evidence_card_ids": [1],
      "rationale": "short visible reason"
    }}
  ]
}}
"""


def _plan_revision_prompt(
    *,
    plan: ResearchPlan,
    reasoning_state: dict[str, Any],
    search_intent_results: list[dict[str, Any]],
) -> str:
    return f"""Suggest additive plan revisions for a deep research agent.

Return only JSON. Do not reveal hidden chain-of-thought.

Rules:
- You may add branch queries, required terms, completion criteria, raise source targets, or add one missing branch.
- You may not delete branches, rename existing branch IDs, invent source IDs, or rewrite the whole plan.
- Keep revisions minimal and grounded in current reasoning gaps.

Current plan:
{json_dumps(plan.to_dict())}

Reasoning state:
{json_dumps(_compact_reasoning(reasoning_state))}

Search intent results:
{json_dumps({"search_intent_results": search_intent_results[:20]})}

Return this schema:
{{
  "add_queries": [{{"branch_id": "existing branch id", "queries": ["specific query"]}}],
  "add_required_terms": [{{"branch_id": "existing branch id", "terms": ["term"]}}],
  "add_completion_criteria": [{{"branch_id": "existing branch id", "criteria": ["criterion"]}}],
  "raise_min_sources": [{{"branch_id": "existing branch id", "min_sources": 3}}],
  "add_branch": {{"title": "optional new branch", "objective": "why needed", "queries": ["specific query"], "required_terms": ["term"], "completion_criteria": ["criterion"], "min_sources": 1}}
}}
"""


def _validate_and_dedupe_intents(
    rows: Any,
    *,
    plan: ResearchPlan,
    evidence_graph: dict[str, Any],
    origin: str,
) -> list[SearchIntent]:
    if not isinstance(rows, list):
        return []
    branch_ids = {branch.id for branch in plan.branches}
    claim_ids = {str(row.get("id")) for row in evidence_graph.get("claims", []) if isinstance(row, dict)}
    source_ids = {
        int(value)
        for row in evidence_graph.get("sources", [])
        if isinstance(row, dict)
        for value in [row.get("id")]
        if isinstance(value, int)
    }
    counts: dict[str, int] = {}
    accepted: list[SearchIntent] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        branch_id = str(row.get("branch_id") or "")
        if branch_id not in branch_ids:
            continue
        query = _trim(str(row.get("query") or ""), 220)
        expected = _trim(str(row.get("expected_evidence") or ""), 260)
        if not _valid_query(query) or not expected:
            continue
        if counts.get(branch_id, 0) >= MAX_INTENTS_PER_BRANCH:
            continue
        if any(_near_duplicate(query, intent.query) for intent in accepted):
            continue
        raw_claim_ids = _string_list(row.get("claim_ids"))
        raw_source_ids = _int_list(row.get("source_ids"))
        intent = SearchIntent(
            id=f"intent_{len(accepted) + 1}",
            branch_id=branch_id,
            gap=_trim(str(row.get("gap") or row.get("description") or expected), 260),
            query=query,
            expected_evidence=expected,
            success_criteria=_trim(str(row.get("success_criteria") or expected), 260),
            source_preference=_trim(str(row.get("source_preference") or ""), 120),
            priority=_priority(row.get("priority")),
            origin=origin,
            rationale=_trim(str(row.get("rationale") or row.get("reason") or ""), 320),
            claim_ids=[claim_id for claim_id in raw_claim_ids if claim_id in claim_ids],
            source_ids=[source_id for source_id in raw_source_ids if source_id in source_ids],
        )
        accepted.append(intent)
        counts[branch_id] = counts.get(branch_id, 0) + 1
        if len(accepted) >= MAX_INTENTS_TOTAL:
            break
    return accepted


def _validate_intent_results(
    rows: Any,
    *,
    intents: list[SearchIntent],
    sources: list[SourceRecordV2],
    evidence_cards: list[EvidenceCard],
    fallback: list[SearchIntentResult],
) -> list[SearchIntentResult]:
    if not isinstance(rows, list):
        return fallback
    intent_by_id = {intent.id: intent for intent in intents}
    source_ids = {source.id for source in sources}
    card_ids = {card.id for card in evidence_cards}
    fallback_by_id = {row.intent_id: row for row in fallback}
    results: list[SearchIntentResult] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        intent_id = str(row.get("intent_id") or "")
        intent = intent_by_id.get(intent_id)
        if intent is None or intent_id in seen:
            continue
        accepted_source_ids = [source_id for source_id in _int_list(row.get("accepted_source_ids")) if source_id in source_ids]
        evidence_card_ids = [card_id for card_id in _int_list(row.get("evidence_card_ids")) if card_id in card_ids]
        status = str(row.get("status") or "unsatisfied")
        if status not in {"satisfied", "partially_satisfied", "unsatisfied"}:
            status = "unsatisfied"
        if status == "satisfied" and not evidence_card_ids:
            status = "partially_satisfied" if accepted_source_ids else "unsatisfied"
        results.append(
            SearchIntentResult(
                intent_id=intent.id,
                branch_id=intent.branch_id,
                query=intent.query,
                status=status,
                accepted_source_ids=accepted_source_ids,
                evidence_card_ids=evidence_card_ids,
                rationale=_trim(str(row.get("rationale") or ""), 320),
            )
        )
        seen.add(intent_id)
    for intent in intents:
        if intent.id not in seen and intent.id in fallback_by_id:
            results.append(fallback_by_id[intent.id])
    return results


def _fallback_intent_row(
    row: dict[str, Any],
    *,
    plan: ResearchPlan,
    source_policy: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    branch_id = str(row.get("branch_id") or "")
    branch = next((branch for branch in plan.branches if branch.id == branch_id), None)
    focus = _string_list(row.get("focus_terms"))[:5]
    if not focus and branch is not None:
        focus = branch.required_terms[:5] + branch.queries[:1]
    gap = _trim(str(row.get("description") or row.get("claim") or row.get("reason") or "missing evidence"), 220)
    query_terms = " ".join(_dedupe([plan.question, branch.title if branch else branch_id, " ".join(focus), gap]))
    source_pref = _source_preference(source_policy)
    return {
        "branch_id": branch_id,
        "gap": gap,
        "query": _trim(query_terms, 220),
        "expected_evidence": _trim(str(row.get("reason") or gap), 220),
        "success_criteria": "Find accepted source text and evidence cards that directly address the gap.",
        "source_preference": source_pref,
        "priority": "high" if str(row.get("severity") or "").lower() == "high" else "medium",
        "rationale": f"Deterministic fallback intent for unresolved reasoning gap {index}.",
    }


def _apply_plan_revisions(plan: ResearchPlan, payload: dict[str, Any]) -> tuple[ResearchPlan, list[dict[str, Any]]]:
    branches = [replace(branch) for branch in plan.branches]
    by_id = {branch.id: branch for branch in branches}
    applied: list[dict[str, Any]] = []
    for row in payload.get("add_queries", []) if isinstance(payload.get("add_queries"), list) else []:
        branch = by_id.get(str(row.get("branch_id") or "")) if isinstance(row, dict) else None
        values = _string_list(row.get("queries")) if isinstance(row, dict) else []
        if branch and values:
            by_id[branch.id] = replace(branch, queries=_dedupe(branch.queries + values)[:20])
            applied.append({"type": "add_queries", "branch_id": branch.id, "count": len(values)})
    for row in payload.get("add_required_terms", []) if isinstance(payload.get("add_required_terms"), list) else []:
        branch = by_id.get(str(row.get("branch_id") or "")) if isinstance(row, dict) else None
        values = _string_list(row.get("terms")) if isinstance(row, dict) else []
        if branch and values:
            by_id[branch.id] = replace(branch, required_terms=_dedupe(branch.required_terms + values)[:30])
            applied.append({"type": "add_required_terms", "branch_id": branch.id, "count": len(values)})
    for row in payload.get("add_completion_criteria", []) if isinstance(payload.get("add_completion_criteria"), list) else []:
        branch = by_id.get(str(row.get("branch_id") or "")) if isinstance(row, dict) else None
        values = _string_list(row.get("criteria")) if isinstance(row, dict) else []
        if branch and values:
            by_id[branch.id] = replace(branch, completion_criteria=_dedupe(branch.completion_criteria + values)[:20])
            applied.append({"type": "add_completion_criteria", "branch_id": branch.id, "count": len(values)})
    for row in payload.get("raise_min_sources", []) if isinstance(payload.get("raise_min_sources"), list) else []:
        branch = by_id.get(str(row.get("branch_id") or "")) if isinstance(row, dict) else None
        target = int(row.get("min_sources", 0) or 0) if isinstance(row, dict) else 0
        if branch and target > branch.min_sources:
            by_id[branch.id] = replace(branch, min_sources=min(target, 6))
            applied.append({"type": "raise_min_sources", "branch_id": branch.id, "min_sources": min(target, 6)})
    branches = [by_id[branch.id] for branch in branches]
    add_branch = payload.get("add_branch")
    if isinstance(add_branch, dict) and str(add_branch.get("title") or "").strip():
        branch_id = _new_branch_id(add_branch.get("title"), branches)
        branches.append(
            ResearchBranch(
                id=branch_id,
                title=_trim(str(add_branch.get("title") or ""), 90),
                objective=_trim(str(add_branch.get("objective") or ""), 220),
                queries=_string_list(add_branch.get("queries"))[:8],
                source_types=[],
                min_sources=max(1, min(int(add_branch.get("min_sources", 1) or 1), 4)),
                required_terms=_string_list(add_branch.get("required_terms"))[:12],
                completion_criteria=_string_list(add_branch.get("completion_criteria"))[:8],
            )
        )
        applied.append({"type": "add_branch", "branch_id": branch_id})
    return replace(plan, branches=branches[: len(plan.branches) + MAX_PLAN_ADDED_BRANCHES]), applied


def _compact_reasoning(reasoning_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "readiness_status": reasoning_state.get("readiness_status"),
        "summary": reasoning_state.get("summary", {}),
        "weak_claims": reasoning_state.get("weak_claims", [])[:20],
        "unknowns": reasoning_state.get("unknowns", [])[:20],
        "contradictions": reasoning_state.get("contradictions", [])[:12],
        "model_recommended_action": reasoning_state.get("model_recommended_action", {}),
    }


def _compact_evidence_graph(evidence_graph: dict[str, Any]) -> dict[str, Any]:
    return {
        "claims": [
            {
                "id": row.get("id"),
                "branch_id": row.get("branch_id"),
                "claim": row.get("claim"),
                "source_ids": row.get("source_ids", []),
                "evidence_card_ids": row.get("evidence_card_ids", []),
                "weak": row.get("weak"),
                "high_impact": row.get("high_impact"),
                "weakness_reasons": row.get("weakness_reasons", []),
            }
            for row in evidence_graph.get("claims", [])[:60]
            if isinstance(row, dict)
        ],
        "sources": evidence_graph.get("sources", [])[:80],
        "metrics": evidence_graph.get("metrics", {}),
    }


def _compact_source_policy(source_policy: dict[str, Any]) -> dict[str, Any]:
    policy = source_policy.get("policy", {}) if isinstance(source_policy.get("policy"), dict) else source_policy
    return {
        "label": source_policy.get("label") or policy.get("label"),
        "task_type": source_policy.get("task_type") or policy.get("task_type"),
        "preferred_source_types": policy.get("preferred_source_types", [])[:12],
        "acceptable_source_types": policy.get("acceptable_source_types", [])[:12],
        "low_trust_source_types": policy.get("low_trust_source_types", [])[:12],
    }


def _compact_source(source: SourceRecordV2) -> dict[str, Any]:
    return {
        "id": source.id,
        "branch_id": source.branch_id,
        "title": source.title[:180],
        "url": source.url,
        "quality_type": source.quality_type,
        "quality_score": source.quality_score,
        "metadata": {
            "search_intent_id": source.metadata.get("search_intent_id"),
            "search_intent_expected_evidence": source.metadata.get("search_intent_expected_evidence"),
        },
    }


def _compact_card(card: EvidenceCard) -> dict[str, Any]:
    return {
        "id": card.id,
        "branch_id": card.branch_id,
        "source_id": card.source_id,
        "claim": card.claim[:240],
        "supporting_excerpt": card.supporting_excerpt[:360],
        "confidence": card.semantic_score if card.semantic_score is not None else card.confidence,
    }


def _branch_lines(plan: ResearchPlan) -> str:
    return "\n".join(
        f"- {branch.id}: {branch.title}; objective={branch.objective}; required_terms={branch.required_terms[:10]}"
        for branch in plan.branches
    )


def _valid_query(query: str) -> bool:
    tokens = _token_set(query)
    return len(tokens) >= 3 and len(tokens - GENERIC_QUERY_TOKENS) >= 2


def _near_duplicate(left: str, right: str) -> bool:
    left_tokens = _token_set(left)
    right_tokens = _token_set(right)
    if not left_tokens or not right_tokens:
        return False
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1) >= 0.82


def _token_set(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}", value.lower()) if token not in {"the", "and", "for", "with"}}


def _source_preference(source_policy: dict[str, Any]) -> str:
    policy = source_policy.get("policy", {}) if isinstance(source_policy.get("policy"), dict) else source_policy
    preferred = policy.get("preferred_source_types", [])
    if isinstance(preferred, list) and preferred:
        return ", ".join(str(value) for value in preferred[:4])
    return str(source_policy.get("label") or "task-appropriate sources")


def _priority(value: Any) -> str:
    priority = str(value or "medium").lower()
    return priority if priority in {"high", "medium", "low"} else "medium"


def _loads_json_object_with_repair(
    text: str,
    *,
    model: BaseChatModel,
    settings: Settings,
    model_spec: str,
    schema_name: str,
) -> tuple[dict[str, Any], bool]:
    try:
        return _loads_json_object(text, schema_name=schema_name), False
    except Exception:
        response = _invoke_with_synthesis_budget(
            model,
            prompt=_json_repair_prompt(text, schema_name=schema_name),
            settings=settings,
            model_spec=model_spec,
        )
        return _loads_json_object(str(response.content), schema_name=schema_name), True


def _loads_json_object(text: str, *, schema_name: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError(f"{schema_name} response did not return a JSON object")
    return payload


def _json_repair_prompt(raw_text: str, *, schema_name: str) -> str:
    return f"""Convert this malformed {schema_name} response into valid JSON.

Return JSON only. Do not add facts, notes, IDs, markdown, comments, or explanation.

Malformed response:
{raw_text[:12000]}
"""


def _new_branch_id(title: Any, branches: list[ResearchBranch]) -> str:
    existing = {branch.id for branch in branches}
    base = "branch_" + re.sub(r"[^a-zA-Z0-9]+", "_", str(title).strip().lower()).strip("_")[:36]
    base = base.strip("_") or f"branch_{len(branches) + 1}"
    candidate = base
    index = 2
    while candidate in existing:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_trim(str(item), 240) for item in value if str(item).strip()]


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _trim(value: str, max_chars: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    boundary = cleaned.rfind(" ", 0, max_chars)
    return cleaned[: boundary if boundary > max_chars // 2 else max_chars].strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = _trim(str(value), 240)
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result

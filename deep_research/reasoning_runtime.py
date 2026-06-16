from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from deep_research.model_router import model_for_role
from deep_research.schemas import EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.settings import Settings
from deep_research.synthesis_formatting import json_dumps
from deep_research.synthesis_runtime import _invoke_with_synthesis_budget, _synthesis_model_spec


MAX_REASONING_REFINEMENT_CLAIMS = 80


def refine_reasoning_state_with_model(
    *,
    reasoning_state: dict[str, Any],
    evidence_graph: dict[str, Any],
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    settings: Settings,
    writing_guidance: str = "",
) -> dict[str, Any]:
    if not settings.llm_synthesis or not evidence_cards:
        return _with_refinement_status(reasoning_state, applied=False, reason="llm_disabled_or_no_evidence")
    model_spec = _synthesis_model_spec(settings, plan, writing_guidance)
    model = model_for_role(settings, "orchestrator", model_spec)
    if not isinstance(model, BaseChatModel):
        return _with_refinement_status(
            reasoning_state,
            applied=False,
            reason=f"orchestrator role did not resolve to a chat model: {model!r}",
        )
    prompt = _reasoning_refinement_prompt(
        reasoning_state=reasoning_state,
        evidence_graph=evidence_graph,
        plan=plan,
        evidence_cards=evidence_cards,
        sources=sources,
        writing_guidance=writing_guidance,
    )
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        payload, json_repair_applied = _loads_json_object_with_repair(
            str(response.content),
            model=model,
            settings=settings,
            model_spec=model_spec,
        )
    except Exception as exc:
        return _with_refinement_status(reasoning_state, applied=False, reason=f"{type(exc).__name__}: {exc}")
    refined = _apply_reasoning_refinement(
        reasoning_state=reasoning_state,
        payload=payload,
        evidence_graph=evidence_graph,
        plan=plan,
    )
    metadata = dict(refined.get("model_refinement", {}) or {})
    metadata.update(
        {
            "applied": True,
            "reason": "llm_reasoning_refinement_clamped",
            "model_spec": model_spec,
            "json_repair_applied": json_repair_applied,
            "proposed_weak_claims": _list_len(payload.get("weak_claims")),
            "proposed_unknowns": _list_len(payload.get("unknowns")),
            "proposed_contradictions": _list_len(payload.get("contradictions")),
        }
    )
    refined["model_refinement"] = metadata
    return refined


def _apply_reasoning_refinement(
    *,
    reasoning_state: dict[str, Any],
    payload: dict[str, Any],
    evidence_graph: dict[str, Any],
    plan: ResearchPlan,
) -> dict[str, Any]:
    refined = json.loads(json.dumps(reasoning_state))
    claims = {str(row.get("id")): row for row in evidence_graph.get("claims", []) if isinstance(row, dict)}
    branches = {branch.id for branch in plan.branches}
    weak_claims = list(refined.get("weak_claims", []) if isinstance(refined.get("weak_claims"), list) else [])
    weak_ids = {str(row.get("claim_id")) for row in weak_claims if isinstance(row, dict)}
    for row in payload.get("weak_claims", []) if isinstance(payload.get("weak_claims"), list) else []:
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("claim_id") or "")
        claim = claims.get(claim_id)
        if claim is None or claim_id in weak_ids:
            continue
        weak_claims.append(
            {
                "claim_id": claim_id,
                "branch_id": str(claim.get("branch_id") or ""),
                "claim": str(claim.get("claim") or ""),
                "source_ids": list(claim.get("source_ids", [])),
                "evidence_card_ids": list(claim.get("evidence_card_ids", [])),
                "confidence": float(claim.get("average_confidence", 0.0) or 0.0),
                "reasons": _string_list(row.get("reasons"))[:5],
                "recommended_action": _safe_action(row.get("recommended_action")),
                "model_added": True,
            }
        )
        weak_ids.add(claim_id)
    refined["weak_claims"] = weak_claims

    contradictions = list(refined.get("contradictions", []) if isinstance(refined.get("contradictions"), list) else [])
    for row in payload.get("contradictions", []) if isinstance(payload.get("contradictions"), list) else []:
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("claim_id") or "")
        claim = claims.get(claim_id)
        if claim is None:
            continue
        contradictions.append(
            {
                "id": f"model_contradiction_{len(contradictions) + 1}",
                "claim_id": claim_id,
                "branch_id": str(claim.get("branch_id") or ""),
                "description": _trim(str(row.get("description") or ""), 500),
                "source_ids": _clamped_ints(row.get("source_ids"), set(claim.get("source_ids", []))),
                "confidence": _float(row.get("confidence"), 0.55),
                "needs_caveat": bool(row.get("needs_caveat", True)),
                "model_added": True,
            }
        )
    refined["contradictions"] = contradictions

    unknowns = list(refined.get("unknowns", []) if isinstance(refined.get("unknowns"), list) else [])
    for row in payload.get("unknowns", []) if isinstance(payload.get("unknowns"), list) else []:
        if not isinstance(row, dict):
            continue
        branch_id = str(row.get("branch_id") or "")
        if branch_id not in branches:
            continue
        unknowns.append(
            {
                "id": f"model_unknown_{len(unknowns) + 1}",
                "branch_id": branch_id,
                "description": _trim(str(row.get("description") or ""), 240),
                "reason": _trim(str(row.get("reason") or "model identified gap"), 240),
                "severity": str(row.get("severity") or "medium"),
                "focus_terms": _string_list(row.get("focus_terms"))[:8],
                "model_added": True,
            }
        )
    refined["unknowns"] = unknowns
    refined["missing_evidence"] = unknowns

    next_action = payload.get("next_action")
    if isinstance(next_action, dict):
        refined["model_recommended_action"] = {
            "action": _safe_action(next_action.get("action")),
            "rationale": _trim(str(next_action.get("rationale") or ""), 320),
            "branch_ids": [branch_id for branch_id in _string_list(next_action.get("branch_ids")) if branch_id in branches],
            "focus_terms": _string_list(next_action.get("focus_terms"))[:10],
            "priority": str(next_action.get("priority") or "medium"),
            "deferred": bool(next_action.get("deferred", False)),
        }
    summary = dict(refined.get("summary", {}) or {})
    summary["weak_claim_count"] = len(weak_claims)
    summary["unknown_count"] = len(unknowns)
    summary["contradiction_count"] = len(contradictions)
    refined["summary"] = summary
    return refined


def _reasoning_refinement_prompt(
    *,
    reasoning_state: dict[str, Any],
    evidence_graph: dict[str, Any],
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    writing_guidance: str,
) -> str:
    branch_lines = "\n".join(f"- {branch.id}: {branch.title}; objective: {branch.objective}" for branch in plan.branches)
    source_titles = {source.id: source.title for source in sources}
    card_lines = "\n".join(
        (
            f"- card {card.id}; branch {card.branch_id}; source [{card.source_id}] "
            f"{source_titles.get(card.source_id, card.source_title)}; claim: {card.claim[:220]}; "
            f"excerpt: {card.supporting_excerpt[:260]}"
        )
        for card in evidence_cards[:MAX_REASONING_REFINEMENT_CLAIMS]
    )
    compact_graph = {
        "claims": [
            {
                "id": claim.get("id"),
                "branch_id": claim.get("branch_id"),
                "claim": claim.get("claim"),
                "source_ids": claim.get("source_ids", []),
                "evidence_card_ids": claim.get("evidence_card_ids", []),
                "support_count": claim.get("support_count"),
                "average_confidence": claim.get("average_confidence"),
                "weak": claim.get("weak"),
                "high_impact": claim.get("high_impact"),
                "weakness_reasons": claim.get("weakness_reasons", []),
            }
            for claim in evidence_graph.get("claims", [])[:MAX_REASONING_REFINEMENT_CLAIMS]
            if isinstance(claim, dict)
        ],
        "contradiction_edges": evidence_graph.get("contradiction_edges", [])[:30],
        "metrics": evidence_graph.get("metrics", {}),
    }
    return f"""You are refining the visible reasoning state for a deep research agent.

Return only JSON. Do not reveal hidden chain-of-thought.

Your job:
- Improve weak-claim, unknown, contradiction, and next-action labels.
- Use only existing branch_id, claim_id, evidence_card_ids, and source_ids shown here.
- Do not invent facts, sources, or IDs.
- If a claim needs more support, cite the existing claim_id and explain the evidence gap.
- If evidence suggests tension or a caveat, attach it to an existing claim_id.

Question:
{plan.question}

Branches:
{branch_lines}

Writing guidance:
{writing_guidance.strip()[:1200] if writing_guidance.strip() else "None"}

Current reasoning state:
{json_dumps(_compact_reasoning_state(reasoning_state))}

Evidence graph:
{json_dumps(compact_graph)}

Evidence cards:
{card_lines}

Return this schema:
{{
  "weak_claims": [
    {{"claim_id": "existing claim id", "reasons": ["specific support gap"], "recommended_action": "search_more"}}
  ],
  "unknowns": [
    {{"branch_id": "existing branch id", "description": "missing evidence", "reason": "why it matters", "severity": "high|medium|low", "focus_terms": ["search focus"]}}
  ],
  "contradictions": [
    {{"claim_id": "existing claim id", "description": "tension or opposing evidence to check", "source_ids": [1], "confidence": 0.0, "needs_caveat": true}}
  ],
  "next_action": {{"action": "search_more|contradiction_search|synthesize|stop", "rationale": "short visible reason", "branch_ids": ["existing branch id"], "focus_terms": ["search focus"], "priority": "high|medium|low"}}
}}
"""


def _compact_reasoning_state(reasoning_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "readiness_status": reasoning_state.get("readiness_status"),
        "summary": reasoning_state.get("summary", {}),
        "weak_claims": reasoning_state.get("weak_claims", [])[:25],
        "unknowns": reasoning_state.get("unknowns", [])[:25],
        "contradictions": reasoning_state.get("contradictions", [])[:25],
        "next_recommended_actions": reasoning_state.get("next_recommended_actions", [])[:5],
    }


def _with_refinement_status(reasoning_state: dict[str, Any], *, applied: bool, reason: str) -> dict[str, Any]:
    refined = json.loads(json.dumps(reasoning_state))
    metadata = dict(refined.get("model_refinement", {}) or {})
    metadata.update({"applied": applied, "reason": reason})
    refined["model_refinement"] = metadata
    return refined


def _loads_json_object(text: str) -> dict[str, Any]:
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
        raise ValueError("reasoning refinement did not return a JSON object")
    return payload


def _loads_json_object_with_repair(
    text: str,
    *,
    model: BaseChatModel,
    settings: Settings,
    model_spec: str,
) -> tuple[dict[str, Any], bool]:
    try:
        return _loads_json_object(text), False
    except Exception:
        response = _invoke_with_synthesis_budget(model, prompt=_json_repair_prompt(text), settings=settings, model_spec=model_spec)
        return _loads_json_object(str(response.content)), True


def _json_repair_prompt(raw_text: str) -> str:
    return f"""Convert this malformed reasoning-refinement response into valid JSON.

Return JSON only. Do not add facts, notes, IDs, markdown, comments, or explanation.
Preserve only fields that fit this schema:
{{
  "weak_claims": [{{"claim_id": "claim_1", "reasons": ["reason"], "recommended_action": "search_more"}}],
  "unknowns": [{{"branch_id": "branch_1", "description": "gap", "reason": "why", "severity": "medium", "focus_terms": ["term"]}}],
  "contradictions": [{{"claim_id": "claim_1", "description": "tension", "source_ids": [1], "confidence": 0.5, "needs_caveat": true}}],
  "next_action": {{"action": "search_more", "rationale": "reason", "branch_ids": ["branch_1"], "focus_terms": ["term"], "priority": "medium"}}
}}

Malformed response:
{raw_text[:12000]}
"""


def _safe_action(value: Any) -> str:
    action = str(value or "search_more")
    if action in {"search_more", "scrape_more", "contradiction_search", "analyze_with_python", "synthesize", "repair", "stop"}:
        return action
    return "search_more"


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_trim(str(item), 240) for item in value if str(item).strip()]


def _clamped_ints(value: Any, allowed: set[int]) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number in allowed and number not in result:
            result.append(number)
    return result


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _trim(value: str, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned[:limit]


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0

from __future__ import annotations

from typing import Any


def render_reasoning_summary(
    *,
    query: str,
    source_policy: dict[str, Any],
    reasoning_state: dict[str, Any],
    reasoning_decision: dict[str, Any],
    contradiction_queries: list[dict[str, Any]] | None = None,
    search_intents: list[dict[str, Any]] | None = None,
    search_intent_results: list[dict[str, Any]] | None = None,
    plan_revisions: dict[str, Any] | None = None,
) -> str:
    summary = dict(reasoning_state.get("summary", {}) or {})
    policy = dict(source_policy.get("policy", {}) or source_policy)
    lines = [
        "# Reasoning Summary",
        "",
        f"Original query: {query}",
        "",
        f"- task type: `{reasoning_state.get('task_type') or policy.get('task_type') or 'general'}`",
        f"- selected source policy: `{reasoning_state.get('source_policy_label') or policy.get('label') or 'general_source_policy'}`",
        f"- reasoning decision: `{reasoning_decision.get('action') or summary.get('primary_action') or 'unknown'}`",
        f"- decision rationale: {reasoning_decision.get('rationale') or 'none recorded'}",
        f"- weak claim count: `{summary.get('weak_claim_count', len(reasoning_state.get('weak_claims', []) or []))}`",
        f"- unknown count: `{summary.get('unknown_count', len(reasoning_state.get('unknowns', []) or []))}`",
        f"- contradiction count: `{summary.get('contradiction_count', len(reasoning_state.get('contradictions', []) or []))}`",
        f"- final readiness status: `{reasoning_state.get('readiness_status', 'unknown')}`",
        f"- synthesis allowed or delayed: `{_synthesis_status(reasoning_decision)}`",
        "",
        "## Top Weak Claims",
        "",
    ]
    lines.extend(_numbered(_weak_claim_line(row) for row in reasoning_state.get("weak_claims", [])[:5]))
    lines.extend(["", "## Top Unknowns", ""])
    lines.extend(_numbered(_unknown_line(row) for row in reasoning_state.get("unknowns", [])[:5]))
    lines.extend(["", "## Top Contradictions", ""])
    lines.extend(_numbered(_contradiction_line(row) for row in reasoning_state.get("contradictions", [])[:5]))
    if contradiction_queries:
        lines.extend(["", "## Contradiction Queries", ""])
        lines.extend(_numbered(str(row.get("query") or "") for row in contradiction_queries[:8]))
    if search_intents:
        lines.extend(["", "## Search Intents", ""])
        lines.extend(_numbered(_search_intent_line(row) for row in search_intents[:10]))
    if search_intent_results:
        lines.extend(["", "## Search Intent Results", ""])
        lines.extend(_numbered(_search_intent_result_line(row) for row in search_intent_results[:10]))
    if plan_revisions:
        lines.extend(["", "## Plan Revisions", ""])
        lines.append(f"- applied: `{bool(plan_revisions.get('applied'))}`")
        lines.append(f"- reason: {plan_revisions.get('reason') or 'none'}")
        if plan_revisions.get("applied_revisions"):
            lines.extend(_numbered(str(row) for row in plan_revisions.get("applied_revisions", [])[:8]))
    lines.extend(["", "## Source Policy", ""])
    lines.append(f"- preferred: {', '.join(policy.get('preferred_source_types', [])[:12]) or 'none'}")
    lines.append(f"- acceptable: {', '.join(policy.get('acceptable_source_types', [])[:12]) or 'none'}")
    lines.append(f"- low trust: {', '.join(policy.get('low_trust_source_types', [])[:12]) or 'none'}")
    return "\n".join(lines).rstrip() + "\n"


def reasoning_brief_for_prompt(reasoning_state: dict[str, Any], reasoning_decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "readiness_status": reasoning_state.get("readiness_status"),
        "decision": {
            "action": reasoning_decision.get("action"),
            "rationale": reasoning_decision.get("rationale"),
            "branch_ids": reasoning_decision.get("branch_ids", []),
            "focus_terms": reasoning_decision.get("focus_terms", [])[:8],
        },
        "weak_claims": [
            {
                "claim_id": row.get("claim_id"),
                "branch_id": row.get("branch_id"),
                "claim": row.get("claim"),
                "source_ids": row.get("source_ids", []),
                "reasons": row.get("reasons", [])[:4],
            }
            for row in reasoning_state.get("weak_claims", [])[:8]
            if isinstance(row, dict)
        ],
        "unknowns": [
            {
                "branch_id": row.get("branch_id"),
                "description": row.get("description"),
                "reason": row.get("reason"),
                "focus_terms": row.get("focus_terms", [])[:6],
            }
            for row in reasoning_state.get("unknowns", [])[:8]
            if isinstance(row, dict)
        ],
        "contradictions": [
            {
                "claim_id": row.get("claim_id"),
                "branch_id": row.get("branch_id"),
                "description": row.get("description"),
                "source_ids": row.get("source_ids", []),
                "needs_caveat": row.get("needs_caveat", True),
            }
            for row in reasoning_state.get("contradictions", [])[:8]
            if isinstance(row, dict)
        ],
    }


def _synthesis_status(decision: dict[str, Any]) -> str:
    action = str(decision.get("action") or "")
    if action in {"search_more", "contradiction_search", "scrape_more"}:
        return f"delayed for {action}"
    if action == "stop":
        return "not allowed"
    return "allowed"


def _weak_claim_line(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    reasons = "; ".join(str(item) for item in row.get("reasons", [])[:3])
    return f"`{row.get('claim_id')}` {row.get('claim')} | sources={row.get('source_ids', [])} | {reasons or 'no reason'}"


def _unknown_line(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return f"`{row.get('branch_id')}` {row.get('description')} | {row.get('reason')}"


def _contradiction_line(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return f"`{row.get('claim_id')}` {row.get('description')} | sources={row.get('source_ids', [])}"


def _search_intent_line(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return (
        f"`{row.get('id')}` branch={row.get('branch_id')} | query={row.get('query')} | "
        f"expects={row.get('expected_evidence')}"
    )


def _search_intent_result_line(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return (
        f"`{row.get('intent_id')}` status={row.get('status')} | "
        f"sources={row.get('accepted_source_ids', [])} | cards={row.get('evidence_card_ids', [])}"
    )


def _numbered(values: Any) -> list[str]:
    rows = [str(value).strip() for value in values if str(value).strip()]
    if not rows:
        return ["None recorded."]
    return [f"{index}. {value}" for index, value in enumerate(rows, start=1)]

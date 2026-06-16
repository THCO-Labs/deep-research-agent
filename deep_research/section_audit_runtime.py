from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from deep_research.model_router import model_for_role
from deep_research.schemas import EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.section_writing import audit_report_sections, refine_section_audits_from_payload
from deep_research.settings import Settings
from deep_research.synthesis_runtime import _invoke_with_synthesis_budget, _synthesis_model_spec


def audit_report_sections_with_model(
    *,
    section_plan: dict[str, Any],
    report_markdown: str,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    source_texts: dict[int, str],
    settings: Settings,
    writing_guidance: str = "",
) -> dict[str, Any]:
    deterministic = audit_report_sections(
        section_plan=section_plan,
        report_markdown=report_markdown,
        evidence_cards=evidence_cards,
        sources=sources,
        source_texts=source_texts,
    )
    if not settings.llm_synthesis or not evidence_cards:
        deterministic["llm_review_applied"] = False
        return deterministic
    model_spec = _synthesis_model_spec(settings, plan, writing_guidance)
    model = model_for_role(settings, "verifier", model_spec)
    if not isinstance(model, BaseChatModel):
        deterministic["llm_review_applied"] = False
        deterministic["llm_review_error"] = f"Verifier role did not resolve to a chat model: {model!r}"
        return deterministic
    prompt = _section_audit_prompt(
        plan=plan,
        report_markdown=report_markdown,
        deterministic_audit=deterministic,
        evidence_cards=evidence_cards,
        writing_guidance=writing_guidance,
    )
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        payload = _loads_json_object(str(response.content))
    except Exception as exc:
        deterministic["llm_review_applied"] = False
        deterministic["llm_review_error"] = f"{type(exc).__name__}: {exc}"
        return deterministic
    return refine_section_audits_from_payload(deterministic, payload)


def section_audit_failures(section_audit: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for audit in section_audit.get("audits", []):
        if not isinstance(audit, dict) or audit.get("locked"):
            continue
        section_id = str(audit.get("section_id") or "section")
        heading = str(audit.get("heading") or "").strip()
        label = f"{section_id}" + (f" ({heading})" if heading else "")
        for failure in audit.get("failures", [])[:4]:
            failures.append(f"Section audit failed for {label}: {failure}")
        guidance = str(audit.get("repair_guidance") or "").strip()
        if guidance:
            failures.append(f"Section repair guidance for {label}: {guidance}")
    for guidance in section_audit.get("global_repair_guidance", [])[:6]:
        failures.append(f"Section-level global repair guidance: {guidance}")
    return failures


def _section_audit_prompt(
    *,
    plan: ResearchPlan,
    report_markdown: str,
    deterministic_audit: dict[str, Any],
    evidence_cards: list[EvidenceCard],
    writing_guidance: str,
) -> str:
    card_lines = "\n".join(
        f"- card {card.id}; source [{card.source_id}]; branch {card.branch_id}; claim: {card.claim[:240]}; excerpt: {card.supporting_excerpt[:260]}"
        for card in evidence_cards[:80]
    )
    compact_audit = _compact_audit_for_prompt(deterministic_audit)
    report_excerpt = _compact_report(report_markdown)
    return f"""You are a strict section-level reviewer for a deep research report.

You are not writing hidden reasoning. Return only concise JSON.

Your job:
- Review whether each internal section contract is actually satisfied by the matched report section.
- Use the deterministic audit as a floor. You may add failures or repair guidance, but do not remove deterministic failures.
- Prefer targeted section repairs over whole-report rewrites.
- Do not require rigid headings. Judge whether the visible section naturally fulfills the contract.
- Mark locked=false if the section has unsupported exact numbers, contradictions, weak citations, missing decision logic, or a recommendation that does not follow from cited evidence.

Question:
{plan.question}

Additional writing guidance:
{writing_guidance.strip()[:1600] if writing_guidance.strip() else "None"}

Evidence cards:
{card_lines}

Deterministic section audit:
{json.dumps(compact_audit, ensure_ascii=True, indent=2, sort_keys=True)}

Report excerpt:
{report_excerpt}

Return JSON only:
{{
  "section_reviews": [
    {{
      "section_id": "same section_id from audit",
      "locked": true,
      "failures": ["additional concrete failure, if any"],
      "warnings": ["non-blocking concern"],
      "repair_guidance": "targeted repair instruction for this section"
    }}
  ],
  "global_repair_guidance": ["cross-section consistency repair, if any"]
}}
"""


def _compact_audit_for_prompt(section_audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": section_audit.get("schema_version"),
        "report_mode": section_audit.get("report_mode"),
        "locked_count": section_audit.get("locked_count"),
        "section_count": section_audit.get("section_count"),
        "audits": [
            {
                "section_id": audit.get("section_id"),
                "heading": audit.get("heading"),
                "locked": audit.get("locked"),
                "citation_support_score": audit.get("citation_support_score"),
                "evidence_linkage_score": audit.get("evidence_linkage_score"),
                "cited_source_ids": audit.get("cited_source_ids"),
                "failures": audit.get("failures", [])[:6],
                "warnings": audit.get("warnings", [])[:4],
                "section_excerpt": str(audit.get("matched_report_section_markdown") or "")[:900],
            }
            for audit in section_audit.get("audits", [])
            if isinstance(audit, dict)
        ],
    }


def _compact_report(report_markdown: str, *, limit: int = 6000) -> str:
    text = re.sub(r"\s+", " ", report_markdown).strip()
    if len(text) <= limit:
        return text
    return f"{text[: int(limit * 0.65)]} ... [middle omitted] ... {text[-int(limit * 0.35):]}"


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
        raise ValueError("section audit review did not return a JSON object")
    return payload

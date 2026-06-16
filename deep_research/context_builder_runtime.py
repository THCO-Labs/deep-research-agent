from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from deep_research.context_builder import refine_knowledge_base_from_payload
from deep_research.model_router import model_for_role
from deep_research.schemas import EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.settings import Settings
from deep_research.synthesis_formatting import json_dumps
from deep_research.synthesis_runtime import _invoke_with_synthesis_budget, _synthesis_model_spec

MAX_CONTEXT_REFINEMENT_CARDS = 80
MAX_CONTEXT_REFINEMENT_PACKETS = 14


def refine_knowledge_base_with_model(
    *,
    knowledge_base: dict[str, Any],
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    settings: Settings,
    writing_guidance: str = "",
) -> dict[str, Any]:
    if not settings.llm_synthesis or not evidence_cards:
        return _with_refinement_status(knowledge_base, applied=False, reason="llm_disabled_or_no_evidence")
    model_spec = _synthesis_model_spec(settings, plan, writing_guidance)
    model = model_for_role(settings, "orchestrator", model_spec)
    if not isinstance(model, BaseChatModel):
        return _with_refinement_status(
            knowledge_base,
            applied=False,
            reason=f"orchestrator role did not resolve to a chat model: {model!r}",
        )
    prompt = _context_refinement_prompt(
        plan=plan,
        knowledge_base=knowledge_base,
        evidence_cards=evidence_cards,
        sources=sources,
        writing_guidance=writing_guidance,
    )
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        raw_response = str(response.content)
        payload, json_repair_applied = _loads_json_object_with_repair(
            raw_response,
            model=model,
            settings=settings,
            model_spec=model_spec,
        )
    except Exception as exc:
        return _with_refinement_status(knowledge_base, applied=False, reason=f"{type(exc).__name__}: {exc}")
    refined = refine_knowledge_base_from_payload(
        knowledge_base,
        payload,
        evidence_cards=evidence_cards,
        sources=sources,
    )
    refinement = dict(refined.get("model_refinement", {}) or {})
    refinement.update(
        {
            "applied": True,
            "reason": "llm_refinement_clamped",
            "model_spec": model_spec,
            "json_repair_applied": json_repair_applied,
            "proposed_branches": _list_len(payload.get("branches")),
            "proposed_section_packets": _list_len(payload.get("section_packets")),
        }
    )
    refined["model_refinement"] = refinement
    return refined


def _context_refinement_prompt(
    *,
    plan: ResearchPlan,
    knowledge_base: dict[str, Any],
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    writing_guidance: str,
) -> str:
    source_titles = {source.id: source.title for source in sources}
    card_lines = "\n".join(
        (
            f"- card {card.id}; branch {card.branch_id}; source [{card.source_id}] "
            f"{source_titles.get(card.source_id, card.source_title)}; claim: {card.claim[:240]}; "
            f"excerpt: {card.supporting_excerpt[:320]}"
        )
        for card in evidence_cards[:MAX_CONTEXT_REFINEMENT_CARDS]
    )
    branch_lines = "\n".join(
        f"- {branch.id}: {branch.title}; objective: {branch.objective}"
        for branch in plan.branches
    )
    return f"""You are refining the visible working context for a deep research writer.

Return only JSON. Do not write hidden reasoning.

Your job:
- Improve the deterministic knowledge base with concise focus notes, open questions, and limitations.
- Use only existing branch_id, section_id, evidence_card_ids, and source_ids shown here.
- Do not invent facts. Factual focus notes must be grounded in the listed evidence card claims or excerpts.
- Keep notes useful for a natural, non-template report: argument flow, tensions, comparisons, missing support, and what to preserve.
- Put uncertain or missing support as open_questions, not as facts.

Question:
{plan.question}

Branches:
{branch_lines}

Additional writing guidance:
{writing_guidance.strip()[:1600] if writing_guidance.strip() else "None"}

Deterministic knowledge base:
{json_dumps(_compact_knowledge_base(knowledge_base))}

Available evidence cards:
{card_lines}

Return this schema:
{{
  "branches": [
    {{
      "branch_id": "existing branch id",
      "evidence_card_ids": [1],
      "source_ids": [1],
      "model_focus": ["concise grounded focus note with citations like [1] when factual"],
      "open_questions": ["unsupported but important question or gap"],
      "limitations": ["grounded limitation with citation when factual"]
    }}
  ],
  "section_packets": [
    {{
      "section_id": "existing section id",
      "evidence_card_ids": [1],
      "source_ids": [1],
      "model_focus": ["what this section should argue or compare, grounded in evidence"],
      "open_questions": ["gap the writer should avoid overclaiming"],
      "limitations": ["grounded limitation or caveat"]
    }}
  ]
}}
"""


def _compact_knowledge_base(knowledge_base: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": knowledge_base.get("schema_version"),
        "question": knowledge_base.get("question"),
        "coverage": knowledge_base.get("coverage"),
        "branches": [
            {
                "branch_id": branch.get("branch_id"),
                "title": branch.get("title"),
                "source_ids": branch.get("source_ids", [])[:10],
                "evidence_card_ids": branch.get("evidence_card_ids", [])[:14],
                "notes": branch.get("notes", [])[:6],
                "limitations": branch.get("limitations", [])[:4],
            }
            for branch in knowledge_base.get("branches", [])[:MAX_CONTEXT_REFINEMENT_PACKETS]
            if isinstance(branch, dict)
        ],
        "section_packets": [
            {
                "section_id": packet.get("section_id"),
                "title_hint": packet.get("title_hint"),
                "role": packet.get("role"),
                "source_ids": packet.get("source_ids", [])[:10],
                "evidence_card_ids": packet.get("evidence_card_ids", [])[:10],
                "packet": packet.get("packet", [])[:5],
                "limitations": packet.get("limitations", [])[:4],
            }
            for packet in knowledge_base.get("section_packets", [])[:MAX_CONTEXT_REFINEMENT_PACKETS]
            if isinstance(packet, dict)
        ],
    }


def _with_refinement_status(
    knowledge_base: dict[str, Any],
    *,
    applied: bool,
    reason: str,
) -> dict[str, Any]:
    refined = {
        **knowledge_base,
        "coverage": dict(knowledge_base.get("coverage", {}) or {}),
        "branches": [dict(row) for row in knowledge_base.get("branches", []) if isinstance(row, dict)],
        "section_packets": [dict(row) for row in knowledge_base.get("section_packets", []) if isinstance(row, dict)],
    }
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
        raise ValueError("context refinement did not return a JSON object")
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
        repair_prompt = _json_repair_prompt(text)
        response = _invoke_with_synthesis_budget(model, prompt=repair_prompt, settings=settings, model_spec=model_spec)
        return _loads_json_object(str(response.content)), True


def _json_repair_prompt(raw_text: str) -> str:
    return f"""Convert this malformed context-refinement response into valid JSON.

Return JSON only. Do not add facts, notes, IDs, markdown, comments, or explanation.
Preserve only fields that fit this schema:
{{
  "branches": [
    {{
      "branch_id": "existing branch id",
      "evidence_card_ids": [1],
      "source_ids": [1],
      "model_focus": ["note"],
      "open_questions": ["question"],
      "limitations": ["limitation"]
    }}
  ],
  "section_packets": [
    {{
      "section_id": "existing section id",
      "evidence_card_ids": [1],
      "source_ids": [1],
      "model_focus": ["note"],
      "open_questions": ["question"],
      "limitations": ["limitation"]
    }}
  ]
}}

Malformed response:
{raw_text[:12000]}
"""


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0

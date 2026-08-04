from __future__ import annotations

from typing import Any

from deep_research.core.schemas import EvidenceCard, ResearchPlan
from deep_research.core.settings import Settings


def judge_report_sections_batched(
    *,
    report_markdown: str,
    section_plan: dict[str, Any],
    evidence_cards: list[EvidenceCard],
    plan: ResearchPlan,
    settings: Settings,
    writing_guidance: str = "",
) -> dict[str, Any]:
    """Return an advisory section audit payload.

    The graph gates this behind ``section_batch_judge_enabled`` and treats all
    findings as advisory. This lightweight implementation keeps the import path
    valid and records section/card coverage without adding another LLM pass.
    """
    sections = section_plan.get("sections", []) if isinstance(section_plan, dict) else []
    section_payloads: dict[str, dict[str, Any]] = {}
    for index, section in enumerate(sections):
        if not isinstance(section, dict):
            continue
        section_id = str(section.get("id") or section.get("title_hint") or f"section_{index + 1}")
        card_ids = [int(card_id) for card_id in section.get("evidence_card_ids", []) if str(card_id).isdigit()]
        source_ids = [int(source_id) for source_id in section.get("source_ids", []) if str(source_id).isdigit()]
        section_payloads[section_id] = {
            "locked": True,
            "status": "advisory_only",
            "evidence_card_ids": card_ids,
            "source_ids": source_ids,
            "failures": [],
        }
    return {
        "all_locked": True,
        "locked_count": len(section_payloads),
        "section_count": len(section_payloads),
        "sections": section_payloads,
        "evidence_card_count": len(evidence_cards),
        "question": plan.question,
        "writing_guidance_present": bool(writing_guidance.strip()),
        "report_chars": len(report_markdown),
    }


def batched_section_failures(batch_audit: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    sections = batch_audit.get("sections", {})
    if not isinstance(sections, dict):
        return failures
    for section_id, payload in sections.items():
        if not isinstance(payload, dict):
            continue
        for failure in payload.get("failures", []) or []:
            failures.append(f"{section_id}: {failure}")
    return failures

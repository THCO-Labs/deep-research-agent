from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from deep_research.core.schemas import EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.core.settings import Settings
from deep_research.models.model_router import model_for_role
from deep_research.synthesis.section_writing import AdaptiveSectionPlan
from deep_research.synthesis.synthesis_repair import _normalize_report_markdown
from deep_research.synthesis.synthesis_runtime import _invoke_with_synthesis_budget

MAX_CITATION_CARDS = 120
MAX_EXCERPT_CHARS = 520


def apply_citations(
    *,
    report_markdown: str,
    section_plan: AdaptiveSectionPlan | dict[str, Any] | None,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    plan: ResearchPlan,
    settings: Settings,
    writing_guidance: str = "",
    citation_failures: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Insert source citations in a separate LLM pass."""
    model_spec = getattr(settings, "citation_model", "") or settings.verifier_model
    model = model_for_role(settings, "citation", model_spec)
    diagnostics: dict[str, Any] = {
        "model_spec": model_spec,
        "status": "started",
        "sections": {},
        "card_count": len(evidence_cards),
    }
    if not isinstance(model, BaseChatModel):
        diagnostics["status"] = "skipped"
        diagnostics["error"] = f"Citation role did not resolve to a chat model: {model!r}"
        return _normalize_report_markdown(report_markdown, sources, citations_enabled=True), diagnostics

    prompt = _citation_prompt(
        report_markdown=report_markdown,
        section_plan=section_plan,
        evidence_cards=evidence_cards,
        sources=sources,
        plan=plan,
        writing_guidance=writing_guidance,
        citation_failures=citation_failures or [],
    )
    response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
    content = str(getattr(response, "content", response) or "").strip()
    cited_report = _strip_markdown_fence(content)
    if not cited_report:
        diagnostics["status"] = "empty_response"
        return _normalize_report_markdown(report_markdown, sources, citations_enabled=True), diagnostics

    normalized = _normalize_report_markdown(cited_report, sources, citations_enabled=True)
    diagnostics["status"] = "cited"
    diagnostics["sections"] = _section_diagnostics(normalized)
    return normalized, diagnostics


def _citation_prompt(
    *,
    report_markdown: str,
    section_plan: AdaptiveSectionPlan | dict[str, Any] | None,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    plan: ResearchPlan,
    writing_guidance: str,
    citation_failures: list[str],
) -> str:
    source_ids = ", ".join(str(source.id) for source in sources if source.usable)
    return f"""You are the citation agent for a deep research report.

Task:
- Insert numeric inline citations like [3] into the report.
- Use only the source IDs listed in the evidence cards and allowed source IDs.
- Every factual paragraph must have at least one citation.
- Specific numbers, dates, names, comparisons, and causal claims need citations from evidence cards that directly support them.
- Do not add new facts, change the report's argument, add URLs in the body, or invent source IDs.
- Keep the report in Markdown.
- End with exactly one ## Sources section. Each entry must be exactly: [N] Title: https://url

Allowed source IDs:
{source_ids}

User question:
{plan.question}

Writing guidance:
{writing_guidance.strip() or "(none)"}

Prior citation failures to avoid:
{_format_failures(citation_failures)}

Adaptive section controls:
{_format_section_plan(section_plan)}

Evidence cards:
{_format_cards(evidence_cards)}

Source list:
{_format_sources(sources)}

Citation-free report:
{report_markdown.strip()}

Return only the fully cited Markdown report.
"""


def _format_cards(cards: list[EvidenceCard]) -> str:
    lines: list[str] = []
    ranked = sorted(cards, key=lambda card: (card.confidence, card.quality_score, card.relevance_score), reverse=True)
    for card in ranked[:MAX_CITATION_CARDS]:
        excerpt = re.sub(r"\s+", " ", card.supporting_excerpt).strip()[:MAX_EXCERPT_CHARS]
        claim = re.sub(r"\s+", " ", card.claim).strip()
        lines.append(
            f"- card {card.id}; source [{card.source_id}] {card.source_title}; "
            f"claim: {claim}; excerpt: {excerpt}"
        )
    return "\n".join(lines) if lines else "(no evidence cards)"


def _format_sources(sources: list[SourceRecordV2]) -> str:
    lines = [f"[{source.id}] {source.title}: {source.url}" for source in sources if source.usable]
    return "\n".join(lines) if lines else "(no usable sources)"


def _format_section_plan(section_plan: AdaptiveSectionPlan | dict[str, Any] | None) -> str:
    if section_plan is None:
        return "(none)"
    payload = section_plan.to_dict() if hasattr(section_plan, "to_dict") else section_plan
    sections = payload.get("sections", []) if isinstance(payload, dict) else []
    lines: list[str] = []
    for section in sections[:24]:
        if not isinstance(section, dict):
            continue
        lines.append(
            f"- {section.get('title_hint') or section.get('id')}: "
            f"sources={section.get('source_ids') or []}; cards={section.get('evidence_card_ids') or []}"
        )
    return "\n".join(lines) if lines else "(none)"


def _format_failures(failures: list[str]) -> str:
    if not failures:
        return "(none)"
    return "\n".join(f"- {failure}" for failure in failures[:25])


def _strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _section_diagnostics(report: str) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    current = "opening"
    sections[current] = {"status": "cited", "citation_count": 0}
    for line in report.splitlines():
        heading = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if heading:
            current = heading.group(1).strip()
            sections.setdefault(current, {"status": "cited", "citation_count": 0})
        sections[current]["citation_count"] += len(re.findall(r"\[[0-9]+(?:\s*,\s*[0-9]+)*]", line))
    for payload in sections.values():
        if int(payload["citation_count"]) <= 0:
            payload["status"] = "uncited"
    return sections

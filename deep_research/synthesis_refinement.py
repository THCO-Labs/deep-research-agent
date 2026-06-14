from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.settings import Settings
from deep_research.source_validation import content_terms
from deep_research.synthesis_formatting import (
    _compact_blueprint_for_prompt,
    _compact_target_profile,
    _language_instruction,
    _minimal_blueprint_for_prompt,
    _target_depth_hint,
    json_dumps,
)
from deep_research.synthesis_repair import (
    _append_evidence_coverage_if_needed,
    _clean_malformed_citation_punctuation,
    _normalize_report_markdown,
    _normalize_markdown_headings,
    _numeric_citation_ids,
    _repair_weak_citation_support,
    _remove_existing_source_listing,
    _remove_unknown_numeric_citations,
    _separate_heading_blocks,
    _split_sources,
    _strip_report_chrome_lines,
    _strip_hallucinated_specific_citations,
    _strip_source_artifact_lines,
    _rewrite_low_overlap_cited_sentences,
)
from deep_research.synthesis_runtime import _invoke_with_synthesis_budget
from deep_research.synthesis_selection import _cards_by_branch, _cards_for_synthesis, _rank_cards, _source_diverse_cards
from deep_research.text_terms import cjk_char_count

MAX_DEPTH_EXPANSION_ROUNDS = 2


def _expand_report_depth_if_needed(
    *,
    model: BaseChatModel,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
    report: str,
    target_profile: dict[str, Any],
    verification_failures: list[str],
    writing_guidance: str,
    model_spec: str,
    settings: Settings,
) -> str:
    expanded = report
    for round_index in range(MAX_DEPTH_EXPANSION_ROUNDS):
        if not _report_needs_depth_expansion(expanded, target_profile):
            break
        prompt = _depth_expansion_prompt(
            plan=plan,
            evidence_cards=evidence_cards,
            coverage=coverage,
            sources=sources,
            current_report=expanded,
            target_profile=target_profile,
            verification_failures=verification_failures,
            writing_guidance=writing_guidance,
            round_index=round_index,
        )
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        addition = _clean_depth_expansion_markdown(str(response.content), sources)
        if _is_degenerate_expansion(addition):
            break
        expanded = _insert_before_sources(expanded, addition)
        expanded = _normalize_report_markdown(expanded, sources)
        expanded = _repair_weak_citation_support(expanded, evidence_cards, sources)
        expanded = _strip_hallucinated_specific_citations(expanded, evidence_cards, sources)
        expanded = _rewrite_low_overlap_cited_sentences(
            expanded, evidence_cards, sources,
            model=model, settings=settings, model_spec=model_spec,
        )
        expanded = _append_evidence_coverage_if_needed(expanded, plan, evidence_cards)
    return expanded


def _depth_expansion_prompt(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
    current_report: str,
    target_profile: dict[str, Any],
    verification_failures: list[str],
    writing_guidance: str,
    round_index: int,
) -> str:
    source_lookup = {source.id: source for source in sources}
    evidence_lines = []
    expansion_cards = _cards_for_depth_expansion(
        plan=plan,
        evidence_cards=evidence_cards,
        current_report=current_report,
        round_index=round_index,
    )
    for card in expansion_cards:
        source = source_lookup.get(card.source_id)
        if source is None:
            continue
        evidence_lines.append(
            "\n".join(
                [
                    f"Evidence card {card.id}",
                    f"- branch_id: {card.branch_id}",
                    f"- source_id: {card.source_id}",
                    f"- source_title: {source.title}",
                    f"- claim: {card.claim}",
                    f"- excerpt: {card.supporting_excerpt[:320]}",
                    f"- limitations: {', '.join(card.limitations) or 'none'}",
                ]
            )
        )
    body, _separator, _source_tail = _split_sources(current_report)
    repair_text = "\n".join(f"- {failure}" for failure in verification_failures[:16]) or "None"
    current_excerpt = body[-3500:] if round_index else body[:2500] + "\n\n" + body[-1500:]
    return f"""You are extending a research report that is factually grounded but too shallow for its evidence plan.

User question:
{plan.question}

Output language:
{_language_instruction(plan.question)}

Current depth status:
{json_dumps(_depth_status(current_report, target_profile))}

Target report profile:
{json_dumps(target_profile)}

Coverage status:
- complete: {coverage.complete}
- coverage_score: {coverage.coverage_score}
- missing_branches: {', '.join(coverage.missing_branches) or 'none'}

Prior verification feedback to repair:
{repair_text}

Additional writing guidance:
{writing_guidance.strip()[:4000] if writing_guidance.strip() else 'None'}

Current report excerpt:
{current_excerpt}

Evidence cards available for expansion:
{chr(10).join(evidence_lines)}

Write only additional Markdown sections or paragraphs to insert before the Sources section.

Requirements:
- Do not repeat the full report, title, or Sources section.
- Choose natural headings that fit this question and the evidence; do not force a universal template.
- Every factual paragraph must include source citations like [3].
- Use only source_id values from the evidence cards above.
- Expand underdeveloped mechanisms, evidence strength, disagreements, boundary conditions, implications, limitations, and unresolved questions when supported by evidence.
- Repair weak-citation feedback by rewriting unsupported claims or replacing broad citations with claims that the cited evidence cards directly support.
- Prefer cohesive analytical prose over bullet dumps.
- Do not include raw URLs, markdown links, images, scrape metadata, branch IDs, evidence card IDs, or verification diagnostics in the report text.
"""


def _cards_for_depth_expansion(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    current_report: str,
    round_index: int,
    limit: int = 28,
) -> list[EvidenceCard]:
    if not evidence_cards:
        return []
    body, _separator, _source_tail = _split_sources(current_report)
    report_terms = content_terms(body)
    cited_source_ids = set(_numeric_citation_ids(body))
    selected: list[EvidenceCard] = []
    selected_ids: set[int] = set()
    cards_by_branch = _cards_by_branch(evidence_cards)
    per_branch_limit = 2 if round_index == 0 else 1
    for branch in plan.branches:
        branch_cards = cards_by_branch.get(branch.id, [])
        if not branch_cards:
            continue
        branch_terms = content_terms(branch.title + " " + branch.objective + " " + " ".join(branch.required_terms))
        branch_coverage = len(branch_terms & report_terms) / max(len(branch_terms), 1) if branch_terms else 1.0
        branch_cited = bool({card.source_id for card in branch_cards} & cited_source_ids)
        if branch_coverage >= 0.45 and branch_cited and round_index > 0:
            continue
        for card in _source_diverse_cards(_rank_cards(branch_cards, question=plan.question), limit=per_branch_limit):
            if card.id in selected_ids:
                continue
            selected.append(card)
            selected_ids.add(card.id)
            if len(selected) >= limit:
                return selected

    remaining = [
        card
        for card in _cards_for_synthesis(plan, evidence_cards)
        if card.id not in selected_ids and card.source_id not in cited_source_ids
    ]
    for card in _source_diverse_cards(remaining, limit=limit - len(selected)):
        selected.append(card)
        selected_ids.add(card.id)
        if len(selected) >= limit:
            break
    return selected


def _report_needs_depth_expansion(report: str, target_profile: dict[str, Any]) -> bool:
    status = _depth_status(report, target_profile)
    return bool(
        status["word_count"] < status["minimum_words"]
        or status["cited_paragraphs"] < status["minimum_cited_paragraphs"]
        or status["major_sections_before_sources"] < status["minimum_major_sections_before_sources"]
    )


def _depth_status(report: str, target_profile: dict[str, Any]) -> dict[str, int]:
    body, _separator, _source_tail = _split_sources(report)
    return {
        "word_count": _body_word_count(body),
        "minimum_words": int(target_profile.get("minimum_words", 0) or 0),
        "cited_paragraphs": _cited_paragraph_count(body),
        "minimum_cited_paragraphs": int(target_profile.get("minimum_cited_paragraphs", 0) or 0),
        "major_sections_before_sources": len(re.findall(r"(?m)^##\s+.+$", body)),
        "minimum_major_sections_before_sources": int(target_profile.get("minimum_major_sections_before_sources", 0) or 0),
    }


def _body_word_count(text: str) -> int:
    latin_words = len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9+.-]*\b", text))
    return latin_words + (cjk_char_count(text) // 2)


def _cited_paragraph_count(text: str) -> int:
    count = 0
    for paragraph in re.split(r"\n\s*\n", text):
        stripped = _substantive_paragraph_text(paragraph)
        if stripped and re.search(r"\[[0-9][0-9,;\s]*\]", stripped):
            count += 1
    return count


def _substantive_paragraph_text(paragraph: str) -> str:
    lines: list[str] = []
    for line in paragraph.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("|"):
            continue
        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", stripped):
            continue
        if re.fullmatch(r"\*?\s*(?:end of report|end)\s*\*?", stripped, flags=re.I):
            continue
        lines.append(stripped)
    return " ".join(lines)


def _clean_depth_expansion_markdown(text: str, sources: list[SourceRecordV2]) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = _remove_existing_source_listing(cleaned)
    cleaned = re.sub(r"(?m)^#\s+.+$", "", cleaned)
    cleaned = _normalize_markdown_headings(cleaned)
    cleaned = _separate_heading_blocks(cleaned)
    cleaned = _clean_malformed_citation_punctuation(cleaned)
    cleaned = _strip_report_chrome_lines(cleaned)
    cleaned = _strip_source_artifact_lines(cleaned)
    cleaned = _remove_unknown_numeric_citations(cleaned, sources)
    return cleaned.strip()


def _is_degenerate_expansion(text: str) -> bool:
    body = "\n".join(line for line in text.splitlines() if not line.strip().startswith("#")).strip()
    normalized = body.lower().strip(" .`'\"")
    if not normalized or normalized in {"none", "null", "n/a", "na"}:
        return True
    return len(content_terms(body)) < 20 or not _numeric_citation_ids(body)


def _insert_before_sources(report: str, addition: str) -> str:
    if not addition.strip():
        return report
    body, separator, source_tail = _split_sources(report)
    if separator:
        return body.rstrip() + "\n\n" + addition.strip() + "\n\n" + separator + source_tail
    return report.rstrip() + "\n\n" + addition.strip() + "\n"


def _build_argumentative_outline(
    *,
    model: BaseChatModel,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    settings: Settings,
    model_spec: str,
) -> dict[str, str]:
    """Pre-synthesis pass: ask the model to commit to one thesis per branch.

    Returns a {branch_id: thesis_sentence} dict. The thesis sentences are injected
    into the main synthesis prompt so the writer has argumentative direction, not just
    a coverage checklist. Falls back silently to an empty dict on any error.
    """
    source_lookup = {source.id: source for source in sources}
    cards_by_branch: dict[str, list[EvidenceCard]] = {}
    for card in evidence_cards:
        cards_by_branch.setdefault(card.branch_id, []).append(card)

    branch_summaries = []
    for branch in plan.branches:
        branch_cards = cards_by_branch.get(branch.id, [])[:5]
        card_claims = "; ".join(f'"{c.claim}"' for c in branch_cards)
        branch_summaries.append(
            f"- {branch.id} ({branch.title}): objective={branch.objective[:200]}; top claims: {card_claims or 'none'}"
        )

    prompt = f"""You are planning a research report. For each branch below, write one sentence that states the main argument or finding the section should defend — not a topic label, but a claim the evidence supports.

User question: {plan.question[:400]}

Branches and their top evidence:
{chr(10).join(branch_summaries)}

Return ONLY a JSON object mapping branch_id to one thesis sentence. Example:
{{"branch_1": "The evidence consistently shows X because Y.", "branch_2": "Despite claims of Z, the data supports W."}}

JSON only. No prose, no markdown."""
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        text = str(response.content).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return {
                str(k): str(v).strip()
                for k, v in parsed.items()
                if isinstance(v, str) and v.strip()
            }
    except Exception:
        pass
    return {}


def _rewrite_opening_paragraph(
    *,
    model: BaseChatModel,
    report: str,
    plan: ResearchPlan,
    opening_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    settings: Settings,
    model_spec: str,
) -> str:
    """Targeted rewrite of the first substantive paragraph so it leads with a direct answer.

    Only fires if the opening paragraph looks like hedging, a topic statement, or a
    "this report examines..." preamble. Falls back to the original report on any error
    or if the opening is already strong.
    """
    # Extract the first paragraph after headings.
    lines = report.split("\n")
    para_start = -1
    para_end = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if para_start == -1:
            if stripped and not stripped.startswith("#"):
                para_start = i
        elif para_start >= 0:
            if not stripped:
                para_end = i
                break
    if para_start < 0:
        return report
    if para_end < 0:
        para_end = min(para_start + 8, len(lines))
    opening_para = "\n".join(lines[para_start:para_end]).strip()
    if not opening_para:
        return report

    # Skip rewrite if the paragraph already looks like a direct answer
    # (contains a citation or is short and declarative).
    if re.search(r"\[\d+\]", opening_para) and len(opening_para.split()) >= 30:
        return report

    source_lookup = {source.id: source for source in sources}
    card_snippets = []
    for card in opening_cards[:4]:
        src = source_lookup.get(card.source_id)
        if src:
            card_snippets.append(f'[{card.source_id}] {card.claim} — "{card.supporting_excerpt[:200]}"')

    prompt = f"""Rewrite the opening paragraph of this research report so that it immediately answers the user's question in one or two sentences, then develops the answer with cited evidence.

User question: {plan.question[:400]}

Current opening paragraph:
{opening_para}

Key evidence for the opening (use these source IDs for citations):
{chr(10).join(card_snippets) or 'None'}

Rules:
- Start with the direct answer or main finding — not "This report examines..." or "Research has shown..."
- Cite at least one source using [N] inline notation
- Keep it 2–4 sentences maximum
- Do not add section headings
- Return ONLY the rewritten paragraph, nothing else"""
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        new_para = str(response.content).strip()
        if not new_para or len(new_para) < 40 or len(new_para) > len(opening_para) * 4:
            return report
        # Splice the new paragraph back in.
        new_lines = list(lines)
        replacement = new_para.split("\n")
        new_lines[para_start:para_end] = replacement
        return "\n".join(new_lines)
    except Exception:
        return report



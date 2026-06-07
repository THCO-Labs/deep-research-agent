from __future__ import annotations

from collections import defaultdict
from datetime import date
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deep_research.model_router import model_for_role
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2
from deep_research.settings import GOOGLE_DEFAULT_MODEL, Settings
from deep_research.source_validation import content_terms
from deep_research.text_terms import cjk_char_count, preferred_output_language

MAX_SYNTHESIS_CARDS_PER_BRANCH = 8
MAX_SYNTHESIS_CARDS_TOTAL = 20
MAX_SYNTHESIS_EXCERPT_CHARS = 160
MAX_DEPTH_EXPANSION_ROUNDS = 2
INDIVIDUAL_CITATION_REPAIR_THRESHOLD = 0.22
LOW_TPM_PROVIDER_BUDGET_TOKENS = 7_600
MIN_SYNTHESIS_COMPLETION_TOKENS = 768
REPORT_STYLE_EXAMPLES = (
    {
        "name": "Analytical explainer",
        "best_for": "Questions that ask what something is, why it matters, or how it works.",
        "shape": [
            "Lead with the direct answer in one or two cited paragraphs.",
            "Use concept-specific headings that teach the reader in a logical order.",
            "Separate mechanisms, evidence strength, debates, and implications when the evidence supports them.",
            "Close with uncertainty, gaps, or what would change the conclusion.",
        ],
    },
    {
        "name": "Comparative decision report",
        "best_for": "Questions asking versus, which option, trade-offs, or decision guidance.",
        "shape": [
            "Open with the recommended choice or decision frame.",
            "Compare options on dimensions native to the topic, not generic labels.",
            "Use a cited table only when it clarifies trade-offs.",
            "Make assumptions and edge cases explicit.",
        ],
    },
    {
        "name": "Evidence review",
        "best_for": "Research, medical, policy, legal, scientific, or literature-review prompts.",
        "shape": [
            "Start with the bottom-line evidence assessment.",
            "Organize by study types, populations, mechanisms, outcomes, or chronology as appropriate.",
            "Distinguish strong findings, suggestive findings, conflicts, and missing evidence.",
            "Avoid overstating causality when evidence is correlational.",
        ],
    },
    {
        "name": "Technical implementation brief",
        "best_for": "Engineering, methods, architecture, algorithm, or operational prompts.",
        "shape": [
            "Explain the system or method at the right abstraction level first.",
            "Then cover components, workflow, failure modes, costs, and alternatives.",
            "Use diagrams, tables, or numbered procedures only when grounded in evidence.",
            "End with practical constraints and verification checks.",
        ],
    },
)


def build_report_blueprint(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
) -> dict[str, Any]:
    cards_by_branch = _cards_by_branch(evidence_cards)
    branch_sections = []
    for branch in plan.branches:
        branch_cards = cards_by_branch.get(branch.id, [])
        branch_sections.append(
            {
                "branch_id": branch.id,
                "heading": branch.title,
                "objective": branch.objective,
                "evidence_card_count": len(branch_cards),
                "source_count": len({card.source_id for card in branch_cards}),
                "required_points": branch.required_terms,
                "writing_task": (
                    "Turn this branch's evidence into analytical prose: explain what the evidence means, "
                    "where sources agree or differ, and why it matters for the user's question."
                ),
            }
        )
    return {
        "schema_version": 1,
        "report_title": _report_title(plan.question),
        "output_language": _language_label(plan.question),
        "question": plan.question,
        "audience": plan.audience,
        "key_message_task": "State the direct answer early, then develop it through evidence-backed sections.",
        "source_summary": {
            "usable_source_count": len(sources),
            "evidence_card_count": len(evidence_cards),
            "coverage_complete": coverage.complete,
            "coverage_score": coverage.coverage_score,
            "missing_branches": coverage.missing_branches,
        },
        "acceptance_criteria": plan.acceptance_criteria,
        "branch_writing_briefs": branch_sections,
        "style_examples": REPORT_STYLE_EXAMPLES,
        "quality_contract": _report_quality_contract(plan, evidence_cards, coverage),
        "structure_guidance": [
            "Choose headings that fit this specific question and evidence set.",
            "Do not force a universal report template.",
            "Use branch titles as raw material, not mandatory section titles.",
            "Include comparison tables, numbered recommendations, or visual assets only when they improve the answer.",
            "Always end with one parseable Sources section.",
        ],
        "visual_assets": _visual_assets_from_sources(sources),
    }


def synthesize_report(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
) -> str:
    cards_by_branch = _cards_by_branch(evidence_cards)
    blueprint = build_report_blueprint(plan=plan, evidence_cards=evidence_cards, coverage=coverage, sources=sources)
    labels = _report_labels(plan.question)

    cited_source_ids: set[int] = set()
    lines = [
        f"# {blueprint['report_title']}",
        "",
        f"## {labels['bottom_line']}",
        "",
    ]
    summary_cards = _opening_cards(plan, evidence_cards)[:5]
    if summary_cards:
        summary = _executive_summary_sentence(plan.question, summary_cards)
        cited_source_ids.update(card.source_id for card in summary_cards)
        lines.extend([summary, ""])
    else:
        lines.extend([_no_evidence_sentence(plan.question), ""])

    for branch in plan.branches:
        branch_cards = cards_by_branch.get(branch.id, [])
        if not branch_cards:
            continue
        lines.extend([f"## {branch.title}", ""])
        if branch.id in {"comparison", "comparisons"} and branch_cards:
            lines.extend(_comparison_table(branch_cards, plan=plan))
            cited_source_ids.update(card.source_id for card in branch_cards[:4])
            lines.append("")
        for card in _source_diverse_cards(_rank_cards(branch_cards, question=plan.question), limit=max(4, min(8, len(branch_cards)))):
            lines.extend([_sentence_with_citation(card), ""])
            cited_source_ids.add(card.source_id)

    lines.extend([f"## {labels['synthesis']}", ""])
    if summary_cards:
        synthesis_cards = summary_cards[:3]
        lines.extend([_synthesis_sentence(synthesis_cards, question=plan.question), ""])
        cited_source_ids.update(card.source_id for card in synthesis_cards)
    else:
        lines.extend([labels["no_synthesis"], ""])

    lines.extend([f"## {labels['comparison']}", ""])
    if evidence_cards:
        table_cards = _cards_for_synthesis(plan, evidence_cards)[:6]
        lines.extend(_comparison_table(table_cards, plan=plan))
        cited_source_ids.update(card.source_id for card in table_cards)
        lines.append("")
    else:
        lines.extend([labels["no_table"], ""])

    breadth_cards = _cards_needed_for_source_breadth(plan, evidence_cards, cited_source_ids)
    if breadth_cards:
        lines.extend([f"## {labels['additional_evidence']} {_report_subject(plan.question)}", ""])
        for card in breadth_cards:
            lines.extend([_sentence_with_citation(card), ""])
            cited_source_ids.add(card.source_id)

    lines.extend([f"## {labels['implications']}", ""])
    if summary_cards:
        takeaway_cards = summary_cards[:3]
        lines.extend([_takeaway_sentence(takeaway_cards, question=plan.question), ""])
        cited_source_ids.update(card.source_id for card in takeaway_cards)
    else:
        lines.extend([labels["no_takeaways"], ""])

    lines.extend([f"## {labels['limits']}", ""])
    if evidence_cards:
        lines.extend([_limitations_sentence(coverage, question=plan.question), ""])
    else:
        lines.extend([labels["no_confidence"], ""])

    if coverage.missing_branches:
        lines.extend([f"## {labels['gaps']}", ""])
        lines.extend([labels["gaps_sentence"], ""])

    cited_sources = [source for source in sources if source.id in cited_source_ids]
    lines.extend(["## Sources", ""])
    for source in sorted(cited_sources, key=lambda item: item.id):
        lines.append(f"[{source.id}] {source.title}: {source.url}")
    return "\n".join(lines).rstrip() + "\n"


def synthesize_report_with_model(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
    settings: Settings,
    previous_report: str = "",
    verification_failures: list[str] | None = None,
    blueprint: dict[str, Any] | None = None,
    writing_guidance: str = "",
) -> str:
    if not evidence_cards:
        return synthesize_report(plan=plan, evidence_cards=evidence_cards, coverage=coverage, sources=sources)
    model_spec = _synthesis_model_spec(settings, plan, writing_guidance)
    model = model_for_role(settings, "orchestrator", model_spec)
    if not isinstance(model, BaseChatModel):
        raise RuntimeError(f"Synthesis role did not resolve to a chat model: {model!r}")
    blocked_source_ids = _blocked_source_ids_from_failures(verification_failures or [])
    candidate_cards = _without_blocked_sources(evidence_cards, blocked_source_ids)
    if not candidate_cards:
        candidate_cards = evidence_cards
    synthesis_cards = _cards_for_synthesis(plan, candidate_cards)
    evidence_sources = _evidence_backed_sources(sources, synthesis_cards)
    report_blueprint = blueprint or build_report_blueprint(
        plan=plan,
        evidence_cards=synthesis_cards,
        coverage=coverage,
        sources=evidence_sources,
    )
    target_profile = _target_report_profile(
        plan=plan,
        evidence_cards=synthesis_cards,
        writing_guidance=writing_guidance,
    )
    prompt = _synthesis_prompt(
        plan=plan,
        evidence_cards=synthesis_cards,
        coverage=coverage,
        sources=evidence_sources,
        previous_report=previous_report,
        verification_failures=verification_failures or [],
        blueprint=report_blueprint,
        writing_guidance=writing_guidance,
    )
    response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
    text = str(response.content).strip()
    if not text:
        if previous_report.strip():
            return _normalize_report_markdown(previous_report, evidence_sources)
        return synthesize_report(plan=plan, evidence_cards=synthesis_cards, coverage=coverage, sources=evidence_sources)
    if _is_degenerate_model_report(text, synthesis_cards):
        return synthesize_report(plan=plan, evidence_cards=synthesis_cards, coverage=coverage, sources=evidence_sources)
    normalized = _normalize_report_markdown(text, evidence_sources)
    citation_repaired = _repair_weak_citation_support(normalized, synthesis_cards, evidence_sources)
    coverage_repaired = _append_evidence_coverage_if_needed(citation_repaired, plan, synthesis_cards)
    normalized_report = _normalize_report_markdown(coverage_repaired, evidence_sources)
    expanded_report = _expand_report_depth_if_needed(
        model=model,
        plan=plan,
        evidence_cards=synthesis_cards,
        coverage=coverage,
        sources=evidence_sources,
        report=normalized_report,
        target_profile=target_profile,
        verification_failures=verification_failures or [],
        writing_guidance=writing_guidance,
        model_spec=model_spec,
        settings=settings,
    )
    return _normalize_report_markdown(expanded_report, evidence_sources)


def _synthesis_model_spec(settings: Settings, plan: ResearchPlan, writing_guidance: str = "") -> str:
    if not _criteria_rich_plan(plan, writing_guidance=writing_guidance):
        return settings.model
    provider, _, _model = settings.model.partition(":")
    if provider == "google_genai":
        return settings.model
    if settings.provider == "hybrid" and settings.google_key_pool:
        return GOOGLE_DEFAULT_MODEL
    return settings.model


def _invoke_with_synthesis_budget(
    model: BaseChatModel,
    *,
    prompt: str,
    settings: Settings,
    model_spec: str,
) -> Any:
    kwargs = _synthesis_request_kwargs(settings=settings, prompt=prompt, model_spec=model_spec)
    try:
        return model.invoke([HumanMessage(content=prompt)], **kwargs)
    except TypeError as exc:
        if kwargs and "unexpected" in str(exc).lower() and "keyword" in str(exc).lower():
            return model.invoke([HumanMessage(content=prompt)])
        raise


def _synthesis_request_kwargs(*, settings: Settings, prompt: str, model_spec: str) -> dict[str, int]:
    provider, _separator, _model_name = model_spec.partition(":")
    if provider != "groq":
        return {}
    prompt_tokens = _rough_token_count(prompt)
    requested_completion_tokens = min(
        settings.model_max_output_tokens,
        max(MIN_SYNTHESIS_COMPLETION_TOKENS, LOW_TPM_PROVIDER_BUDGET_TOKENS - prompt_tokens),
    )
    return {"max_tokens": requested_completion_tokens}


def _rough_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _synthesis_prompt(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
    previous_report: str,
    verification_failures: list[str],
    blueprint: dict[str, Any] | None = None,
    writing_guidance: str = "",
) -> str:
    source_lookup = {source.id: source for source in sources}
    evidence_lines = []
    selected_cards = _cards_for_synthesis(plan, evidence_cards)
    for card in selected_cards:
        source = source_lookup.get(card.source_id)
        if source is None:
            continue
        evidence_lines.append(
            (
                f"- card {card.id}; branch {card.branch_id}; source [{card.source_id}] {source.title}; "
                f"claim: {card.claim}; excerpt: {card.supporting_excerpt[:MAX_SYNTHESIS_EXCERPT_CHARS]}; "
                f"limits: {', '.join(card.limitations[:2]) or 'none'}"
            )
        )
    source_lines = "\n".join(f"[{source.id}] {source.title}" for source in sorted(sources, key=lambda item: item.id))
    branch_lines = "\n".join(
        f"- {branch.id}: {branch.title}; objective: {branch.objective[:220]}"
        for branch in plan.branches
    )
    acceptance_criteria_lines = "\n".join(f"- {criterion}" for criterion in plan.acceptance_criteria[:8]) or "None"
    repair_text = "\n".join(f"- {failure}" for failure in verification_failures[:8]) or "None"
    previous_text = previous_report[:800] if previous_report else "None"
    report_blueprint = blueprint or build_report_blueprint(plan=plan, evidence_cards=evidence_cards, coverage=coverage, sources=sources)
    visual_assets = report_blueprint.get("visual_assets", [])
    visual_text = "\n".join(
        f"- {asset['alt']}: {asset['url']} (source_id {asset['source_id']})"
        for asset in visual_assets[:6]
    ) or "None"
    language_instruction = _language_instruction(plan.question)
    opening_cards = _opening_cards(plan, evidence_cards)[:8]
    opening_priority = ", ".join(f"card {card.id} from source [{card.source_id}]" for card in opening_cards) or "None"
    required_source_breadth = min(17, len({card.source_id for card in evidence_cards}))
    target_profile = _target_report_profile(plan=plan, evidence_cards=evidence_cards, writing_guidance=writing_guidance)
    target_depth_hint = _target_depth_hint(
        plan=plan,
        evidence_cards=evidence_cards,
        writing_guidance=writing_guidance,
        target_profile=target_profile,
    )
    return f"""You are writing a professional deep research report from verified evidence cards.

Current date: {date.today().isoformat()}

User question:
{plan.question}

Research branches:
{branch_lines}

Acceptance criteria to satisfy in the report:
{acceptance_criteria_lines}

Report depth and structure target:
{json_dumps(_compact_target_profile(target_profile))}

Coverage status:
- complete: {coverage.complete}
- coverage_score: {coverage.coverage_score}
- missing_branches: {', '.join(coverage.missing_branches) or 'none'}

Prior verification failures to repair:
{repair_text}

Additional report-writing guidance:
{writing_guidance.strip()[:2500] if writing_guidance.strip() else 'None'}

Previous draft, if any:
{previous_text}

Evidence cards:
{chr(10).join(evidence_lines)}

Allowed source IDs and titles:
{source_lines}

Evidence-backed visual assets, if any:
{visual_text}

Output language:
{language_instruction}

Opening-answer evidence priority:
{opening_priority}

Write the final report in Markdown.

Report style examples to learn from, not copy:
- Analytical explainer: direct answer, mechanisms, evidence strength, debates, implications.
- Evidence review: bottom-line evidence assessment, study/evidence distinctions, conflicts, gaps.
- Technical or decision brief: system/choice framing, trade-offs, constraints, verification checks.

Hard requirements:
- Answer the exact user question in the first substantive paragraph, using the opening-answer evidence priority.
- Use only the evidence cards above; every factual paragraph needs inline source citations like [3].
- Cite source IDs only, never evidence-card IDs, and cite at least {required_source_breadth} distinct evidence-backed sources when available.
- Depth target: {target_depth_hint}
- Cover the branches, acceptance criteria, and depth profile as report coverage requirements, but do not print them as checklists.
- Use natural question-specific headings and polished analytical prose; synthesize across sources instead of dumping evidence cards.
- Define central constructs, explain mechanisms and trade-offs, compare evidence strength/tensions, and state limitations or unresolved questions when supported.
- Keep the title, opening, and body centered on the requested scope; ignore any previous-draft drift.
- Do not include raw URLs, markdown links/media, page chrome, scrape metadata, branch IDs, evidence-card IDs, or verification diagnostics outside Sources.
- Include images only when listed as evidence-backed visual assets and useful for inspection.
- End with exactly one ## Sources section. Each entry must be exactly: [N] Title: https://url
"""


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


def _normalize_report_markdown(report: str, sources: list[SourceRecordV2]) -> str:
    cleaned = report.strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = _normalize_markdown_headings(cleaned)
    cleaned = _separate_heading_blocks(cleaned)
    cleaned = _clean_malformed_citation_punctuation(cleaned)
    cleaned = _strip_report_chrome_lines(cleaned)
    cleaned = _remove_existing_source_listing(cleaned)
    cleaned = _strip_source_artifact_lines(cleaned)
    cleaned = _remove_unknown_numeric_citations(cleaned, sources)
    cleaned = _repair_uncited_body_paragraphs(cleaned, sources)
    if not re.search(r"(?im)^#\s+", cleaned):
        cleaned = "# Research Report\n\n" + cleaned
    cleaned = _remove_existing_source_listing(cleaned)
    cleaned = _clean_malformed_citation_punctuation(cleaned)
    cleaned = _strip_report_chrome_lines(cleaned)
    cleaned = _strip_source_artifact_lines(cleaned)
    source_section = _sources_section(sources, cleaned)
    if re.search(r"(?ims)^##\s+Sources\s*$", cleaned):
        cleaned = re.sub(r"(?ims)^##\s+Sources\s*$.*\Z", source_section, cleaned).strip()
    else:
        cleaned = cleaned.rstrip() + "\n\n" + source_section
    return cleaned.rstrip() + "\n"


def _clean_malformed_citation_punctuation(report: str) -> str:
    cleaned = report
    cleaned = re.sub(r"\((\s*\[[0-9][0-9,;\s]*\])\s*;\s*\)", r"(\1)", cleaned)
    cleaned = re.sub(r"\[\s*([0-9][0-9,;\s]*)\s*;\s*]", r"[\1]", cleaned)
    cleaned = re.sub(r"\[\s*([0-9][0-9,;\s]*)\s*,\s*]", r"[\1]", cleaned)
    cleaned = re.sub(r"\[\s*([0-9][0-9,;\s]*)\s+]", r"[\1]", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def _is_degenerate_model_report(report: str, evidence_cards: list[EvidenceCard]) -> bool:
    body, _separator, _source_tail = _split_sources(report)
    body_without_headings = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("#")
    ).strip()
    normalized = body_without_headings.lower().strip(" .`'\"")
    if normalized in {"none", "null", "n/a", "na"}:
        return True
    if evidence_cards and len(content_terms(body_without_headings)) < 20:
        return True
    if evidence_cards and not _numeric_citation_ids(body_without_headings) and len(body_without_headings) < 1200:
        return True
    return False


def _normalize_markdown_headings(report: str) -> str:
    normalized_lines: list[str] = []
    for line in report.splitlines():
        stripped = line.strip()
        bold_match = re.fullmatch(r"\*\*([^*]{2,120})\*\*", stripped)
        if bold_match:
            heading = bold_match.group(1).strip()
            normalized_lines.append(f"## {heading}")
            continue
        heading_match = re.fullmatch(r"(#{1,6})[ \t]+(.+?)\s*", stripped)
        if heading_match:
            level = heading_match.group(1)
            heading = re.sub(r"\*\*", "", heading_match.group(2)).strip()
            normalized_lines.append(f"{level} {heading}")
            continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _separate_heading_blocks(report: str) -> str:
    return re.sub(r"(?m)^(#{1,6}\s+.+?)\s*\n(?!\s*\n)", r"\1\n\n", report)


def _strip_report_chrome_lines(report: str) -> str:
    cleaned_lines: list[str] = []
    for line in report.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", stripped):
            continue
        if re.fullmatch(r"\*?\s*(?:end of report|end)\s*\*?", stripped, flags=re.I):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _remove_existing_source_listing(report: str) -> str:
    if re.search(r"(?ims)^##\s+Sources\s*$", report):
        return re.sub(r"(?ims)^##\s+Sources\s*$.*\Z", "", report).strip()
    bibliography_heading = _malformed_bibliography_heading_match(report)
    if bibliography_heading is not None:
        return report[: bibliography_heading.start()].strip()

    lines = report.splitlines()
    index = len(lines) - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    if index < 0:
        return report

    source_line_count = 0
    block_start = index + 1
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped:
            block_start = index
            index -= 1
            continue
        if _looks_like_source_entry(stripped):
            source_line_count += 1
            block_start = index
            index -= 1
            continue
        break

    if source_line_count == 0:
        return report
    if index >= 0 and _looks_like_bibliography_heading(lines[index].strip()):
        block_start = index
    return "\n".join(lines[:block_start]).rstrip()


def _looks_like_source_entry(line: str) -> bool:
    stripped = line.strip()
    if not re.search(r"https?://\S+", stripped, flags=re.I):
        return False
    return bool(
        re.search(r"^\s*\[[0-9]+]\s+.+https?://\S+", stripped, flags=re.I)
        or re.search(r"^\s*(?:[-*]\s*)?[^:]{2,240}:\s+https?://\S+", stripped, flags=re.I)
        or re.search(r"^\s*(?:[-*]\s*)?https?://\S+", stripped, flags=re.I)
    )


def _malformed_bibliography_heading_match(report: str) -> re.Match[str] | None:
    for match in re.finditer(r"(?im)^#{1,6}\s+(.+?)\s*$", report):
        if not _looks_like_bibliography_heading(match.group(0)):
            continue
        tail = report[match.end() :]
        preview_lines = [line.strip() for line in tail.splitlines()[:12] if line.strip()]
        if _looks_like_source_entry(match.group(0)) or any(_looks_like_source_entry(line) for line in preview_lines):
            return match
    return None


def _looks_like_bibliography_heading(line: str) -> bool:
    stripped = re.sub(r"^\s*#{1,6}\s*", "", line.strip())
    stripped = re.sub(r"^\*\*|\*\*$", "", stripped).strip()
    return bool(
        re.match(
            r"^(?:sources|references|bibliography|works cited)\s*(?:$|[:\-]|\[[0-9]+])",
            stripped,
            flags=re.I,
        )
    )


def _strip_source_artifact_lines(report: str) -> str:
    body, separator, source_tail = _split_sources(report)
    cleaned_lines = [
        line
        for line in body.splitlines()
        if not _looks_like_source_artifact_line(line)
    ]
    cleaned_body = "\n".join(cleaned_lines).strip()
    return cleaned_body + (separator + source_tail if separator else "")


def _looks_like_source_artifact_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _looks_like_bibliography_heading(stripped) and (
        _looks_like_source_entry(stripped) or re.search(r"\[[0-9]+]\s+.+https?://", stripped, flags=re.I)
    ):
        return True
    if _looks_like_source_entry(stripped):
        return True
    if re.search(r"https?://\S+", stripped, flags=re.I):
        word_count = len(re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", stripped))
        url_chars = sum(len(match.group(0)) for match in re.finditer(r"https?://\S+", stripped, flags=re.I))
        if word_count <= 24 or url_chars / max(len(stripped), 1) > 0.10:
            return True
    return False


def _remove_unknown_numeric_citations(report: str, sources: list[SourceRecordV2]) -> str:
    allowed = {source.id for source in sources}

    def replace(match: re.Match[str]) -> str:
        kept = [
            int(value)
            for value in re.findall(r"\d+", match.group(1))
            if int(value) in allowed
        ]
        if not kept:
            return ""
        return "[" + ", ".join(str(value) for value in sorted(set(kept))) + "]"

    return re.sub(r"\[([0-9][0-9,;\s]*)\]", replace, report)


def _repair_uncited_body_paragraphs(report: str, sources: list[SourceRecordV2]) -> str:
    body, separator, source_tail = _split_sources(report)
    fallback = _fallback_citations(body, sources)
    if not fallback:
        return report
    repaired: list[str] = []
    for paragraph in re.split(r"(\n\s*\n)", body):
        if not paragraph.strip() or re.fullmatch(r"\n\s*\n", paragraph):
            repaired.append(paragraph)
            continue
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if _paragraph_needs_repair(text):
            paragraph = paragraph.rstrip() + f" {fallback}"
        repaired.append(paragraph)
    return "".join(repaired) + (separator + source_tail if separator else "")


def _repair_weak_citation_support(
    report: str,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    *,
    threshold: float = 0.35,
) -> str:
    source_ids = {source.id for source in sources}
    terms_by_source = _evidence_terms_by_source(evidence_cards, source_ids)
    if not terms_by_source:
        return report
    body, separator, source_tail = _split_sources(report)
    repaired: list[str] = []
    for paragraph in re.split(r"(\n\s*\n)", body):
        if not paragraph.strip() or re.fullmatch(r"\n\s*\n", paragraph):
            repaired.append(paragraph)
            continue
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not _paragraph_can_receive_support_repair(text):
            repaired.append(paragraph)
            continue
        claim_terms = content_terms(_strip_numeric_citations(text))
        if len(claim_terms) < 6:
            repaired.append(paragraph)
            continue
        cited_ids = {source_id for source_id in _numeric_citation_ids(text) if source_id in source_ids}
        if not cited_ids:
            repaired.append(paragraph)
            continue
        individual_scores = _individual_source_support_scores(claim_terms, terms_by_source, cited_ids)
        weak_ids = {
            source_id
            for source_id, score in individual_scores.items()
            if score < INDIVIDUAL_CITATION_REPAIR_THRESHOLD
        }
        strong_ids = cited_ids - weak_ids
        support_terms = set().union(*(terms_by_source.get(source_id, set()) for source_id in cited_ids))
        support_score = len(claim_terms & support_terms) / max(len(claim_terms), 1)
        if support_score >= threshold and not weak_ids:
            repaired.append(paragraph)
            continue
        additions = _best_supporting_source_ids(claim_terms, terms_by_source, cited_ids, threshold)
        if weak_ids and (strong_ids or additions):
            paragraph = _remove_numeric_citation_ids(paragraph, weak_ids)
        if additions:
            paragraph = paragraph.rstrip() + " " + " ".join(f"[{source_id}]" for source_id in additions)
        repaired.append(paragraph)
    return "".join(repaired) + (separator + source_tail if separator else "")


def _append_evidence_coverage_if_needed(
    report: str,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    *,
    max_added_cards: int = 24,
) -> str:
    if not evidence_cards:
        return report
    if _criteria_rich_plan(plan):
        return report
    body, separator, source_tail = _split_sources(report)
    cited_source_ids = set(_numeric_citation_ids(body))
    report_terms = content_terms(body)
    cards_by_branch = _cards_by_branch(evidence_cards)
    additions: list[tuple[ResearchBranch | None, EvidenceCard]] = []
    selected_card_ids: set[int] = set()

    for branch in plan.branches:
        branch_cards = cards_by_branch.get(branch.id, [])
        if not branch_cards:
            continue
        branch_terms = content_terms(branch.title + " " + branch.objective + " " + " ".join(branch.required_terms))
        branch_coverage = len(branch_terms & report_terms) / max(len(branch_terms), 1) if branch_terms else 1.0
        branch_source_ids = {card.source_id for card in branch_cards}
        branch_is_cited = bool(branch_source_ids & cited_source_ids)
        if branch_coverage >= 0.35 and branch_is_cited:
            continue
        for card in _source_diverse_cards(_rank_cards(branch_cards, question=plan.question), limit=2):
            if card.id in selected_card_ids:
                continue
            additions.append((branch, card))
            selected_card_ids.add(card.id)
            cited_source_ids.add(card.source_id)
            if len(additions) >= max_added_cards:
                break
        if len(additions) >= max_added_cards:
            break

    target_sources = min(17, len({card.source_id for card in evidence_cards}))
    if len(cited_source_ids) < target_sources and len(additions) < max_added_cards:
        needed = target_sources - len(cited_source_ids)
        breadth_cards = [
            card
            for card in _cards_for_synthesis(plan, evidence_cards)
            if card.source_id not in cited_source_ids and card.id not in selected_card_ids
        ]
        for card in _source_diverse_cards(breadth_cards, limit=min(needed, max_added_cards - len(additions))):
            additions.append((None, card))
            selected_card_ids.add(card.id)
            cited_source_ids.add(card.source_id)

    if not additions:
        return report

    labels = _coverage_repair_labels(plan.question)
    lines = [
        "",
        f"## {labels['heading']} {_report_subject(plan.question)}",
        "",
    ]
    for branch, card in additions:
        if branch is not None:
            prefix = f"{labels['branch_prefix']} {branch.title}, {labels['evidence_adds']}"
        else:
            prefix = labels["breadth_prefix"]
        lines.append(f"{prefix} {card.claim.rstrip('. ')}. [{card.source_id}]")
        lines.append("")
    repaired_body = body.rstrip() + "\n" + "\n".join(lines).rstrip() + "\n"
    return repaired_body + (separator + source_tail if separator else "")


def _paragraph_can_receive_support_repair(text: str) -> bool:
    if len(text) < 80:
        return False
    if text.startswith("#") or text.startswith("|"):
        return False
    if not re.search(r"\[[0-9][0-9,;\s]*\]", text):
        return False
    return True


def _evidence_terms_by_source(
    evidence_cards: list[EvidenceCard],
    source_ids: set[int],
) -> dict[int, set[str]]:
    terms: dict[int, set[str]] = {}
    for card in evidence_cards:
        if card.source_id not in source_ids:
            continue
        terms.setdefault(card.source_id, set()).update(content_terms(card.claim + " " + card.supporting_excerpt))
    return terms


def _individual_source_support_scores(
    claim_terms: set[str],
    terms_by_source: dict[int, set[str]],
    cited_ids: set[int],
) -> dict[int, float]:
    return {
        source_id: len(claim_terms & terms_by_source.get(source_id, set())) / max(len(claim_terms), 1)
        for source_id in cited_ids
    }


def _remove_numeric_citation_ids(paragraph: str, remove_ids: set[int]) -> str:
    if not remove_ids:
        return paragraph

    def replace(match: re.Match[str]) -> str:
        kept = [
            int(value)
            for value in re.findall(r"\d+", match.group(1))
            if int(value) not in remove_ids
        ]
        if not kept:
            return ""
        return "[" + ", ".join(str(value) for value in sorted(set(kept))) + "]"

    cleaned = re.sub(r"\[([0-9][0-9,;\s]*)\]", replace, paragraph)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def _best_supporting_source_ids(
    claim_terms: set[str],
    terms_by_source: dict[int, set[str]],
    cited_ids: set[int],
    threshold: float,
) -> list[int]:
    ranked = sorted(
        (
            (
                len(claim_terms & terms) / max(len(claim_terms), 1),
                len(claim_terms & terms),
                source_id,
            )
            for source_id, terms in terms_by_source.items()
            if source_id not in cited_ids
        ),
        reverse=True,
    )
    additions: list[int] = []
    support_terms = set().union(*(terms_by_source.get(source_id, set()) for source_id in cited_ids))
    for score, _hits, source_id in ranked:
        if score <= 0:
            continue
        additions.append(source_id)
        support_terms |= terms_by_source.get(source_id, set())
        support_score = len(claim_terms & support_terms) / max(len(claim_terms), 1)
        if support_score >= threshold or len(additions) >= 3:
            break
    return sorted(additions)


def _split_sources(report: str) -> tuple[str, str, str]:
    match = re.search(r"(?ims)^##\s+Sources\s*$", report)
    if not match:
        return report, "", ""
    return report[: match.start()], report[match.start() : match.end()], report[match.end() :]


def _strip_numeric_citations(text: str) -> str:
    return re.sub(r"\[([0-9][0-9,;\s]*)\]", "", text)


def _numeric_citation_ids(text: str) -> list[int]:
    return [
        int(value)
        for block in re.findall(r"\[([0-9][0-9,;\s]*)\]", text)
        for value in re.findall(r"\d+", block)
    ]


def _fallback_citations(body: str, sources: list[SourceRecordV2]) -> str:
    cited = [
        source_id
        for source_id in _numeric_citation_ids(body)
        if any(source.id == source_id for source in sources)
    ]
    if not cited:
        cited = [source.id for source in sorted(sources, key=lambda item: (-item.quality_score, item.id))[:2]]
    return " ".join(f"[{source_id}]" for source_id in sorted(set(cited))[:3])


def _paragraph_needs_repair(text: str) -> bool:
    if len(text) < 80:
        return False
    if text.startswith("#") or text.startswith("|"):
        return False
    if re.search(r"\[[0-9][0-9,;\s]*\]", text):
        return False
    return True


def _sources_section(sources: list[SourceRecordV2], report: str) -> str:
    cited_ids = sorted(
        {
            int(value)
            for block in re.findall(r"\[([0-9][0-9,;\s]*)\]", report)
            for value in re.findall(r"\d+", block)
        }
    )
    source_by_id = {source.id: source for source in sources}
    listed = [source_by_id[source_id] for source_id in cited_ids if source_id in source_by_id]
    if not listed:
        listed = sorted(sources, key=lambda item: item.id)
    lines = ["## Sources", ""]
    for source in listed:
        lines.append(f"[{source.id}] {source.title}: {source.url}")
    return "\n".join(lines)


def _cards_by_branch(evidence_cards: list[EvidenceCard]) -> dict[str, list[EvidenceCard]]:
    cards_by_branch: dict[str, list[EvidenceCard]] = defaultdict(list)
    for card in evidence_cards:
        cards_by_branch[card.branch_id].append(card)
    return cards_by_branch


def _evidence_backed_sources(
    sources: list[SourceRecordV2],
    evidence_cards: list[EvidenceCard],
) -> list[SourceRecordV2]:
    evidence_source_ids = {card.source_id for card in evidence_cards}
    return [source for source in sources if source.id in evidence_source_ids]


def _blocked_source_ids_from_failures(failures: list[str]) -> set[int]:
    blocked: set[int] = set()
    for failure in failures:
        if not re.search(r"\bfails current branch/request alignment\b", failure, flags=re.I):
            continue
        blocked.update(int(value) for value in re.findall(r"Cited source \[([0-9]+)]", failure, flags=re.I))
    return blocked


def _without_blocked_sources(evidence_cards: list[EvidenceCard], blocked_source_ids: set[int]) -> list[EvidenceCard]:
    if not blocked_source_ids:
        return evidence_cards
    return [card for card in evidence_cards if card.source_id not in blocked_source_ids]


def _cards_for_synthesis(plan: ResearchPlan, evidence_cards: list[EvidenceCard]) -> list[EvidenceCard]:
    if not evidence_cards:
        return []
    cards_by_branch = _cards_by_branch(evidence_cards)
    selected: list[EvidenceCard] = []
    selected_ids: set[int] = set()
    branch_count = max(len(plan.branches), 1)
    per_branch_limit = max(2, min(MAX_SYNTHESIS_CARDS_PER_BRANCH, MAX_SYNTHESIS_CARDS_TOTAL // branch_count))
    for branch in plan.branches:
        branch_cards = sorted(
            cards_by_branch.get(branch.id, []),
            key=lambda item: _card_rank_key(item, question=plan.question),
        )
        for card in _source_diverse_cards(branch_cards, limit=per_branch_limit):
            if len(selected) >= MAX_SYNTHESIS_CARDS_TOTAL:
                return selected
            if card.id in selected_ids:
                continue
            selected.append(card)
            selected_ids.add(card.id)

    if len(selected) >= MAX_SYNTHESIS_CARDS_TOTAL:
        return selected[:MAX_SYNTHESIS_CARDS_TOTAL]

    remaining = [card for card in evidence_cards if card.id not in selected_ids]
    selected.extend(
        _marginal_coverage_cards(
            plan=plan,
            candidates=remaining,
            selected=selected,
            limit=MAX_SYNTHESIS_CARDS_TOTAL - len(selected),
        )
    )
    return selected[:MAX_SYNTHESIS_CARDS_TOTAL]


def _marginal_coverage_cards(
    *,
    plan: ResearchPlan,
    candidates: list[EvidenceCard],
    selected: list[EvidenceCard],
    limit: int,
) -> list[EvidenceCard]:
    if limit <= 0 or not candidates:
        return []

    branch_terms = _plan_branch_terms(plan)
    global_terms = _plan_global_terms(plan, branch_terms)
    card_terms = {card.id: _card_terms(card) for card in candidates + selected}
    covered_global_terms: set[str] = set()
    covered_branch_terms: dict[str, set[str]] = defaultdict(set)
    selected_term_sets: list[set[str]] = []
    branch_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[int, int] = defaultdict(int)

    for card in selected:
        terms = card_terms[card.id]
        covered_global_terms.update(terms & global_terms)
        covered_branch_terms[card.branch_id].update(terms & branch_terms.get(card.branch_id, set()))
        selected_term_sets.append(terms)
        branch_counts[card.branch_id] += 1
        source_counts[card.source_id] += 1

    remaining = list(candidates)
    chosen: list[EvidenceCard] = []
    while remaining and len(chosen) < limit:
        best_index = 0
        best_score: tuple[float, float, float, int] | None = None
        for index, card in enumerate(remaining):
            terms = card_terms[card.id]
            global_gain = len(terms & (global_terms - covered_global_terms))
            branch_gain = len(terms & (branch_terms.get(card.branch_id, set()) - covered_branch_terms[card.branch_id]))
            new_source_bonus = 1.0 if source_counts[card.source_id] == 0 else 1.0 / (source_counts[card.source_id] + 2)
            branch_balance_bonus = 1.0 / (branch_counts[card.branch_id] + 1)
            redundancy_penalty = _max_term_overlap(terms, selected_term_sets)
            relevance = _card_relevance_score(card, question=plan.question)
            score = (
                (branch_gain * 4.0)
                + (global_gain * 3.0)
                + (new_source_bonus * 1.25)
                + (branch_balance_bonus * 0.75)
                + relevance
                - (redundancy_penalty * 1.5),
                relevance,
                card.confidence,
                -card.id,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        card = remaining.pop(best_index)
        terms = card_terms[card.id]
        chosen.append(card)
        covered_global_terms.update(terms & global_terms)
        covered_branch_terms[card.branch_id].update(terms & branch_terms.get(card.branch_id, set()))
        selected_term_sets.append(terms)
        branch_counts[card.branch_id] += 1
        source_counts[card.source_id] += 1

    return chosen


def _plan_branch_terms(plan: ResearchPlan) -> dict[str, set[str]]:
    return {
        branch.id: content_terms(
            " ".join(
                [
                    branch.title,
                    branch.objective,
                    " ".join(branch.queries),
                    " ".join(branch.required_terms),
                    " ".join(branch.completion_criteria),
                ]
            )
        )
        for branch in plan.branches
    }


def _plan_global_terms(plan: ResearchPlan, branch_terms: dict[str, set[str]]) -> set[str]:
    terms = content_terms(
        " ".join(
            [
                plan.question,
                " ".join(plan.report_outline),
                " ".join(plan.acceptance_criteria),
                " ".join(requirement.source_type + " " + requirement.rationale for requirement in plan.source_requirements),
            ]
        )
    )
    for values in branch_terms.values():
        terms.update(values)
    return terms


def _card_terms(card: EvidenceCard) -> set[str]:
    return content_terms(" ".join([card.claim, card.supporting_excerpt, card.source_title, " ".join(card.limitations)]))


def _card_relevance_score(card: EvidenceCard, *, question: str) -> float:
    return (
        _question_phrase_score(question, f"{card.claim} {card.supporting_excerpt} {card.source_title}") * 2.0
        + _question_term_score(question, f"{card.claim} {card.supporting_excerpt} {card.source_title}")
        + card.confidence
        + card.quality_score
        + card.relevance_score
        + (card.semantic_score or 0.0)
    )


def _max_term_overlap(terms: set[str], selected_term_sets: list[set[str]]) -> float:
    if not terms or not selected_term_sets:
        return 0.0
    return max(
        (len(terms & selected_terms) / max(len(terms | selected_terms), 1) for selected_terms in selected_term_sets),
        default=0.0,
    )


def _source_diverse_cards(cards: list[EvidenceCard], *, limit: int) -> list[EvidenceCard]:
    selected: list[EvidenceCard] = []
    seen_sources: set[int] = set()
    for card in cards:
        if len(selected) >= limit:
            break
        if card.source_id in seen_sources:
            continue
        selected.append(card)
        seen_sources.add(card.source_id)
    if len(selected) < limit:
        selected_ids = {card.id for card in selected}
        for card in cards:
            if len(selected) >= limit:
                break
            if card.id in selected_ids:
                continue
            selected.append(card)
    return selected


def _rank_cards(cards: list[EvidenceCard], *, question: str = "") -> list[EvidenceCard]:
    return sorted(
        cards,
        key=lambda item: _card_rank_key(item, question=question),
    )


def _card_rank_key(card: EvidenceCard, *, question: str = "") -> tuple[float, float, float, float, float, int]:
    card_text = f"{card.claim} {card.supporting_excerpt} {card.source_title}"
    return (
        -_question_phrase_score(question, card_text),
        -_question_term_score(question, card_text),
        -card.confidence,
        -(card.semantic_score or 0.0),
        -card.quality_score,
        card.id,
    )


def _question_phrase_score(question: str, text: str) -> float:
    question_terms = content_terms(question)
    if len(question_terms) < 2:
        return _question_term_score(question, text)
    ordered_question_terms = [term for term in question.lower().replace("-", " ").split() if term in question_terms]
    if len(ordered_question_terms) < 2:
        ordered_question_terms = sorted(question_terms)
    text_terms = content_terms(text)
    phrase_terms: set[str] = set()
    for size in (3, 2):
        for index in range(0, max(0, len(ordered_question_terms) - size + 1)):
            window = ordered_question_terms[index : index + size]
            if all(term in text_terms for term in window):
                phrase_terms.update(window)
    return round(len(phrase_terms) / max(len(question_terms), 1), 4)


def _question_term_score(question: str, text: str) -> float:
    question_terms = content_terms(question)
    if not question_terms:
        return 1.0
    text_terms = content_terms(text)
    return round(len(question_terms & text_terms) / max(len(question_terms), 1), 4)


def _opening_cards(plan: ResearchPlan, evidence_cards: list[EvidenceCard]) -> list[EvidenceCard]:
    ranked = _cards_for_synthesis(plan, evidence_cards)
    if not ranked:
        return []
    question_terms = content_terms(plan.question)
    if not question_terms:
        return ranked
    if len(question_terms) <= 3:
        threshold = 0.67
    elif len(question_terms) <= 8:
        threshold = 0.50
    else:
        threshold = 0.35
    direct = [
        card
        for card in ranked
        if _question_term_score(plan.question, f"{card.claim} {card.supporting_excerpt} {card.source_title}") >= threshold
    ]
    return direct + [card for card in ranked if card not in direct]


def _language_label(question: str) -> str:
    return "Simplified Chinese" if preferred_output_language(question) == "zh" else "English"


def _language_instruction(question: str) -> str:
    if preferred_output_language(question) == "zh":
        return (
            "Use Simplified Chinese prose because the user request is Chinese. "
            "Keep source titles, URLs, citations, acronyms, product names, and quoted technical terms in their original language when useful."
        )
    return (
        "Use English prose because the user request is English or predominantly Latin-script. "
        "Keep source titles, URLs, citations, acronyms, product names, and quoted technical terms in their original language when useful."
    )


def _report_quality_contract(
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
) -> dict[str, Any]:
    evidence_source_count = len({card.source_id for card in evidence_cards})
    evidence_card_count = len(evidence_cards)
    branch_count = len(plan.branches)
    criteria_count = len(plan.acceptance_criteria)
    return {
        "direct_answer": [
            "Open with the answer, not background.",
            "State the central mechanism, trade-off, or decision frame that organizes the report.",
            "Keep the opening tied to the exact task scope and language.",
        ],
        "coverage": [
            f"Cover {branch_count} planned research branch(es) and {criteria_count} report-level criterion/criteria when present.",
            "Make every important sub-question visible in prose even when related branches are grouped.",
            "Separate direct evidence from adjacent examples or narrower case studies.",
        ],
        "depth_and_insight": [
            "Explain causes, mechanisms, boundary conditions, trade-offs, and consequences instead of only listing facts.",
            "Compare where sources agree, disagree, or answer different parts of the question.",
            "Add implications or future directions only when they follow from cited evidence.",
        ],
        "evidence_use": [
            f"Use the strongest {min(evidence_card_count, MAX_SYNTHESIS_CARDS_TOTAL)} evidence cards from {evidence_source_count} source(s).",
            "Attach citations to the exact claims they support; avoid broad citation stacks that include weakly related sources.",
            "Name evidence strength and limitations when the source set is uneven, indirect, or incomplete.",
        ],
        "readability": [
            "Use natural headings that match the argument and make scanning easy.",
            "Use cohesive paragraphs with topic sentences, interpretation, and transitions.",
            "Use tables only when they clarify question-specific comparisons; otherwise prefer prose synthesis.",
        ],
        "coverage_status": {
            "complete": coverage.complete,
            "score": coverage.coverage_score,
            "missing_branches": coverage.missing_branches,
        },
    }


def _compact_quality_contract(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            compact[key] = [str(item)[:240] for item in value[:5]]
        elif isinstance(value, dict):
            compact[key] = value
        else:
            compact[key] = value
    return compact


def _report_labels(question: str) -> dict[str, str]:
    if preferred_output_language(question) == "zh":
        return {
            "title": "\u7814\u7a76\u62a5\u544a",
            "bottom_line": "\u6838\u5fc3\u7ed3\u8bba",
            "synthesis": "\u7efc\u5408\u5206\u6790",
            "comparison": "\u5bf9\u6bd4\u5206\u6790",
            "additional_evidence": "\u8865\u5145\u8bc1\u636e\uff1a",
            "implications": "\u542b\u4e49",
            "limits": "\u5c40\u9650\u4e0e\u4fe1\u5fc3",
            "gaps": "\u8bc1\u636e\u7f3a\u53e3",
            "no_synthesis": "\u7531\u4e8e\u6ca1\u6709\u8db3\u591f\u901a\u8fc7\u9a8c\u8bc1\u7684\u8bc1\u636e\u5361\uff0c\u65e0\u6cd5\u5b8c\u6210\u8de8\u6765\u6e90\u7efc\u5408\u3002",
            "no_table": "\u7531\u4e8e\u6ca1\u6709\u8db3\u591f\u901a\u8fc7\u9a8c\u8bc1\u7684\u8bc1\u636e\u5361\uff0c\u65e0\u6cd5\u751f\u6210\u5bf9\u6bd4\u8868\u3002",
            "no_takeaways": "\u6ca1\u6709\u5e72\u51c0\u8bc1\u636e\u5361\u65f6\uff0c\u62a5\u544a\u4e0d\u80fd\u7ed9\u51fa\u6709\u8bc1\u636e\u652f\u6491\u7684\u5efa\u8bae\u3002",
            "no_confidence": "\u7cfb\u7edf\u6ca1\u6709\u6536\u96c6\u5230\u8db3\u591f\u5e72\u51c0\u7684\u8bc1\u636e\u6765\u652f\u6301\u9ad8\u4fe1\u5fc3\u7ed3\u8bba\u3002",
            "gaps_sentence": "\u90e8\u5206\u8ba1\u5212\u7814\u7a76\u8303\u56f4\u4ecd\u7136\u8bc1\u636e\u4e0d\u8db3\u3002",
        }
    return {
        "title": "Research Report",
        "bottom_line": "Bottom Line",
        "synthesis": "What the Sources Show Together",
        "comparison": "Comparison Table",
        "additional_evidence": "Additional Evidence on",
        "implications": "Implications",
        "limits": "Limits and Confidence",
        "gaps": "Evidence Gaps",
        "no_synthesis": "Cross-source synthesis could not be completed because no evidence cards passed the gates.",
        "no_table": "No comparison table could be generated because no evidence cards passed hygiene gates.",
        "no_takeaways": "The report cannot make evidence-backed recommendations without clean evidence cards.",
        "no_confidence": "The system did not gather enough clean evidence to support a confident answer.",
        "gaps_sentence": "Some planned evidence areas remained under-supported.",
    }


def _coverage_repair_labels(question: str) -> dict[str, str]:
    if preferred_output_language(question) == "zh":
        return {
            "heading": "\u8865\u5145\u5206\u6790\uff1a",
            "branch_prefix": "\u5173\u4e8e",
            "evidence_adds": "\u8bc1\u636e\u8fdb\u4e00\u6b65\u8868\u660e",
            "breadth_prefix": "\u8865\u5145\u8bc1\u636e\u6269\u5c55\u4e86\u6765\u6e90\u57fa\u7840\uff0c\u663e\u793a",
        }
    return {
        "heading": "Additional Source-Backed Analysis:",
        "branch_prefix": "On",
        "evidence_adds": "the evidence further indicates that",
        "breadth_prefix": "Additional evidence broadens the source base by showing that",
    }


def _report_title(question: str) -> str:
    labels = _report_labels(question)
    cleaned = re.sub(r"\s+", " ", question.strip(" ?.!")).strip()
    if not cleaned:
        return labels["title"]
    if len(cleaned) > 90:
        boundary = cleaned.rfind(" ", 0, 90)
        cleaned = cleaned[: boundary if boundary > 40 else 90].strip()
    return f"{labels['title']}: {cleaned}"


def _report_subject(question: str) -> str:
    title = _report_title(question)
    prefix = f"{_report_labels(question)['title']}: "
    return title.replace(prefix, "", 1)


def _question_label(question: str) -> str:
    cleaned = re.sub(r"\s+", " ", question.strip()).strip()
    if not cleaned:
        return "the user's question"
    return cleaned.rstrip("?!.")


def _executive_summary_sentence(question: str, cards: list[EvidenceCard]) -> str:
    central_cards = sorted(cards, key=lambda card: _card_rank_key(card, question=question))[:3]
    claims = "; ".join(card.claim.rstrip(". ") for card in central_cards)
    if not claims:
        return _no_evidence_sentence(question)
    if preferred_output_language(question) == "zh":
        return (
            f"\u56f4\u7ed5\u7528\u6237\u95ee\u9898\u201c{_question_label(question)}\u201d\uff0c"
            f"\u8bc1\u636e\u5e94\u9996\u5148\u56de\u7b54\u539f\u59cb\u5173\u7cfb\u672c\u8eab\uff0c"
            f"\u800c\u4e0d\u662f\u53ea\u56de\u7b54\u67d0\u4e2a\u6848\u4f8b\u6216\u76f8\u90bb\u4e3b\u9898\u3002"
            f"\u6700\u6709\u652f\u6491\u7684\u8981\u70b9\u662f\uff1a{claims}\u3002{_citation_group(central_cards)}"
        )
    return (
        f"In answer to the question, the evidence should be read around the central request: "
        f"{_question_label(question)}. The strongest source-backed points are that {claims}. "
        f"{_citation_group(central_cards)}"
    )


def _sentence_with_citation(card: EvidenceCard) -> str:
    claim = card.claim.rstrip(". ")
    return f"{claim}. [{card.source_id}]"


def _synthesis_sentence(cards: list[EvidenceCard], *, question: str = "") -> str:
    claims = "; ".join(card.claim.rstrip(". ") for card in cards)
    if preferred_output_language(question) == "zh":
        return f"\u7efc\u5408\u6765\u770b\uff0c\u8bc1\u636e\u663e\u793a\u8fd9\u4e9b\u53d1\u73b0\u4e0d\u662f\u5b64\u7acb\u7684\uff1a{claims}\u3002{_citation_group(cards)}"
    return f"Taken together, the evidence indicates a linked pattern rather than isolated findings: {claims}. {_citation_group(cards)}"


def _takeaway_sentence(cards: list[EvidenceCard], *, question: str = "") -> str:
    claims = "; ".join(card.claim.rstrip(". ") for card in cards[:3])
    if preferred_output_language(question) == "zh":
        return f"\u5b9e\u9645\u542b\u4e49\u5e94\u4ece\u6700\u5f3a\u652f\u6491\u70b9\u51fa\u53d1\uff1a{claims}\u3002{_citation_group(cards[:3])}"
    return f"Practical takeaways should follow the strongest supported points: {claims}. {_citation_group(cards[:3])}"


def _limitations_sentence(coverage: CoverageMatrix, *, question: str = "") -> str:
    if preferred_output_language(question) == "zh":
        if coverage.missing_branches:
            return "\u7531\u4e8e\u90e8\u5206\u8ba1\u5212\u7814\u7a76\u8303\u56f4\u4ecd\u7f3a\u5c11\u8db3\u591f\u8bc1\u636e\uff0c\u7ed3\u8bba\u4fe1\u5fc3\u6709\u9650\u3002"
        return "\u7ed3\u8bba\u4fe1\u5fc3\u53d6\u51b3\u4e8e\u6765\u6e90\u8d28\u91cf\u3001\u7814\u7a76\u8303\u56f4\u3001\u8d44\u6599\u65f6\u6548\u6027\u548c\u5206\u652f\u8986\u76d6\u5b8c\u6574\u5ea6\u3002"
    if coverage.missing_branches:
        return "Confidence is limited because some planned branches remained incomplete."
    return "Confidence depends on source quality, scope, recency, and branch coverage."


def _no_evidence_sentence(question: str) -> str:
    if preferred_output_language(question) == "zh":
        return f"\u5f53\u524d\u6ca1\u6709\u8db3\u591f\u901a\u8fc7\u9a8c\u8bc1\u7684\u8bc1\u636e\u6765\u56de\u7b54\uff1a{_question_label(question)}\u3002"
    return f"No sufficient evidence was gathered to answer: {_question_label(question)}."


def _cards_needed_for_source_breadth(
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    cited_source_ids: set[int],
) -> list[EvidenceCard]:
    target = min(17, len({card.source_id for card in evidence_cards}))
    if len(cited_source_ids) >= target:
        return []
    needed = target - len(cited_source_ids)
    cards = [
        card
        for card in _cards_for_synthesis(plan, evidence_cards)
        if card.source_id not in cited_source_ids
    ]
    return _source_diverse_cards(cards, limit=needed)


def _comparison_table(cards: list[EvidenceCard], *, plan: ResearchPlan | None = None) -> list[str]:
    branch_title_by_id = {branch.id: branch.title for branch in (plan.branches if plan else [])}
    if plan and preferred_output_language(plan.question) == "zh":
        rows = [
            "| \u7ef4\u5ea6 | \u8bc1\u636e | \u6765\u6e90 |",
            "| --- | --- | --- |",
        ]
    else:
        rows = [
            "| Dimension | Evidence | Source |",
            "| --- | --- | --- |",
        ]
    for card in cards[:6]:
        dimension = branch_title_by_id.get(card.branch_id) or "Evidence area"
        rows.append(f"| {_escape_table(dimension[:90])} | {_escape_table(card.claim[:180])} | [{card.source_id}] |")
    return rows


def _escape_table(text: str) -> str:
    return text.replace("|", " ").replace("\n", " ").strip()


def _citation_group(cards: list[EvidenceCard]) -> str:
    source_ids = sorted({card.source_id for card in cards})
    return " ".join(f"[{source_id}]" for source_id in source_ids)


def _visual_assets_from_sources(sources: list[SourceRecordV2]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for source in sources:
        for raw_asset in _metadata_images(source.metadata):
            url = str(raw_asset.get("url") or "").strip()
            if not re.match(r"^https?://", url, flags=re.I):
                continue
            alt = str(raw_asset.get("alt") or raw_asset.get("title") or source.title).strip()
            assets.append({"source_id": source.id, "url": url, "alt": alt[:160]})
    return assets


def _metadata_images(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    images = metadata.get("images") or metadata.get("image_urls") or metadata.get("visual_assets") or []
    if isinstance(images, str):
        return [{"url": images}]
    if isinstance(images, list):
        result = []
        for item in images:
            if isinstance(item, str):
                result.append({"url": item})
            elif isinstance(item, dict):
                result.append(item)
        return result
    return []


def _compact_blueprint_for_prompt(blueprint: dict[str, Any]) -> dict[str, Any]:
    branch_briefs = []
    for row in blueprint.get("branch_writing_briefs", []):
        if not isinstance(row, dict):
            continue
        branch_briefs.append(
            {
                "branch_id": row.get("branch_id"),
                "heading": row.get("heading"),
                "objective": row.get("objective"),
                "evidence_card_count": row.get("evidence_card_count"),
                "source_count": row.get("source_count"),
                "required_points": list(row.get("required_points", []))[:8],
            }
        )
    return {
        "report_title": blueprint.get("report_title"),
        "question": blueprint.get("question"),
        "audience": blueprint.get("audience"),
        "source_summary": blueprint.get("source_summary"),
        "acceptance_criteria": list(blueprint.get("acceptance_criteria", []))[:12],
        "branch_writing_briefs": branch_briefs,
        "structure_guidance": blueprint.get("structure_guidance"),
    }


def _minimal_blueprint_for_prompt(blueprint: dict[str, Any]) -> dict[str, Any]:
    branch_briefs = []
    for row in blueprint.get("branch_writing_briefs", []):
        if not isinstance(row, dict):
            continue
        branch_briefs.append(
            {
                "heading": row.get("heading"),
                "objective": str(row.get("objective") or "")[:180],
                "source_count": row.get("source_count"),
            }
        )
    return {
        "report_title": blueprint.get("report_title"),
        "audience": blueprint.get("audience"),
        "source_summary": blueprint.get("source_summary"),
        "branch_writing_briefs": branch_briefs[:10],
    }


def _compact_target_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "minimum_words": profile.get("minimum_words"),
        "target_words": profile.get("target_words"),
        "minimum_cited_paragraphs": profile.get("minimum_cited_paragraphs"),
        "minimum_major_sections_before_sources": profile.get("minimum_major_sections_before_sources"),
        "criteria_rich": profile.get("criteria_rich"),
        "style": profile.get("style"),
        "section_purposes": [
            str(row.get("purpose", ""))[:120]
            for row in profile.get("section_plan", [])
            if isinstance(row, dict)
        ][:12],
    }


def _target_report_profile(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    writing_guidance: str,
) -> dict[str, Any]:
    evidence_sources = len({card.source_id for card in evidence_cards})
    branch_count = len(plan.branches)
    criteria_count = len(_report_level_criteria(plan))
    criteria_rich = _criteria_rich_plan(plan, writing_guidance=writing_guidance)
    if criteria_rich:
        min_words = min(
            9000,
            max(
                6500,
                criteria_count * 220,
                branch_count * 520,
                evidence_sources * 150,
            ),
        )
        target_words = min(11000, max(min_words + 1500, int(min_words * 1.25)))
        min_cited_paragraphs = min(52, max(28, criteria_count + 4, branch_count * 3))
        min_major_sections = min(30, max(16, branch_count + 7, criteria_count // 2))
    elif evidence_sources >= 30 or branch_count >= 8:
        min_words = 3200
        target_words = 5600
        min_cited_paragraphs = min(30, max(14, branch_count * 2))
        min_major_sections = min(12, max(6, branch_count))
    elif evidence_sources >= 17 or branch_count >= 5:
        min_words = 2600
        target_words = 4500
        min_cited_paragraphs = min(24, max(10, branch_count * 2))
        min_major_sections = min(10, max(5, branch_count))
    else:
        min_words = 1400
        target_words = 2800
        min_cited_paragraphs = max(5, min(14, branch_count * 2 + 2))
        min_major_sections = max(3, min(7, branch_count + 2))

    return {
        "minimum_words": min_words,
        "target_words": target_words,
        "minimum_cited_paragraphs": min_cited_paragraphs,
        "minimum_major_sections_before_sources": min_major_sections,
        "criteria_rich": criteria_rich,
        "criteria_count": criteria_count,
        "branch_count": branch_count,
        "evidence_source_count": evidence_sources,
        "section_plan": _dynamic_section_plan(plan, criteria_rich=criteria_rich),
        "style": (
            "reference-grade analytical report"
            if criteria_rich
            else "substantial evidence-backed report"
            if min_words >= 2600
            else "focused evidence-backed report"
        ),
    }


def _target_depth_hint(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    writing_guidance: str,
    target_profile: dict[str, Any] | None = None,
) -> str:
    profile = target_profile or _target_report_profile(
        plan=plan,
        evidence_cards=evidence_cards,
        writing_guidance=writing_guidance,
    )
    evidence_sources = len({card.source_id for card in evidence_cards})
    branch_count = len(plan.branches)
    criteria_rich = bool(profile.get("criteria_rich"))
    min_words = int(profile.get("minimum_words", 0) or 0)
    target_words = int(profile.get("target_words", 0) or 0)
    if criteria_rich:
        return (
            f"write a reference-grade long-form report of at least {min_words} words, targeting about {target_words} words "
            "when the evidence supports it; cover each task-specific criterion with substantive analysis, cross-source "
            "synthesis, and clear implications, not a checklist."
        )
    if evidence_sources >= 30 or branch_count >= 8:
        return f"write a thorough report of at least {min_words} words, targeting about {target_words} words when the evidence supports it."
    if evidence_sources >= 17 or branch_count >= 5:
        return f"write a substantial report of at least {min_words} words, targeting about {target_words} words when the evidence supports it."
    return "write enough detail to answer the question fully without padding; prefer depth over brevity when evidence supports it."


def _report_level_criteria(plan: ResearchPlan) -> list[str]:
    criteria: list[str] = []
    for criterion in plan.acceptance_criteria:
        cleaned = re.sub(r"^Cover this task-specific criterion in synthesis:\s*", "", criterion.strip(), flags=re.I)
        cleaned = re.sub(r"^Task-specific\s+[^:]{1,80}\s+criterion:\s*", "", cleaned, flags=re.I)
        cleaned = cleaned.strip(" .:")
        if cleaned and len(content_terms(cleaned)) >= 2:
            criteria.append(cleaned)
    return _dedupe_text(criteria)


def _criteria_rich_plan(plan: ResearchPlan, *, writing_guidance: str = "") -> bool:
    if re.search(r"\bDeepResearch Bench evaluation guidance\b|\bdimension weight\b", writing_guidance, flags=re.I):
        return True
    if len(_report_level_criteria(plan)) >= 8:
        return True
    return any("task-specific" in criterion.lower() and "criterion" in criterion.lower() for criterion in plan.acceptance_criteria)


def _dynamic_section_plan(plan: ResearchPlan, *, criteria_rich: bool) -> list[dict[str, str]]:
    sections: list[dict[str, str]] = [
        {
            "purpose": "Direct answer and thesis",
            "must_do": "Answer the user's question in the first substantive paragraph, state confidence, and name the organizing mechanism or decision frame.",
        }
    ]
    if criteria_rich:
        sections.extend(
            [
                {
                    "purpose": "Definitions and conceptual grounding",
                    "must_do": "Define the central constructs and any essential theory or measurement concepts before making complex claims.",
                },
                {
                    "purpose": "Mechanisms and causal logic",
                    "must_do": "Explain how the relationship, system, process, or comparison works, including mediators, moderators, trade-offs, or boundary conditions where relevant.",
                },
                {
                    "purpose": "Evidence review",
                    "must_do": "Synthesize the strongest direct evidence and distinguish direct evidence from adjacent examples or indirect context.",
                },
            ]
        )
    for branch in plan.branches[:10]:
        sections.append(
            {
                "purpose": branch.title,
                "must_do": branch.objective[:260],
            }
        )
    sections.extend(
        [
            {
                "purpose": "Cross-source synthesis",
                "must_do": "Compare agreement, disagreement, evidence strength, and what the sources imply together.",
            },
            {
                "purpose": "Implications and future directions",
                "must_do": "State practical, theoretical, or decision-relevant implications and forward-looking questions supported by evidence.",
            },
            {
                "purpose": "Limitations and confidence",
                "must_do": "Name limits, uncertainty, source constraints, and what evidence would change the conclusion.",
            },
        ]
    )
    return sections[:16]


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", value.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)

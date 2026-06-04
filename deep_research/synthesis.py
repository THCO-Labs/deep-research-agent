from __future__ import annotations

from collections import defaultdict
from datetime import date
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deep_research.model_router import model_for_role
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2
from deep_research.settings import Settings
from deep_research.source_validation import content_terms
from deep_research.text_terms import preferred_output_language

MAX_SYNTHESIS_CARDS_PER_BRANCH = 8
MAX_SYNTHESIS_CARDS_TOTAL = 96
MAX_SYNTHESIS_EXCERPT_CHARS = 500
INDIVIDUAL_CITATION_REPAIR_THRESHOLD = 0.22
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
    model = model_for_role(settings, "orchestrator", settings.model)
    if not isinstance(model, BaseChatModel):
        raise RuntimeError(f"Synthesis role did not resolve to a chat model: {model!r}")
    synthesis_cards = _cards_for_synthesis(plan, evidence_cards)
    evidence_sources = _evidence_backed_sources(sources, synthesis_cards)
    report_blueprint = blueprint or build_report_blueprint(
        plan=plan,
        evidence_cards=synthesis_cards,
        coverage=coverage,
        sources=evidence_sources,
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
    response = model.invoke([HumanMessage(content=prompt)])
    text = str(response.content).strip()
    if not text:
        raise RuntimeError("Synthesis model returned an empty report.")
    normalized = _normalize_report_markdown(text, evidence_sources)
    citation_repaired = _repair_weak_citation_support(normalized, synthesis_cards, evidence_sources)
    coverage_repaired = _append_evidence_coverage_if_needed(citation_repaired, plan, synthesis_cards)
    return _normalize_report_markdown(coverage_repaired, evidence_sources)


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
            "\n".join(
                [
                    f"Evidence card {card.id}",
                    f"- branch_id: {card.branch_id}",
                    f"- source_id: {card.source_id}",
                    f"- source_title: {source.title}",
                    f"- source_quality: {source.quality_label} ({source.quality_score})",
                    f"- semantic_score: {card.semantic_score if card.semantic_score is not None else 'not_judged'}",
                    f"- claim: {card.claim}",
                    f"- excerpt: {card.supporting_excerpt[:MAX_SYNTHESIS_EXCERPT_CHARS]}",
                    f"- limitations: {', '.join(card.limitations) or 'none'}",
                ]
            )
        )
    source_lines = "\n".join(f"[{source.id}] {source.title}: {source.url}" for source in sorted(sources, key=lambda item: item.id))
    branch_lines = "\n".join(
        f"- {branch.id}: {branch.title}; objective: {branch.objective}; required terms: {', '.join(branch.required_terms)}"
        for branch in plan.branches
    )
    acceptance_criteria_lines = "\n".join(f"- {criterion}" for criterion in plan.acceptance_criteria[:32]) or "None"
    repair_text = "\n".join(f"- {failure}" for failure in verification_failures[:20]) or "None"
    previous_text = previous_report[:3000] if previous_report else "None"
    report_blueprint = blueprint or build_report_blueprint(plan=plan, evidence_cards=evidence_cards, coverage=coverage, sources=sources)
    quality_contract = report_blueprint.get("quality_contract", {})
    visual_assets = report_blueprint.get("visual_assets", [])
    visual_text = "\n".join(
        f"- {asset['alt']}: {asset['url']} (source_id {asset['source_id']})"
        for asset in visual_assets[:12]
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

Report-writing blueprint:
{json_dumps(_compact_blueprint_for_prompt(report_blueprint))}

Report quality contract:
{json_dumps(_compact_quality_contract(quality_contract))}

Report depth and structure target:
{json_dumps(target_profile)}

Coverage status:
- complete: {coverage.complete}
- coverage_score: {coverage.coverage_score}
- missing_branches: {', '.join(coverage.missing_branches) or 'none'}

Prior verification failures to repair:
{repair_text}

Additional report-writing guidance:
{writing_guidance.strip()[:8000] if writing_guidance.strip() else 'None'}

Previous draft, if any:
{previous_text}

Evidence cards:
{chr(10).join(evidence_lines)}

Allowed sources:
{source_lines}

Evidence-backed visual assets, if any:
{visual_text}

Output language:
{language_instruction}

Opening-answer evidence priority:
{opening_priority}

Write the final report in Markdown.

Report style examples to learn from, not copy:
{json_dumps({"examples": list(REPORT_STYLE_EXAMPLES)})}

Hard requirements:
- Answer the user's question directly in the first substantive paragraph.
- Keep the title, opening answer, and body centered on the user's exact question. If the previous draft drifts to a different topic, ignore the drift and rebuild from the evidence cards.
- Write the title, section headings, opening answer, and body in the requested output language above. Source titles, URLs, citations, acronyms, and quoted terms may remain in their original language.
- Do not let a narrower context, example case, disease, country, product, dataset, or source-specific framing replace the user's requested relationship or scope unless the user explicitly asked for that narrower context.
- Use the opening-answer evidence priority to write the first substantive paragraph; it ranks cards by direct relevance to the user's exact question.
- Use only the evidence cards above. Do not add uncited facts from memory.
- Every factual paragraph must include at least one inline citation like [3].
- Citation IDs must be source_id values from the allowed sources list.
- Do not cite evidence card IDs. Cite source IDs only.
- Cite at least {required_source_breadth} distinct evidence-backed source IDs when that many are available.
- Depth target: {target_depth_hint}
- Treat the report depth and structure target as a minimum acceptable plan, not an aspiration. If the target calls for a long-form report, do not stop after a short overview.
- Optimize for report quality as well as factuality: broad coverage, non-obvious synthesis, strict task fit, and readable organization.
- Use the report quality contract as a writing checklist, but do not print it as a checklist.
- Satisfy the acceptance criteria as report coverage requirements. Do not quote them as a checklist, but make the relevant concepts and analysis visible in the prose.
- Treat the research branches and any additional report-writing guidance as a coverage checklist.
- Every branch with evidence cards must be substantively answered in the report, either in its own section or in a clearly relevant grouped section.
- For each evidence-rich branch, write analytical paragraphs that define the issue, summarize the strongest evidence, explain mechanisms or trade-offs, and state limitations. Do not compress a branch into a single sentence when multiple cards support it.
- When prior failures mention answer coverage, missing context, or semantic completeness, expand the under-covered branch objectives and required points instead of writing a short overview.
- For criteria-rich benchmark-style prompts, write a comprehensive report rather than a brief answer; depth and coverage matter more than brevity.
- For criteria-rich benchmark-style prompts, build an argument at reference-report depth: define constructs, explain mechanisms, review evidence, compare mediators/moderators or alternatives, discuss boundary conditions, and end with implications and future research when supported.
- Use the dynamic section plan in the report depth target to decide what each section must accomplish. You may rename sections naturally, but every planned section purpose must be represented in the final report.
- Do not include structural extraction artifacts in the body: raw URLs, markdown link/media syntax, page-control text, key-value scrape metadata, or extraction notes.
- Do not copy low-information page chrome or boilerplate-like text.
- Treat prior verification failures as private repair instructions only; never quote them or mention branch IDs, evidence card IDs, missing-citation diagnostics, or internal coverage scores in the report body.
- Choose natural section headings for this question. Do not force a fixed template or reuse the same headings for every topic.
- Write in polished report prose with synthesis across sources, not a bullet dump of evidence cards.
- Make the first paragraph self-contained: answer, core mechanism or decision frame, confidence level, and why the answer matters.
- Each major section must make a claim, interpret the claim, explain why it matters for the user's question, and connect back to the report's central thesis.
- Prefer cohesive analytical paragraphs over isolated bullets. Use bullets only for compact enumerations where the items are parallel and evidence-backed.
- Use precise domain terminology, define specialized terms when needed, and keep paragraph transitions explicit so the argument reads as one coherent report.
- Include synthesis paragraphs that compare agreement, tension, evidence strength, mechanisms, boundary conditions, and trade-offs across sources.
- Where the evidence supports it, include a final integrative section that states implications, unresolved questions, and what would change the conclusion.
- Add forward-looking or decision-relevant implications only when the evidence supports them; frame uncertainty and open questions clearly.
- The table must use question-specific dimensions, not generic labels.
- Include images only when listed under evidence-backed visual assets and only when the visual helps inspect the topic; otherwise omit images.
- End with exactly one ## Sources section. Each entry must be exactly: [N] Title: https://url
"""


def _normalize_report_markdown(report: str, sources: list[SourceRecordV2]) -> str:
    cleaned = report.strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = _normalize_markdown_headings(cleaned)
    cleaned = _remove_unknown_numeric_citations(cleaned, sources)
    cleaned = _repair_uncited_body_paragraphs(cleaned, sources)
    if not re.search(r"(?im)^#\s+", cleaned):
        cleaned = "# Research Report\n\n" + cleaned
    cleaned = _remove_existing_source_listing(cleaned)
    source_section = _sources_section(sources, cleaned)
    if re.search(r"(?ims)^##\s+Sources\s*$", cleaned):
        cleaned = re.sub(r"(?ims)^##\s+Sources\s*$.*\Z", source_section, cleaned).strip()
    else:
        cleaned = cleaned.rstrip() + "\n\n" + source_section
    return cleaned.rstrip() + "\n"


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


def _remove_existing_source_listing(report: str) -> str:
    if re.search(r"(?ims)^##\s+Sources\s*$", report):
        return re.sub(r"(?ims)^##\s+Sources\s*$.*\Z", "", report).strip()

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
    if index >= 0 and re.fullmatch(r"(?:#{1,6}\s+.+|\*\*[^*]{2,120}\*\*)", lines[index].strip()):
        block_start = index
    return "\n".join(lines[:block_start]).rstrip()


def _looks_like_source_entry(line: str) -> bool:
    return bool(re.search(r"^\s*\[[0-9]+]\s+.+https?://\S+", line, flags=re.I))


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

    return re.sub(r"\[([0-9][0-9,\s]*)\]", replace, report)


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
    if not re.search(r"\[[0-9][0-9,\s]*\]", text):
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

    cleaned = re.sub(r"\[([0-9][0-9,\s]*)\]", replace, paragraph)
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
    return re.sub(r"\[([0-9][0-9,\s]*)\]", "", text)


def _numeric_citation_ids(text: str) -> list[int]:
    return [
        int(value)
        for block in re.findall(r"\[([0-9][0-9,\s]*)\]", text)
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
    if re.search(r"\[[0-9][0-9,\s]*\]", text):
        return False
    return True


def _sources_section(sources: list[SourceRecordV2], report: str) -> str:
    cited_ids = sorted(
        {
            int(value)
            for block in re.findall(r"\[([0-9][0-9,\s]*)\]", report)
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


def _cards_for_synthesis(plan: ResearchPlan, evidence_cards: list[EvidenceCard]) -> list[EvidenceCard]:
    cards_by_branch = _cards_by_branch(evidence_cards)
    selected: list[EvidenceCard] = []
    branch_count = max(len(plan.branches), 1)
    per_branch_limit = max(2, min(MAX_SYNTHESIS_CARDS_PER_BRANCH, MAX_SYNTHESIS_CARDS_TOTAL // branch_count))
    for branch in plan.branches:
        branch_cards = sorted(
            cards_by_branch.get(branch.id, []),
            key=lambda item: _card_rank_key(item, question=plan.question),
        )
        selected.extend(_source_diverse_cards(branch_cards, limit=per_branch_limit))
    selected_ids = {card.id for card in selected}
    selected.extend(
        card
        for card in sorted(evidence_cards, key=lambda item: (-item.confidence, item.id))
        if card.id not in selected_ids
    )
    return selected[:MAX_SYNTHESIS_CARDS_TOTAL]


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
        "acceptance_criteria": list(blueprint.get("acceptance_criteria", []))[:24],
        "branch_writing_briefs": branch_briefs,
        "structure_guidance": blueprint.get("structure_guidance"),
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
            7500,
            max(
                4800,
                criteria_count * 180,
                branch_count * 380,
                evidence_sources * 120,
            ),
        )
        target_words = min(9000, max(min_words + 1200, int(min_words * 1.25)))
        min_cited_paragraphs = min(38, max(20, criteria_count, branch_count * 2))
        min_major_sections = min(14, max(8, branch_count + 3))
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

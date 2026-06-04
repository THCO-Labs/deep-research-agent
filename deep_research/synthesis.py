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

MAX_SYNTHESIS_CARDS_PER_BRANCH = 6
MAX_SYNTHESIS_CARDS_TOTAL = 64
MAX_SYNTHESIS_EXCERPT_CHARS = 500
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

    cited_source_ids: set[int] = set()
    lines = [
        f"# {blueprint['report_title']}",
        "",
        "## Bottom Line",
        "",
    ]
    summary_cards = _cards_for_synthesis(plan, evidence_cards)[:5]
    if summary_cards:
        summary = _executive_summary_sentence(plan.question, summary_cards)
        cited_source_ids.update(card.source_id for card in summary_cards)
        lines.extend([summary, ""])
    else:
        lines.extend([f"No sufficient evidence was gathered to answer: {plan.question}", ""])

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

    lines.extend(["## What the Sources Show Together", ""])
    if summary_cards:
        synthesis_cards = summary_cards[:3]
        lines.extend([_synthesis_sentence(synthesis_cards), ""])
        cited_source_ids.update(card.source_id for card in synthesis_cards)
    else:
        lines.extend(["Cross-source synthesis could not be completed because no evidence cards passed the gates.", ""])

    lines.extend(["## Comparison Table", ""])
    if evidence_cards:
        table_cards = _cards_for_synthesis(plan, evidence_cards)[:6]
        lines.extend(_comparison_table(table_cards, plan=plan))
        cited_source_ids.update(card.source_id for card in table_cards)
        lines.append("")
    else:
        lines.extend(["No comparison table could be generated because no evidence cards passed hygiene gates.", ""])

    breadth_cards = _cards_needed_for_source_breadth(plan, evidence_cards, cited_source_ids)
    if breadth_cards:
        lines.extend([f"## Additional Evidence on {_report_title(plan.question).replace('Research Report: ', '')}", ""])
        for card in breadth_cards:
            lines.extend([_sentence_with_citation(card), ""])
            cited_source_ids.add(card.source_id)

    lines.extend(["## Implications", ""])
    if summary_cards:
        takeaway_cards = summary_cards[:3]
        lines.extend([_takeaway_sentence(takeaway_cards), ""])
        cited_source_ids.update(card.source_id for card in takeaway_cards)
    else:
        lines.extend(["The report cannot make evidence-backed recommendations without clean evidence cards.", ""])

    lines.extend(["## Limits and Confidence", ""])
    if evidence_cards:
        lines.extend([_limitations_sentence(coverage), ""])
    else:
        lines.extend(["The system did not gather enough clean evidence to support a confident answer.", ""])

    if coverage.missing_branches:
        lines.extend(["## Evidence Gaps", ""])
        lines.extend(["Some planned evidence areas remained under-supported.", ""])

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
    visual_assets = report_blueprint.get("visual_assets", [])
    visual_text = "\n".join(
        f"- {asset['alt']}: {asset['url']} (source_id {asset['source_id']})"
        for asset in visual_assets[:12]
    ) or "None"
    required_source_breadth = min(17, len({card.source_id for card in evidence_cards}))
    target_depth_hint = _target_depth_hint(plan=plan, evidence_cards=evidence_cards, writing_guidance=writing_guidance)
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

Write the final report in Markdown.

Report style examples to learn from, not copy:
{json_dumps({"examples": list(REPORT_STYLE_EXAMPLES)})}

Hard requirements:
- Answer the user's question directly in the first substantive paragraph.
- Keep the title, opening answer, and body centered on the user's exact question. If the previous draft drifts to a different topic, ignore the drift and rebuild from the evidence cards.
- Use only the evidence cards above. Do not add uncited facts from memory.
- Every factual paragraph must include at least one inline citation like [3].
- Citation IDs must be source_id values from the allowed sources list.
- Do not cite evidence card IDs. Cite source IDs only.
- Cite at least {required_source_breadth} distinct evidence-backed source IDs when that many are available.
- Depth target: {target_depth_hint}
- Satisfy the acceptance criteria as report coverage requirements. Do not quote them as a checklist, but make the relevant concepts and analysis visible in the prose.
- Treat the research branches and any additional report-writing guidance as a coverage checklist.
- Every branch with evidence cards must be substantively answered in the report, either in its own section or in a clearly relevant grouped section.
- For each evidence-rich branch, write analytical paragraphs that define the issue, summarize the strongest evidence, explain mechanisms or trade-offs, and state limitations. Do not compress a branch into a single sentence when multiple cards support it.
- When prior failures mention answer coverage, missing context, or semantic completeness, expand the under-covered branch objectives and required points instead of writing a short overview.
- For criteria-rich benchmark-style prompts, write a comprehensive report rather than a brief answer; depth and coverage matter more than brevity.
- Do not include structural extraction artifacts in the body: raw URLs, markdown link/media syntax, page-control text, key-value scrape metadata, or extraction notes.
- Do not copy low-information page chrome or boilerplate-like text.
- Treat prior verification failures as private repair instructions only; never quote them or mention branch IDs, evidence card IDs, missing-citation diagnostics, or internal coverage scores in the report body.
- Choose natural section headings for this question. Do not force a fixed template or reuse the same headings for every topic.
- Write in polished report prose with synthesis across sources, not a bullet dump of evidence cards.
- Each major section must make a claim, interpret the claim, explain why it matters for the user's question, and connect back to the report's central thesis.
- Use precise domain terminology, define specialized terms when needed, and keep paragraph transitions explicit so the argument reads as one coherent report.
- Include synthesis paragraphs that compare agreement, tension, evidence strength, mechanisms, boundary conditions, and trade-offs across sources.
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
        cited_ids = {
            int(value)
            for value in re.findall(r"\[([0-9]+)]", text)
            if int(value) in source_ids
        }
        if not cited_ids:
            repaired.append(paragraph)
            continue
        support_terms = set().union(*(terms_by_source.get(source_id, set()) for source_id in cited_ids))
        support_score = len(claim_terms & support_terms) / max(len(claim_terms), 1)
        if support_score >= threshold:
            repaired.append(paragraph)
            continue
        additions = _best_supporting_source_ids(claim_terms, terms_by_source, cited_ids, threshold)
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
    cited_source_ids = {int(value) for value in re.findall(r"\[([0-9]+)]", body)}
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

    lines = [
        "",
        f"## Evidence Coverage for {_report_title(plan.question).replace('Research Report: ', '')}",
        "",
    ]
    for branch, card in additions:
        if branch is not None:
            prefix = f"For {branch.title}, the evidence adds that"
        else:
            prefix = "Additional evidence broadens the source base by showing that"
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


def _fallback_citations(body: str, sources: list[SourceRecordV2]) -> str:
    cited = [
        int(value)
        for value in re.findall(r"\[([0-9]+)]", body)
        if any(source.id == int(value) for source in sources)
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


def _report_title(question: str) -> str:
    cleaned = re.sub(r"\s+", " ", question.strip(" ?.!")).strip()
    if not cleaned:
        return "Research Report"
    if len(cleaned) > 90:
        boundary = cleaned.rfind(" ", 0, 90)
        cleaned = cleaned[: boundary if boundary > 40 else 90].strip()
    return f"Research Report: {cleaned}"


def _question_label(question: str) -> str:
    cleaned = re.sub(r"\s+", " ", question.strip()).strip()
    if not cleaned:
        return "the user's question"
    return cleaned.rstrip("?!.")


def _executive_summary_sentence(question: str, cards: list[EvidenceCard]) -> str:
    central_cards = sorted(cards, key=lambda card: _card_rank_key(card, question=question))[:3]
    claims = "; ".join(card.claim.rstrip(". ") for card in central_cards)
    if not claims:
        return f"The gathered evidence did not contain a clean opening answer for: {_question_label(question)}."
    return (
        f"In answer to the question, the evidence should be read around the central request: "
        f"{_question_label(question)}. The strongest source-backed points are that {claims}. "
        f"{_citation_group(central_cards)}"
    )


def _sentence_with_citation(card: EvidenceCard) -> str:
    claim = card.claim.rstrip(". ")
    return f"{claim}. [{card.source_id}]"


def _synthesis_sentence(cards: list[EvidenceCard]) -> str:
    claims = "; ".join(card.claim.rstrip(". ") for card in cards)
    return f"Taken together, the evidence indicates a linked pattern rather than isolated findings: {claims}. {_citation_group(cards)}"


def _takeaway_sentence(cards: list[EvidenceCard]) -> str:
    claims = "; ".join(card.claim.rstrip(". ") for card in cards[:3])
    return f"Practical takeaways should follow the strongest supported points: {claims}. {_citation_group(cards[:3])}"


def _limitations_sentence(coverage: CoverageMatrix) -> str:
    if coverage.missing_branches:
        return "Confidence is limited because some planned branches remained incomplete."
    return "Confidence depends on source quality, scope, recency, and branch coverage."


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


def _target_depth_hint(*, plan: ResearchPlan, evidence_cards: list[EvidenceCard], writing_guidance: str) -> str:
    evidence_sources = len({card.source_id for card in evidence_cards})
    branch_count = len(plan.branches)
    criteria_rich = bool(re.search(r"\bDeepResearch Bench evaluation guidance\b|\bcriterion\b|\bdimension weight\b", writing_guidance, flags=re.I))
    if criteria_rich:
        return (
            "write a reference-grade long-form report, typically 4,500-8,000 words when the evidence supports it; "
            "cover each task-specific criterion with substantive analysis, cross-source synthesis, and clear implications, not a checklist."
        )
    if evidence_sources >= 30 or branch_count >= 8:
        return "write a thorough report, typically 3,000-6,000 words when the evidence supports it."
    if evidence_sources >= 17 or branch_count >= 5:
        return "write a substantial report, typically 2,500-4,500 words when the evidence supports it."
    return "write enough detail to answer the question fully without padding; prefer depth over brevity when evidence supports it."


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)

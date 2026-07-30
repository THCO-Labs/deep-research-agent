from __future__ import annotations

from collections import defaultdict
from datetime import date
import json
import re
import threading
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deep_research.models.model_router import model_for_role
from deep_research.graph.context_builder import format_knowledge_packets_for_prompt
from deep_research.core.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2
from deep_research.core.settings import GOOGLE_DEFAULT_MODEL, Settings
from deep_research.synthesis.section_writing import (
    AdaptiveSectionPlan,
    build_adaptive_section_plan,
    format_section_plan_for_prompt,
    refine_section_plan_from_payload,
)
from deep_research.evidence.source_validation import content_terms
from deep_research.synthesis.synthesis_refinement import (
    _build_argumentative_outline,
    _expand_report_depth_if_needed,
    _rewrite_opening_paragraph,
)
from deep_research.synthesis.synthesis_repair import (
    _append_evidence_coverage_if_needed,
    _is_degenerate_model_report,
    _normalize_report_markdown,
    _repair_weak_citation_support,
    _split_sources,
    _strip_hallucinated_specific_citations,
    _rewrite_low_overlap_cited_sentences,
)
from deep_research.synthesis.synthesis_runtime import (
    SynthesisTimeoutError,
    _invoke_with_synthesis_budget,
    _rough_token_count,
    _synthesis_model_spec,
    _synthesis_request_kwargs,
)
from deep_research.synthesis.synthesis_planning import (
    build_claim_ledger,
    build_sentence_plan,
    _format_sentence_plan_for_prompt,
)
from deep_research.synthesis.synthesis_formatting import (
    _cards_needed_for_source_breadth,
    _citation_group,
    _compact_blueprint_for_prompt,
    _compact_quality_contract,
    _compact_target_profile,
    _comparison_table,
    _coverage_repair_labels,
    _criteria_rich_plan,
    _dynamic_section_plan,
    _executive_summary_sentence,
    _language_instruction,
    _language_label,
    _limitations_sentence,
    _metadata_images,
    _minimal_blueprint_for_prompt,
    _no_evidence_sentence,
    _report_labels,
    _report_level_criteria,
    _report_quality_contract,
    _report_subject,
    _report_title,
    _sentence_with_citation,
    _synthesis_sentence,
    _takeaway_sentence,
    _target_depth_hint,
    _target_report_profile,
    _visual_assets_from_sources,
    json_dumps,
)
from deep_research.synthesis.synthesis_selection import (
    MAX_SYNTHESIS_CARDS_TOTAL,
    _blocked_source_ids_from_failures,
    _cards_by_branch,
    _cards_for_synthesis,
    _evidence_backed_sources,
    _opening_cards,
    _rank_cards,
    _source_diverse_cards,
    _without_blocked_sources,
)
from deep_research.evidence.text_terms import cjk_char_count, preferred_output_language

# Bumped again from 280 → 600 (default) and 480 → 1000 (top-cards): the same
# logic taken further. Citation accuracy improves when the writer has the
# actual source excerpt to quote from rather than a truncated fragment, and
# these were still cutting most excerpts well short of their supporting_excerpt
# length. Evidence-card budgets were raised at the same time (see
# synthesis_selection.py), so total prompt size stays well within the context
# window of the models routed to synthesis even at these larger per-card sizes.
MAX_SYNTHESIS_EXCERPT_CHARS = 600
MAX_SYNTHESIS_TOP_EXCERPT_CHARS = 1000
# Tightened from 0.22 → 0.30: a citation must show meaningful term overlap with
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
    sentence_plan: dict[str, Any] | None = None,
    section_plan: AdaptiveSectionPlan | dict[str, Any] | None = None,
    knowledge_base: dict[str, Any] | None = None,
    writing_guidance: str = "",
    citations_enabled: bool = True,
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
    report_section_plan = section_plan or build_adaptive_section_plan(
        plan=plan,
        evidence_cards=synthesis_cards,
        coverage=coverage,
        sources=evidence_sources,
    )
    report_section_plan = _refine_section_plan_with_model(
        model=model,
        plan=plan,
        evidence_cards=synthesis_cards,
        coverage=coverage,
        sources=evidence_sources,
        base_section_plan=report_section_plan,
        settings=settings,
        model_spec=model_spec,
        writing_guidance=writing_guidance,
    )
    target_profile = _target_report_profile(
        plan=plan,
        evidence_cards=synthesis_cards,
        writing_guidance=writing_guidance,
    )
    # Build an argumentative outline first so the writer has a thesis per section,
    # not just a list of terms to cover. Only for multi-branch questions.
    outline: dict[str, str] = {}
    if len(plan.branches) >= 2:
        outline = _build_argumentative_outline(
            model=model,
            plan=plan,
            evidence_cards=synthesis_cards,
            sources=evidence_sources,
            settings=settings,
            model_spec=model_spec,
        )
    prompt = _synthesis_prompt(
        plan=plan,
        evidence_cards=synthesis_cards,
        coverage=coverage,
        sources=evidence_sources,
        previous_report=previous_report,
        verification_failures=verification_failures or [],
        blueprint=report_blueprint,
        sentence_plan=sentence_plan,
        section_plan=report_section_plan,
        knowledge_base=knowledge_base,
        writing_guidance=writing_guidance,
        argumentative_outline=outline,
        citations_enabled=citations_enabled,
    )
    response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
    text = str(response.content).strip()
    if not text:
        if previous_report.strip():
            return _normalize_report_markdown(previous_report, evidence_sources, citations_enabled=citations_enabled)
        return synthesize_report(plan=plan, evidence_cards=synthesis_cards, coverage=coverage, sources=evidence_sources)
    if citations_enabled and _is_degenerate_model_report(text, synthesis_cards):
        return synthesize_report(plan=plan, evidence_cards=synthesis_cards, coverage=coverage, sources=evidence_sources)
    normalized = _normalize_report_markdown(text, evidence_sources, citations_enabled=citations_enabled)
    if not citations_enabled:
        # Citation-format burden is deliberately absent from this pass -- the
        # citation agent (a separate, later pass) inserts [N] markers and its
        # own Sources section. None of the citation-repair/coverage-stuffing
        # steps below have anything to operate on yet, and running them here
        # would inject premature, unreviewed citations or filler bullets
        # before the writer's own prose has a chance to stand on its own.
        expanded_report = _expand_report_depth_if_needed(
            model=model,
            plan=plan,
            evidence_cards=synthesis_cards,
            coverage=coverage,
            sources=evidence_sources,
            report=normalized,
            target_profile=target_profile,
            verification_failures=verification_failures or [],
            writing_guidance=writing_guidance,
            model_spec=model_spec,
            settings=settings,
            citations_enabled=False,
        )
        return _normalize_report_markdown(expanded_report, evidence_sources, citations_enabled=False)
    citation_repaired = _repair_weak_citation_support(normalized, synthesis_cards, evidence_sources)
    # Strip citations attached to specific facts (numbers, %, $, years) when no
    # cited card actually contains that fact — the FACT pipeline catches this
    # as "unsupported" and rejects the citation. Better to drop the false
    # attribution than keep a misleading citation.
    citation_repaired = _strip_hallucinated_specific_citations(citation_repaired, synthesis_cards, evidence_sources)
    # NEW: ask the model to rewrite low-overlap cited sentences using the
    # cited card's verbatim excerpt language. Single batched LLM call.
    citation_repaired = _rewrite_low_overlap_cited_sentences(
        citation_repaired,
        synthesis_cards,
        evidence_sources,
        model=model,
        settings=settings,
        model_spec=model_spec,
    )
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
    # Targeted rewrite of the opening paragraph to ensure a direct answer leads.
    final_report = _rewrite_opening_paragraph(
        model=model,
        report=_normalize_report_markdown(expanded_report, evidence_sources),
        plan=plan,
        opening_cards=_opening_cards(plan, synthesis_cards)[:6],
        sources=evidence_sources,
        settings=settings,
        model_spec=model_spec,
    )
    return _normalize_report_markdown(final_report, evidence_sources)


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _previous_report_excerpt(previous_report: str, *, limit: int = 2200) -> str:
    if not previous_report.strip():
        return "None"
    body, _separator, _source_tail = _split_sources(previous_report)
    excerpt = body.strip()
    if len(excerpt) <= limit:
        return excerpt
    head = excerpt[: int(limit * 0.6)].rstrip()
    tail = excerpt[-int(limit * 0.4) :].lstrip()
    return f"{head}\n\n...[middle of previous draft omitted]...\n\n{tail}"


def _repair_preservation_contract(
    previous_report: str, verification_failures: list[str], *, citations_enabled: bool = True
) -> str:
    if not previous_report.strip():
        return "None; this is the first synthesis pass."
    stable_map = _stable_previous_draft_map(previous_report, verification_failures, citations_enabled=citations_enabled)
    preserve_line = (
        "- Preserve the previous draft's title, section order, useful citations, and non-failing analysis "
        "unless a verification failure directly names that text."
        if citations_enabled
        else "- Preserve the previous draft's title, section order, and non-failing analysis unless a "
        "verification failure directly names that text. This draft has no citations yet -- don't worry about "
        "citations here, only about the writing itself."
    )
    lines = [
        "- Treat this as an incremental repair, not a fresh report.",
        preserve_line,
        "- Rewrite only paragraphs implicated by the failures, plus narrowly required connective text around them.",
        "- If a failure says a paragraph is weakly supported, either make that paragraph match the cited evidence card exactly or remove only the unsupported sentence.",
        "- Do not add new broad claims, new source IDs, or new sections just to compensate for a localized failure.",
        "- Preserve stable previous-draft material:",
        stable_map,
    ]
    return "\n".join(lines)


def _stable_previous_draft_map(
    previous_report: str, verification_failures: list[str], *, limit: int = 12, citations_enabled: bool = True
) -> str:
    body, _separator, _source_tail = _split_sources(previous_report)
    failure_keys = _failure_match_keys(verification_failures)
    stable_lines: list[str] = []
    for block in re.split(r"\n{2,}", body):
        text = re.sub(r"\s+", " ", block).strip()
        if not text:
            continue
        if text.startswith("#"):
            stable_lines.append(f"  - keep heading: {text[:140]}")
        elif citations_enabled and "[" in text and "]" in text and not _text_matches_failure(text, failure_keys):
            stable_lines.append(f"  - keep cited paragraph unless adjacent repair requires edits: {text[:220]}")
        elif not citations_enabled and len(text) > 40 and not _text_matches_failure(text, failure_keys):
            # No citation markers to key off of in an uncited draft -- any
            # substantive paragraph not named by a failure is presumed stable.
            stable_lines.append(f"  - keep paragraph unless adjacent repair requires edits: {text[:220]}")
        if len(stable_lines) >= limit:
            break
    return "\n".join(stable_lines) if stable_lines else "  - No stable paragraphs identified from the previous draft excerpt."


def _failure_match_keys(verification_failures: list[str]) -> list[str]:
    keys: list[str] = []
    for failure in verification_failures:
        cleaned = re.sub(r"[^a-z0-9]+", " ", str(failure).lower()).strip()
        words = [word for word in cleaned.split() if len(word) > 3]
        if len(words) >= 6:
            keys.append(" ".join(words[-12:]))
    return keys


def _text_matches_failure(text: str, failure_keys: list[str]) -> bool:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not cleaned:
        return False
    text_words = set(cleaned.split())
    for key in failure_keys:
        if key and (key in cleaned or cleaned[:120] in key):
            return True
        key_words = set(key.split())
        if key_words and len(key_words.intersection(text_words)) >= min(6, len(key_words)):
            return True
    return False


def _synthesis_prompt(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
    previous_report: str,
    verification_failures: list[str],
    blueprint: dict[str, Any] | None = None,
    sentence_plan: dict[str, Any] | None = None,
    section_plan: AdaptiveSectionPlan | dict[str, Any] | None = None,
    knowledge_base: dict[str, Any] | None = None,
    writing_guidance: str = "",
    argumentative_outline: dict[str, str] | None = None,
    citations_enabled: bool = True,
) -> str:
    source_lookup = {source.id: source for source in sources}
    evidence_lines = []
    criteria_rich = _criteria_rich_plan(plan, writing_guidance=writing_guidance)
    # No criteria_rich-specific override: MAX_SYNTHESIS_CARDS_TOTAL/PER_BRANCH are
    # already generous enough (150/25) to cover typical runs without a separate cap.
    selected_cards = _cards_for_synthesis(plan, evidence_cards)
    # Track the top 2 cards per branch so we can give them richer excerpts.
    branch_card_count: dict[str, int] = {}
    for card in selected_cards:
        source = source_lookup.get(card.source_id)
        if source is None:
            continue
        branch_card_count[card.branch_id] = branch_card_count.get(card.branch_id, 0) + 1
        # Top 2 cards per branch get the full excerpt so the writer can quote
        # verbatim from real source language rather than paraphrasing — which
        # is the single most effective way to improve FACT citation accuracy.
        excerpt_limit = (
            MAX_SYNTHESIS_TOP_EXCERPT_CHARS
            if branch_card_count[card.branch_id] <= 2
            else MAX_SYNTHESIS_EXCERPT_CHARS
        )
        evidence_lines.append(
            (
                f"- card {card.id}; branch {card.branch_id}; source [{card.source_id}] {source.title}; "
                f"claim: {card.claim}; excerpt (VERBATIM source text — quote from here for specific facts): "
                f"{card.supporting_excerpt[:excerpt_limit]}; "
                f"limits: {', '.join(card.limitations[:2]) or 'none'}; "
                f"WRITING RULE: prefer excerpt's exact wording over the claim paraphrase for specific facts from this card"
            )
        )
    source_lines = "\n".join(f"[{source.id}] {source.title}: {source.url}" for source in sorted(sources, key=lambda item: item.id))
    branch_lines = "\n".join(
        f"- {branch.id}: {branch.title}; objective: {branch.objective[:220]}"
        for branch in plan.branches
    )
    acceptance_criteria_lines = "\n".join(f"- {criterion}" for criterion in plan.acceptance_criteria[:8]) or "None"
    repair_text = "\n".join(f"- {failure}" for failure in verification_failures[:8]) or "None"
    previous_text = _previous_report_excerpt(previous_report)
    repair_contract = _repair_preservation_contract(previous_report, verification_failures, citations_enabled=citations_enabled)
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
    writer_persona = (plan.writer_persona.strip() if plan.writer_persona else "").strip()
    persona_line = writer_persona if writer_persona else (
        "You are writing a professional deep research report from verified evidence cards. "
        "For every specific fact (numbers, percentages, dates, prices, named entities), "
        "use the exact wording from the evidence card's supporting excerpt. "
        "State one factual claim per sentence so each sentence maps cleanly to a single source citation."
    )
    outline_text = "None"
    if argumentative_outline:
        outline_lines = [
            f"- {bid}: {thesis}"
            for bid, thesis in argumentative_outline.items()
            if thesis.strip()
        ]
        outline_text = "\n".join(outline_lines) if outline_lines else "None"
    claim_ledger = build_claim_ledger(plan=plan, evidence_cards=evidence_cards, sources=sources)
    content_plan = sentence_plan or build_sentence_plan(
        plan=plan,
        evidence_cards=evidence_cards,
        sources=sources,
        coverage=coverage,
        claim_ledger=claim_ledger,
    )
    claim_ledger_lines = []
    for entry in claim_ledger["claims"]:
        limitations = ", ".join(entry["limitations"]) or "none"
        claim_ledger_lines.append(
            (
                f"- {entry['claim_id']} ({entry['claim_type']}) -> cite [{entry['source_id']}]; branch {entry['branch_id']}; "
                f"allowed claim: {entry['allowed_claim']}; supporting quote: {entry['supporting_quote']}; "
                f"limits: {limitations}"
            )
        )
    claim_ledger_text = "\n".join(claim_ledger_lines) or "None"
    section_brief_lines = []
    for brief in claim_ledger["section_briefs"]:
        section_brief_lines.append(
            (
                f"- {brief['section_focus']} ({brief['branch_id']}): use claims "
                f"{', '.join(brief['claim_ids'][:10])}; source IDs "
                f"{', '.join(str(source_id) for source_id in brief['source_ids'][:10])}; "
                f"claim types: {', '.join(brief['claim_types'])}; task: {brief['writing_task']}"
            )
        )
    section_brief_text = "\n".join(section_brief_lines) or "None"
    synthesis_frame_lines = []
    for frame in claim_ledger["safe_synthesis_frames"]:
        synthesis_frame_lines.append(
            (
                f"- {frame['section_focus']}: combine claims {', '.join(frame['claim_ids'])}; "
                f"cite source IDs {', '.join(str(source_id) for source_id in frame['source_ids'])}; "
                f"{frame['allowed_move']}"
            )
        )
    synthesis_frame_text = "\n".join(synthesis_frame_lines) or "None"
    sentence_plan_text = _format_sentence_plan_for_prompt(content_plan)
    report_section_plan = section_plan or build_adaptive_section_plan(
        plan=plan,
        evidence_cards=evidence_cards,
        coverage=coverage,
        sources=sources,
    )
    section_plan_text = format_section_plan_for_prompt(report_section_plan)
    knowledge_packet_text = format_knowledge_packets_for_prompt(knowledge_base or {})
    minimum_ledger_claims = min(28, int(claim_ledger["claim_count"]))
    benchmark_argument_note = (
        "- Once citation grounding and coverage requirements above are satisfied, optimize for a coherent, "
        "insight-driven article: prioritize comparative analysis, higher-level synthesis across sections, and "
        "smooth transitions over enumerating grounded facts one after another.\n"
        if criteria_rich
        else ""
    )
    return f"""{persona_line}

Current date: {date.today().isoformat()}

User question:
{plan.question}

Research branches:
{branch_lines}

Acceptance criteria to satisfy in the report:
{acceptance_criteria_lines}

Report depth and structure target:
{json_dumps(_compact_target_profile(target_profile))}

Report quality contract:
{json_dumps(_compact_quality_contract(report_blueprint.get("quality_contract", {})))}

Coverage status:
- complete: {coverage.complete}
- coverage_score: {coverage.coverage_score}
- missing_branches: {', '.join(coverage.missing_branches) or 'none'}

Prior verification failures to repair:
{repair_text}

Repair preservation contract:
{repair_contract}

Additional report-writing guidance:
{writing_guidance.strip()[:2500] if writing_guidance.strip() else 'None'}

Previous draft, if any:
{previous_text}

Evidence cards:
{chr(10).join(evidence_lines)}

Claim ledger - factual ground for the report:
{claim_ledger_text}

Section-level claim plan:
{section_brief_text}

Safe synthesis moves:
{synthesis_frame_text}

Sentence-level content plan:
{sentence_plan_text}

Adaptive section contracts - internal controls, not mandatory visible headings:
{section_plan_text}

Knowledge-base section packets - durable workspace notes distilled from evidence:
{knowledge_packet_text}

Allowed source IDs and titles:
{source_lines}

Evidence-backed visual assets, if any:
{visual_text}

Output language:
{language_instruction}

Opening-answer evidence priority:
{opening_priority}

Argumentative outline — thesis to defend in each section (use this to structure your argument, not just list findings):
{outline_text}

Write the final report in Markdown.

Report style examples to learn from, not copy:
- Analytical explainer: direct answer, mechanisms, evidence strength, debates, implications.
- Evidence review: bottom-line evidence assessment, study/evidence distinctions, conflicts, gaps.
- Technical or decision brief: system/choice framing, trade-offs, constraints, verification checks.

Formatting tools — use when the evidence supports it:
- Use markdown tables when comparing specifications, features, prices, or options across multiple items (e.g., manufacturer × model specs, year-over-year data, pros/cons). Cite each data cell to its source when available.
- Use LaTeX math ($...$ for inline, $$...$$ for display) for formulas, growth equations, ratios, or technical calculations. Cite the source of each formula or numerical coefficient.
- Use tables and equations only when they clarify the analysis; never create generic placeholder tables with "N/A" or invented numbers.

Hard requirements:
- Answer the exact user question in the first substantive paragraph, using the opening-answer evidence priority.
- Use only the evidence cards above as your factual ground.{" Every factual paragraph needs inline source citations like [3]." if citations_enabled else " Do not insert any citation markers ([N], author-year, footnotes, or otherwise) -- a separate citation pass adds those after you finish. Write as if citations do not exist; focus entirely on comprehensiveness, insight, and structure."}
- Use the claim ledger as factual ground, not as a cage: write a complete literature-review argument, but every factual sentence must be traceable to one or more ledger entries. Do not print claim IDs in the report.
- Use the sentence-level content plan as a guide for the report's factual moves, but prioritize readability, smooth transitions, and natural prose. You may merge or reorder adjacent points to construct cohesive paragraphs, provided all factual claims remain anchored to the claim ledger.
- Use the knowledge-base packets to keep each section focused; they summarize what the durable workspace knows.{"" if citations_enabled else " Do not worry about which source ID backs which sentence -- that mapping is handled afterward."}
- Use the reasoning brief inside the knowledge-base packets as a visible control layer: caveat weak claims, mention unresolved contradictions/tensions when relevant, and avoid confident wording for branches marked unknown or low-confidence.
- If the reasoning brief lists contradiction/tension items, incorporate the caveat in the relevant analytical section; do not add unsupported outside facts.
- If weak claims remain in the reasoning brief, either strengthen them with directly supporting evidence cards or write them as limitations/uncertainties rather than settled conclusions.
- When the ledger contains enough material, draw on at least {minimum_ledger_claims} distinct ledger claims across the report, distributed across the section-level claim plan.
- Draw on at least {required_source_breadth} distinct evidence-backed sources when available, spread naturally across the report.
- Depth target: {target_depth_hint}
- Cover the branches, acceptance criteria, and depth profile as report coverage requirements, but do not print them as checklists.
{benchmark_argument_note}- Follow the adaptive section contracts as quality gates while choosing natural question-specific headings; do not make the report look templated.
- Use natural question-specific headings and polished analytical prose; synthesize across sources instead of dumping evidence cards.
- Define central constructs, explain mechanisms and trade-offs, compare evidence strength/tensions, and state limitations or unresolved questions when supported.
- Use markdown tables whenever comparing specifications, options, or data points across multiple items. Use a Mermaid diagram (```mermaid fence) whenever a process, architecture, or multi-entity relationship would be clearer as a diagram. Use LaTeX math ($...$, $$...$$) for formulas or ratios. Never invent numbers to fill a table or diagram; only use what the evidence supports.
- Keep the title, opening, and body centered on the requested scope; ignore any previous-draft drift.
- Do not create a generic "Additional Source-Backed Analysis" section; integrate evidence into question-specific sections instead.
- Do not include raw URLs, markdown links/media, page chrome, scrape metadata, branch IDs, evidence-card IDs, or verification diagnostics{" outside Sources" if citations_enabled else ""}.
- Include images only when listed as evidence-backed visual assets and useful for inspection.
{"- End with exactly one ## Sources section. Each entry must be exactly: [N] Title: https://url" if citations_enabled else "- Do not include a References or Sources section -- that is added automatically in a later pass."}
{_citation_grounding_block() if citations_enabled else ""}
SOURCE GROUNDING — for specific facts:
- When you state a specific fact (such as a number, dollar amount, percentage, year, count, or named-entity attribute), ensure it matches the evidence card's data exactly. Maintain accuracy in all numbers and names.
- Write your sentences in a natural, cohesive, and professional style. When stating specific facts (numbers, percentages, dates, prices, named entities), prefer the exact wording from the supporting excerpt — verbatim precision for data claims is required. Synthesis across multiple facts is for analysis and interpretation paragraphs, not data claims. One factual claim per sentence.
"""


def _citation_grounding_block() -> str:
    return """CITATION GROUNDING — STRICT RULES (each violation makes the report fail verification):
- Use ONLY bracket-numeral citations like [3] or [3][7]. Never use author-year, parenthetical, or textual citation formats (e.g. "(Soucy et al., 2015)", "Arnold et al., 2022", "according to Watanabe (2004)"). Even if the source excerpts you read use author-year style, your output must always use [N] bracket format.
- Cite [N] only when an evidence card from source N contains the specific claim being cited. Do NOT cite [N] for general topic relevance, related background, or "this source talks about X" — only when the card's claim or excerpt literally supports the sentence being cited.
- If a sentence states a fact (number, name, date, quote, mechanism, comparison), it must cite a source whose card actually contains that fact. If no card states the fact, rewrite the sentence to remove the unsupported claim, OR drop the citation rather than misattribute.
- Synthesis sentences that combine facts from multiple cards should cite each contributing source separately, e.g. "Buffett evolved from Graham [3] to incorporating Munger's quality focus [12]" — not "[3] [12]" stacked at the end without tying each citation to its claim.
- Cross-source interpretation is allowed and expected for RACE quality when it is cautious and anchored: compare, contrast, or generalize across ledger claims, then cite the contributing source IDs near the specific facts they support.
- Never cite a source for a claim that is more specific than what its card's excerpt or claim field actually says. If the card says "AI investment is significant" and you write "Accenture invested $3 billion in AI", do NOT cite that card for the $3 billion figure.
- When a sentence has no card that directly supports it, write it without a citation (mark as "synthesis" or general framing) rather than padding with a loosely related citation.

"""


def _refine_section_plan_with_model(
    *,
    model: BaseChatModel,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
    base_section_plan: AdaptiveSectionPlan | dict[str, Any],
    settings: Settings,
    model_spec: str,
    writing_guidance: str,
) -> AdaptiveSectionPlan | dict[str, Any]:
    if not evidence_cards:
        return base_section_plan
    base_plan = (
        base_section_plan
        if isinstance(base_section_plan, AdaptiveSectionPlan)
        else build_adaptive_section_plan(plan=plan, evidence_cards=evidence_cards, coverage=coverage, sources=sources)
    )
    prompt = _section_plan_refinement_prompt(
        plan=plan,
        evidence_cards=evidence_cards,
        base_section_plan=base_plan,
        writing_guidance=writing_guidance,
    )
    try:
        response = _invoke_with_synthesis_budget(model, prompt=prompt, settings=settings, model_spec=model_spec)
        payload = _loads_json_object(str(response.content))
    except Exception:
        return base_section_plan
    return refine_section_plan_from_payload(base_plan, payload, evidence_cards=evidence_cards, sources=sources)


def _section_plan_refinement_prompt(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    base_section_plan: AdaptiveSectionPlan,
    writing_guidance: str,
) -> str:
    branch_lines = "\n".join(
        f"- {branch.id}: {branch.title}; objective: {branch.objective}"
        for branch in plan.branches
    )
    card_lines = "\n".join(
        f"- card {card.id}; source [{card.source_id}]; branch {card.branch_id}; claim: {card.claim[:220]}"
        for card in evidence_cards[:80]
    )
    return f"""You are designing the internal section contracts for a deep research report.

Do not write the report. Improve the section contracts so the final report can be natural, non-template, and question-specific.

Rules:
- You may rename, merge, split, or reorder section contracts when it improves the report.
- Do not force fixed headings such as Executive Summary, Background, Analysis, Recommendation unless the question truly needs them.
- Do not invent evidence_card_ids or source_ids. Use only IDs listed below.
- Every section should have enough source_ids to be auditable.
- Return JSON only, no markdown.

Question:
{plan.question}

Branches:
{branch_lines}

Acceptance criteria:
{json_dumps({"criteria": plan.acceptance_criteria[:12]})}

Additional writing guidance:
{writing_guidance.strip()[:1800] if writing_guidance.strip() else "None"}

Base deterministic section plan:
{json_dumps(base_section_plan.to_dict())}

Available evidence cards:
{card_lines}

Return this schema:
{{
  "report_mode": "short adaptive mode label",
  "structure_style": "short phrase describing the natural report flow",
  "sections": [
    {{
      "id": "stable_snake_case_id",
      "title_hint": "natural title hint, not mandatory visible text",
      "role": "answer_frame|analysis_body|decision_dimension|mechanism|evidence_body|risk_or_limit|cross_source_synthesis",
      "purpose": "what this section must accomplish",
      "branch_ids": ["branch id"],
      "required_terms": ["term"],
      "acceptance_criteria": ["criterion"],
      "evidence_card_ids": [1],
      "source_ids": [1],
      "writing_task": "specific instruction for this section",
      "quality_gates": ["gate"]
    }}
  ]
}}
"""


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
        raise ValueError("section plan refinement did not return a JSON object")
    return payload

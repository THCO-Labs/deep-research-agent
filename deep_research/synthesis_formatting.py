from __future__ import annotations

import json
import re
from typing import Any

from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.source_validation import content_terms
from deep_research.text_terms import cjk_char_count, preferred_output_language
from deep_research.synthesis_selection import (
    MAX_SYNTHESIS_CARDS_TOTAL,
    _card_rank_key,
    _cards_for_synthesis,
    _source_diverse_cards,
)


def _language_label(question: str) -> str:
    return "Simplified Chinese" if preferred_output_language(question) == "zh" else "English"


def _language_instruction(question: str) -> str:
    if preferred_output_language(question) == "zh":
        return (
            "Use Simplified Chinese prose because the user request is Chinese. "
            "Do not let a narrower context or English-language source title override the user's requested answer language. "
            "Keep source titles, URLs, citations, acronyms, product names, and quoted technical terms in their original language when useful."
        )
    return (
        "Use English prose because the user request is English or predominantly Latin-script. "
        "Do not let a narrower context or non-English source title override the user's requested answer language. "
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
        "heading": "Evidence Coverage Expansion:",
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
                3200,
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

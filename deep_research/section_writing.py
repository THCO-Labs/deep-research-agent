from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.source_validation import content_terms
from deep_research.verifier import parse_inline_citations

EXACT_CLAIM_RE = re.compile(
    r"(?:"
    r"\b\d{4}\b"
    r"|\b\d{1,3}(?:,\d{3})+\b"
    r"|\b\d{2,}(?:[.]\d+)?\b"
    r"|\b\d+(?:[.,]\d+)?\s*(?:%|percent|percentage\s+points?|x|times|fold|hours?|days?|years?|"
    r"months?|usd|eur|gbp|\$|kw|hp|rpm|nm|mm|kg|lb|psi|bar)\b"
    r"|[$\u20ac\u00a3]\s*\d+(?:[.,]\d+)?"
    r")",
    flags=re.I,
)

SECTION_SUPPORT_THRESHOLD = 0.28
SECTION_EVIDENCE_LINKAGE_THRESHOLD = 0.45

@dataclass(frozen=True)
class SectionContract:
    id: str
    title_hint: str
    role: str
    purpose: str
    branch_ids: list[str] = field(default_factory=list)
    required_terms: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    evidence_card_ids: list[int] = field(default_factory=list)
    source_ids: list[int] = field(default_factory=list)
    writing_task: str = ""
    quality_gates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class AdaptiveSectionPlan:
    schema_version: int
    report_mode: str
    structure_style: str
    guidance: list[str]
    sections: list[SectionContract]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sections"] = [section.to_dict() for section in self.sections]
        return payload

@dataclass(frozen=True)
class SectionAudit:
    section_id: str
    locked: bool
    citation_support_score: float
    evidence_linkage_score: float
    cited_source_ids: list[int]
    failures: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class SectionDraft:
    section_id: str
    heading: str
    markdown: str
    match_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def build_adaptive_section_plan(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    sources: list[SourceRecordV2],
) -> AdaptiveSectionPlan:
    mode = _report_mode(plan)
    cards_by_branch: dict[str, list[EvidenceCard]] = {}
    for card in evidence_cards:
        cards_by_branch.setdefault(card.branch_id, []).append(card)

    sections: list[SectionContract] = []
    opening_cards = _top_cards(evidence_cards, limit=8)
    sections.append(
        SectionContract(
            id="opening_answer",
            title_hint=_opening_title(mode),
            role="answer_frame",
            purpose="Give the direct answer, define the decision or evidence frame, and state confidence without forcing a generic heading.",
            branch_ids=[branch.id for branch in plan.branches[:3]],
            required_terms=_dedupe_terms(_question_terms(plan.question)[:12]),
            acceptance_criteria=plan.acceptance_criteria[:4],
            evidence_card_ids=[card.id for card in opening_cards],
            source_ids=_source_ids(opening_cards),
            writing_task="Open with the answer and the organizing logic; avoid background-first prose.",
            quality_gates=_default_quality_gates(exact_numbers=True),
        )
    )

    for index, branch in enumerate(plan.branches, start=1):
        branch_cards = _top_cards(cards_by_branch.get(branch.id, []), limit=10)
        if not branch_cards:
            continue
        sections.append(
            SectionContract(
                id=_slug_id(f"branch_{index}_{branch.id}"),
                title_hint=branch.title,
                role=_branch_role(branch.title, branch.objective, mode),
                purpose=branch.objective,
                branch_ids=[branch.id],
                required_terms=_dedupe_terms(branch.required_terms[:12]),
                acceptance_criteria=branch.completion_criteria[:5],
                evidence_card_ids=[card.id for card in branch_cards],
                source_ids=_source_ids(branch_cards),
                writing_task=(
                    "Write this as natural analytical prose using a heading that fits the question. "
                    "Compare evidence, explain implications, and avoid a list-of-cards structure."
                ),
                quality_gates=_default_quality_gates(exact_numbers=True),
            )
        )

    if len(sections) > 2:
        synthesis_cards = _top_cards(evidence_cards, limit=12)
        sections.append(
            SectionContract(
                id="cross_source_synthesis",
                title_hint=_synthesis_title(mode),
                role="cross_source_synthesis",
                purpose="Resolve the branches into the final interpretation, recommendation, or evidence judgment.",
                branch_ids=[branch.id for branch in plan.branches],
                required_terms=_dedupe_terms(_coverage_terms(plan, coverage)),
                acceptance_criteria=plan.acceptance_criteria[-6:],
                evidence_card_ids=[card.id for card in synthesis_cards],
                source_ids=_source_ids(synthesis_cards),
                writing_task="Synthesize across sections; do not introduce new unsupported facts.",
                quality_gates=_default_quality_gates(exact_numbers=True)
                + ["Recommendation or conclusion must follow from cited evidence and earlier assumptions."],
            )
        )

    return AdaptiveSectionPlan(
        schema_version=1,
        report_mode=mode,
        structure_style=_structure_style(mode),
        guidance=[
            "These contracts are internal controls, not mandatory visible headings.",
            "Use natural section titles for the user question; do not print contract IDs.",
            "A section can only be locked when cited facts are source-backed and exact numbers are cited or calculated.",
            "Prefer repairing failed sections over rewriting the entire report.",
        ],
        sections=sections[:14],
    )

def refine_section_plan_from_payload(
    base_plan: AdaptiveSectionPlan,
    payload: dict[str, Any],
    *,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
) -> AdaptiveSectionPlan:
    """Apply an LLM-proposed section plan under deterministic evidence constraints."""
    if not isinstance(payload, dict):
        return base_plan
    allowed_source_ids = {source.id for source in sources}
    allowed_card_ids = {card.id for card in evidence_cards}
    base_by_id = {section.id: section for section in base_plan.sections}
    proposed_sections = payload.get("sections", [])
    if not isinstance(proposed_sections, list):
        return base_plan

    sections: list[SectionContract] = []
    for index, row in enumerate(proposed_sections[:14], start=1):
        if not isinstance(row, dict):
            continue
        base = base_by_id.get(str(row.get("id") or ""))
        title = _clean_text(row.get("title_hint") or row.get("title") or (base.title_hint if base else ""), limit=120)
        purpose = _clean_text(row.get("purpose") or (base.purpose if base else ""), limit=360)
        if not title or not purpose:
            continue
        source_ids = _filtered_ints(row.get("source_ids"), allowed_source_ids)
        card_ids = _filtered_ints(row.get("evidence_card_ids"), allowed_card_ids)
        if not source_ids and base:
            source_ids = list(base.source_ids)
        if not card_ids and base:
            card_ids = list(base.evidence_card_ids)
        if evidence_cards and not source_ids:
            continue
        sections.append(
            SectionContract(
                id=_slug_id(str(row.get("id") or title or f"section_{index}")),
                title_hint=title,
                role=_clean_role(row.get("role") or (base.role if base else "analysis_body")),
                purpose=purpose,
                branch_ids=_filtered_strings(row.get("branch_ids")) or (list(base.branch_ids) if base else []),
                required_terms=_filtered_strings(row.get("required_terms"))[:12]
                or (list(base.required_terms) if base else []),
                acceptance_criteria=_filtered_strings(row.get("acceptance_criteria"))[:6]
                or (list(base.acceptance_criteria) if base else []),
                evidence_card_ids=card_ids,
                source_ids=source_ids,
                writing_task=_clean_text(row.get("writing_task") or (base.writing_task if base else ""), limit=320),
                quality_gates=_filtered_strings(row.get("quality_gates"))[:8]
                or (list(base.quality_gates) if base else _default_quality_gates(exact_numbers=True)),
            )
        )

    if not sections:
        return base_plan
    return AdaptiveSectionPlan(
        schema_version=1,
        report_mode=_clean_role(payload.get("report_mode") or base_plan.report_mode),
        structure_style=_clean_text(payload.get("structure_style") or base_plan.structure_style, limit=120),
        guidance=base_plan.guidance,
        sections=sections,
    )

def format_section_plan_for_prompt(section_plan: AdaptiveSectionPlan | dict[str, Any]) -> str:
    payload = section_plan.to_dict() if isinstance(section_plan, AdaptiveSectionPlan) else section_plan
    sections = payload.get("sections", []) if isinstance(payload, dict) else []
    lines = [
        f"- report_mode: {payload.get('report_mode', 'general') if isinstance(payload, dict) else 'general'}",
        f"- structure_style: {payload.get('structure_style', 'adaptive') if isinstance(payload, dict) else 'adaptive'}",
        "- section contracts:",
    ]
    for section in sections[:14]:
        if not isinstance(section, dict):
            continue
        lines.append(
            "  - "
            f"{section.get('id')}: title hint={section.get('title_hint')}; "
            f"role={section.get('role')}; purpose={section.get('purpose')}; "
            f"use sources={_format_ids(section.get('source_ids', []))}; "
            f"task={section.get('writing_task')}"
        )
    return "\n".join(lines)


def audit_report_sections(
    *,
    section_plan: AdaptiveSectionPlan | dict[str, Any],
    report_markdown: str,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    source_texts: dict[int, str],
) -> dict[str, Any]:
    plan = _section_plan_from_any(section_plan)
    drafts = extract_report_sections(report_markdown, section_plan=plan)
    audits = []
    for section in plan.sections:
        draft = _best_section_draft(section, drafts)
        audits.append(
            audit_section_draft(
                section=section,
                markdown=draft.markdown if draft else "",
                evidence_cards=evidence_cards,
                sources=sources,
                source_texts=source_texts,
            ).to_dict()
            | {
                "heading": draft.heading if draft else "",
                "matched_report_section_score": draft.match_score if draft else 0.0,
                "matched_report_section_markdown": draft.markdown if draft else "",
            }
        )
    locked_count = sum(1 for audit in audits if audit.get("locked"))
    return {
        "schema_version": 1,
        "report_mode": plan.report_mode,
        "section_count": len(plan.sections),
        "locked_count": locked_count,
        "all_locked": bool(audits) and locked_count == len(audits),
        "report_sections": [draft.to_dict() for draft in drafts],
        "audits": audits,
    }


def _heading_match(heading: str, sections: list[SectionContract], used_contracts: set[str]) -> SectionContract | None:
    clean_heading = re.sub(r"[^a-z0-9\s]+", "", heading.lower()).strip()
    if not clean_heading or clean_heading in (
        "opening",
        "bottom line",
        "sources",
        "what the sources show together",
        "comparison table",
        "implications",
        "limits and confidence",
        "evidence gaps",
        "核心结论",
        "综合分析",
        "对比分析",
        "补充证据",
        "含义",
        "局限与信心",
        "证据缺口",
        "研究报告",
    ):
        return None
    words_heading = set(clean_heading.split())
    best_section = None
    best_score = 0.0
    for section in sections:
        if section.id in used_contracts:
            continue
        clean_title = re.sub(r"[^a-z0-9\s]+", "", section.title_hint.lower()).strip()
        if not clean_title:
            continue
        if clean_heading == clean_title:
            return section
        words_title = set(clean_title.split())
        intersection = words_heading & words_title
        union = words_heading | words_title
        jaccard = len(intersection) / len(union) if union else 0.0
        is_substring = len(words_heading) >= 2 and (clean_heading in clean_title or clean_title in clean_heading)
        if (jaccard >= 0.35 or is_substring) and jaccard > best_score:
            best_score = jaccard
            best_section = section
    return best_section


def extract_report_sections(
    report_markdown: str,
    *,
    section_plan: AdaptiveSectionPlan | dict[str, Any],
) -> list[SectionDraft]:
    plan = _section_plan_from_any(section_plan)
    body = _strip_sources(report_markdown)
    blocks = _heading_blocks(body)
    if not blocks:
        blocks = [("", body.strip())] if body.strip() else []

    drafts_dict: dict[int, tuple[str, float]] = {}
    used_contracts: set[str] = set()

    # Pass 1: Match by strong heading / title_hint similarity
    for i, (heading, markdown) in enumerate(blocks):
        matched_section = _heading_match(heading, plan.sections, used_contracts)
        if matched_section is not None:
            used_contracts.add(matched_section.id)
            drafts_dict[i] = (matched_section.id, 1.0)

    # Pass 2: Match remaining blocks using content overlap score
    for i, (heading, markdown) in enumerate(blocks):
        if i in drafts_dict:
            continue
        section, score = _match_contract(heading, markdown, plan.sections, used_contracts)
        if section is not None:
            used_contracts.add(section.id)
            drafts_dict[i] = (section.id, score)
        else:
            drafts_dict[i] = (_slug_id(heading or "unmatched_section"), 0.0)

    drafts: list[SectionDraft] = []
    for i, (heading, markdown) in enumerate(blocks):
        section_id, score = drafts_dict[i]
        drafts.append(
            SectionDraft(
                section_id=section_id,
                heading=heading,
                markdown=markdown.strip(),
                match_score=score,
            )
        )
    return drafts


def audit_section_draft(
    *,
    section: SectionContract | dict[str, Any],
    markdown: str,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    source_texts: dict[int, str],
) -> SectionAudit:
    section_dict = section.to_dict() if isinstance(section, SectionContract) else dict(section)
    section_id = str(section_dict.get("id") or "section")
    expected_source_ids = {int(value) for value in section_dict.get("source_ids", []) if isinstance(value, int)}
    source_by_id = {source.id: source for source in sources}
    card_sources = {card.source_id for card in evidence_cards}
    cited_source_ids = sorted(set(parse_inline_citations(markdown)))
    failures: list[str] = []
    warnings: list[str] = []

    if not markdown.strip():
        failures.append("Section is empty.")
    if not cited_source_ids:
        failures.append("Section has no citations.")
    for source_id in cited_source_ids:
        if source_id not in source_by_id:
            failures.append(f"Cited source [{source_id}] is not a usable source record.")
        if source_id not in card_sources:
            failures.append(f"Cited source [{source_id}] has no evidence card.")
    for sentence in _sentences(markdown):
        if EXACT_CLAIM_RE.search(sentence) and not parse_inline_citations(sentence):
            failures.append(f"Exact factual claim lacks a nearby citation: {sentence[:160]}")

    support_scores = _citation_support_scores(markdown, source_texts, evidence_cards)
    citation_support_score = round(sum(support_scores) / len(support_scores), 4) if support_scores else 0.0
    if support_scores and citation_support_score < SECTION_SUPPORT_THRESHOLD:
        failures.append(f"Section citation support below threshold: {citation_support_score}")
    elif not support_scores and cited_source_ids:
        warnings.append("Citations were present, but support scoring could not evaluate enough claim terms.")

    if expected_source_ids:
        linked = len(set(cited_source_ids) & expected_source_ids)
        evidence_linkage_score = round(linked / max(min(len(expected_source_ids), 4), 1), 4)
        if evidence_linkage_score < SECTION_EVIDENCE_LINKAGE_THRESHOLD:
            failures.append(f"Section cites too little of its assigned evidence: {evidence_linkage_score}")
    else:
        evidence_linkage_score = 1.0 if cited_source_ids else 0.0

    return SectionAudit(
        section_id=section_id,
        locked=not failures,
        citation_support_score=citation_support_score,
        evidence_linkage_score=evidence_linkage_score,
        cited_source_ids=cited_source_ids,
        failures=failures,
        warnings=warnings,
    )


def refine_section_audits_from_payload(
    base_payload: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Apply an LLM audit review while preserving deterministic failures."""
    if not isinstance(payload, dict):
        return base_payload
    base_audits = {
        str(audit.get("section_id")): dict(audit)
        for audit in base_payload.get("audits", [])
        if isinstance(audit, dict)
    }
    for review in payload.get("section_reviews", []):
        if not isinstance(review, dict):
            continue
        section_id = str(review.get("section_id") or "")
        if section_id not in base_audits:
            continue
        audit = base_audits[section_id]
        deterministic_failures = [str(item) for item in audit.get("failures", [])]
        llm_failures = _filtered_strings(review.get("failures"))
        llm_warnings = _filtered_strings(review.get("warnings"))
        repair_guidance = _clean_text(review.get("repair_guidance"), limit=420)
        audit["failures"] = _dedupe_terms(deterministic_failures + llm_failures)
        audit["warnings"] = _dedupe_terms([str(item) for item in audit.get("warnings", [])] + llm_warnings)
        if repair_guidance:
            audit["repair_guidance"] = repair_guidance
        # The LLM can make a passing deterministic section stricter, but cannot
        # erase deterministic failures.
        audit["locked"] = not audit["failures"] and bool(review.get("locked", audit.get("locked", False)))
    audits = list(base_audits.values())
    locked_count = sum(1 for audit in audits if audit.get("locked"))
    result = dict(base_payload)
    result["audits"] = audits
    result["locked_count"] = locked_count
    result["all_locked"] = bool(audits) and locked_count == len(audits)
    global_guidance = _filtered_strings(payload.get("global_repair_guidance"))
    if global_guidance:
        result["global_repair_guidance"] = global_guidance[:8]
    result["llm_review_applied"] = True
    return result


def _report_mode(plan: ResearchPlan) -> str:
    text = " ".join(
        [
            plan.question,
            " ".join(plan.report_outline),
            " ".join(plan.acceptance_criteria),
            " ".join(f"{branch.title} {branch.objective}" for branch in plan.branches),
        ]
    ).lower()
    signals = {
        "comparative_decision": ("compare", "versus", "vs", "choose", "recommend", "trade-off", "vendor", "product"),
        "evidence_review": ("literature", "clinical", "trial", "evidence", "meta-analysis", "systematic", "study"),
        "technical_brief": ("architecture", "implementation", "technical", "spec", "system", "integration", "engineering"),
        "market_strategy": ("market", "competitive", "strategy", "investment", "pricing", "customer", "growth"),
        "policy_brief": ("policy", "regulation", "governance", "stakeholder", "public", "legal"),
    }
    scored = {
        mode: sum(1 for term in terms if term in text)
        for mode, terms in signals.items()
    }
    best_mode, best_score = max(scored.items(), key=lambda item: item[1])
    return best_mode if best_score > 0 else "analytical_report"


def _structure_style(mode: str) -> str:
    return {
        "comparative_decision": "decision-first, trade-off driven",
        "evidence_review": "evidence-first, uncertainty-aware",
        "technical_brief": "constraint-first, mechanism-driven",
        "market_strategy": "thesis-first, scenario-aware",
        "policy_brief": "problem-options-feasibility",
    }.get(mode, "adaptive analytical narrative")


def _opening_title(mode: str) -> str:
    return {
        "comparative_decision": "Bottom-line decision frame",
        "evidence_review": "Evidence judgment",
        "technical_brief": "Technical answer",
        "market_strategy": "Strategic thesis",
        "policy_brief": "Policy answer",
    }.get(mode, "Direct answer")


def _synthesis_title(mode: str) -> str:
    return {
        "comparative_decision": "Recommendation logic",
        "evidence_review": "What the evidence supports",
        "technical_brief": "Engineering implications",
        "market_strategy": "Strategic implications",
        "policy_brief": "Feasibility and trade-offs",
    }.get(mode, "Synthesis")


def _branch_role(title: str, objective: str, mode: str) -> str:
    text = f"{title} {objective}".lower()
    if mode == "comparative_decision" and re.search(r"\b(compare|option|vendor|cost|risk|service|support)\b", text):
        return "decision_dimension"
    if re.search(r"\b(method|mechanism|why|how|causal|pathway)\b", text):
        return "mechanism"
    if re.search(r"\b(evidence|study|trial|data|source|literature)\b", text):
        return "evidence_body"
    if re.search(r"\b(risk|limit|uncertain|constraint|failure)\b", text):
        return "risk_or_limit"
    return "analysis_body"


def _default_quality_gates(*, exact_numbers: bool) -> list[str]:
    gates = [
        "Every concrete claim has a nearby source citation.",
        "Cited source IDs must have evidence cards assigned to this section or a justified cross-section role.",
        "Citations support the specific sentence, not just the general topic.",
    ]
    if exact_numbers:
        gates.append("Every exact number, date, unit, price, or percentage is cited or explicitly calculated.")
    return gates


def _top_cards(cards: list[EvidenceCard], *, limit: int) -> list[EvidenceCard]:
    return sorted(
        cards,
        key=lambda card: (
            -(card.semantic_score if card.semantic_score is not None else card.confidence),
            -card.confidence,
            -card.quality_score,
            -card.relevance_score,
            card.id,
        ),
    )[:limit]


def _source_ids(cards: list[EvidenceCard]) -> list[int]:
    return sorted({card.source_id for card in cards})


def _question_terms(question: str) -> list[str]:
    return sorted(content_terms(question), key=lambda term: (-len(term), term))


def _coverage_terms(plan: ResearchPlan, coverage: CoverageMatrix) -> list[str]:
    terms: list[str] = []
    for branch in plan.branches:
        terms.extend(branch.required_terms[:4])
    terms.extend(coverage.missing_branches)
    return _dedupe_terms(terms)[:16]


def _dedupe_terms(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", str(value).strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(str(value).strip())
    return result


def _format_ids(values: Any) -> str:
    if not isinstance(values, list) or not values:
        return "none"
    return ", ".join(str(value) for value in values[:10])


def _filtered_ints(values: Any, allowed: set[int]) -> list[int]:
    if not isinstance(values, list):
        return []
    result: list[int] = []
    for value in values:
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item in allowed and item not in result:
            result.append(item)
    return result


def _filtered_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        text = _clean_text(value, limit=220)
        if text and text not in result:
            result.append(text)
    return result


def _clean_text(value: Any, *, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _clean_role(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9_ -]+", "", str(value or "")).strip().lower().replace(" ", "_")
    return text[:80] or "analysis_body"


def _sentences(markdown: str) -> list[str]:
    body = re.sub(r"(?m)^#{1,6}\s+.*$", "", markdown)
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]


def _citation_support_scores(
    markdown: str,
    source_texts: dict[int, str],
    evidence_cards: list[EvidenceCard],
) -> list[float]:
    fallback_source_texts: dict[int, str] = {}
    for card in evidence_cards:
        fallback_source_texts.setdefault(card.source_id, "")
        fallback_source_texts[card.source_id] += f" {card.claim} {card.supporting_excerpt}"
    scores: list[float] = []
    for sentence in _sentences(markdown):
        cited_ids = sorted(set(parse_inline_citations(sentence)))
        if not cited_ids:
            continue
        claim_terms = content_terms(re.sub(r"\[\d+]", "", sentence))
        if len(claim_terms) < 4:
            continue
        source_terms: set[str] = set()
        for source_id in cited_ids:
            text = source_texts.get(source_id) or fallback_source_texts.get(source_id, "")
            source_terms.update(content_terms(text))
        if not source_terms:
            continue
        scores.append(round(len(claim_terms & source_terms) / max(len(claim_terms), 1), 4))
    return scores


def _slug_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "section"


def _section_plan_from_any(section_plan: AdaptiveSectionPlan | dict[str, Any]) -> AdaptiveSectionPlan:
    if isinstance(section_plan, AdaptiveSectionPlan):
        return section_plan
    sections = []
    for row in section_plan.get("sections", []) if isinstance(section_plan, dict) else []:
        if not isinstance(row, dict):
            continue
        sections.append(
            SectionContract(
                id=str(row.get("id") or _slug_id(str(row.get("title_hint") or "section"))),
                title_hint=str(row.get("title_hint") or row.get("title") or "Section"),
                role=str(row.get("role") or "analysis_body"),
                purpose=str(row.get("purpose") or ""),
                branch_ids=[str(item) for item in row.get("branch_ids", []) if isinstance(item, str)],
                required_terms=[str(item) for item in row.get("required_terms", []) if isinstance(item, str)],
                acceptance_criteria=[str(item) for item in row.get("acceptance_criteria", []) if isinstance(item, str)],
                evidence_card_ids=[int(item) for item in row.get("evidence_card_ids", []) if isinstance(item, int)],
                source_ids=[int(item) for item in row.get("source_ids", []) if isinstance(item, int)],
                writing_task=str(row.get("writing_task") or ""),
                quality_gates=[str(item) for item in row.get("quality_gates", []) if isinstance(item, str)],
            )
        )
    return AdaptiveSectionPlan(
        schema_version=1,
        report_mode=str(section_plan.get("report_mode") or "general") if isinstance(section_plan, dict) else "general",
        structure_style=str(section_plan.get("structure_style") or "adaptive") if isinstance(section_plan, dict) else "adaptive",
        guidance=[str(item) for item in section_plan.get("guidance", [])] if isinstance(section_plan, dict) else [],
        sections=sections,
    )


def _strip_sources(markdown: str) -> str:
    return re.split(r"(?im)^##\s+Sources\s*$", markdown, maxsplit=1)[0].strip()


def _is_just_title_preface(preface: str) -> bool:
    lines = [line.strip() for line in preface.splitlines() if line.strip()]
    content_lines = [line for line in lines if not line.startswith("#")]
    if not content_lines:
        return True
    content_text = " ".join(content_lines).strip()
    return len(content_text.split()) < 10


def _heading_blocks(markdown: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"(?m)^##\s+(.+?)\s*$", markdown))
    if not matches:
        return []
    blocks: list[tuple[str, str]] = []
    preface = markdown[: matches[0].start()].strip()
    if preface and not _is_just_title_preface(preface):
        blocks.append(("Opening", preface))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        heading = match.group(1).strip()
        blocks.append((heading, markdown[start:end].strip()))
    return blocks


def _best_section_draft(section: SectionContract, drafts: list[SectionDraft]) -> SectionDraft | None:
    candidates = [draft for draft in drafts if draft.section_id == section.id]
    if candidates:
        return max(candidates, key=lambda draft: draft.match_score)
    return None


def _match_contract(
    heading: str,
    markdown: str,
    sections: list[SectionContract],
    used_contracts: set[str],
) -> tuple[SectionContract | None, float]:
    haystack_terms = content_terms(f"{heading} {markdown[:1200]}")
    best: tuple[SectionContract | None, float] = (None, 0.0)
    for section in sections:
        if section.id in used_contracts:
            continue
        contract_terms = content_terms(
            " ".join(
                [
                    section.title_hint,
                    section.role,
                    section.purpose,
                    " ".join(section.required_terms),
                    " ".join(section.acceptance_criteria),
                ]
            )
        )
        if not contract_terms:
            continue
        score = len(haystack_terms & contract_terms) / max(len(contract_terms), 1)
        if score > best[1]:
            best = (section, round(score, 4))
    if best[1] < 0.08:
        return None, 0.0
    return best

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2
from deep_research.synthesis_selection import (
    _cards_by_branch,
    _cards_for_synthesis,
    _marginal_coverage_cards,
    _rank_cards,
    _source_diverse_cards,
)

MAX_CLAIM_LEDGER_CLAIMS_TOTAL = 48
MAX_SENTENCE_PLAN_CLAIMS_PER_SECTION = 10
MAX_SENTENCE_PLAN_SPECS_PER_SECTION = 7
MAX_SYNTHESIS_TOP_EXCERPT_CHARS = 480


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def build_claim_ledger(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
) -> dict[str, Any]:
    source_lookup = {source.id: source for source in sources}
    claims: list[dict[str, Any]] = []
    for index, card in enumerate(_cards_for_claim_ledger(plan, evidence_cards), start=1):
        source = source_lookup.get(card.source_id)
        if source is None:
            continue
        claim_type = _claim_type(card)
        claims.append(
            {
                "claim_id": f"C{index:03d}",
                "claim_type": claim_type,
                "source_id": card.source_id,
                "source_title": source.title,
                "source_url": source.url,
                "branch_id": card.branch_id,
                "allowed_claim": _compact_whitespace(card.claim),
                "supporting_quote": _compact_whitespace(card.supporting_excerpt)[:MAX_SYNTHESIS_TOP_EXCERPT_CHARS],
                "limitations": [_compact_whitespace(item) for item in card.limitations[:2]],
                "confidence": round(float(card.confidence or 0.0), 4),
                "relevance_score": round(float(card.relevance_score or 0.0), 4),
            }
        )
    return {
        "schema_version": 1,
        "question": plan.question,
        "rules": [
            "Use these entries as the factual ground for synthesis.",
            "A cited sentence must stay within the allowed_claim and supporting_quote for its cited source.",
            "Cross-source synthesis is allowed when it combines ledger entries and cites the contributing sources.",
            "If the ledger does not support a specific fact, omit or soften that fact instead of adding a loose citation.",
        ],
        "claim_count": len(claims),
        "section_briefs": _claim_ledger_section_briefs(plan, claims),
        "safe_synthesis_frames": _safe_synthesis_frames(plan, claims),
        "claims": claims,
    }


def build_sentence_plan(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    coverage: CoverageMatrix,
    claim_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = claim_ledger or build_claim_ledger(plan=plan, evidence_cards=evidence_cards, sources=sources)
    claims = [claim for claim in ledger.get("claims", []) if isinstance(claim, dict)]
    claims_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        claims_by_branch[str(claim.get("branch_id", ""))].append(claim)

    sections: list[dict[str, Any]] = []
    planned_claim_ids: set[str] = set()
    planned_source_ids: set[int] = set()
    role_distribution: dict[str, int] = defaultdict(int)

    for branch in plan.branches:
        branch_claims = claims_by_branch.get(branch.id, [])
        if not branch_claims:
            continue
        selected_claims = _source_and_role_diverse_claims(
            branch_claims,
            limit=min(MAX_SENTENCE_PLAN_CLAIMS_PER_SECTION, len(branch_claims)),
        )
        sentence_specs = _sentence_specs_for_section(branch, selected_claims)
        for spec in sentence_specs:
            for claim_id in spec["claim_ids"]:
                planned_claim_ids.add(str(claim_id))
            for source_id in spec["source_ids"]:
                planned_source_ids.add(int(source_id))
            role_distribution[str(spec["purpose"])] += 1
        sections.append(
            {
                "branch_id": branch.id,
                "heading_basis": branch.title,
                "section_role": _section_role(branch),
                "objective": branch.objective,
                "sentence_specs": sentence_specs,
            }
        )

    conclusion_claims = _source_and_role_diverse_claims(claims, limit=min(8, len(claims)))
    conclusion_specs = _sentence_specs_for_conclusion(conclusion_claims)
    if conclusion_specs:
        for spec in conclusion_specs:
            for claim_id in spec["claim_ids"]:
                planned_claim_ids.add(str(claim_id))
            for source_id in spec["source_ids"]:
                planned_source_ids.add(int(source_id))
            role_distribution[str(spec["purpose"])] += 1
        sections.append(
            {
                "branch_id": "cross_branch_synthesis",
                "heading_basis": "Overall synthesis",
                "section_role": "cross_branch_synthesis",
                "objective": "Integrate the strongest supported patterns, limits, and implications across sections.",
                "sentence_specs": conclusion_specs,
            }
        )

    return {
        "schema_version": 1,
        "question": plan.question,
        "rules": [
            "Use this as the sentence-level content plan for the report.",
            "Each planned point must be written only from its claim_ids and cited with its source_ids.",
            "You may merge adjacent planned points into polished paragraphs, but do not introduce new factual claims outside the plan or claim ledger.",
            "If a planned point feels too narrow, write it cautiously rather than padding it with unsupported facts.",
        ],
        "coverage": {
            "complete": coverage.complete,
            "coverage_score": coverage.coverage_score,
            "missing_branches": list(coverage.missing_branches),
        },
        "planned_claim_count": len(planned_claim_ids),
        "planned_source_count": len(planned_source_ids),
        "role_distribution": dict(sorted(role_distribution.items())),
        "sections": sections,
    }


def _cards_for_claim_ledger(plan: ResearchPlan, evidence_cards: list[EvidenceCard]) -> list[EvidenceCard]:
    if not evidence_cards:
        return []
    base = _cards_for_synthesis(plan, evidence_cards)
    selected: list[EvidenceCard] = []
    selected_ids: set[int] = set()
    for card in base:
        selected.append(card)
        selected_ids.add(card.id)

    cards_by_branch = _cards_by_branch(evidence_cards)
    branch_count = max(len(plan.branches), 1)
    per_branch_target = max(4, min(10, MAX_CLAIM_LEDGER_CLAIMS_TOTAL // branch_count))
    for branch in plan.branches:
        branch_cards = _rank_cards(cards_by_branch.get(branch.id, []), question=plan.question)
        branch_selected = [card for card in selected if card.branch_id == branch.id]
        for card in _source_diverse_cards(branch_cards, limit=per_branch_target):
            if len(selected) >= MAX_CLAIM_LEDGER_CLAIMS_TOTAL:
                return selected
            if len(branch_selected) >= per_branch_target:
                break
            if card.id in selected_ids:
                continue
            selected.append(card)
            selected_ids.add(card.id)
            branch_selected.append(card)

    if len(selected) >= MAX_CLAIM_LEDGER_CLAIMS_TOTAL:
        return selected[:MAX_CLAIM_LEDGER_CLAIMS_TOTAL]

    remaining = [card for card in evidence_cards if card.id not in selected_ids]
    selected.extend(
        _marginal_coverage_cards(
            plan=plan,
            candidates=remaining,
            selected=selected,
            limit=MAX_CLAIM_LEDGER_CLAIMS_TOTAL - len(selected),
        )
    )
    return selected[:MAX_CLAIM_LEDGER_CLAIMS_TOTAL]


def _claim_type(card: EvidenceCard) -> str:
    text = f"{card.claim} {card.supporting_excerpt}".lower()
    if re.search(r"\b(?:limitation|uncertain|uncertainty|gap|mixed|caution|risk|constraint|weakness|boundary)\b", text):
        return "limitation"
    if re.search(r"\b(?:compared?|versus|whereas|while|relative|difference|similar|stronger|weaker|higher|lower)\b", text):
        return "comparison"
    if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|percentage|million|billion|trillion|fold|x|years?)?\b", text):
        return "quantitative_result"
    if re.search(r"\b(?:because|through|via|driven by|associated with|linked to|explains?|mechanism|pathway|framework)\b", text):
        return "mechanism"
    if re.search(r"\b(?:define[sd]?|definition|refers to|means|concept|construct|describes?)\b", text):
        return "definition_or_scope"
    if re.search(r"\b(?:therefore|implies|suggests|recommend|should|policy|practice|decision|strategy|intervention)\b", text):
        return "implication"
    return "background_evidence"


def _claim_ledger_section_briefs(plan: ResearchPlan, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims_by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        claims_by_branch[str(claim["branch_id"])].append(claim)
    briefs: list[dict[str, Any]] = []
    for branch in plan.branches:
        branch_claims = claims_by_branch.get(branch.id, [])
        if not branch_claims:
            continue
        claim_ids = [str(claim["claim_id"]) for claim in branch_claims]
        claim_types = sorted({str(claim["claim_type"]) for claim in branch_claims})
        source_ids = sorted({int(claim["source_id"]) for claim in branch_claims})
        briefs.append(
            {
                "branch_id": branch.id,
                "section_focus": branch.title,
                "writing_task": (
                    "Write this as an analytical literature-review section: state the pattern, "
                    "compare evidence strength, then explain implications and limitations using only these claim IDs."
                ),
                "claim_ids": claim_ids,
                "source_ids": source_ids,
                "claim_types": claim_types,
            }
        )
    return briefs


def _safe_synthesis_frames(plan: ResearchPlan, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for brief in _claim_ledger_section_briefs(plan, claims):
        branch_claims = [claim for claim in claims if str(claim["branch_id"]) == str(brief["branch_id"])]
        selected = _source_and_role_diverse_claims(branch_claims, limit=8)
        claim_ids = [str(claim["claim_id"]) for claim in selected]
        source_ids = sorted({int(claim["source_id"]) for claim in selected})
        if len(claim_ids) < 2:
            continue
        frames.append(
            {
                "section_focus": brief["section_focus"],
                "claim_ids": claim_ids,
                "source_ids": source_ids,
                "allowed_move": (
                    "You may synthesize these claims into a cautious pattern statement, "
                    "as long as the sentence cites the contributing source IDs near the facts they support."
                ),
            }
        )
    return frames


def _source_and_role_diverse_claims(claims: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    seen_sources: set[int] = set()
    seen_types: set[str] = set()
    ordered = sorted(
        claims,
        key=lambda claim: (
            -float(claim.get("confidence", 0.0) or 0.0),
            -float(claim.get("relevance_score", 0.0) or 0.0),
            str(claim.get("claim_id", "")),
        ),
    )
    passes = (
        lambda claim: int(claim.get("source_id", -1)) not in seen_sources
        and str(claim.get("claim_type", "")) not in seen_types,
        lambda claim: int(claim.get("source_id", -1)) not in seen_sources,
        lambda claim: str(claim.get("claim_type", "")) not in seen_types,
        lambda _claim: True,
    )
    for predicate in passes:
        for claim in ordered:
            if len(selected) >= limit:
                return selected
            if claim in selected or not predicate(claim):
                continue
            selected.append(claim)
            seen_sources.add(int(claim.get("source_id", -1)))
            seen_types.add(str(claim.get("claim_type", "")))
    return selected


def _section_role(branch: ResearchBranch) -> str:
    text = f"{branch.title} {branch.objective}".lower()
    if re.search(r"\b(?:method|implementation|system|technical|architecture|workflow)\b", text):
        return "method_or_system_section"
    if re.search(r"\b(?:compare|comparison|trade[- ]?off|versus|alternative)\b", text):
        return "comparison_section"
    if re.search(r"\b(?:evidence|study|trial|review|literature|empirical)\b", text):
        return "evidence_review_section"
    if re.search(r"\b(?:limit|risk|gap|uncertain|challenge)\b", text):
        return "limitations_section"
    if re.search(r"\b(?:policy|practice|recommend|intervention|strategy|implication)\b", text):
        return "implications_section"
    return "analytical_section"


def _sentence_specs_for_section(branch: ResearchBranch, claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not claims:
        return []
    specs: list[dict[str, Any]] = []
    overview_claims = claims[: min(3, len(claims))]
    specs.append(
        _sentence_spec(
            purpose="section_thesis",
            claims=overview_claims,
            planned_point=(
                f"Open the {branch.title} section with the cautious pattern these claims support, "
                "not a broader claim than the listed evidence allows."
            ),
        )
    )

    claims_by_purpose: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in claims:
        claims_by_purpose[_sentence_purpose(str(claim.get("claim_type", "")))].append(claim)

    purpose_order = [
        "definition_or_scope",
        "mechanism",
        "empirical_result",
        "comparison_or_tension",
        "limitation_or_gap",
        "implication",
        "background_pattern",
    ]
    used_spec_keys = {_claim_group_key(overview_claims)}
    for purpose in purpose_order:
        purpose_claims = claims_by_purpose.get(purpose, [])
        for group in _chunk_claims_for_sentence_plan(purpose_claims):
            if len(specs) >= MAX_SENTENCE_PLAN_SPECS_PER_SECTION:
                return specs
            key = _claim_group_key(group)
            if not group or key in used_spec_keys:
                continue
            specs.append(
                _sentence_spec(
                    purpose=purpose,
                    claims=group,
                    planned_point=_planned_point_for_claim_group(purpose, group),
                )
            )
            used_spec_keys.add(key)
    return specs


def _sentence_specs_for_conclusion(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(claims) < 2:
        return []
    specs = [
        _sentence_spec(
            purpose="cross_source_synthesis",
            claims=claims[: min(4, len(claims))],
            planned_point=(
                "Synthesize the strongest supported answer across sections, citing each source next to the fact it contributes."
            ),
        )
    ]
    limitation_claims = [claim for claim in claims if str(claim.get("claim_type")) == "limitation"]
    implication_claims = [claim for claim in claims if str(claim.get("claim_type")) == "implication"]
    if limitation_claims:
        specs.append(
            _sentence_spec(
                purpose="final_limitations",
                claims=limitation_claims[:3],
                planned_point="Close with the main supported limitation or uncertainty without overstating it.",
            )
        )
    if implication_claims:
        specs.append(
            _sentence_spec(
                purpose="final_implication",
                claims=implication_claims[:3],
                planned_point="State the practical or research implication that is directly supported by these claims.",
            )
        )
    return specs


def _sentence_spec(*, purpose: str, claims: list[dict[str, Any]], planned_point: str) -> dict[str, Any]:
    claim_ids = [str(claim.get("claim_id")) for claim in claims if claim.get("claim_id")]
    source_ids = sorted({int(claim.get("source_id")) for claim in claims if claim.get("source_id") is not None})
    support_quotes = [
        {
            "claim_id": str(claim.get("claim_id")),
            "source_id": int(claim.get("source_id")),
            "quote": _compact_whitespace(str(claim.get("supporting_quote", "")))[:220],
        }
        for claim in claims
        if claim.get("claim_id") and claim.get("source_id") is not None
    ]
    return {
        "purpose": purpose,
        "claim_ids": claim_ids,
        "source_ids": source_ids,
        "planned_point": planned_point,
        "citation_requirements": (
            "Cite each source ID next to the exact fact drawn from that claim; do not stack loose citations at paragraph end."
        ),
        "support_quotes": support_quotes,
    }


def _sentence_purpose(claim_type: str) -> str:
    return {
        "definition_or_scope": "definition_or_scope",
        "mechanism": "mechanism",
        "quantitative_result": "empirical_result",
        "comparison": "comparison_or_tension",
        "limitation": "limitation_or_gap",
        "implication": "implication",
        "background_evidence": "background_pattern",
    }.get(claim_type, "background_pattern")


def _chunk_claims_for_sentence_plan(claims: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not claims:
        return []
    ordered = _source_and_role_diverse_claims(claims, limit=len(claims))
    groups: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(ordered):
        group = ordered[index : index + 2]
        if group:
            groups.append(group)
        index += 2
    return groups


def _claim_group_key(claims: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(str(claim.get("claim_id", "")) for claim in claims))


def _planned_point_for_claim_group(purpose: str, claims: list[dict[str, Any]]) -> str:
    claim_summaries = "; ".join(_compact_whitespace(str(claim.get("allowed_claim", "")))[:180] for claim in claims)
    instructions = {
        "definition_or_scope": "Define or scope the central construct using only these claims",
        "mechanism": "Explain the supported causal pathway or mechanism",
        "empirical_result": "Report the empirical result with exact source wording for numbers, dates, or named facts",
        "comparison_or_tension": "Compare the claims and make the tension explicit without resolving beyond the evidence",
        "limitation_or_gap": "State the supported limitation, uncertainty, or evidence gap",
        "implication": "Explain the supported implication for practice, policy, strategy, or future research",
        "background_pattern": "Add a background pattern that connects to the section objective",
    }
    prefix = instructions.get(purpose, "Use these claims")
    return f"{prefix}: {claim_summaries}"


def _format_sentence_plan_for_prompt(sentence_plan: dict[str, Any]) -> str:
    sections = sentence_plan.get("sections", [])
    if not isinstance(sections, list) or not sections:
        return "None"
    lines: list[str] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        heading = str(section.get("heading_basis") or "Section")
        role = str(section.get("section_role") or "analytical_section")
        branch_id = str(section.get("branch_id") or "")
        lines.append(f"- Section basis: {heading} ({role}; {branch_id})")
        specs = section.get("sentence_specs", [])
        if not isinstance(specs, list):
            continue
        for spec in specs[:MAX_SENTENCE_PLAN_SPECS_PER_SECTION]:
            if not isinstance(spec, dict):
                continue
            claim_ids = ", ".join(str(value) for value in spec.get("claim_ids", []))
            source_ids = ", ".join(str(value) for value in spec.get("source_ids", []))
            planned_point = _compact_whitespace(str(spec.get("planned_point", "")))[:360]
            lines.append(
                f"  - {spec.get('purpose', 'planned_point')}: {planned_point} "
                f"(claim_ids: {claim_ids or 'none'}; cite sources: {source_ids or 'none'})"
            )
    return "\n".join(lines) if lines else "None"



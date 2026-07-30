from __future__ import annotations

import re
from typing import Any

from deep_research.runtime.artifacts_v2 import ResearchArtifactsV2
from deep_research.core.schemas import CoverageMatrix, EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.evidence.source_validation import content_terms

MAX_PACKET_CARDS = 8
MAX_BRANCH_CARDS = 14
MAX_EXCERPT_CHARS = 360
MAX_MODEL_NOTES = 6
MAX_MODEL_NOTE_CHARS = 420


def build_knowledge_base(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    coverage: CoverageMatrix,
    section_plan: dict[str, Any],
) -> dict[str, Any]:
    source_by_id = {source.id: source for source in sources}
    cards_by_branch: dict[str, list[EvidenceCard]] = {}
    for card in evidence_cards:
        cards_by_branch.setdefault(card.branch_id, []).append(card)

    branches = []
    for branch in plan.branches:
        cards = _rank_cards(cards_by_branch.get(branch.id, []))[:MAX_BRANCH_CARDS]
        branches.append(
            {
                "branch_id": branch.id,
                "title": branch.title,
                "objective": branch.objective,
                "source_ids": _source_ids(cards),
                "evidence_card_ids": [card.id for card in cards],
                "coverage_terms": list(branch.required_terms),
                "notes": _branch_notes(cards, source_by_id),
                "limitations": _limitations(cards),
            }
        )

    section_packets = []
    for section in section_plan.get("sections", []) if isinstance(section_plan, dict) else []:
        if not isinstance(section, dict):
            continue
        packet_cards = _section_cards(section, evidence_cards)[:MAX_PACKET_CARDS]
        if not packet_cards:
            continue
        section_packets.append(
            {
                "section_id": str(section.get("id") or ""),
                "title_hint": str(section.get("title_hint") or ""),
                "role": str(section.get("role") or ""),
                "purpose": str(section.get("purpose") or ""),
                "source_ids": _source_ids(packet_cards),
                "evidence_card_ids": [card.id for card in packet_cards],
                "packet": _packet_notes(packet_cards, source_by_id),
                "exact_claims": _exact_claims(packet_cards),
                "limitations": _limitations(packet_cards),
            }
        )

    return {
        "schema_version": 1,
        "question": plan.question,
        "coverage": {
            "complete": coverage.complete,
            "coverage_score": coverage.coverage_score,
            "missing_branches": list(coverage.missing_branches),
        },
        "branches": branches,
        "section_packets": section_packets,
    }


def write_knowledge_base(
    *,
    artifacts: ResearchArtifactsV2,
    knowledge_base: dict[str, Any],
) -> None:
    artifacts.write_json("knowledge_base/manifest.json", knowledge_base)
    artifacts.write_text("knowledge_base/index.md", render_knowledge_index(knowledge_base))
    for branch in knowledge_base.get("branches", []):
        if not isinstance(branch, dict):
            continue
        branch_id = _safe_id(str(branch.get("branch_id") or "branch"))
        artifacts.write_text(f"knowledge_base/branches/{branch_id}.md", render_branch_note(branch))
    for packet in knowledge_base.get("section_packets", []):
        if not isinstance(packet, dict):
            continue
        section_id = _safe_id(str(packet.get("section_id") or "section"))
        artifacts.write_text(f"knowledge_base/section_packets/{section_id}.md", render_section_packet(packet))


def refine_knowledge_base_from_payload(
    base: dict[str, Any],
    payload: dict[str, Any],
    *,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
) -> dict[str, Any]:
    """Apply model-proposed focus notes without letting it invent evidence.

    The deterministic packet remains the factual floor. The model can add
    section focus, synthesis questions, and limitations only when the target IDs
    already exist and the note overlaps with assigned evidence text.
    """
    if not isinstance(payload, dict):
        return base
    refined = _copy_knowledge_base(base)
    source_ids = {source.id for source in sources}
    card_by_id = {card.id: card for card in evidence_cards}
    branch_rows = {
        str(row.get("branch_id") or ""): row
        for row in payload.get("branches", [])
        if isinstance(row, dict)
    }
    for branch in refined.get("branches", []):
        if not isinstance(branch, dict):
            continue
        proposed = branch_rows.get(str(branch.get("branch_id") or ""))
        if not proposed:
            continue
        allowed_cards = _allowed_cards(branch, card_by_id)
        allowed_source_ids = _allowed_source_ids(branch, source_ids, allowed_cards)
        _merge_model_fields(branch, proposed, allowed_cards=allowed_cards, allowed_source_ids=allowed_source_ids)

    packet_rows = {
        str(row.get("section_id") or ""): row
        for row in payload.get("section_packets", [])
        if isinstance(row, dict)
    }
    for packet in refined.get("section_packets", []):
        if not isinstance(packet, dict):
            continue
        proposed = packet_rows.get(str(packet.get("section_id") or ""))
        if not proposed:
            continue
        allowed_cards = _allowed_cards(packet, card_by_id)
        allowed_source_ids = _allowed_source_ids(packet, source_ids, allowed_cards)
        _merge_model_fields(packet, proposed, allowed_cards=allowed_cards, allowed_source_ids=allowed_source_ids)

    refinement = refined.setdefault("model_refinement", {})
    refinement["applied"] = True
    refinement["source"] = "llm_clamped"
    return refined


def format_knowledge_packets_for_prompt(knowledge_base: dict[str, Any], *, limit: int = 8) -> str:
    packets = [row for row in knowledge_base.get("section_packets", []) if isinstance(row, dict)]
    reasoning_brief = knowledge_base.get("reasoning_brief", {}) if isinstance(knowledge_base.get("reasoning_brief"), dict) else {}
    if not packets and not reasoning_brief:
        return "None"
    lines = []
    if reasoning_brief:
        lines.extend(
            [
                "- reasoning brief:",
                f"  readiness: {reasoning_brief.get('readiness_status')}",
                f"  decision: {reasoning_brief.get('decision', {}).get('action')} - {reasoning_brief.get('decision', {}).get('rationale')}",
                f"  weak claims to caveat or avoid overclaiming: {_brief_items(reasoning_brief.get('weak_claims', []))}",
                f"  unknowns to avoid overstating: {_brief_items(reasoning_brief.get('unknowns', []))}",
                f"  contradictions/tensions requiring caveats: {_brief_items(reasoning_brief.get('contradictions', []))}",
            ]
        )
    for packet in packets[:limit]:
        lines.append(
            "\n".join(
                [
                    f"- section {packet.get('section_id')}: {packet.get('title_hint')} ({packet.get('role')})",
                    f"  purpose: {packet.get('purpose')}",
                    f"  source_ids: {', '.join(str(item) for item in packet.get('source_ids', [])[:10]) or 'none'}",
                    f"  evidence_card_ids: {', '.join(str(item) for item in packet.get('evidence_card_ids', [])[:10]) or 'none'}",
                    f"  model_source_ids: {', '.join(str(item) for item in packet.get('model_source_ids', [])[:10]) or 'none'}",
                    f"  model_evidence_card_ids: {', '.join(str(item) for item in packet.get('model_evidence_card_ids', [])[:10]) or 'none'}",
                    f"  packet notes: {' '.join(str(item) for item in packet.get('packet', [])[:5])}",
                    f"  exact claims: {' '.join(str(item) for item in packet.get('exact_claims', [])[:4]) or 'none'}",
                    f"  limitations: {' '.join(str(item) for item in packet.get('limitations', [])[:4]) or 'none'}",
                    f"  model focus: {' '.join(str(item) for item in packet.get('model_focus', [])[:4]) or 'none'}",
                    f"  open questions: {' '.join(str(item) for item in packet.get('open_questions', [])[:4]) or 'none'}",
                ]
            )
        )
    return "\n".join(lines)


def _brief_items(rows: Any, *, limit: int = 5) -> str:
    if not isinstance(rows, list):
        return "none"
    values = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        text = row.get("claim") or row.get("description") or row.get("reason")
        source_ids = row.get("source_ids", [])
        suffix = f" sources={source_ids}" if source_ids else ""
        if text:
            values.append(f"{text}{suffix}")
    return " | ".join(values) if values else "none"


def render_knowledge_index(knowledge_base: dict[str, Any]) -> str:
    lines = [
        "# Knowledge Base",
        "",
        f"Question: {knowledge_base.get('question', '')}",
        "",
        "## Coverage",
        "",
        f"- complete: `{knowledge_base.get('coverage', {}).get('complete')}`",
        f"- score: `{knowledge_base.get('coverage', {}).get('coverage_score')}`",
        f"- missing: {', '.join(knowledge_base.get('coverage', {}).get('missing_branches', [])) or 'none'}",
        "",
        "## Branch Notes",
        "",
    ]
    for branch in knowledge_base.get("branches", []):
        if isinstance(branch, dict):
            lines.append(f"- [{branch.get('title') or branch.get('branch_id')}](branches/{_safe_id(str(branch.get('branch_id') or 'branch'))}.md)")
    lines.extend(["", "## Section Packets", ""])
    for packet in knowledge_base.get("section_packets", []):
        if isinstance(packet, dict):
            lines.append(
                f"- [{packet.get('title_hint') or packet.get('section_id')}](section_packets/{_safe_id(str(packet.get('section_id') or 'section'))}.md)"
            )
    return "\n".join(lines).rstrip() + "\n"


def render_branch_note(branch: dict[str, Any]) -> str:
    lines = [
        f"# {branch.get('title') or branch.get('branch_id')}",
        "",
        f"Objective: {branch.get('objective', '')}",
        "",
        f"Source IDs: {', '.join(str(item) for item in branch.get('source_ids', [])) or 'none'}",
        f"Evidence card IDs: {', '.join(str(item) for item in branch.get('evidence_card_ids', [])) or 'none'}",
        f"Model source IDs: {', '.join(str(item) for item in branch.get('model_source_ids', [])) or 'none'}",
        f"Model evidence card IDs: {', '.join(str(item) for item in branch.get('model_evidence_card_ids', [])) or 'none'}",
        "",
        "## Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in branch.get("notes", []))
    if branch.get("model_focus"):
        lines.extend(["", "## Model Focus", ""])
        lines.extend(f"- {item}" for item in branch.get("model_focus", []))
    if branch.get("open_questions"):
        lines.extend(["", "## Open Questions", ""])
        lines.extend(f"- {item}" for item in branch.get("open_questions", []))
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in branch.get("limitations", []) or ["No explicit limitations captured."])
    return "\n".join(lines).rstrip() + "\n"


def render_section_packet(packet: dict[str, Any]) -> str:
    lines = [
        f"# {packet.get('title_hint') or packet.get('section_id')}",
        "",
        f"Role: `{packet.get('role', '')}`",
        f"Purpose: {packet.get('purpose', '')}",
        f"Source IDs: {', '.join(str(item) for item in packet.get('source_ids', [])) or 'none'}",
        f"Evidence card IDs: {', '.join(str(item) for item in packet.get('evidence_card_ids', [])) or 'none'}",
        f"Model source IDs: {', '.join(str(item) for item in packet.get('model_source_ids', [])) or 'none'}",
        f"Model evidence card IDs: {', '.join(str(item) for item in packet.get('model_evidence_card_ids', [])) or 'none'}",
        "",
        "## Packet Notes",
        "",
    ]
    lines.extend(f"- {note}" for note in packet.get("packet", []))
    if packet.get("model_focus"):
        lines.extend(["", "## Model Focus", ""])
        lines.extend(f"- {item}" for item in packet.get("model_focus", []))
    if packet.get("open_questions"):
        lines.extend(["", "## Open Questions", ""])
        lines.extend(f"- {item}" for item in packet.get("open_questions", []))
    lines.extend(["", "## Exact Claims", ""])
    lines.extend(f"- {item}" for item in packet.get("exact_claims", []) or ["None captured."])
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in packet.get("limitations", []) or ["No explicit limitations captured."])
    return "\n".join(lines).rstrip() + "\n"


def _branch_notes(cards: list[EvidenceCard], source_by_id: dict[int, SourceRecordV2]) -> list[str]:
    notes = []
    for card in cards:
        source = source_by_id.get(card.source_id)
        title = source.title if source else card.source_title
        notes.append(f"[{card.source_id}] {title}: {_compact(card.claim)}")
    return notes


def _packet_notes(cards: list[EvidenceCard], source_by_id: dict[int, SourceRecordV2]) -> list[str]:
    notes = []
    for card in cards:
        source = source_by_id.get(card.source_id)
        title = source.title if source else card.source_title
        notes.append(
            f"[{card.source_id}] {title}: claim={_compact(card.claim)}; quote={_compact(card.supporting_excerpt)[:MAX_EXCERPT_CHARS]}"
        )
    return notes


def _exact_claims(cards: list[EvidenceCard]) -> list[str]:
    return [
        f"[{card.source_id}] {_compact(card.claim)}"
        for card in cards
        if re.search(r"\b\d{2,}(?:[.,]\d+)?\b|[%$]|\b(?:percent|million|billion|hours?|years?|kw|hp|rpm|nm|mm)\b", card.claim, flags=re.I)
    ][:8]


def _limitations(cards: list[EvidenceCard]) -> list[str]:
    result = []
    for card in cards:
        result.extend(_compact(item) for item in card.limitations if _compact(item))
    return _dedupe(result)[:8]


def _section_cards(section: dict[str, Any], evidence_cards: list[EvidenceCard]) -> list[EvidenceCard]:
    card_ids = {int(item) for item in section.get("evidence_card_ids", []) if isinstance(item, int)}
    source_ids = {int(item) for item in section.get("source_ids", []) if isinstance(item, int)}
    branch_ids = {str(item) for item in section.get("branch_ids", []) if str(item).strip()}
    selected = [
        card
        for card in evidence_cards
        if card.id in card_ids or card.source_id in source_ids or card.branch_id in branch_ids
    ]
    return _rank_cards(selected)


def _rank_cards(cards: list[EvidenceCard]) -> list[EvidenceCard]:
    return sorted(
        cards,
        key=lambda card: (
            -(card.semantic_score if card.semantic_score is not None else card.confidence),
            -card.confidence,
            -card.quality_score,
            -card.relevance_score,
            card.id,
        ),
    )


def _source_ids(cards: list[EvidenceCard]) -> list[int]:
    return sorted({card.source_id for card in cards})


def _copy_knowledge_base(base: dict[str, Any]) -> dict[str, Any]:
    return {
        **base,
        "coverage": dict(base.get("coverage", {}) or {}),
        "branches": [dict(row) for row in base.get("branches", []) if isinstance(row, dict)],
        "section_packets": [dict(row) for row in base.get("section_packets", []) if isinstance(row, dict)],
        "model_refinement": dict(base.get("model_refinement", {}) or {}),
    }


def _allowed_cards(row: dict[str, Any], card_by_id: dict[int, EvidenceCard]) -> list[EvidenceCard]:
    card_ids = _int_set(row.get("evidence_card_ids", []))
    return [card_by_id[card_id] for card_id in sorted(card_ids) if card_id in card_by_id]


def _allowed_source_ids(row: dict[str, Any], source_ids: set[int], cards: list[EvidenceCard]) -> set[int]:
    row_source_ids = _int_set(row.get("source_ids", [])) & source_ids
    card_source_ids = {card.source_id for card in cards}
    return row_source_ids | card_source_ids


def _merge_model_fields(
    target: dict[str, Any],
    proposed: dict[str, Any],
    *,
    allowed_cards: list[EvidenceCard],
    allowed_source_ids: set[int],
) -> None:
    proposed_card_ids = _int_set(proposed.get("evidence_card_ids", []))
    proposed_source_ids = _int_set(proposed.get("source_ids", []))
    scoped_cards = [card for card in allowed_cards if not proposed_card_ids or card.id in proposed_card_ids]
    scoped_source_ids = ({card.source_id for card in scoped_cards} | proposed_source_ids) & allowed_source_ids
    if scoped_cards:
        target["model_evidence_card_ids"] = [card.id for card in scoped_cards]
    if scoped_source_ids:
        target["model_source_ids"] = sorted(scoped_source_ids)
    evidence_text = _evidence_text(scoped_cards or allowed_cards)
    for field in ("model_focus", "open_questions", "limitations"):
        notes = _validated_notes(
            proposed.get(field, []),
            evidence_text=evidence_text,
            allowed_source_ids=allowed_source_ids,
            require_overlap=field != "open_questions",
        )
        if not notes:
            continue
        if field == "limitations":
            target[field] = _dedupe([str(item) for item in target.get(field, [])] + notes)[:8]
        else:
            target[field] = notes


def _validated_notes(
    values: Any,
    *,
    evidence_text: str,
    allowed_source_ids: set[int],
    require_overlap: bool,
) -> list[str]:
    result = []
    evidence_terms = content_terms(evidence_text)
    for value in values if isinstance(values, list) else []:
        note = _compact(str(value))[:MAX_MODEL_NOTE_CHARS]
        if not note:
            continue
        cited_ids = set(int(match) for match in re.findall(r"\[(\d+)]", note))
        if cited_ids and not cited_ids <= allowed_source_ids:
            continue
        if require_overlap:
            note_terms = content_terms(note)
            if len(note_terms & evidence_terms) < min(3, max(1, len(note_terms))):
                continue
        result.append(note)
        if len(result) >= MAX_MODEL_NOTES:
            break
    return _dedupe(result)


def _evidence_text(cards: list[EvidenceCard]) -> str:
    return " ".join(f"{card.claim} {card.supporting_excerpt}" for card in cards)


def _int_set(values: Any) -> set[int]:
    result: set[int] = set()
    for value in values if isinstance(values, list) else []:
        if isinstance(value, int):
            result.add(value)
    return result


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _safe_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return slug or "item"

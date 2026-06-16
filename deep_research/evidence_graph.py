from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from deep_research.lexical_expansion import term_matches
from deep_research.schemas import EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.source_validation import content_terms


HIGH_IMPACT_SEEDS = (
    "cause",
    "effect",
    "risk",
    "effective",
    "prevent",
    "require",
    "recommend",
    "superior",
    "inferior",
    "change",
    "increase",
    "decrease",
)
CONTRADICTION_SEEDS = (
    "contradict",
    "conflict",
    "mixed",
    "inconsistent",
    "uncertain",
    "limited",
    "caution",
    "dispute",
    "challenge",
)


@dataclass(frozen=True)
class SourceNode:
    id: int
    branch_id: str
    title: str
    url: str
    quality_score: float
    quality_type: str
    relevance_score: float
    word_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SupportEdge:
    claim_id: str
    source_id: int
    evidence_card_id: int
    confidence: float
    passage: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClaimNode:
    id: str
    branch_id: str
    claim: str
    evidence_card_ids: list[int]
    source_ids: list[int]
    support_count: int
    average_confidence: float
    average_source_quality: float
    high_impact: bool
    weak: bool
    weakness_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContradictionEdge:
    id: str
    claim_id: str
    branch_id: str
    description: str
    source_ids: list[int]
    confidence: float
    needs_caveat: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceGraph:
    schema_version: int
    question: str
    sources: list[SourceNode]
    claims: list[ClaimNode]
    support_edges: list[SupportEdge]
    contradiction_edges: list[ContradictionEdge]
    weak_claim_ids: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "question": self.question,
            "sources": [source.to_dict() for source in self.sources],
            "claims": [claim.to_dict() for claim in self.claims],
            "support_edges": [edge.to_dict() for edge in self.support_edges],
            "contradiction_edges": [edge.to_dict() for edge in self.contradiction_edges],
            "weak_claim_ids": self.weak_claim_ids,
            "metrics": self.metrics,
        }


def build_evidence_graph(
    *,
    plan: ResearchPlan,
    sources: list[SourceRecordV2],
    evidence_cards: list[EvidenceCard],
    source_texts: dict[int, str] | None = None,
    semantic_judgments: dict[str, Any] | None = None,
) -> EvidenceGraph:
    source_by_id = {source.id: source for source in sources}
    source_nodes = [_source_node(source) for source in sources]
    groups = _group_cards_by_claim(evidence_cards)
    claims: list[ClaimNode] = []
    support_edges: list[SupportEdge] = []
    for index, cards in enumerate(groups, start=1):
        claim = _representative_claim(cards)
        source_ids = sorted({card.source_id for card in cards})
        card_ids = sorted(card.id for card in cards)
        confidences = [_card_confidence(card) for card in cards]
        qualities = [source_by_id[source_id].quality_score for source_id in source_ids if source_id in source_by_id]
        high_impact = _high_impact_claim(claim)
        weakness = _weakness_reasons(
            cards=cards,
            source_ids=source_ids,
            source_by_id=source_by_id,
            high_impact=high_impact,
        )
        claim_id = f"claim_{index}"
        claims.append(
            ClaimNode(
                id=claim_id,
                branch_id=cards[0].branch_id,
                claim=claim,
                evidence_card_ids=card_ids,
                source_ids=source_ids,
                support_count=len(source_ids),
                average_confidence=round(sum(confidences) / max(len(confidences), 1), 4),
                average_source_quality=round(sum(qualities) / max(len(qualities), 1), 4) if qualities else 0.0,
                high_impact=high_impact,
                weak=bool(weakness),
                weakness_reasons=weakness,
            )
        )
        for card in cards:
            support_edges.append(
                SupportEdge(
                    claim_id=claim_id,
                    source_id=card.source_id,
                    evidence_card_id=card.id,
                    confidence=_card_confidence(card),
                    passage=card.supporting_excerpt[:500],
                )
            )
    contradictions = _contradictions_from_cards(claims, evidence_cards) + _contradictions_from_semantics(
        claims, semantic_judgments or {}
    )
    weak_claim_ids = [claim.id for claim in claims if claim.weak]
    metrics = {
        "source_count": len(source_nodes),
        "claim_count": len(claims),
        "support_edge_count": len(support_edges),
        "contradiction_count": len(contradictions),
        "weak_claim_count": len(weak_claim_ids),
        "source_text_count": len(source_texts or {}),
    }
    return EvidenceGraph(
        schema_version=1,
        question=plan.question,
        sources=source_nodes,
        claims=claims,
        support_edges=support_edges,
        contradiction_edges=contradictions,
        weak_claim_ids=weak_claim_ids,
        metrics=metrics,
    )


def _source_node(source: SourceRecordV2) -> SourceNode:
    return SourceNode(
        id=source.id,
        branch_id=source.branch_id,
        title=source.title,
        url=source.url,
        quality_score=round(source.quality_score, 4),
        quality_type=source.quality_type,
        relevance_score=round(source.relevance_score, 4),
        word_count=source.word_count,
    )


def _group_cards_by_claim(cards: list[EvidenceCard]) -> list[list[EvidenceCard]]:
    groups: list[list[EvidenceCard]] = []
    for card in sorted(cards, key=lambda row: row.id):
        card_terms = content_terms(card.claim)
        placed = False
        for group in groups:
            group_terms = content_terms(_representative_claim(group))
            if _term_overlap(card_terms, group_terms) >= 0.72:
                group.append(card)
                placed = True
                break
        if not placed:
            groups.append([card])
    return groups


def _representative_claim(cards: list[EvidenceCard]) -> str:
    return max((card.claim.strip() for card in cards), key=len, default="")


def _term_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left | right), 1)


def _card_confidence(card: EvidenceCard) -> float:
    if card.semantic_score is not None:
        return round(max(0.0, min(float(card.semantic_score), 1.0)), 4)
    return round(max(0.0, min(float(card.confidence), 1.0)), 4)


def _high_impact_claim(claim: str) -> bool:
    return bool(
        re.search(
            r"\b\d{4}\b|\b\d+(?:[.,]\d+)?\s*(?:%|percent|usd|eur|gbp|\$|kw|hp|rpm|mm|kg|years?)\b",
            claim,
            flags=re.I,
        )
        or term_matches(claim, HIGH_IMPACT_SEEDS)
    )


def _weakness_reasons(
    *,
    cards: list[EvidenceCard],
    source_ids: list[int],
    source_by_id: dict[int, SourceRecordV2],
    high_impact: bool,
) -> list[str]:
    reasons: list[str] = []
    if not source_ids:
        reasons.append("no supporting source")
    if high_impact and len(source_ids) < 2:
        reasons.append("high-impact claim has fewer than two independent sources")
    avg_conf = sum(_card_confidence(card) for card in cards) / max(len(cards), 1)
    if avg_conf < 0.45:
        reasons.append("low evidence confidence")
    avg_quality = sum(source_by_id[source_id].quality_score for source_id in source_ids if source_id in source_by_id) / max(
        len([source_id for source_id in source_ids if source_id in source_by_id]), 1
    )
    if avg_quality < 0.50:
        reasons.append("low source quality")
    if any(card.limitations for card in cards):
        reasons.append("evidence card includes limitations")
    return reasons


def _contradictions_from_cards(claims: list[ClaimNode], cards: list[EvidenceCard]) -> list[ContradictionEdge]:
    card_by_id = {card.id: card for card in cards}
    edges: list[ContradictionEdge] = []
    for claim in claims:
        notes: list[str] = []
        for card_id in claim.evidence_card_ids:
            card = card_by_id.get(card_id)
            if card is None:
                continue
            notes.extend(card.limitations)
            notes.extend(card.semantic_notes)
        text = " ".join(notes)
        if term_matches(text, CONTRADICTION_SEEDS):
            edges.append(
                ContradictionEdge(
                    id=f"contradiction_{len(edges) + 1}",
                    claim_id=claim.id,
                    branch_id=claim.branch_id,
                    description=text[:500],
                    source_ids=claim.source_ids,
                    confidence=0.55,
                    needs_caveat=True,
                )
            )
    return edges


def _contradictions_from_semantics(claims: list[ClaimNode], semantic_judgments: dict[str, Any]) -> list[ContradictionEdge]:
    rows = semantic_judgments.get("judgments", []) if isinstance(semantic_judgments, dict) else []
    descriptions: list[str] = []
    for row in rows:
        if isinstance(row, dict):
            descriptions.extend(str(value) for value in row.get("contradictions", []) if str(value).strip())
    edges: list[ContradictionEdge] = []
    for description in descriptions[:10]:
        claim = _best_claim_for_text(claims, description)
        if claim is None:
            continue
        edges.append(
            ContradictionEdge(
                id=f"semantic_contradiction_{len(edges) + 1}",
                claim_id=claim.id,
                branch_id=claim.branch_id,
                description=description[:500],
                source_ids=claim.source_ids,
                confidence=0.6,
                needs_caveat=True,
            )
        )
    return edges


def _best_claim_for_text(claims: list[ClaimNode], text: str) -> ClaimNode | None:
    terms = content_terms(text)
    if not terms or not claims:
        return claims[0] if claims else None
    return max(claims, key=lambda claim: _term_overlap(terms, content_terms(claim.claim)))

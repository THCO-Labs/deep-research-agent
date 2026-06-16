from deep_research.evidence_graph import build_evidence_graph
from deep_research.schemas import EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2


def test_evidence_graph_groups_claims_and_flags_single_source_high_impact_claim() -> None:
    plan = _plan()
    source = _source(1, 0.45)
    cards = [
        _card(1, 1, "Spring Boot reduces configuration time by 40 percent for teams."),
        _card(2, 1, "Spring Boot reduces configuration time by 40 percent for teams."),
    ]

    graph = build_evidence_graph(plan=plan, sources=[source], evidence_cards=cards, source_texts={1: "text"})
    payload = graph.to_dict()

    assert payload["metrics"]["claim_count"] == 1
    assert payload["claims"][0]["support_count"] == 1
    assert payload["claims"][0]["weak"] is True
    assert "high-impact claim" in " ".join(payload["claims"][0]["weakness_reasons"])


def test_evidence_graph_records_contradiction_from_limitations() -> None:
    plan = _plan()
    source = _source(1, 0.8)
    card = _card(1, 1, "Servlet containers improved request handling.")
    card = EvidenceCard(**{**card.to_dict(), "limitations": ["Evidence is mixed and may conflict across sources."]})

    graph = build_evidence_graph(plan=plan, sources=[source], evidence_cards=[card])

    assert graph.to_dict()["metrics"]["contradiction_count"] == 1
    assert graph.to_dict()["contradiction_edges"][0]["needs_caveat"] is True


def _plan() -> ResearchPlan:
    return ResearchPlan(
        question="Trace Java enterprise architecture evolution.",
        intent="general",
        audience="technical generalist",
        report_outline=[],
        branches=[ResearchBranch(id="branch_1", title="Boot", objective="Explain Boot.", queries=["boot"])],
    )


def _source(source_id: int, quality: float) -> SourceRecordV2:
    return SourceRecordV2(
        id=source_id,
        branch_id="branch_1",
        title="Source",
        url="https://example.com/source",
        canonical_url="https://example.com/source",
        provenance="web",
        content_path="source_docs/source.md",
        content_hash="hash",
        extraction_method="httpx",
        word_count=500,
        quality_score=quality,
        quality_label="medium",
        quality_type="official_docs",
        relevance_score=0.9,
    )


def _card(card_id: int, source_id: int, claim: str) -> EvidenceCard:
    return EvidenceCard(
        id=card_id,
        source_id=source_id,
        branch_id="branch_1",
        claim=claim,
        supporting_excerpt=claim,
        source_url="https://example.com/source",
        source_title="Source",
        quality_score=0.8,
        relevance_score=0.9,
        confidence=0.8,
    )

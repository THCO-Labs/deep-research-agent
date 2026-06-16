from pathlib import Path
from types import SimpleNamespace

from deep_research.acquisition import acquire_sources
from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.coverage import build_coverage_matrix
from deep_research.search_intents import (
    SearchIntent,
    _apply_plan_revisions,
    _validate_and_dedupe_intents,
    _validate_intent_results,
    fallback_search_intent_results,
    fallback_search_intents,
)
from deep_research.schemas import EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2


def test_search_intent_guardrails_drop_invalid_ids_generic_queries_and_duplicates() -> None:
    plan = _plan()
    graph = {
        "claims": [{"id": "claim_1", "branch_id": "branch_1"}],
        "sources": [{"id": 1}],
    }
    rows = [
        {
            "branch_id": "branch_1",
            "gap": "Need official torque evidence.",
            "query": "NLX 2500SY torque curve official brochure",
            "expected_evidence": "Official torque curve or spindle specification.",
            "success_criteria": "Accepted source has torque numbers.",
            "source_preference": "official specs",
            "priority": "high",
            "claim_ids": ["claim_1", "claim_fake"],
            "source_ids": [1, 99],
        },
        {
            "branch_id": "branch_fake",
            "query": "Mazak official specification",
            "expected_evidence": "Official machine specification.",
        },
        {
            "branch_id": "branch_1",
            "query": "research evidence overview",
            "expected_evidence": "Something useful.",
        },
        {
            "branch_id": "branch_1",
            "query": "NLX 2500SY torque curve official brochure",
            "expected_evidence": "Duplicate.",
        },
    ]

    intents = _validate_and_dedupe_intents(rows, plan=plan, evidence_graph=graph, origin="llm")

    assert len(intents) == 1
    assert intents[0].claim_ids == ["claim_1"]
    assert intents[0].source_ids == [1]


def test_fallback_search_intents_are_gap_specific() -> None:
    plan = _plan()
    coverage = build_coverage_matrix(branches=plan.branches, evidence_cards=[], sources=[])
    reasoning = {
        "unknowns": [
            {
                "branch_id": "branch_1",
                "description": "Missing thermal compensation evidence.",
                "reason": "No accepted source covers thermal drift.",
                "focus_terms": ["thermal compensation", "thermal drift"],
            }
        ]
    }

    intents = fallback_search_intents(
        plan=plan,
        coverage=coverage,
        reasoning_state=reasoning,
        source_policy={"policy": {"preferred_source_types": ["official_docs", "datasheet"]}},
    )

    assert intents
    assert intents[0].branch_id == "branch_1"
    assert "thermal compensation" in intents[0].query.lower()
    assert intents[0].origin == "deterministic_fallback"


def test_intent_result_guardrails_require_real_evidence_ids() -> None:
    intent = SearchIntent(
        id="intent_1",
        branch_id="branch_1",
        gap="Need torque evidence.",
        query="NLX torque official",
        expected_evidence="Torque spec",
        success_criteria="Evidence card cites torque spec.",
        source_preference="official",
        priority="high",
        origin="llm",
        rationale="gap",
    )
    source = _source(1, intent_id="intent_1")
    card = _card(1, source_id=1)
    fallback = fallback_search_intent_results(intents=[intent], sources=[source], evidence_cards=[card])
    rows = [
        {
            "intent_id": "intent_1",
            "status": "satisfied",
            "accepted_source_ids": [1, 999],
            "evidence_card_ids": [999],
            "rationale": "invented card",
        }
    ]

    results = _validate_intent_results(rows, intents=[intent], sources=[source], evidence_cards=[card], fallback=fallback)

    assert results[0].status == "partially_satisfied"
    assert results[0].accepted_source_ids == [1]
    assert results[0].evidence_card_ids == []


def test_plan_revisions_are_additive_and_clamped() -> None:
    plan = _plan()

    revised, applied = _apply_plan_revisions(
        plan,
        {
            "add_queries": [
                {"branch_id": "branch_1", "queries": ["official thermal compensation manual"]},
                {"branch_id": "fake", "queries": ["bad query"]},
            ],
            "add_required_terms": [{"branch_id": "branch_1", "terms": ["thermal compensation"]}],
            "raise_min_sources": [{"branch_id": "branch_1", "min_sources": 5}],
            "add_branch": {
                "title": "Service Infrastructure",
                "objective": "Find local support evidence.",
                "queries": ["machine tool service Mexico official"],
                "required_terms": ["service infrastructure"],
                "completion_criteria": ["local support evidence"],
            },
        },
    )

    branch = revised.branches[0]
    assert "official thermal compensation manual" in branch.queries
    assert "thermal compensation" in branch.required_terms
    assert branch.min_sources == 5
    assert len(revised.branches) == len(plan.branches) + 1
    assert {row["type"] for row in applied} >= {"add_queries", "add_required_terms", "raise_min_sources", "add_branch"}


def test_acquisition_executes_intents_and_preserves_trace_metadata(tmp_path: Path) -> None:
    class FakeSearchClient:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query: str, **kwargs):
            self.queries.append(query)
            content = (
                "The NLX 2500SY official specification includes torque, spindle power, rigidity, "
                "thermal compensation, CAM integration, aerospace titanium machining constraints, "
                "traceability support, and service infrastructure details. "
            ) * 10
            return {
                "results": [
                    {
                        "url": "https://example.com/nlx-spec",
                        "title": "NLX 2500SY official specification",
                        "content": content,
                        "raw_content": content,
                        "score": 0.95,
                    }
                ]
            }

    branch = _plan().branches[0]
    intent = SearchIntent(
        id="intent_1",
        branch_id="branch_1",
        gap="Need official torque and thermal compensation specs.",
        query="NLX 2500SY official torque thermal compensation specification",
        expected_evidence="Official torque and thermal compensation specification.",
        success_criteria="Accepted source contains torque and thermal compensation.",
        source_preference="official_docs",
        priority="high",
        origin="llm",
        rationale="gap",
    )
    client = FakeSearchClient()
    settings = SimpleNamespace(
        scrape_timeout_ms=1_000,
        scrape_retries=0,
        min_source_words=40,
        min_relevant_chunks=1,
        max_candidates=20,
        max_sources=17,
        min_usable_sources=17,
        search_depth="advanced",
        allow_raw_content=True,
        max_followup_queries_per_branch=12,
        max_browser_scrapes_per_query=0,
    )

    result = acquire_sources(
        question=_plan().question,
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "intent acquisition"),
        settings=settings,
        search_client=client,
        search_intents=[intent],
        active_branch_ids={"branch_1"},
    )

    assert client.queries == [intent.query]
    assert result.candidates[0].search_intent_id == "intent_1"
    assert result.sources[0].metadata["search_intent_id"] == "intent_1"
    assert result.sources[0].metadata["search_intent_expected_evidence"] == intent.expected_evidence


def _plan() -> ResearchPlan:
    return ResearchPlan(
        question="Compare machine tools for aerospace titanium machining.",
        intent="general",
        audience="technical buyer",
        report_outline=["Comparison"],
        branches=[
            ResearchBranch(
                id="branch_1",
                title="Machine specifications",
                objective="Compare torque, rigidity, thermal compensation, and integration.",
                queries=["machine official specification"],
                min_sources=1,
                required_terms=["torque", "rigidity", "thermal compensation", "CAM integration"],
                completion_criteria=["official specification evidence"],
            )
        ],
        acceptance_criteria=["Use official specifications."],
    )


def _source(source_id: int, *, intent_id: str = "") -> SourceRecordV2:
    return SourceRecordV2(
        id=source_id,
        branch_id="branch_1",
        title="Official specification",
        url="https://example.com/spec",
        canonical_url="https://example.com/spec",
        provenance="web",
        content_path="sources/source_1.md",
        content_hash="hash",
        extraction_method="raw_content",
        word_count=120,
        quality_score=0.8,
        quality_label="high",
        quality_type="official_docs",
        relevance_score=0.8,
        metadata={"search_intent_id": intent_id} if intent_id else {},
    )


def _card(card_id: int, *, source_id: int) -> EvidenceCard:
    return EvidenceCard(
        id=card_id,
        source_id=source_id,
        branch_id="branch_1",
        claim="The machine has documented torque specifications.",
        supporting_excerpt="Official torque specifications are documented.",
        source_url="https://example.com/spec",
        source_title="Official specification",
        quality_score=0.8,
        relevance_score=0.8,
        confidence=0.8,
    )

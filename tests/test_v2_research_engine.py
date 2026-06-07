import json
from pathlib import Path
from types import SimpleNamespace

from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.evidence import build_evidence_cards
from deep_research.evidence_hygiene import apply_evidence_hygiene, report_quality_issues
from deep_research.acquisition import TavilySearchClientPool, acquire_sources, _branch_queries, _trim_search_query
from deep_research.ingestion import ingest_local_paths, ingest_mcp_manifest
from deep_research.managed import run_gemini_managed_research
from deep_research.planning import build_research_plan
from deep_research.coverage import build_coverage_matrix
from deep_research.guidance import format_criteria_guidance_block
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2, VerificationResultV2
from deep_research.semantic import (
    _load_json_object,
    apply_semantic_report_result,
    enrich_evidence_cards_with_semantics,
    verify_report_with_semantics,
)
from deep_research.semantic_planning import build_or_enrich_research_plan
from deep_research.semantic_planning import _loads_json_object
from deep_research.settings import Settings
from deep_research.source_validation import validate_source_content
from deep_research.synthesis import (
    _append_evidence_coverage_if_needed,
    _cards_for_synthesis,
    _evidence_backed_sources,
    _normalize_report_markdown,
    _repair_weak_citation_support,
    _synthesis_model_spec,
    _synthesis_prompt,
    _synthesis_request_kwargs,
    _target_report_profile,
    build_report_blueprint,
    synthesize_report,
    synthesize_report_with_model,
)
from deep_research.verifier_v2 import _report_depth_score, _report_level_criteria, verify_report_v2
from deep_research.research_graph import _acquire_route, _coverage_route, _focus_terms_from_state, _verification_route


def test_generic_plan_handles_paragraph_length_medical_question() -> None:
    question = (
        "I am trying to understand whether Mediterranean-style diets help adults with hypertension. "
        "Please cover what the diet means, how it may affect blood pressure, evidence from studies, "
        "who it may not be enough for, and practical decision guidance."
    )
    plan = build_research_plan(question)

    branch_ids = [branch.id for branch in plan.branches]
    all_queries = " ".join(query for branch in plan.branches for query in branch.queries)
    branch_text = " ".join(branch.title + " " + branch.objective + " " + " ".join(branch.required_terms) for branch in plan.branches)
    assert len(branch_ids) >= 4
    assert all(branch_id.startswith("branch_") for branch_id in branch_ids)
    assert "definition" not in branch_ids
    assert sum(branch.min_sources for branch in plan.branches) >= 17
    assert "mediterranean" in all_queries.lower()
    assert "hypertension" in all_queries.lower()
    assert "blood pressure" in branch_text.lower() or "pressure" in branch_text.lower()
    assert "urban heat" not in branch_text.lower()


def test_generic_plan_uses_cjk_chunks_without_character_ngram_requirements() -> None:
    question = (
        "收集整理目前国际综合实力前十的保险公司的相关资料，横向比较各公司的融资情况、信誉度、"
        "过往五年的增长幅度、实际分红、未来在中国发展潜力等维度，并为我评估出最有可能"
        "在未来资产排名靠前的2-3家公司"
    )

    plan = build_research_plan(question)

    assert len(plan.branches) >= 5
    assert sum(branch.min_sources for branch in plan.branches) >= 17
    all_required = [term for branch in plan.branches for term in branch.required_terms]
    assert "保险公司" in all_required
    assert "融资" in all_required
    assert "实际分红" in all_required
    assert "际综合实" not in all_required
    assert "向比较各" not in all_required
    assert all(question not in query for branch in plan.branches for query in branch.queries)


def test_semantic_planning_json_loader_repairs_near_json() -> None:
    payload = _loads_json_object(
        """
        {
          audience: 'analyst',
          branches: [
            {title: 'Market context', objective: 'Find market context', queries: ['market context']}
          ],
        }
        """
    )

    assert payload["audience"] == "analyst"
    assert payload["branches"][0]["title"] == "Market context"


def test_acquisition_trims_overlong_search_queries() -> None:
    long_query = " ".join(["criterion coverage phrase"] * 40)

    trimmed = _trim_search_query(long_query)

    assert len(trimmed) <= 380
    assert trimmed.split()[-1] in {"criterion", "coverage", "phrase"}


def test_acquisition_followup_queries_stay_evidence_neutral() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Theoretical mechanism",
        objective="Explain a theoretical mechanism.",
        queries=["theoretical mechanism evidence"],
        required_terms=["mediating factor"],
    )

    queries = _branch_queries(
        branch,
        forced_terms=["source credibility"],
        question="What is the relationship between two constructs?",
    )
    query_text = " ".join(queries).lower()

    assert "source credibility" in query_text
    assert "implementation" not in query_text
    assert "best practices" not in query_text
    assert len(queries) > 4
    assert any(query.endswith("source credibility") for query in queries)


def test_tavily_key_pool_rotates_when_key_hits_usage_limit(monkeypatch) -> None:
    calls: list[str] = []

    class FakeTavilyClient:
        def __init__(self, *, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str, **kwargs):
            calls.append(self.api_key)
            if self.api_key == "tavily-one":
                raise RuntimeError("This request exceeds your plan's set usage limit.")
            return {"results": [{"url": "https://example.com", "title": query, "content": "ok"}]}

    monkeypatch.setattr("deep_research.acquisition.TavilyClient", FakeTavilyClient)
    settings = SimpleNamespace(tavily_key_pool=("tavily-one", "tavily-two"), tavily_api_key="tavily-one")

    client = TavilySearchClientPool(settings)
    response = client.search("test query", max_results=1)

    assert response["results"][0]["url"] == "https://example.com"
    assert calls == ["tavily-one", "tavily-two"]


def test_acquisition_uses_configured_scrape_timeout_and_emits_progress(tmp_path: Path, monkeypatch) -> None:
    captured_scraper_kwargs: dict[str, int] = {}

    class CapturingScraper:
        def __init__(self, *, timeout_ms: int, retries: int) -> None:
            captured_scraper_kwargs["timeout_ms"] = timeout_ms
            captured_scraper_kwargs["retries"] = retries

        def fetch(self, url: str):  # pragma: no cover - raw Tavily content should bypass fetch.
            raise AssertionError(f"unexpected scrape for {url}")

    class FakeSearchClient:
        def search(self, query: str, **kwargs):
            raw_content = (
                "Need for closure is an epistemic motivation linked to fast belief formation, "
                "misinformation acceptance, ambiguity reduction, and reliance on early cues. "
                "Empirical research discusses how closure motivation shapes information processing, "
                "confidence, evidence scrutiny, source evaluation, and false belief acceptance. "
            ) * 8
            return {
                "results": [
                    {
                        "url": "https://example.com/need-for-closure-misinformation",
                        "title": "Need for closure and misinformation",
                        "content": raw_content,
                        "raw_content": raw_content,
                        "score": 0.9,
                    }
                ]
            }

    monkeypatch.setattr("deep_research.acquisition.PlaywrightScraper", CapturingScraper)
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure shapes misinformation acceptance.",
        queries=["need for closure misinformation acceptance evidence"],
        source_types=["academic"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    settings = SimpleNamespace(
        scrape_timeout_ms=7_000,
        scrape_retries=2,
        min_source_words=40,
        min_relevant_chunks=1,
        max_candidates=20,
        max_sources=17,
        min_usable_sources=17,
        search_depth="advanced",
        allow_raw_content=True,
    )
    progress_events: list[tuple[str, dict]] = []

    result = acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "acquisition timeout"),
        settings=settings,
        search_client=FakeSearchClient(),
        progress_callback=lambda message, data: progress_events.append((message, data)),
    )

    assert captured_scraper_kwargs == {"timeout_ms": 7_000, "retries": 2}
    assert len(result.sources) == 1
    assert any(message == "searching source candidates" for message, _ in progress_events)
    assert any(message.startswith("accepted source") for message, _ in progress_events)


def test_acquisition_skips_configured_blocked_source_patterns(tmp_path: Path) -> None:
    class FakeSearchClient:
        def search(self, query: str, **kwargs):
            blocked_content = "benchmark prompt dataset reference row " * 50
            accepted_content = (
                "Need for closure is an epistemic motivation linked to fast belief formation, "
                "misinformation acceptance, ambiguity reduction, evidence scrutiny, source evaluation, "
                "confidence, and false belief acceptance. "
            ) * 10
            return {
                "results": [
                    {
                        "url": "https://huggingface.co/datasets/example/deep_research_bench_eval",
                        "title": "Deep Research Bench eval dataset",
                        "content": blocked_content,
                        "raw_content": blocked_content,
                        "score": 0.99,
                    },
                    {
                        "url": "https://example.com/need-for-closure-study",
                        "title": "Need for closure and misinformation acceptance",
                        "content": accepted_content,
                        "raw_content": accepted_content,
                        "score": 0.9,
                    },
                ]
            }

    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure shapes misinformation acceptance.",
        queries=["need for closure misinformation acceptance evidence"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    settings = SimpleNamespace(
        min_source_words=40,
        min_relevant_chunks=1,
        max_candidates=20,
        max_sources=17,
        min_usable_sources=17,
        search_depth="advanced",
        allow_raw_content=True,
        blocked_source_patterns=(r"deep[_-]?research[_-]?bench",),
    )
    progress_events: list[tuple[str, dict]] = []

    result = acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "blocked source"),
        settings=settings,
        search_client=FakeSearchClient(),
        progress_callback=lambda message, data: progress_events.append((message, data)),
    )

    assert [source.url for source in result.sources] == ["https://example.com/need-for-closure-study"]
    assert len(result.candidates) == 1
    assert any(message == "blocked source candidate" for message, _ in progress_events)


def test_acquisition_respects_browser_fallback_budget(tmp_path: Path) -> None:
    class FakeSearchClient:
        def search(self, query: str, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.com/no-raw",
                        "title": "Need for closure short snippet",
                        "content": "short snippet without enough raw article content",
                        "score": 0.8,
                    }
                ]
            }

    class UnexpectedScraper:
        def fetch(self, url: str):  # pragma: no cover - budget should skip browser fallback.
            raise AssertionError(f"unexpected browser scrape for {url}")

    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure shapes misinformation acceptance.",
        queries=["need for closure misinformation acceptance evidence"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    settings = SimpleNamespace(
        min_source_words=40,
        min_relevant_chunks=1,
        max_candidates=20,
        max_sources=17,
        min_usable_sources=17,
        search_depth="advanced",
        allow_raw_content=True,
        max_browser_scrapes_per_query=0,
    )
    progress_events: list[tuple[str, dict]] = []

    result = acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "browser budget"),
        settings=settings,
        search_client=FakeSearchClient(),
        scraper=UnexpectedScraper(),
        progress_callback=lambda message, data: progress_events.append((message, data)),
    )

    assert result.sources == []
    assert result.candidates == []
    assert any(message == "skipped source candidate" for message, _ in progress_events)


def test_acquisition_followup_targets_only_missing_coverage_branches(tmp_path: Path) -> None:
    searched_queries: list[str] = []

    class FakeSearchClient:
        def search(self, query: str, **kwargs):
            searched_queries.append(query)
            raw_content = (
                "Misinformation acceptance research examines need for closure, belief formation, "
                "future research limitations, mixed evidence, boundary conditions, and uncertainty. "
            ) * 10
            return {
                "results": [
                    {
                        "url": f"https://example.com/{len(searched_queries)}",
                        "title": query,
                        "content": raw_content,
                        "raw_content": raw_content,
                        "score": 0.9,
                    }
                ]
            }

    class UnexpectedScraper:
        def fetch(self, url: str):  # pragma: no cover - raw Tavily content should bypass fetch.
            raise AssertionError(f"unexpected scrape for {url}")

    branches = [
        ResearchBranch(
            id="branch_1",
            title="Completed mechanism branch",
            objective="Already covered branch.",
            queries=["completed branch should not run"],
            min_sources=1,
            required_terms=["need for closure"],
        ),
        ResearchBranch(
            id="branch_2",
            title="Missing limitations branch",
            objective="Cover limitations and future research.",
            queries=["need for closure misinformation limitations future research"],
            min_sources=1,
            required_terms=["limitations", "future research"],
        ),
    ]
    settings = SimpleNamespace(
        min_source_words=40,
        min_relevant_chunks=1,
        max_candidates=20,
        max_sources=17,
        min_usable_sources=17,
        search_depth="advanced",
        allow_raw_content=True,
    )
    existing_source = SourceRecordV2(
        id=1,
        branch_id="branch_2",
        title="Existing limitations source",
        url="https://example.com/existing",
        canonical_url="https://example.com/existing",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=100,
        quality_score=0.8,
        quality_label="strong",
        quality_type="general_web",
        relevance_score=0.8,
    )

    acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=branches,
        artifacts=ResearchArtifactsV2.create(tmp_path, "active branches"),
        settings=settings,
        search_client=FakeSearchClient(),
        scraper=UnexpectedScraper(),
        existing_sources=[existing_source],
        existing_source_texts={1: "existing source text"},
        active_branch_ids={"branch_2"},
    )

    assert searched_queries
    assert "completed branch should not run" not in searched_queries
    assert searched_queries[0] == "need for closure misinformation limitations future research"


def test_acquisition_interleaves_missing_branch_followups(tmp_path: Path) -> None:
    searched_queries: list[str] = []

    class FakeSearchClient:
        def search(self, query: str, **kwargs):
            searched_queries.append(query)
            raw_content = (
                "Need for closure, misinformation acceptance, mediators, limitations, future research, "
                "source credibility, cognitive load, and evidence quality are discussed in context. "
            ) * 10
            return {
                "results": [
                    {
                        "url": f"https://example.com/interleave-{len(searched_queries)}",
                        "title": query,
                        "content": raw_content,
                        "raw_content": raw_content,
                        "score": 0.9,
                    }
                ]
            }

    branches = [
        ResearchBranch(
            id="branch_1",
            title="Mediators",
            objective="Explain mediators.",
            queries=["mediator query one", "mediator query two"],
            min_sources=1,
            required_terms=["source credibility", "cognitive load"],
        ),
        ResearchBranch(
            id="branch_2",
            title="Limitations",
            objective="Explain limitations.",
            queries=["limitations query one", "limitations query two"],
            min_sources=1,
            required_terms=["future research", "mixed evidence"],
        ),
    ]
    existing_sources = [
        SourceRecordV2(
            id=index,
            branch_id=branch.id,
            title=f"Existing {branch.id}",
            url=f"https://example.com/existing-{branch.id}",
            canonical_url=f"https://example.com/existing-{branch.id}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash=f"hash-{index}",
            extraction_method="test",
            word_count=100,
            quality_score=0.8,
            quality_label="strong",
            quality_type="general_web",
            relevance_score=0.8,
        )
        for index, branch in enumerate(branches, start=1)
    ]
    settings = SimpleNamespace(
        min_source_words=40,
        min_relevant_chunks=1,
        max_candidates=20,
        max_sources=17,
        min_usable_sources=17,
        max_followup_queries_per_branch=1,
        search_depth="advanced",
        allow_raw_content=True,
    )

    acquire_sources(
        question="How does need for closure affect misinformation acceptance?",
        branches=branches,
        artifacts=ResearchArtifactsV2.create(tmp_path, "interleaved branches"),
        settings=settings,
        search_client=FakeSearchClient(),
        existing_sources=existing_sources,
        existing_source_texts={1: "existing one", 2: "existing two"},
        active_branch_ids={"branch_1", "branch_2"},
    )

    assert searched_queries == ["mediator query one", "limitations query one"]


def test_generic_plan_handles_food_question_without_topic_specific_branches() -> None:
    plan = build_research_plan("What makes sourdough bread different from regular yeast bread?")

    branch_ids = [branch.id for branch in plan.branches]
    branch_text = " ".join(branch.title + " " + branch.objective + " " + " ".join(branch.required_terms) for branch in plan.branches)
    assert len(branch_ids) >= 1
    assert all(branch_id.startswith("branch_") for branch_id in branch_ids)
    assert "definition" not in branch_ids
    assert sum(branch.min_sources for branch in plan.branches) >= 17
    assert "sourdough" in " ".join(plan.branches[0].queries).lower()
    assert "bread" in branch_text.lower()
    assert "hypertension" not in branch_text.lower()


def test_generic_plan_handles_chinese_question_without_empty_research_branch() -> None:
    question = "请为我整合近几年有关中性粒细胞在脑缺血急性期和慢性期的功能和发展变化的研究成果。"
    plan = build_research_plan(question)

    assert plan.branches
    assert sum(branch.min_sources for branch in plan.branches) >= 17
    assert plan.branches[0].title != "Research"
    assert "中性粒细胞" in " ".join(plan.branches[0].queries)


def test_llm_semantic_planning_accepts_valid_domain_specific_plan(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "public health policy analyst",
            "report_outline": [
                "Executive Summary",
                "Exposure Pathways",
                "Evidence Strength",
                "Intervention Choices",
                "Limitations",
                "Sources",
            ],
            "source_requirements": [
                {"source_type": "academic", "min_count": 3, "rationale": "study evidence"},
                {"source_type": "government", "min_count": 2, "rationale": "public health guidance"},
            ],
            "acceptance_criteria": ["Explain exposure, health outcomes, interventions, and uncertainty."],
            "branches": [
                {
                    "title": "Urban heat exposure pathways",
                    "objective": "Explain how urban heat islands change local heat exposure and who is exposed.",
                    "queries": ["urban heat island exposure pathways public health", "urban heat exposure vulnerable groups"],
                    "source_types": ["academic", "government"],
                    "min_sources": 2,
                    "required_terms": ["heat exposure", "vulnerable groups"],
                    "completion_criteria": ["Evidence explains exposure pathways."],
                },
                {
                    "title": "Health outcomes and burden",
                    "objective": "Assess heat illness, mortality, respiratory stress, and other public health outcomes.",
                    "queries": ["urban heat island health outcomes mortality", "urban heat respiratory cardiovascular risk"],
                    "source_types": ["academic"],
                    "min_sources": 3,
                    "required_terms": ["mortality", "heat illness"],
                    "completion_criteria": ["Evidence covers health outcomes."],
                },
                {
                    "title": "Measurement and mapping methods",
                    "objective": "Compare surface temperature, air temperature, satellite, and neighborhood mapping approaches.",
                    "queries": ["urban heat island measurement mapping methods", "satellite urban heat mapping air temperature"],
                    "source_types": ["academic", "government"],
                    "min_sources": 2,
                    "required_terms": ["temperature mapping", "satellite"],
                    "completion_criteria": ["Evidence covers measurement methods."],
                },
                {
                    "title": "Mitigation interventions",
                    "objective": "Evaluate tree canopy, cool roofs, reflective surfaces, shade, and urban design options.",
                    "queries": ["urban heat island mitigation tree canopy cool roofs", "urban heat reflective surfaces shade evidence"],
                    "source_types": ["academic", "government"],
                    "min_sources": 3,
                    "required_terms": ["tree canopy", "cool roofs"],
                    "completion_criteria": ["Evidence compares interventions."],
                },
                {
                    "title": "Emergency response and adaptation",
                    "objective": "Research warning systems, cooling centers, outreach, and emergency public health operations.",
                    "queries": ["heat wave cooling centers emergency response public health", "urban heat warning systems public health"],
                    "source_types": ["government", "general_web"],
                    "min_sources": 2,
                    "required_terms": ["cooling centers", "warnings"],
                    "completion_criteria": ["Evidence covers emergency response."],
                },
                {
                    "title": "Equity, costs, and implementation trade-offs",
                    "objective": "Analyze distributional effects, costs, maintenance, governance, and trade-offs.",
                    "queries": ["urban heat mitigation equity costs implementation", "urban heat island intervention tradeoffs equity"],
                    "source_types": ["academic", "government"],
                    "min_sources": 3,
                    "required_terms": ["equity", "costs"],
                    "completion_criteria": ["Evidence covers equity and implementation."],
                },
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "How do urban heat islands affect public health, and which interventions matter most?",
        settings=settings,
        model=planner,
    )

    assert result.accepted is True
    assert result.used_model is True
    assert len(result.plan.branches) == 6
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17
    assert "Exposure Pathways" in result.plan.report_outline
    assert "urban heat island exposure pathways public health" in result.plan.branches[0].queries


def test_planning_guidance_criteria_survive_without_llm_planner(tmp_path: Path) -> None:
    guidance = format_criteria_guidance_block(
        {
            "dimension_weight": {"insight": 0.4},
            "criterions": {
                "insight": [
                    {
                        "criterion": "Explain the central mechanism",
                        "explanation": "Describe the causal pathway, evidence strength, and uncertainty.",
                        "weight": 1.0,
                    }
                ]
            },
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=False)

    result = build_or_enrich_research_plan(
        "How do cooling centers reduce heat illness risk during heat waves?",
        settings=settings,
        planning_guidance=guidance,
    )

    criteria_text = "\n".join(result.plan.acceptance_criteria)
    assert "Task-specific insight criterion: Explain the central mechanism" in criteria_text
    assert "causal pathway" in criteria_text


def test_llm_semantic_planning_rejects_invalid_output_without_losing_fallback(tmp_path: Path) -> None:
    planner = FakeSemanticJudge({"branches": []})
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What makes sourdough bread different from regular yeast bread?",
        settings=settings,
        model=planner,
    )

    assert result.accepted is False
    assert result.used_model is True
    assert result.failures
    assert len(result.plan.branches) >= 1
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17


def test_llm_semantic_planning_rejects_unresolved_placeholder_queries(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "financial analyst",
            "report_outline": ["Ranking", "Comparison", "Sources"],
            "branches": [
                {
                    "title": "Identify top global insurers",
                    "objective": "Determine the strongest global insurers by assets and financial strength.",
                    "queries": ["top global insurers by assets financial strength"],
                    "min_sources": 4,
                    "required_terms": ["global insurers", "financial strength"],
                },
                {
                    "title": "Company metrics",
                    "objective": "Collect financial metrics for each insurer.",
                    "queries": ["Company X 2022 annual report", "Company X credit rating 2023"],
                    "min_sources": 13,
                    "required_terms": ["financing", "credit rating"],
                },
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "收集整理目前国际综合实力前十的保险公司的相关资料，横向比较各公司的融资情况、信誉度、过往五年的增长幅度、实际分红、未来在中国发展潜力等维度。",
        settings=settings,
        model=planner,
    )

    assert result.accepted is False
    assert result.used_model is True
    assert any("unresolved placeholder" in failure.lower() for failure in result.failures)
    assert all("Company X" not in " ".join(branch.queries) for branch in result.plan.branches)


def test_llm_semantic_planning_accepts_focused_single_branch_with_guidance(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "social psychology researcher",
            "report_outline": ["Direct Answer", "Evidence and Mechanisms", "Sources"],
            "branches": [
                {
                    "title": "Need for closure and misinformation acceptance",
                    "objective": "Explain how need for closure shapes misinformation acceptance using theory and empirical evidence.",
                    "queries": [
                        "need for closure misinformation acceptance empirical evidence",
                        "need for cognitive closure false information belief formation",
                    ],
                    "min_sources": 17,
                    "required_terms": ["need for closure", "misinformation acceptance", "mechanisms", "empirical evidence"],
                }
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What is the role of need for closure on misinformation acceptance?",
        settings=settings,
        model=planner,
        planning_guidance="Benchmark criterion: cover the psychological mechanisms and direct empirical evidence.",
    )

    assert result.accepted is True
    assert len(result.plan.branches) == 1
    assert result.plan.branches[0].min_sources >= 17
    assert "Benchmark criterion" in planner.prompts[0]


def test_llm_semantic_planning_does_not_append_title_words_to_required_terms(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "social psychology researcher",
            "report_outline": ["Mechanisms", "Sources"],
            "branches": [
                {
                    "title": "Mediating variables in misinformation acceptance",
                    "objective": "Identify how need for closure can shape misinformation acceptance through intermediate variables.",
                    "queries": ["need for closure misinformation mediating variables"],
                    "min_sources": 17,
                    "required_terms": [
                        "mediating variables",
                        "depth of processing",
                        "source credibility assessment",
                        "trust in sources",
                        "emotional regulation",
                        "anxiety",
                        "cognitive load",
                        "mediating",
                        "variables",
                        "explaining",
                        "NFC",
                        "impacts",
                    ],
                }
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What is the role of need for closure on misinformation acceptance?",
        settings=settings,
        model=planner,
    )

    terms = result.plan.branches[0].required_terms
    lowered = {term.lower() for term in terms}
    assert "depth of processing" in lowered
    assert "source credibility assessment" in lowered
    assert "nfc" in lowered
    assert "mediating" not in lowered
    assert "variables" not in lowered
    assert "explaining" not in lowered
    assert "impacts" not in lowered
    assert "role" not in lowered


def test_llm_semantic_planning_keeps_guidance_out_of_search_branches(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "social psychology researcher",
            "report_outline": ["Direct Answer", "Sources"],
            "branches": [
                {
                    "title": "Need for closure theory",
                    "objective": "Define need for closure and its theoretical foundations.",
                    "queries": ["need for closure theory lay epistemology"],
                    "min_sources": 17,
                    "required_terms": ["need for closure", "lay epistemology"],
                }
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What is the role of need for closure on misinformation acceptance?",
        settings=settings,
        model=planner,
        planning_guidance=(
            "DeepResearch Bench evaluation guidance for this task:\n"
            "- Consideration of Diverse Contexts and Misinformation Types (weight: 0.1)\n"
            "- Discussion of Research Limitations, Nuances, and Future Directions (weight: 0.05)\n"
        ),
    )

    assert result.accepted is True
    assert len(result.plan.branches) == 1
    titles = " ".join(branch.title for branch in result.plan.branches).lower()
    queries = " ".join(query for branch in result.plan.branches for query in branch.queries).lower()
    assert "diverse contexts" not in titles
    assert "limitations" not in titles
    assert "weight:" not in queries
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17
    assert any("diverse contexts" in criterion.lower() for criterion in result.plan.acceptance_criteria)
    assert any("limitations" in criterion.lower() for criterion in result.plan.acceptance_criteria)


def test_llm_semantic_planning_does_not_use_lexical_scope_blacklists(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "social psychology researcher",
            "report_outline": ["Executive Summary", "Findings", "Sources"],
            "branches": [
                {
                    "title": "Need for closure definition",
                    "objective": "Define need for closure and its role in information processing.",
                    "queries": ["need for closure definition information processing", "need for closure epistemic motivation"],
                    "min_sources": 2,
                    "required_terms": ["need for closure", "information processing"],
                },
                {
                    "title": "Misinformation acceptance",
                    "objective": "Explain misinformation acceptance and belief formation.",
                    "queries": ["misinformation acceptance belief formation", "false information belief psychology"],
                    "min_sources": 2,
                    "required_terms": ["misinformation acceptance", "belief formation"],
                },
                {
                    "title": "Mechanisms linking closure and false beliefs",
                    "objective": "Analyze how closure motivation changes scrutiny, heuristics, and ambiguity tolerance.",
                    "queries": ["need for closure heuristics ambiguity misinformation", "closure motivation false beliefs"],
                    "min_sources": 2,
                    "required_terms": ["heuristics", "ambiguity tolerance"],
                },
                {
                    "title": "Empirical evidence for the relationship",
                    "objective": "Review studies linking need for closure to misinformation or conspiracy belief.",
                    "queries": ["need for closure misinformation empirical study", "need for closure conspiracy beliefs evidence"],
                    "min_sources": 2,
                    "required_terms": ["empirical evidence", "conspiracy beliefs"],
                },
                {
                    "title": "Interventions and mitigation strategies",
                    "objective": "Recommend interventions, mitigation, and media literacy strategies.",
                    "queries": ["need for closure misinformation intervention mitigation", "media literacy need for closure strategy"],
                    "min_sources": 2,
                    "required_terms": ["interventions", "mitigation"],
                },
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What is the role of need for closure on misinformation acceptance?",
        settings=settings,
        model=planner,
    )

    assert result.accepted is True
    assert any("Interventions" in branch.title for branch in result.plan.branches)
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17


def test_semantic_evidence_reuses_prior_judgments_without_recalling_model(tmp_path: Path) -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Direct relationship",
        objective="Explain the relationship.",
        queries=["relationship evidence"],
        required_terms=["relationship evidence"],
    )
    plan = ResearchPlan(
        question="How does one construct affect another?",
        intent="general",
        audience="general",
        report_outline=[],
        branches=[branch],
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="The evidence directly explains the relationship between the constructs.",
        supporting_excerpt="The evidence directly explains the relationship between the constructs.",
        source_url="https://example.com",
        source_title="Source",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    first_model = FakeSemanticJudge(
        {
            "cards": [
                {
                    "id": 1,
                    "keep": True,
                    "branch_alignment_score": 1.0,
                    "entailment_score": 1.0,
                    "evidence_relevance_score": 1.0,
                    "normalized_claim": "The evidence directly explains the relationship between the constructs.",
                    "key_points": ["direct relationship"],
                    "limitations": [],
                    "failure_reasons": [],
                }
            ]
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    first = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=first_model,
    )
    second = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=RaisingSemanticJudge(),
        prior_judgments=first.judgments,
    )

    assert len(first_model.prompts) == 1
    assert len(second.cards) == 1
    assert second.metrics["semantic_evidence_cached_judgment_count"] == 1
    assert second.metrics["semantic_evidence_batches"] == 0


def test_semantic_evidence_uses_deterministic_fallback_on_provider_quota(tmp_path: Path) -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Direct relationship",
        objective="Explain the relationship.",
        queries=["relationship evidence"],
        required_terms=["relationship evidence"],
    )
    plan = ResearchPlan(
        question="How does one construct affect another?",
        intent="general",
        audience="general",
        report_outline=[],
        branches=[branch],
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="The evidence directly explains the relationship between the constructs.",
        supporting_excerpt="The evidence directly explains the relationship between the constructs.",
        source_url="https://example.com",
        source_title="Source",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    result = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=QuotaSemanticJudge(),
    )

    assert len(result.cards) == 1
    assert result.metrics["semantic_evidence_provider_failure_count"] == 1
    assert any("deterministic fallback" in failure for failure in result.failures)


def test_generic_source_validation_accepts_medical_topic() -> None:
    plan = build_research_plan(
        "Do Mediterranean diets help adults with hypertension? Cover blood pressure mechanisms and evidence."
    )
    branch = next(branch for branch in plan.branches if "pressure" in " ".join(branch.required_terms + branch.queries).lower())
    content = (
        "Mediterranean diets emphasize vegetables, fruits, legumes, whole grains, nuts, olive oil, "
        "and moderate fish intake. For adults with hypertension, the mechanism may involve lower "
        "sodium intake, higher potassium and fiber intake, improved endothelial function, and better "
        "weight control. Clinical nutrition studies often evaluate systolic and diastolic blood pressure, "
        "adherence, medication use, and cardiovascular risk. This dietary pattern can help some patients, "
        "but it does not replace medical evaluation, antihypertensive medication when indicated, or monitoring."
    )

    result = validate_source_content(
        title="Mediterranean diet and hypertension",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
    )

    assert result.usable is True
    assert result.relevance_score >= 0.30


def test_source_validation_accepts_chinese_sentence_chunks() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="中性粒细胞与脑缺血急性期",
        objective="分析中性粒细胞在脑缺血急性期的功能变化。",
        queries=["中性粒细胞 脑缺血急性期 功能变化"],
        min_sources=1,
        required_terms=["中性粒细胞", "脑缺血急性期", "功能变化"],
    )
    content = (
        "近年研究显示，中性粒细胞在脑缺血急性期会快速募集到损伤区域，"
        "并通过炎症因子释放、血脑屏障影响、微血管阻塞和免疫细胞互作改变局部组织环境。"
        "这些功能变化与梗死扩大、神经炎症强度以及后续修复窗口密切相关。"
        "慢性期研究还提示，中性粒细胞亚群可能参与免疫调节和组织重塑。"
    )

    result = validate_source_content(
        title="中性粒细胞在脑缺血急性期的功能变化",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="请整合中性粒细胞在脑缺血急性期和慢性期的功能变化研究。",
    )

    assert result.usable is True
    assert result.relevant_chunk_count >= 1


def test_source_validation_accepts_translated_branch_source_context() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Top global insurers by assets",
        objective="Identify the strongest global insurance companies by assets and financial strength.",
        queries=["largest insurance companies worldwide by assets"],
        min_sources=1,
        required_terms=["global insurers", "assets", "financial strength"],
    )
    content = (
        "A ranking of the largest insurance companies worldwide by total assets identifies major global insurers "
        "and explains how asset scale, financial strength, and market position differ across companies. "
        "The report compares life insurers and diversified insurance groups across regions. "
    ) * 6

    result = validate_source_content(
        title="Largest insurance companies worldwide by assets",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question=(
            "\u6536\u96c6\u6574\u7406\u76ee\u524d\u56fd\u9645\u7efc\u5408\u5b9e\u529b\u524d\u5341"
            "\u7684\u4fdd\u9669\u516c\u53f8\u7684\u76f8\u5173\u8d44\u6599\uff0c\u5e76\u6a2a"
            "\u5411\u6bd4\u8f83\u878d\u8d44\u3001\u4fe1\u8a89\u5ea6\u3001\u589e\u957f"
            "\u3001\u5206\u7ea2\u548c\u4e2d\u56fd\u53d1\u5c55\u6f5c\u529b\u3002"
        ),
    )

    assert result.usable is True
    assert result.relevance_score >= 0.30


def test_build_evidence_cards_splits_chinese_sentences() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="中性粒细胞与脑缺血急性期",
        objective="分析中性粒细胞在脑缺血急性期的功能变化。",
        queries=["中性粒细胞 脑缺血急性期 功能变化"],
        min_sources=1,
        required_terms=["中性粒细胞", "脑缺血急性期", "功能变化"],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="中性粒细胞脑缺血研究",
        url="https://example.com/neutrophil-stroke",
        canonical_url="https://example.com/neutrophil-stroke",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    text = (
        "背景信息介绍研究设计。"
        "中性粒细胞在脑缺血急性期会快速募集到缺血区域，并通过炎症因子释放、"
        "血脑屏障损伤、微血管阻塞和免疫细胞互作影响神经炎症强度与临床结局。"
        "其他段落讨论统计方法。"
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question="中性粒细胞在脑缺血急性期的功能变化是什么？",
    )

    assert cards
    assert "中性粒细胞" in cards[0].claim
    assert "脑缺血急性期" in cards[0].claim


def test_evidence_builder_accepts_translated_branch_context() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Global insurance companies by assets",
        objective="Identify the largest insurance companies worldwide by assets.",
        queries=["top insurance companies by assets"],
        min_sources=1,
        required_terms=["global insurance companies", "insurance companies assets"],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Largest global insurers by assets",
        url="https://example.com/insurers-assets",
        canonical_url="https://example.com/insurers-assets",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="industry",
        relevance_score=0.8,
    )
    text = (
        "A ranking of global insurance companies by total assets identifies major insurers worldwide "
        "and compares their asset scale, market position, and balance sheet strength across regions."
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question=(
            "\u6536\u96c6\u6574\u7406\u76ee\u524d\u56fd\u9645\u7efc\u5408\u5b9e\u529b"
            "\u524d\u5341\u7684\u4fdd\u9669\u516c\u53f8\u7684\u76f8\u5173\u8d44\u6599"
        ),
    )

    assert cards
    assert cards[0].branch_id == branch.id
    assert "global insurance companies" in cards[0].claim.lower()


def test_source_validation_rejects_generic_content_that_misses_original_question() -> None:
    plan = build_research_plan("What is the role of need for closure on misinformation acceptance?")
    branch = plan.branches[0]
    content = (
        "A purchase approval workflow starts with request submission, budget validation, manager approval, "
        "vendor onboarding, payment release, exception handling, and audit trails. The workflow improves "
        "budget control and operational efficiency for finance teams. "
    ) * 4

    result = validate_source_content(
        title="Operations workflow guide",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question=plan.question,
    )

    assert result.usable is False
    assert any("original question" in reason for reason in result.reasons)


def test_source_validation_rejects_near_neighbor_question_anchor() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    content = (
        "Need for cognition is a motivation to engage in effortful thinking. "
        "Researchers sometimes examine need for cognition and misinformation acceptance, "
        "but this passage discusses effortful cognition rather than certainty seeking. "
    ) * 4

    result = validate_source_content(
        title="Need for cognition and misinformation acceptance",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is False
    assert any("complete phrase" in reason for reason in result.reasons)


def test_source_validation_rejects_neighboring_concept_dominance_with_incidental_target_mentions() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["NFC and misinformation acceptance studies"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    content = (
        "Need for cognition is a motivation to engage in effortful thinking. "
        "Need for cognition appears in studies about false memories, cognitive effort, and misinformation. "
        "Some authors briefly mention need for closure as a related but different construct. "
    ) * 6

    result = validate_source_content(
        title="Need for cognition and misinformation acceptance",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is False
    assert any("neighboring concept" in reason for reason in result.reasons)


def test_source_validation_rejects_acronym_expansion_collision() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how Need for Closure (NFC) affects misinformation acceptance.",
        queries=["NFC and systematic analysis"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    content = (
        "Near Field Communication (NFC) enables short range wireless communication. "
        "Near field communication payment systems analyze cyber threats, transactions, tags, and devices. "
        "The source discusses systematic analysis, security mitigation, and communication protocols. "
    ) * 8

    result = validate_source_content(
        title="Near-Field Communication (NFC) Cyber Threats and Mitigation Solutions",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is False
    assert any("acronym" in reason for reason in result.reasons)


def test_source_validation_does_not_treat_protected_aliases_as_neighbors() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and cognitive closure",
        objective="Define need for closure as a form of cognitive closure.",
        queries=["need for closure cognitive closure"],
        required_terms=["need for closure", "cognitive closure"],
    )
    content = (
        "Need for closure is a desire for definite cognitive closure instead of prolonged ambiguity. "
        "The need for closure framework explains why people seek certainty and stable answers. "
        "Cognitive closure is therefore part of the same construct rather than a competing topic. "
    ) * 4

    result = validate_source_content(
        title="Need for Closure: influence user behaviour",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is True
    assert not any("neighboring concept" in reason for reason in result.reasons)


def test_source_validation_accepts_source_that_substantively_compares_neighboring_constructs() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["need for closure need for cognition misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    content = (
        "Need for closure is a desire for definite answers and reduced ambiguity. "
        "Need for closure can increase misinformation acceptance when quick certainty displaces careful checking. "
        "The article contrasts need for closure with need for cognition, explaining that need for cognition concerns effortful thinking. "
        "This comparison clarifies why need for closure and misinformation acceptance form a distinct pathway. "
    ) * 4

    result = validate_source_content(
        title="Need for closure, need for cognition, and misinformation acceptance",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is True


def test_source_validation_accepts_direct_single_concept_context_source() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Misinformation acceptance definitions",
        objective="Define misinformation acceptance and belief formation.",
        queries=["misinformation acceptance definition psychology"],
        required_terms=["misinformation acceptance", "belief formation"],
    )
    content = (
        "Misinformation acceptance describes the process by which people endorse inaccurate claims. "
        "The psychology of belief formation includes source credibility, repetition, prior attitudes, "
        "and cognitive shortcuts that shape whether people accept false information. "
    ) * 4

    result = validate_source_content(
        title="Misinformation acceptance and belief formation",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is True


def test_coverage_allows_partial_soft_required_term_coverage() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mechanisms and evidence",
        objective="Explain the mechanisms and evidence for a relationship.",
        queries=["mechanisms evidence relationship"],
        min_sources=1,
        required_terms=[
            "mechanisms",
            "empirical evidence",
            "mediating factors",
            "moderating factors",
            "boundary conditions",
            "limitations",
            "future research",
            "source credibility",
            "information processing",
            "uncertainty",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Mechanism Source",
        url="https://example.com/mechanism",
        canonical_url="https://example.com/mechanism",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim=(
            "The evidence explains mechanisms, empirical evidence, mediating factors, "
            "source credibility, information processing, and uncertainty."
        ),
        supporting_excerpt=(
            "The evidence explains mechanisms, empirical evidence, mediating factors, "
            "source credibility, information processing, and uncertainty."
        ),
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=[card], sources=[source])

    assert coverage.complete is True
    assert coverage.missing_branches == []
    assert any("required term coverage" in point for point in coverage.branches[0].covered_points)


def test_coverage_keeps_sparse_required_term_coverage_incomplete() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mechanisms and evidence",
        objective="Explain the mechanisms and evidence for a relationship.",
        queries=["mechanisms evidence relationship"],
        min_sources=1,
        required_terms=[
            "mechanisms",
            "empirical evidence",
            "mediating factors",
            "moderating factors",
            "boundary conditions",
            "limitations",
            "future research",
            "source credibility",
            "information processing",
            "uncertainty",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Sparse Source",
        url="https://example.com/sparse",
        canonical_url="https://example.com/sparse",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="The evidence mentions mechanisms only.",
        supporting_excerpt="The evidence mentions mechanisms only.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=[card], sources=[source])

    assert coverage.complete is False
    assert coverage.missing_branches == [branch.id]
    assert any("actual" in point for point in coverage.branches[0].missing_points)


def test_coverage_accepts_strong_semantic_evidence_without_exact_phrase_matches() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mediators and boundary conditions",
        objective="Explain factors that mediate and moderate the relationship.",
        queries=["mediators moderators relationship"],
        min_sources=3,
        required_terms=[
            "source credibility reliance",
            "emotional states",
            "cognitive load",
            "situational urgency",
            "message complexity",
            "prior knowledge",
        ],
    )
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=branch.id,
            title=f"Semantic Source {index}",
            url=f"https://example.com/semantic/{index}",
            canonical_url=f"https://example.com/semantic/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash=f"hash-{index}",
            extraction_method="test",
            word_count=400,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        )
        for index in range(1, 4)
    ]
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=branch.id,
            claim=f"Study {index} describes a distinct pathway that shapes the relationship.",
            supporting_excerpt=f"Study {index} describes a distinct pathway that shapes the relationship.",
            source_url=sources[index - 1].url,
            source_title=sources[index - 1].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
            semantic_score=0.86,
            semantic_notes=[f"semantic pathway {index}", "boundary condition"],
        )
        for index in range(1, 4)
    ]

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=cards, sources=sources)

    assert coverage.complete is True
    assert coverage.missing_branches == []
    assert any("semantic evidence sufficiency" in point for point in coverage.branches[0].covered_points)


def test_coverage_allows_evidence_limited_synthesis_when_direct_cards_are_sparse() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Nuanced relationship and evidence limits",
        objective="Synthesize what available evidence can and cannot establish about the relationship.",
        queries=["relationship evidence limitations"],
        min_sources=2,
        required_terms=[
            "relationship evidence",
            "methodological limitations",
            "future studies",
            "boundary conditions",
            "alternative explanations",
            "context dependence",
        ],
    )
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=branch.id,
            title=f"Limited Source {index}",
            url=f"https://example.com/limited/{index}",
            canonical_url=f"https://example.com/limited/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash=f"hash-{index}",
            extraction_method="test",
            word_count=500,
            quality_score=0.88,
            quality_label="good",
            quality_type="academic",
            relevance_score=0.72,
        )
        for index in range(1, 7)
    ]
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id=branch.id,
            claim="The evidence identifies relationship evidence and methodological limitations without proving a direct pathway.",
            supporting_excerpt="The evidence identifies relationship evidence and methodological limitations without proving a direct pathway.",
            source_url=sources[0].url,
            source_title=sources[0].title,
            quality_score=0.88,
            relevance_score=0.62,
            confidence=0.62,
            semantic_score=0.50,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id=branch.id,
            claim="The review calls for future studies because boundary conditions and context dependence remain unresolved.",
            supporting_excerpt="The review calls for future studies because boundary conditions and context dependence remain unresolved.",
            source_url=sources[1].url,
            source_title=sources[1].title,
            quality_score=0.88,
            relevance_score=0.62,
            confidence=0.62,
            semantic_score=0.50,
        ),
    ]

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=cards, sources=sources)

    assert coverage.complete is True
    assert coverage.missing_branches == []
    assert "evidence-limited synthesis readiness" in coverage.branches[0].covered_points
    assert not any("strong semantic evidence cards" in point for point in coverage.branches[0].covered_points)


def test_coverage_accepts_evidence_rich_branch_with_planner_phrase_drift() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Five-year growth analysis",
        objective="Compare growth patterns across sources.",
        queries=["growth analysis"],
        min_sources=3,
        required_terms=[
            "growth analysis calculate",
            "analysis calculate compare",
            "calculate compare revenue",
            "growth analysis",
        ],
    )
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=branch.id,
            title=f"Growth Source {index}",
            url=f"https://example.com/growth/{index}",
            canonical_url=f"https://example.com/growth/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash=f"hash-{index}",
            extraction_method="test",
            word_count=500,
            quality_score=0.72,
            quality_label="usable",
            quality_type="industry",
            relevance_score=0.3,
        )
        for index in range(1, 5)
    ]
    cards = [
        EvidenceCard(
            id=index,
            source_id=((index - 1) % 4) + 1,
            branch_id=branch.id,
            claim="Revenue and asset growth are compared across insurers over several years.",
            supporting_excerpt="Revenue and asset growth are compared across insurers over several years.",
            source_url=sources[((index - 1) % 4)].url,
            source_title=sources[((index - 1) % 4)].title,
            quality_score=0.72,
            relevance_score=0.3,
            confidence=0.63,
        )
        for index in range(1, 13)
    ]

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=cards, sources=sources)

    assert coverage.complete is True
    assert "evidence-limited synthesis readiness" in coverage.branches[0].covered_points


def test_evidence_builder_uses_heading_windows_for_ranking_sources() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Future asset ranking prediction",
        objective="Forecast likely future asset ranking among major insurers.",
        queries=["predict insurance industry asset rankings"],
        min_sources=1,
        required_terms=["asset ranking"],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Best's Rankings and World's Largest Insurance Companies",
        url="https://example.com/rankings",
        canonical_url="https://example.com/rankings",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.8,
        quality_label="strong",
        quality_type="industry",
        relevance_score=0.7,
    )
    text = "\n".join(
        [
            "# Best's Rankings and World's Largest Insurance Companies",
            "## By assets",
            "The annual ranking compares insurance groups by assets and premium scale.",
            "The table gives a basis for assessing future asset leadership.",
        ]
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question="Which insurers are likely to lead future asset ranking?",
    )

    assert cards
    assert any("assets" in card.claim.lower() for card in cards)


def test_evidence_builder_uses_strong_branch_overlap_without_exact_anchor_phrase() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mediating and moderating factors",
        objective="Identify factors that influence a misinformation acceptance relationship.",
        queries=["source credibility information complexity misinformation acceptance"],
        min_sources=1,
        required_terms=[
            "source cues",
            "information processing depth",
            "emotional responses",
            "mediating factors",
            "moderating factors",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Source Credibility Study",
        url="https://example.com/source-credibility",
        canonical_url="https://example.com/source-credibility",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    text = (
        "Credibility, information complexity, and platform context can influence misinformation "
        "acceptance by changing how much people scrutinize false claims before accepting them. "
        "This sentence intentionally does not repeat the planned anchor phrases verbatim."
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question="What factors influence misinformation acceptance?",
    )

    assert cards
    assert "misinformation acceptance" in cards[0].claim


def test_evidence_builder_skips_stale_neighboring_concept_source() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how Need for Closure (NFC) affects misinformation acceptance.",
        queries=["NFC misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    sources = [
        SourceRecordV2(
            id=1,
            branch_id=branch.id,
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash-1",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id=branch.id,
            title="Near-Field Communication (NFC) Cyber Threats and Mitigation Solutions",
            url="https://example.com/near-field",
            canonical_url="https://example.com/near-field",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash-2",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]

    cards = build_evidence_cards(
        branches=[branch],
        sources=sources,
        source_texts={
            1: (
                "Need for closure can increase misinformation acceptance when people seek quick certainty. "
                "Need for closure encourages premature judgment when misinformation acceptance offers a simple answer."
            ),
            2: (
                "Near Field Communication (NFC) enables short range wireless communication. "
                "Near field communication payment systems analyze cyber threats, transactions, tags, and devices. "
            )
            * 4,
        },
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert cards
    assert {card.source_id for card in cards} == {1}


def test_evidence_extraction_prefers_branch_anchor_sentences() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Mixed psychological constructs",
        url="https://example.com/mixed",
        canonical_url="https://example.com/mixed",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    text = (
        "Need for cognition is associated with effortful thinking and may affect how people evaluate misinformation acceptance. "
        "Need for closure is linked to misinformation acceptance when people seek quick certainty and stop evaluating alternatives."
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert cards
    assert "need for closure" in cards[0].claim.lower()


def test_stanford_hai_bad_scrape_is_rejected() -> None:
    branch = build_research_plan("What are urban heat islands and how do they affect public health?").branches[0]
    bad_content = (
        "Explore Similar Terms. Stanford HAI. Your browser does not support the video tag. "
        "Subscribe to newsletter. Main navigation. Search. Related content. "
    ) * 8

    result = validate_source_content(
        title="Stanford HAI",
        content=bad_content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
    )

    assert result.usable is False
    assert any("boilerplate" in reason or "relevance" in reason for reason in result.reasons)


def test_report_quality_rejects_internal_research_artifact_language() -> None:
    report = (
        "# Report\n\n"
        "No verified evidence cards were available for branch_1, so the report cannot answer this section. [1]\n\n"
        "## Sources\n\n"
        "[1] Example: https://example.com\n"
    )

    issues = report_quality_issues(report)

    assert any("internal research artifact" in issue for issue in issues)


def test_irrelevant_recovery_fails_answer_coverage() -> None:
    plan = build_research_plan("What are urban heat islands and how do they affect public health?")
    branch_id = plan.branches[0].id
    source = SourceRecordV2(
        id=1,
        branch_id=branch_id,
        title="Invoice Processing Workflow",
        url="https://example.com/invoices",
        canonical_url="https://example.com/invoices",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=100,
        quality_score=0.9,
        quality_label="high",
        quality_type="academic",
        relevance_score=0.1,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch_id,
        claim="Invoice processing workflows reconcile purchase orders, receipts, approvals, and payment records.",
        supporting_excerpt="Invoice processing workflows reconcile purchase orders, receipts, approvals, and payment records.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.1,
        confidence=0.2,
    )
    coverage = CoverageMatrix(branches=[], complete=False, coverage_score=0.0, missing_branches=[branch_id])
    report = (
        "# Research Report\n\n"
        "## Mismatched Recovery Output\n\n"
        "Invoice processing workflows reconcile purchase orders, receipts, approvals, and payment records. [1]\n\n"
        "## Sources\n\n"
        "[1] Invoice Processing Workflow: https://example.com/invoices\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=coverage,
        source_texts={1: card.supporting_excerpt},
    )

    assert result.valid is False
    assert result.answer_coverage_score < 0.5
    assert any("Answer coverage" in failure for failure in result.failures)


def test_verifier_rejects_report_that_drifts_from_original_question() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="COVID Misinformation Source",
        url="https://example.com/covid-misinformation",
        canonical_url="https://example.com/covid-misinformation",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="COVID-19 misinformation can spread through social media during public health crises.",
        supporting_excerpt="COVID-19 misinformation can spread through social media during public health crises.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# COVID-19 Misinformation Report\n\n"
        "## Public Health Effects\n\n"
        "COVID-19 misinformation can spread through social media during public health crises and affect public behavior. [1]\n\n"
        "## Evidence Pattern\n\n"
        "The evidence indicates that public health communication can be challenged by viral false claims online. [1]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because this evidence does not resolve every public health context. [1]\n\n"
        "## Sources\n\n"
        "[1] COVID Misinformation Source: https://example.com/covid-misinformation\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt},
    )

    assert result.valid is False
    assert result.request_alignment_score < 0.45
    assert any("topic alignment" in failure.lower() for failure in result.failures)


def test_verifier_rejects_correct_title_with_contextual_case_as_opening_answer() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    contextual_source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Contextual Conspiracy Belief Study",
        url="https://example.com/context",
        canonical_url="https://example.com/context",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash-1",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.8,
    )
    direct_source = SourceRecordV2(
        id=2,
        branch_id=branch.id,
        title="Need for Closure and Misinformation Acceptance",
        url="https://example.com/direct",
        canonical_url="https://example.com/direct",
        provenance="web",
        content_path="source_docs/source_2.md",
        content_hash="hash-2",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    contextual_card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="A study examined the role of need for cognitive closure for COVID-19 conspiracy beliefs.",
        supporting_excerpt="A study examined the role of need for cognitive closure for COVID-19 conspiracy beliefs.",
        source_url=contextual_source.url,
        source_title=contextual_source.title,
        quality_score=0.9,
        relevance_score=0.8,
        confidence=0.9,
    )
    direct_card = EvidenceCard(
        id=2,
        source_id=2,
        branch_id=branch.id,
        claim="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        supporting_excerpt="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        source_url=direct_source.url,
        source_title=direct_source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "A study examined the role of need for cognitive closure for COVID-19 conspiracy beliefs. [1]\n\n"
        "## The Broader Relationship\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty and stop evaluating alternatives. [2]\n\n"
        "## Cross-Source Pattern\n\n"
        "Taken together, the evidence indicates that contextual conspiracy-belief evidence must be interpreted as one case within the broader relationship. [1, 2]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because contextual examples do not substitute for direct evidence about misinformation acceptance. [1, 2]\n\n"
        "## Sources\n\n"
        "[1] Contextual Conspiracy Belief Study: https://example.com/context\n"
        "[2] Need for Closure and Misinformation Acceptance: https://example.com/direct\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[contextual_source, direct_source],
        evidence_cards=[contextual_card, direct_card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={
            1: contextual_card.supporting_excerpt,
            2: direct_card.supporting_excerpt,
        },
    )

    assert result.valid is False
    assert any("opening answer topic alignment" in failure.lower() for failure in result.failures)


def test_verifier_rejects_cited_paragraphs_that_drift_inside_otherwise_aligned_report() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Mixed Source",
        url="https://example.com/mixed",
        canonical_url="https://example.com/mixed",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        supporting_excerpt="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    off_topic = (
        "An invoice approval workflow routes purchase orders, vendor onboarding, budget validation, "
        "exception handling, payment scheduling, and audit records through finance operations."
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty and stop evaluating alternatives. [1]\n\n"
        f"{off_topic} [1]\n\n"
        "## Evidence Strength and Limits\n\n"
        "Confidence is limited because the cited evidence must be interpreted within the specific question about closure and misinformation. [1]\n\n"
        "## Sources\n\n"
        "[1] Mixed Source: https://example.com/mixed\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt + "\n\n" + off_topic},
    )

    assert result.valid is False
    assert any("topic-drift" in failure.lower() for failure in result.failures)


def test_verifier_rejects_stale_neighboring_concept_sources_even_if_cited() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how Need for Closure (NFC) affects misinformation acceptance.",
        queries=["NFC and misinformation acceptance studies"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    direct_source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash-1",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    stale_source = SourceRecordV2(
        id=2,
        branch_id=branch.id,
        title="Need for cognition and misinformation acceptance",
        url="https://example.com/need-for-cognition",
        canonical_url="https://example.com/need-for-cognition",
        provenance="web",
        content_path="source_docs/source_2.md",
        content_hash="hash-2",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id=branch.id,
            claim="Need for closure can increase misinformation acceptance when people seek quick certainty.",
            supporting_excerpt="Need for closure can increase misinformation acceptance when people seek quick certainty.",
            source_url=direct_source.url,
            source_title=direct_source.title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id=branch.id,
            claim="Need for cognition concerns effortful thinking and appears in misinformation studies.",
            supporting_excerpt="Need for cognition concerns effortful thinking and appears in misinformation studies.",
            source_url=stale_source.url,
            source_title=stale_source.title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty and stop evaluating alternatives. [1, 2]\n\n"
        "## Evidence Strength and Limits\n\n"
        "Taken together, the evidence indicates that neighboring constructs should not replace the specific need-for-closure pathway. [1, 2]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
        "[2] Need for cognition and misinformation acceptance: https://example.com/need-for-cognition\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[direct_source, stale_source],
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={
            1: "Need for closure can increase misinformation acceptance when people seek quick certainty.",
            2: (
                "Need for cognition is a motivation to engage in effortful thinking. "
                "Need for cognition appears in studies about false memories, cognitive effort, and misinformation. "
                "Some authors briefly mention need for closure as a related but different construct. "
            )
            * 4,
        },
    )

    assert result.valid is False
    assert result.cited_source_alignment_score < 1.0
    assert any("fails current branch/request alignment" in failure for failure in result.failures)


def test_verifier_rejects_individual_unsupported_citation_in_supported_group() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    sources = [
        SourceRecordV2(
            id=1,
            branch_id=branch.id,
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash-1",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id=branch.id,
            title="Misinformation Source",
            url="https://example.com/misinformation",
            canonical_url="https://example.com/misinformation",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash-2",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=3,
            branch_id=branch.id,
            title="Unrelated Source",
            url="https://example.com/unrelated",
            canonical_url="https://example.com/unrelated",
            provenance="web",
            content_path="source_docs/source_3.md",
            content_hash="hash-3",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id=branch.id,
            claim="Need for closure is a desire for quick certainty and firm answers.",
            supporting_excerpt="Need for closure is a desire for quick certainty and firm answers.",
            source_url=sources[0].url,
            source_title=sources[0].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id=branch.id,
            claim="Misinformation acceptance involves believing inaccurate information.",
            supporting_excerpt="Misinformation acceptance involves believing inaccurate information.",
            source_url=sources[1].url,
            source_title=sources[1].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=3,
            source_id=3,
            branch_id=branch.id,
            claim="Urban heat islands raise local temperature exposure.",
            supporting_excerpt="Urban heat islands raise local temperature exposure.",
            source_url=sources[2].url,
            source_title=sources[2].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can shape misinformation acceptance because people seeking quick certainty may believe inaccurate information. [1, 2, 3]\n\n"
        "## Synthesis\n\n"
        "Taken together, the evidence indicates that closure motivation and misinformation acceptance should be interpreted as a linked cognitive pattern. [1, 2]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because the evidence should be interpreted cautiously across information contexts. [1, 2]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
        "[2] Misinformation Source: https://example.com/misinformation\n"
        "[3] Unrelated Source: https://example.com/unrelated\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=sources,
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={
            1: cards[0].supporting_excerpt,
            2: cards[1].supporting_excerpt,
            3: cards[2].supporting_excerpt,
        },
    )

    assert result.valid is False
    assert any(
        claim.get("support_kind") == "individual_citation" and claim.get("cited_source_ids") == [3]
        for claim in result.weakly_supported_claims
    )


def test_verifier_rejects_chinese_prompt_answered_in_english() -> None:
    question = "\u8bf7\u5206\u6790\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u5f71\u54cd\u516c\u5171\u5065\u5eb7"
    branch = ResearchBranch(
        id="branch_1",
        title="Urban heat and public health",
        objective="Explain how urban heat islands affect public health.",
        queries=["urban heat public health"],
        min_sources=1,
        required_terms=["urban heat", "public health"],
    )
    plan = ResearchPlan(
        question=question,
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Urban Heat Source",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Urban heat islands increase public health risks by raising local heat exposure.",
        supporting_excerpt="Urban heat islands increase public health risks by raising local heat exposure.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Urban Heat and Public Health\n\n"
        "Urban heat islands increase public health risks by raising local heat exposure, making heat illness and related health burdens more likely during extreme temperatures. [1]\n\n"
        "## Evidence Pattern\n\n"
        "Taken together, the evidence indicates that urban heat and public health risks are linked through exposure, vulnerability, and limits in local cooling access. [1]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because the available evidence should be interpreted cautiously across neighborhoods, populations, and weather conditions. [1]\n\n"
        "## Sources\n\n"
        "[1] Urban Heat Source: https://example.com/urban-heat\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: report},
    )

    assert result.language_alignment_score < 0.80
    assert any("language alignment" in failure.lower() for failure in result.failures)


def test_verifier_accepts_chinese_report_structure_language_signals() -> None:
    question = "\u8bf7\u5206\u6790\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u5f71\u54cd\u516c\u5171\u5065\u5eb7"
    branch = ResearchBranch(
        id="branch_1",
        title="\u57ce\u5e02\u70ed\u5c9b\u4e0e\u516c\u5171\u5065\u5eb7",
        objective="\u8bf4\u660e\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u901a\u8fc7\u70ed\u66b4\u9732\u5f71\u54cd\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        queries=["\u57ce\u5e02\u70ed\u5c9b \u516c\u5171\u5065\u5eb7"],
        min_sources=1,
        required_terms=["\u57ce\u5e02\u70ed\u5c9b", "\u516c\u5171\u5065\u5eb7", "\u70ed\u66b4\u9732"],
    )
    plan = ResearchPlan(
        question=question,
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="\u57ce\u5e02\u70ed\u5c9b\u8bc1\u636e",
        url="https://example.com/urban-heat-zh",
        canonical_url="https://example.com/urban-heat-zh",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="\u57ce\u5e02\u70ed\u5c9b\u4f1a\u63d0\u9ad8\u5c40\u5730\u6e29\u5ea6\uff0c\u5e76\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        supporting_excerpt="\u57ce\u5e02\u70ed\u5c9b\u4f1a\u63d0\u9ad8\u5c40\u5730\u6e29\u5ea6\uff0c\u5e76\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# \u57ce\u5e02\u70ed\u5c9b\u4e0e\u516c\u5171\u5065\u5eb7\n\n"
        "\u57ce\u5e02\u70ed\u5c9b\u4f1a\u63d0\u9ad8\u5c40\u5730\u6e29\u5ea6\uff0c\u5e76\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\uff1b\u8bc1\u636e\u663e\u793a\uff0c\u8fd9\u4e00\u5f71\u54cd\u9700\u8981\u56f4\u7ed5\u70ed\u66b4\u9732\u3001\u6613\u611f\u4eba\u7fa4\u548c\u5730\u533a\u5dee\u5f02\u7efc\u5408\u7406\u89e3\u3002 [1]\n\n"
        "## \u7efc\u5408\u5206\u6790\n\n"
        "\u7efc\u5408\u6765\u770b\uff0c\u7814\u7a76\u8bc1\u636e\u8868\u660e\u57ce\u5e02\u70ed\u5c9b\u4e0d\u662f\u5b64\u7acb\u7684\u73af\u5883\u73b0\u8c61\uff0c\u800c\u662f\u4e0e\u516c\u5171\u5065\u5eb7\u98ce\u9669\u5171\u540c\u4f5c\u7528\u7684\u66b4\u9732\u673a\u5236\u3002 [1]\n\n"
        "## \u5c40\u9650\u4e0e\u4fe1\u5fc3\n\n"
        "\u5c40\u9650\u5728\u4e8e\u5355\u4e00\u8bc1\u636e\u6e90\u53ea\u80fd\u652f\u6301\u57fa\u672c\u5173\u7cfb\uff0c\u56e0\u6b64\u5bf9\u4e0d\u540c\u57ce\u5e02\u3001\u4eba\u7fa4\u548c\u5929\u6c14\u6761\u4ef6\u7684\u56e0\u679c\u63a8\u65ad\u5e94\u4fdd\u6301\u8c28\u614e\u3002 [1]\n\n"
        "## Sources\n\n"
        "[1] \u57ce\u5e02\u70ed\u5c9b\u8bc1\u636e: https://example.com/urban-heat-zh\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: report},
    )

    assert result.language_alignment_score >= 0.80
    assert result.report_structure_score >= 0.60
    assert not any("language alignment" in failure.lower() for failure in result.failures)


def test_verifier_rejects_undercovered_acceptance_criteria() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            "Explain mediating variables such as heuristic processing and source credibility assessment.",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        supporting_excerpt="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty and stop evaluating alternatives. [1]\n\n"
        "## Evidence Strength and Limits\n\n"
        "The evidence is limited because it supports the broad relationship, not every pathway or boundary condition. [1]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt},
    )

    assert result.valid is False
    assert result.criteria_coverage_score < 0.65
    assert result.undercovered_criteria
    assert any("acceptance criteria coverage" in failure.lower() for failure in result.failures)


def test_verifier_does_not_treat_readability_rubrics_as_content_criteria() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            "Task-specific comprehensiveness criterion: Explain mechanisms linking need for closure and misinformation acceptance.",
            "Task-specific readability criterion: Effective Use of Formatting for Readability - Assesses headings, paragraphing, sentence variety, and navigation.",
            "Clarity, Precision, and Appropriate Use of Psychological Terminology - Assesses whether psychological terms (e.g., 'need for closure,' 'epistemic motivation,' 'heuristics,' 'misinformation susceptibility') are clearly defined, used accurately and consistently, and explained appropriately for an academic audience.",
            "Paragraph Cohesion, Clarity, and Transitions - Assesses if each paragraph focuses on a single, clear idea related to the topic, with well-structured sentences.",
            "Overall Textual Fluency and Engagement - Considers the overall quality of the writing style, including sentence variety and an engaging academic tone.",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure affects misinformation acceptance by encouraging quick certainty and reduced scrutiny.",
        supporting_excerpt="Need for closure affects misinformation acceptance by encouraging quick certainty and reduced scrutiny.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure affects misinformation acceptance by encouraging quick certainty, heuristic processing, and reduced scrutiny of false claims. [1]\n\n"
        "## Evidence Strength and Limits\n\n"
        "The evidence supports a mechanism-focused interpretation, while remaining cautious about causal strength and boundary conditions. [1]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: report},
    )

    assert result.criteria_coverage_score >= 0.65
    assert not any("readability" in row["criterion"].lower() for row in result.undercovered_criteria)
    assert not any("acceptance criteria coverage" in failure.lower() for failure in result.failures)


def test_verifier_rejects_shallow_report_even_with_supported_citation() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Cooling centers and heat illness risk",
        objective="Explain how cooling centers reduce heat illness risk during heat waves.",
        queries=["cooling centers heat illness risk"],
        min_sources=1,
        required_terms=["cooling centers", "heat illness", "heat waves"],
    )
    plan = ResearchPlan(
        question="How do cooling centers reduce heat illness risk during heat waves?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Cooling Center Source",
        url="https://example.com/cooling",
        canonical_url="https://example.com/cooling",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves.",
        supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Cooling Centers\n\n"
        "Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves. [1]\n\n"
        "## Sources\n\n"
        "[1] Cooling Center Source: https://example.com/cooling\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt},
    )

    assert result.valid is False
    assert result.report_depth_score < 0.45
    assert any("report depth" in failure.lower() for failure in result.failures)


def test_verifier_accepts_covered_acceptance_criteria_gate() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance", "heuristic processing", "source credibility"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            "Explain mediating variables such as heuristic processing and source credibility assessment.",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure can increase misinformation acceptance through heuristic processing and source credibility assessment.",
        supporting_excerpt="Need for closure can increase misinformation acceptance through heuristic processing and source credibility assessment.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty, rely on heuristic processing, and use source credibility assessment as a shortcut. [1]\n\n"
        "## Evidence Strength and Limits\n\n"
        "The strongest supported mediating variables in this narrow evidence set are heuristic processing and source credibility assessment. [1]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt},
    )

    assert result.criteria_coverage_score >= 0.65
    assert not result.undercovered_criteria
    assert not any("acceptance criteria coverage" in failure.lower() for failure in result.failures)


def test_verifier_requires_broad_citation_use_when_evidence_deck_is_broad() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Cooling centers and heat illness risk",
        objective="Explain how cooling centers reduce heat illness risk during heat waves.",
        queries=["cooling centers heat illness risk"],
        min_sources=17,
        required_terms=["cooling centers", "heat illness risk", "heat waves", "cooler indoor spaces"],
    )
    plan = ResearchPlan(
        question="How do cooling centers reduce heat illness risk during heat waves?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    sources: list[SourceRecordV2] = []
    cards: list[EvidenceCard] = []
    for source_id in range(1, 18):
        source = SourceRecordV2(
            id=source_id,
            branch_id=branch.id,
            title=f"Cooling Center Evidence {source_id}",
            url=f"https://example.com/cooling/{source_id}",
            canonical_url=f"https://example.com/cooling/{source_id}",
            provenance="web",
            content_path=f"source_docs/source_{source_id}.md",
            content_hash=f"hash-{source_id}",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        )
        sources.append(source)
        cards.append(
            EvidenceCard(
                id=source_id,
                source_id=source_id,
                branch_id=branch.id,
                claim="Cooling centers reduce heat illness risk during heat waves by providing cooler indoor spaces.",
                supporting_excerpt="Cooling centers reduce heat illness risk during heat waves by providing cooler indoor spaces.",
                source_url=source.url,
                source_title=source.title,
                quality_score=0.9,
                relevance_score=0.9,
                confidence=0.9,
            )
        )
    body = (
        "Cooling centers reduce heat illness risk during heat waves by providing cooler indoor spaces. "
        "The evidence indicates that access to cooler indoor spaces matters for heat exposure and public health confidence."
    )
    report = (
        "# Cooling Centers and Heat Illness Risk\n\n"
        f"## Direct Answer\n\n{body} [1]\n\n"
        f"## Evidence Pattern\n\n{body} [1]\n\n"
        f"## Limits and Confidence\n\n{body} [1]\n\n"
        "## Sources\n\n"
        "[1] Cooling Center Evidence 1: https://example.com/cooling/1\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=sources,
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: cards[0].supporting_excerpt},
    )

    assert result.valid is False
    assert result.source_breadth_score < 1.0
    assert any("cited evidence-backed source count" in failure.lower() for failure in result.failures)


def test_verifier_rejects_incomplete_branch_coverage_even_when_average_score_is_high() -> None:
    branches = [
        ResearchBranch(
            id=f"branch_{index}",
            title=f"Coverage branch {index}",
            objective=f"Explain coverage branch {index}.",
            queries=[f"coverage branch {index}"],
            min_sources=1,
            required_terms=[f"coverage branch {index}"],
        )
        for index in range(1, 6)
    ]
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=f"branch_{index}",
            title=f"Coverage Source {index}",
            url=f"https://example.com/coverage/{index}",
            canonical_url=f"https://example.com/coverage/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash="hash",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        )
        for index in range(1, 5)
    ]
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=f"branch_{index}",
            claim=f"Coverage branch {index} is supported by evidence.",
            supporting_excerpt=f"Coverage branch {index} is supported by evidence.",
            source_url=f"https://example.com/coverage/{index}",
            source_title=f"Coverage Source {index}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 5)
    ]
    coverage = build_coverage_matrix(branches=branches, evidence_cards=cards, sources=sources)
    plan = ResearchPlan(
        question="How should all coverage branches be handled?",
        intent="general",
        audience="general",
        report_outline=[branch.title for branch in branches],
        branches=branches,
    )
    report = (
        "# Coverage Branches\n\n"
        "Coverage branch 1, coverage branch 2, coverage branch 3, and coverage branch 4 are supported by evidence. [1, 2, 3, 4]\n\n"
        "## Evidence Pattern\n\n"
        "The evidence indicates that the covered branches have source support, but one planned branch is not represented. [1, 2, 3, 4]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because one planned branch lacks evidence coverage. [1, 2, 3, 4]\n\n"
        "## Sources\n\n"
        + "\n".join(
            f"[{index}] Coverage Source {index}: https://example.com/coverage/{index}"
            for index in range(1, 5)
        )
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=sources,
        evidence_cards=cards,
        coverage=coverage,
        source_texts={card.source_id: card.supporting_excerpt for card in cards},
    )

    assert coverage.coverage_score >= 0.80
    assert coverage.complete is False
    assert result.valid is False
    assert any("coverage incomplete" in failure.lower() for failure in result.failures)


def test_synthesis_appends_evidence_coverage_for_missing_branches_and_sources() -> None:
    branches = [
        ResearchBranch(
            id="branch_1",
            title="Cooling center access",
            objective="Explain access to cooling centers during heat waves.",
            queries=["cooling center access heat waves"],
            required_terms=["cooling centers", "access"],
        ),
        ResearchBranch(
            id="branch_2",
            title="Transportation barriers",
            objective="Explain transportation barriers that limit cooling center use.",
            queries=["cooling center transportation barriers"],
            required_terms=["transportation barriers", "cooling center use"],
        ),
    ]
    plan = ResearchPlan(
        question="How do cooling centers reduce heat illness risk, and what limits their use?",
        intent="general",
        audience="general",
        report_outline=[branch.title for branch in branches],
        branches=branches,
    )
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id="branch_1",
            claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
            supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
            source_url="https://example.com/1",
            source_title="Source 1",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id="branch_2",
            claim="Transportation barriers can limit whether residents can use cooling centers during heat waves.",
            supporting_excerpt="Transportation barriers can limit whether residents can use cooling centers during heat waves.",
            source_url="https://example.com/2",
            source_title="Source 2",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Cooling Centers\n\n"
        "Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces. [1]\n\n"
        "## Sources\n\n"
        "[1] Source 1: https://example.com/1\n"
    )

    repaired = _append_evidence_coverage_if_needed(report, plan, cards)

    assert "Transportation barriers" in repaired
    assert "[2]" in repaired.split("## Sources")[0]


def test_synthesis_does_not_append_fragmented_coverage_for_criteria_rich_reports() -> None:
    branches = [
        ResearchBranch(
            id="branch_1",
            title="Cooling centers",
            objective="Explain how cooling centers reduce heat illness risk.",
            queries=["cooling centers heat illness risk"],
            required_terms=["cooling centers", "heat illness risk"],
        ),
        ResearchBranch(
            id="branch_2",
            title="Access barriers",
            objective="Explain barriers that limit access.",
            queries=["cooling center access barriers"],
            required_terms=["access barriers", "heat illness risk"],
        ),
    ]
    plan = ResearchPlan(
        question="How do cooling centers reduce heat illness risk, and what limits their use?",
        intent="general",
        audience="general",
        report_outline=[branch.title for branch in branches],
        branches=branches,
        acceptance_criteria=[
            f"Cover this task-specific insight criterion in synthesis: Criterion {index} requires evidence, mechanisms, limits, and implications."
            for index in range(1, 10)
        ],
    )
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id="branch_1",
            claim="Cooling centers reduce heat illness risk by giving residents cooler indoor spaces.",
            supporting_excerpt="Cooling centers reduce heat illness risk by giving residents cooler indoor spaces.",
            source_url="https://example.com/1",
            source_title="Source 1",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id="branch_2",
            claim="Access barriers limit whether residents can use cooling centers during heat waves.",
            supporting_excerpt="Access barriers limit whether residents can use cooling centers during heat waves.",
            source_url="https://example.com/2",
            source_title="Source 2",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Cooling Centers\n\n"
        "Cooling centers reduce heat illness risk by giving residents cooler indoor spaces. [1]\n\n"
        "## Sources\n\n"
        "[1] Source 1: https://example.com/1\n"
    )

    repaired = _append_evidence_coverage_if_needed(report, plan, cards)

    assert repaired == report
    assert "Additional Source-Backed Analysis" not in repaired
    assert "[2]" not in repaired.split("## Sources")[0]


def test_evidence_hygiene_rejects_boilerplate_and_metadata_cards() -> None:
    clean = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="Clean evidence states one supported claim in natural prose without copied page chrome.",
        supporting_excerpt="Clean evidence states one supported claim in natural prose without copied page chrome.",
        source_url="https://example.com/clean",
        source_title="Clean",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    noisy = EvidenceCard(
        id=2,
        source_id=2,
        branch_id="branch_1",
        claim="URL: https://example.com Canonical URL: https://example.com Branch: branch_1 Extraction method: raw",
        supporting_excerpt="URL: https://example.com Canonical URL: https://example.com Branch: branch_1 Extraction method: raw",
        source_url="https://example.com/noisy",
        source_title="Noisy",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    result = apply_evidence_hygiene([clean, noisy])

    assert [card.id for card in result.kept] == [1]
    assert result.rejected[0]["card"]["id"] == 2
    assert any("artifact" in reason or "metadata" in reason for reason in result.rejected[0]["reasons"])


def test_verifier_rejects_report_boilerplate_even_with_valid_citation() -> None:
    plan = build_research_plan("What are model adaptation methods?")
    branch_id = plan.branches[0].id
    source = SourceRecordV2(
        id=1,
        branch_id=branch_id,
        title="Model Adaptation Source",
        url="https://example.com/source",
        canonical_url="https://example.com/source",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="official_docs",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch_id,
        claim="Model adaptation changes pretrained systems so they perform better on specific tasks and formats.",
        supporting_excerpt="Model adaptation changes pretrained systems so they perform better on specific tasks and formats.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    coverage = CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[])
    report = (
        "# Research Report\n\n"
        "## Main Finding\n\n"
        "Model adaptation changes pretrained systems so they perform better on specific tasks and formats. [1]\n\n"
        "Share [](https://www.linkedin.com/shareArticle?url=https://example.com). [1]\n\n"
        "## Evidence Handling\n\n"
        "Scope note.\n\n"
        "## Adaptation Effects\n\n"
        "Model adaptation changes pretrained systems so they perform better on specific tasks and formats. [1]\n\n"
        "## Confidence Notes\n\n"
        "Confidence note.\n\n"
        "## Sources\n\n"
        "[1] Model Adaptation Source: https://example.com/source\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=coverage,
        source_texts={1: card.supporting_excerpt},
    )

    assert result.valid is False
    assert result.report_cleanliness_score < 1.0
    assert any("artifact" in failure or "URL-heavy" in failure for failure in result.failures)


def test_verifier_accepts_flexible_non_template_report_headings() -> None:
    plan = build_research_plan("How do urban heat islands affect public health?")
    all_plan_text = " ".join(
        branch.title + " " + branch.objective + " " + " ".join(branch.required_terms)
        for branch in plan.branches
    )
    source = SourceRecordV2(
        id=1,
        branch_id=plan.branches[0].id,
        title="Urban Heat Source",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=300,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=plan.branches[0].id,
        claim=(
            "Urban heat islands affect public health through heat exposure, vulnerable populations, "
            "health outcomes, mitigation choices, evidence tradeoffs, applications, risks, limitations, costs, safeguards, "
            "and decision guidance."
        ),
        supporting_excerpt=all_plan_text,
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    coverage = CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[])
    body = (
        "Urban heat islands affect public health through heat exposure, vulnerable populations, health outcomes, "
        "mitigation choices, evidence tradeoffs, applications, risks, limitations, costs, safeguards, and decision guidance. "
        f"{all_plan_text} [1]"
    )
    report = (
        "# Urban Heat and Public Health\n\n"
        "## The Core Relationship\n\n"
        f"{body}\n\n"
        "## How the Evidence Connects\n\n"
        f"Taken together, the evidence indicates that public health risk, methods, implementation, performance, "
        f"applications, examples, risks, limitations, costs, safeguards, best practices, decision criteria, and alternatives "
        f"must be read as connected factors rather than isolated findings. {all_plan_text} [1]\n\n"
        "## What Remains Uncertain\n\n"
        f"Confidence is limited by source scope and by evidence gaps around local context, but the available evidence "
        f"supports the main public-health pattern. {all_plan_text} [1]\n\n"
        "## Sources\n\n"
        "[1] Urban Heat Source: https://example.com/urban-heat\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=coverage,
        source_texts={1: body + " " + all_plan_text},
    )

    assert result.valid is True
    assert result.report_structure_score >= 0.60


def test_semantic_evidence_enrichment_filters_unentailed_cards(tmp_path: Path) -> None:
    plan = build_research_plan("How do cooling centers reduce heat illness risk during heat waves?")
    branch = plan.branches[0]
    strong = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
        supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves.",
        source_url="https://example.com/cooling",
        source_title="Cooling Centers",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    weak = EvidenceCard(
        id=2,
        source_id=2,
        branch_id=branch.id,
        claim="Invoice systems reconcile approvals and payments.",
        supporting_excerpt="Invoice systems reconcile approvals and payments.",
        source_url="https://example.com/invoices",
        source_title="Invoices",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    judge = FakeSemanticJudge(
        {
            "cards": [
                {
                    "id": 1,
                    "keep": True,
                    "branch_alignment_score": 0.94,
                    "entailment_score": 0.96,
                    "evidence_relevance_score": 0.95,
                    "normalized_claim": "Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
                    "key_points": ["cooler indoor spaces"],
                    "limitations": [],
                    "failure_reasons": [],
                },
                {
                    "id": 2,
                    "keep": False,
                    "branch_alignment_score": 0.05,
                    "entailment_score": 0.80,
                    "evidence_relevance_score": 0.02,
                    "normalized_claim": "Invoice systems reconcile approvals and payments.",
                    "key_points": [],
                    "limitations": [],
                    "failure_reasons": ["irrelevant to the research question"],
                },
            ]
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    result = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[strong, weak],
        settings=settings,
        model=judge,
    )

    assert [card.id for card in result.cards] == [1]
    assert result.cards[0].semantic_score == 0.95
    assert result.rejected[0]["card"]["id"] == 2
    assert result.failures == []
    assert "Return exactly one JSON object" in judge.prompts[0]


def test_semantic_evidence_enrichment_falls_back_on_invalid_json(tmp_path: Path) -> None:
    plan = build_research_plan("How does need for closure affect misinformation acceptance?")
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=plan.branches[0].id,
        claim="Need for closure can increase reliance on quick judgments when evaluating misinformation.",
        supporting_excerpt="Need for closure can increase reliance on quick judgments when evaluating misinformation.",
        source_url="https://example.com/nfc",
        source_title="Need for Closure Evidence",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    judge = InvalidSemanticJudge()
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    result = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=judge,
    )

    assert [kept.id for kept in result.cards] == [1]
    assert result.failures == []
    assert result.metrics["semantic_evidence_parse_failure_count"] == 1
    assert result.judgments[0]["fallback"] == "deterministic_after_invalid_judge_output"


def test_semantic_evidence_enrichment_uses_large_deck_fallback(tmp_path: Path) -> None:
    plan = build_research_plan("How does need for closure affect misinformation acceptance?")
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=plan.branches[0].id,
            claim=f"Need for closure evidence card {index} explains misinformation acceptance through quick judgments.",
            supporting_excerpt=f"Need for closure evidence card {index} explains misinformation acceptance through quick judgments.",
            source_url=f"https://example.com/{index}",
            source_title=f"Source {index}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 4)
    ]
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        semantic_verification=True,
        semantic_evidence_max_llm_cards=2,
    )

    result = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=cards,
        settings=settings,
        model=RaisingSemanticJudge(),
    )

    assert [card.id for card in result.cards] == [1, 2, 3]
    assert result.failures == []
    assert result.metrics["semantic_evidence_large_deck_fallback"] is True
    assert result.metrics["semantic_evidence_batches"] == 0


def test_semantic_report_verification_can_fail_valid_deterministic_result(tmp_path: Path) -> None:
    plan = build_research_plan("How do cooling centers reduce heat illness risk during heat waves?")
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=plan.branches[0].id,
        claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
        supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves.",
        source_url="https://example.com/cooling",
        source_title="Cooling Centers",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    deterministic = VerificationResultV2(
        valid=True,
        citation_validity_score=1.0,
        source_support_score=1.0,
        answer_coverage_score=1.0,
        branch_coverage_score=1.0,
        evidence_linkage_score=1.0,
        source_quality_score=1.0,
        report_structure_score=1.0,
    )
    judge = FakeSemanticJudge(
        {
            "answer_completeness_score": 0.4,
            "citation_entailment_score": 0.5,
            "evidence_use_score": 0.6,
            "contradiction_safety_score": 1.0,
            "overall_score": 0.5,
            "failures": ["missing context about who can access cooling centers"],
            "missing_context": ["access barriers"],
            "unsupported_claims": ["Cooling centers eliminate heat-wave risk."],
            "contradictions": [],
            "search_focus": ["cooling center access barriers"],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    semantic = verify_report_with_semantics(
        report_markdown="## Executive Summary\n\nCooling centers eliminate heat-wave risk. [1]",
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=judge,
    )
    merged = apply_semantic_report_result(deterministic, semantic)

    assert merged.valid is False
    assert merged.semantic_verification_score == 0.5
    assert any("unsupported claim" in failure.lower() for failure in merged.failures)
    assert merged.semantic_verification["search_focus"] == ["cooling center access barriers"]


def test_semantic_json_loader_accepts_python_literal_objects() -> None:
    payload = _load_json_object(
        "{'overall_score': 0.82, 'failures': [], 'unsupported_claims': [], "
        "'contradictions': [], 'missing_context': [], 'search_focus': []}"
    )

    assert payload["overall_score"] == 0.82
    assert payload["failures"] == []


def test_semantic_report_verification_accepts_json_like_failure_entries(tmp_path: Path) -> None:
    plan = build_research_plan("How do cooling centers reduce heat illness risk during heat waves?")
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=plan.branches[0].id,
        claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
        supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
        source_url="https://example.com/cooling",
        source_title="Cooling Centers",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    judge = FakeSemanticJudge(
        {
            "answer_completeness_score": 0.8,
            "citation_entailment_score": 0.8,
            "evidence_use_score": 0.8,
            "contradiction_safety_score": 1.0,
            "overall_score": 0.8,
            "failures": [{"issue": "minor scope note"}],
            "missing_context": [],
            "unsupported_claims": [],
            "contradictions": [],
            "search_focus": [],
        }
    )

    semantic = verify_report_with_semantics(
        report_markdown="Cooling centers reduce heat illness risk. [1]",
        plan=plan,
        evidence_cards=[card],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True),
        model=judge,
    )

    assert semantic.score == 0.8
    assert '{"issue": "minor scope note"}' in semantic.failures


def test_normalize_report_sources_section_includes_multi_citation_ids() -> None:
    sources = [
        SourceRecordV2(
            id=index,
            branch_id="branch_1",
            title=f"Source {index}",
            url=f"https://example.com/{index}",
            canonical_url=f"https://example.com/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash="hash",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="high",
            quality_type="academic",
            relevance_score=0.9,
        )
        for index in (1, 10)
    ]
    report = "# Report\n\nThe evidence combines two source records. [1, 10]\n"

    normalized = _normalize_report_markdown(report, sources)

    assert "[1] Source 1: https://example.com/1" in normalized
    assert "[10] Source 10: https://example.com/10" in normalized


def test_normalize_report_removes_malformed_source_dump_before_canonical_sources() -> None:
    sources = [
        SourceRecordV2(
            id=1,
            branch_id="branch_1",
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="high",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id="branch_1",
            title="Misinformation Source",
            url="https://example.com/misinfo",
            canonical_url="https://example.com/misinfo",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="high",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]
    raw = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can shape misinformation acceptance through quick certainty and reduced scrutiny. [1]\n\n"
        "## Sources  [1] Old Source: https://old.example/source\n"
        "[2] Misinformation Source: https://example.com/misinfo\n"
        "Unnumbered Source: https://example.com/unlisted\n"
        "More copied metadata: https://example.com/metadata\n\n"
        "---\n\n"
        "*End of Report*\n"
    )

    normalized = _normalize_report_markdown(raw, sources)
    body = normalized.split("## Sources")[0]

    assert "https://old.example/source" not in normalized
    assert "Unnumbered Source" not in normalized
    assert "*End of Report*" not in body
    assert normalized.count("## Sources") == 1
    assert "[1] Need for Closure Source: https://example.com/nfc" in normalized
    assert "[2] Misinformation Source: https://example.com/misinfo" not in normalized


def test_report_blueprint_and_writer_use_professional_report_sections() -> None:
    plan = build_research_plan("What are urban heat islands and how do they affect public health?")
    branch = plan.branches[0]
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Urban Heat Evidence",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="official_docs",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Urban heat islands raise local temperatures and increase heat-related public health risks.",
        supporting_excerpt="Urban heat islands raise local temperatures and increase heat-related public health risks.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    coverage = CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[])

    blueprint = build_report_blueprint(plan=plan, evidence_cards=[card], coverage=coverage, sources=[source])
    report = synthesize_report(plan=plan, evidence_cards=[card], coverage=coverage, sources=[source])

    examples = [example["name"] for example in blueprint["style_examples"]]
    assert "Analytical explainer" in examples
    assert "Evidence review" in examples
    assert "quality_contract" in blueprint
    assert "depth_and_insight" in blueprint["quality_contract"]
    assert blueprint["source_summary"]["evidence_card_count"] == 1
    assert "section_contract" not in blueprint
    assert "## What the Sources Show Together" in report
    assert "## Implications" in report
    assert "## Sources" in report
    assert "[1] Urban Heat Evidence: https://example.com/urban-heat" in report


def test_model_synthesis_falls_back_when_model_returns_degenerate_report(tmp_path: Path, monkeypatch) -> None:
    class NullReportModel:
        def invoke(self, _messages):
            return SimpleNamespace(content="None")

    monkeypatch.setattr("deep_research.synthesis.model_for_role", lambda *_args, **_kwargs: NullReportModel())
    monkeypatch.setattr("deep_research.synthesis.BaseChatModel", object)
    branch = ResearchBranch(
        id="branch_1",
        title="Urban heat health effects",
        objective="Explain how urban heat affects public health.",
        queries=["urban heat health effects"],
        required_terms=["urban heat", "public health"],
    )
    plan = ResearchPlan(
        question="How does urban heat affect public health?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Urban Heat Evidence",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="official_docs",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Urban heat increases public health risks by raising local temperatures and heat exposure.",
        supporting_excerpt="Urban heat increases public health risks by raising local temperatures and heat exposure.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    report = synthesize_report_with_model(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path),
    )

    assert "None" not in report.split("## Sources")[0]
    assert "Urban heat increases public health risks" in report
    assert "[1]" in report.split("## Sources")[0]


def test_model_synthesis_preserves_previous_report_when_repair_response_is_empty(
    tmp_path: Path, monkeypatch
) -> None:
    class EmptyReportModel:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(content="")

    monkeypatch.setattr("deep_research.synthesis.model_for_role", lambda *_args, **_kwargs: EmptyReportModel())
    monkeypatch.setattr("deep_research.synthesis.BaseChatModel", object)
    branch = ResearchBranch(
        id="branch_1",
        title="Urban heat health effects",
        objective="Explain how urban heat affects public health.",
        queries=["urban heat health effects"],
        required_terms=["urban heat", "public health"],
    )
    plan = ResearchPlan(
        question="How does urban heat affect public health?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Urban Heat Evidence",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="official_docs",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Urban heat increases public health risks by raising local temperatures and heat exposure.",
        supporting_excerpt="Urban heat increases public health risks by raising local temperatures and heat exposure.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    previous = (
        "# Urban Heat\n\n"
        "Urban heat increases public health risks by raising local temperatures and heat exposure. [1]\n\n"
        "## Sources\n\n"
        "[1] Urban Heat Evidence: https://example.com/urban-heat\n"
    )

    report = synthesize_report_with_model(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path),
        previous_report=previous,
        verification_failures=["repair this draft"],
    )

    assert "Urban heat increases public health risks" in report
    assert "[1] Urban Heat Evidence: https://example.com/urban-heat" in report


def test_model_synthesis_excludes_sources_failed_by_alignment_verification(tmp_path: Path, monkeypatch) -> None:
    captured_prompts: list[str] = []

    class CapturingReportModel:
        def invoke(self, messages):
            captured_prompts.append(messages[0].content)
            return SimpleNamespace(
                content=(
                    "# Need for Closure and Misinformation Acceptance\n\n"
                    "Need for closure can shape misinformation acceptance when people seek quick certainty. [1]\n\n"
                    "## Sources\n\n"
                    "[1] Direct Source: https://example.com/direct\n"
                )
            )

    monkeypatch.setattr("deep_research.synthesis.model_for_role", lambda *_args, **_kwargs: CapturingReportModel())
    monkeypatch.setattr("deep_research.synthesis.BaseChatModel", object)
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    direct_source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Direct Source",
        url="https://example.com/direct",
        canonical_url="https://example.com/direct",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash-1",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="academic",
        relevance_score=0.9,
    )
    adjacent_source = SourceRecordV2(
        id=2,
        branch_id=branch.id,
        title="Adjacent Topic Source",
        url="https://example.com/adjacent",
        canonical_url="https://example.com/adjacent",
        provenance="web",
        content_path="source_docs/source_2.md",
        content_hash="hash-2",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="academic",
        relevance_score=0.4,
    )
    direct_card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure can shape misinformation acceptance when people seek quick certainty.",
        supporting_excerpt="Need for closure can shape misinformation acceptance when people seek quick certainty.",
        source_url=direct_source.url,
        source_title=direct_source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    adjacent_card = EvidenceCard(
        id=2,
        source_id=2,
        branch_id=branch.id,
        claim="Need for closure can shape a neighboring attitude outcome.",
        supporting_excerpt="Need for closure can shape a neighboring attitude outcome.",
        source_url=adjacent_source.url,
        source_title=adjacent_source.title,
        quality_score=0.9,
        relevance_score=0.4,
        confidence=0.9,
    )

    report = synthesize_report_with_model(
        plan=plan,
        evidence_cards=[direct_card, adjacent_card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[direct_source, adjacent_source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path),
        verification_failures=[
            "Cited source [2] fails current branch/request alignment: source main topic appears to be a neighboring concept rather than the requested concept",
        ],
    )

    assert captured_prompts
    assert "Adjacent Topic Source" not in captured_prompts[0]
    assert "https://example.com/adjacent" not in captured_prompts[0]
    assert "[2] Adjacent Topic Source" not in report


def test_report_level_criteria_ignores_traceability_quality_gate() -> None:
    criteria = _report_level_criteria(
        [
            "All data points must be traceable to at least one cited source",
            "Compare financing conditions across companies",
        ]
    )

    assert criteria == ["Compare financing conditions across companies"]


def test_synthesis_request_budget_caps_groq_completion_tokens(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:openai/gpt-oss-20b",
        groq_api_keys=("groq-a",),
        model_max_output_tokens=4000,
    )

    kwargs = _synthesis_request_kwargs(
        settings=settings,
        prompt="evidence " * 2500,
        model_spec=settings.model,
    )

    assert kwargs["max_tokens"] < 4000
    assert kwargs["max_tokens"] >= 768


def test_synthesis_request_budget_leaves_google_completion_tokens_unforced(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="google",
        model="google_genai:gemini-2.5-flash",
        google_api_keys=("google-a",),
        model_max_output_tokens=4000,
    )

    kwargs = _synthesis_request_kwargs(
        settings=settings,
        prompt="evidence " * 2500,
        model_spec=settings.model,
    )

    assert kwargs == {}


def test_criteria_rich_synthesis_profile_requires_reference_grade_depth() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain mechanisms, empirical evidence, mediators, moderators, and limitations.",
        queries=["need for closure misinformation acceptance empirical evidence"],
        min_sources=17,
        required_terms=["need for closure", "misinformation acceptance", "mechanisms"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="academic",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            f"Cover this task-specific insight criterion in synthesis: Criterion {index} explains the relationship in depth."
            for index in range(1, 18)
        ],
    )
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=branch.id,
            claim=f"Evidence item {index} links need for closure to misinformation acceptance through cognitive mechanisms.",
            supporting_excerpt=f"Evidence item {index} links need for closure to misinformation acceptance through cognitive mechanisms.",
            source_url=f"https://example.com/{index}",
            source_title=f"Source {index}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 18)
    ]

    profile = _target_report_profile(
        plan=plan,
        evidence_cards=cards,
        writing_guidance="DeepResearch Bench evaluation guidance",
    )

    assert profile["criteria_rich"] is True
    assert profile["minimum_words"] >= 6500
    assert profile["target_words"] >= 8000
    assert profile["target_words"] > profile["minimum_words"]
    assert profile["minimum_cited_paragraphs"] >= 28
    assert profile["minimum_major_sections_before_sources"] >= 16
    assert any(row["purpose"] == "Mechanisms and causal logic" for row in profile["section_plan"])


def test_criteria_rich_synthesis_respects_configured_provider(tmp_path: Path) -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain mechanisms and evidence.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="academic",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            f"Cover this task-specific insight criterion in synthesis: Criterion {index} requires depth."
            for index in range(1, 10)
        ],
    )
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:openai/gpt-oss-20b",
        google_api_keys=("google-a",),
        groq_api_keys=("groq-a",),
    )

    assert _synthesis_model_spec(settings, plan) == "groq:openai/gpt-oss-20b"

    hybrid_settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="hybrid",
        model="groq:openai/gpt-oss-20b",
        google_api_keys=("google-a",),
        groq_api_keys=("groq-a",),
    )

    assert _synthesis_model_spec(hybrid_settings, plan) == "google_genai:gemini-2.5-flash"


def test_normal_synthesis_keeps_configured_model(tmp_path: Path) -> None:
    plan = build_research_plan("What are urban heat islands?")
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:openai/gpt-oss-20b",
        google_api_keys=("google-a",),
        groq_api_keys=("groq-a",),
    )

    assert _synthesis_model_spec(settings, plan) == "groq:openai/gpt-oss-20b"


def test_criteria_rich_depth_score_rewards_rich_natural_sectioning() -> None:
    branches = [
        ResearchBranch(
            id=f"branch_{index}",
            title=f"Analytical branch {index}",
            objective=f"Explain analytical branch {index} with evidence, mechanisms, limits, and implications.",
            queries=[f"analytical branch {index} evidence"],
            min_sources=1,
            required_terms=[f"analytical branch {index}", "evidence", "mechanisms"],
        )
        for index in range(1, 6)
    ]
    plan = ResearchPlan(
        question="How should this benchmark-style relationship be explained?",
        intent="general",
        audience="academic",
        report_outline=[branch.title for branch in branches],
        branches=branches,
        acceptance_criteria=[
            f"Cover this task-specific insight criterion in synthesis: Criterion {index} requires depth, evidence, mechanisms, limitations, and implications."
            for index in range(1, 18)
        ],
    )
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=branches[(index - 1) % len(branches)].id,
            claim=f"Evidence item {index} supports analytical branch mechanisms and implications.",
            supporting_excerpt=f"Evidence item {index} supports analytical branch mechanisms and implications.",
            source_url=f"https://example.com/{index}",
            source_title=f"Source {index}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 18)
    ]
    paragraph = (
        "This cited paragraph explains analytical branch evidence, mechanisms, limitations, implications, "
        "uncertainty, comparison, synthesis, and future research with enough terminology to count as a "
        "substantive report paragraph for a benchmark-style task. [1]"
    )
    thin_heading_report = (
        "# Benchmark Report\n\n"
        "## Direct Answer\n\n"
        + "\n\n".join(paragraph for _ in range(90))
        + "\n\n## Evidence\n\n"
        + "\n\n".join(paragraph for _ in range(20))
        + "\n\n## Sources\n\n[1] Source 1: https://example.com/1\n"
    )

    score = _report_depth_score(thin_heading_report, plan, cards)

    assert score < 0.90


def test_synthesis_prompt_requires_user_request_language() -> None:
    question = "\u8bf7\u5206\u6790\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u5f71\u54cd\u516c\u5171\u5065\u5eb7"
    branch = ResearchBranch(
        id="branch_1",
        title="\u57ce\u5e02\u70ed\u5c9b\u4e0e\u516c\u5171\u5065\u5eb7",
        objective="\u8bf4\u660e\u57ce\u5e02\u70ed\u5c9b\u5bf9\u516c\u5171\u5065\u5eb7\u7684\u5f71\u54cd\u3002",
        queries=["\u57ce\u5e02\u70ed\u5c9b \u516c\u5171\u5065\u5eb7"],
        min_sources=1,
        required_terms=["\u57ce\u5e02\u70ed\u5c9b", "\u516c\u5171\u5065\u5eb7"],
    )
    plan = ResearchPlan(
        question=question,
        intent="general",
        audience="general",
        report_outline=[],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="\u57ce\u5e02\u70ed\u5c9b\u8bc1\u636e",
        url="https://example.com/urban-heat-zh",
        canonical_url="https://example.com/urban-heat-zh",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="\u57ce\u5e02\u70ed\u5c9b\u4f1a\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        supporting_excerpt="\u57ce\u5e02\u70ed\u5c9b\u4f1a\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    prompt = _synthesis_prompt(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
        previous_report="",
        verification_failures=[],
    )

    assert "Use Simplified Chinese prose" in prompt
    assert "Do not let a narrower context" in prompt
    assert "Report quality contract" in prompt
    assert "Open with the answer, not background" in prompt


def test_synthesis_filters_allowed_sources_to_evidence_backed_sources() -> None:
    source_with_card = SourceRecordV2(
        id=1,
        branch_id="branch_1",
        title="Evidence Source",
        url="https://example.com/evidence",
        canonical_url="https://example.com/evidence",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash-1",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    source_without_card = SourceRecordV2(
        id=2,
        branch_id="branch_1",
        title="No Evidence Source",
        url="https://example.com/no-evidence",
        canonical_url="https://example.com/no-evidence",
        provenance="web",
        content_path="source_docs/source_2.md",
        content_hash="hash-2",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="Evidence-backed sources are the only sources the report may cite.",
        supporting_excerpt="Evidence-backed sources are the only sources the report may cite.",
        source_url=source_with_card.url,
        source_title=source_with_card.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    result = _evidence_backed_sources([source_with_card, source_without_card], [card])

    assert [source.id for source in result] == [1]


def test_synthesis_repairs_weak_paragraph_citation_support() -> None:
    sources = [
        SourceRecordV2(
            id=1,
            branch_id="branch_1",
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash-1",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id="branch_1",
            title="Misinformation Source",
            url="https://example.com/misinformation",
            canonical_url="https://example.com/misinformation",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash-2",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id="branch_1",
            claim="Need for closure is a desire for quick certainty and firm answers.",
            supporting_excerpt="Need for closure is a desire for quick certainty and firm answers.",
            source_url=sources[0].url,
            source_title=sources[0].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id="branch_1",
            claim="Misinformation acceptance involves believing inaccurate information and can be shaped by heuristics.",
            supporting_excerpt="Misinformation acceptance involves believing inaccurate information and can be shaped by heuristics.",
            source_url=sources[1].url,
            source_title=sources[1].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Report\n\n"
        "Need for closure can shape misinformation acceptance because people seeking quick certainty may rely on "
        "heuristics and believe inaccurate information. [1]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
    )

    repaired = _repair_weak_citation_support(report, cards, sources, threshold=0.35)

    assert "[2]" in repaired.split("## Sources")[0]


def test_synthesis_repair_removes_individually_unsupported_citation() -> None:
    sources = [
        SourceRecordV2(
            id=1,
            branch_id="branch_1",
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash-1",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id="branch_1",
            title="Unrelated Source",
            url="https://example.com/unrelated",
            canonical_url="https://example.com/unrelated",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash-2",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id="branch_1",
            claim="Need for closure can shape misinformation acceptance through quick certainty and heuristic judgment.",
            supporting_excerpt="Need for closure can shape misinformation acceptance through quick certainty and heuristic judgment.",
            source_url=sources[0].url,
            source_title=sources[0].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id="branch_1",
            claim="Urban heat islands increase local temperature exposure.",
            supporting_excerpt="Urban heat islands increase local temperature exposure.",
            source_url=sources[1].url,
            source_title=sources[1].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Report\n\n"
        "Need for closure can shape misinformation acceptance through quick certainty and heuristic judgment. [1, 2]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
        "[2] Unrelated Source: https://example.com/unrelated\n"
    )

    repaired = _repair_weak_citation_support(report, cards, sources, threshold=0.35)

    assert "[1]" in repaired.split("## Sources")[0]
    assert "[2]" not in repaired.split("## Sources")[0]


def test_synthesis_card_selection_is_bounded_and_branch_balanced() -> None:
    plan = build_research_plan("Compare several approaches to reducing misinformation acceptance.")
    cards: list[EvidenceCard] = []
    card_id = 1
    for branch in plan.branches:
        for index in range(20):
            cards.append(
                EvidenceCard(
                    id=card_id,
                    source_id=card_id,
                    branch_id=branch.id,
                    claim=f"{branch.title} evidence item {index} explains misinformation acceptance.",
                    supporting_excerpt=f"{branch.title} evidence item {index} explains misinformation acceptance.",
                    source_url=f"https://example.com/{card_id}",
                    source_title=f"Source {card_id}",
                    quality_score=0.9,
                    relevance_score=0.9,
                    confidence=0.9,
                )
            )
            card_id += 1

    selected = _cards_for_synthesis(plan, cards)
    selected_branches = {card.branch_id for card in selected}

    assert len(selected) <= 96
    assert selected_branches == {branch.id for branch in plan.branches}


def test_synthesis_card_selection_prefers_question_phrase_overlap() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    broad_overlap = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for cognition can influence how people respond to misinformation.",
        supporting_excerpt="Need for cognition can influence how people respond to misinformation.",
        source_url="https://example.com/broad",
        source_title="Broad Overlap",
        quality_score=1.0,
        relevance_score=1.0,
        confidence=1.0,
    )
    phrase_overlap = EvidenceCard(
        id=2,
        source_id=2,
        branch_id=branch.id,
        claim="Need for closure is linked to misinformation acceptance when people seek certainty.",
        supporting_excerpt="Need for closure is linked to misinformation acceptance when people seek certainty.",
        source_url="https://example.com/phrase",
        source_title="Phrase Overlap",
        quality_score=0.8,
        relevance_score=0.8,
        confidence=0.8,
    )

    selected = _cards_for_synthesis(plan, [broad_overlap, phrase_overlap])

    assert selected[0].id == 2


def test_synthesis_card_selection_keeps_rare_required_term_under_budget() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mechanisms and boundary conditions",
        objective="Explain the evidence, mechanisms, and boundary conditions.",
        queries=["mechanisms evidence boundary conditions"],
        required_terms=["mechanisms", "evidence", "boundary conditions"],
    )
    plan = ResearchPlan(
        question="Explain the mechanisms and boundary conditions for misinformation acceptance.",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=["Cover mechanisms, evidence quality, and boundary conditions."],
    )
    cards = [
        EvidenceCard(
            id=index,
            source_id=1,
            branch_id=branch.id,
            claim="Mechanisms and evidence explain misinformation acceptance through repeated exposure.",
            supporting_excerpt="Mechanisms and evidence explain misinformation acceptance through repeated exposure.",
            source_url="https://example.com/repeated",
            source_title="Repeated Exposure",
            quality_score=1.0,
            relevance_score=1.0,
            confidence=1.0,
        )
        for index in range(1, 130)
    ]
    rare_card = EvidenceCard(
        id=500,
        source_id=1,
        branch_id=branch.id,
        claim="Boundary conditions include uncertainty, prior beliefs, and the information environment.",
        supporting_excerpt="Boundary conditions include uncertainty, prior beliefs, and the information environment.",
        source_url="https://example.com/boundary",
        source_title="Boundary Conditions",
        quality_score=0.45,
        relevance_score=0.45,
        confidence=0.45,
    )

    selected = _cards_for_synthesis(plan, cards + [rare_card])

    assert len(selected) <= 96
    assert rare_card.id in {card.id for card in selected}


def test_synthesis_card_selection_preserves_source_diversity() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Evidence diversity",
        objective="Synthesize findings across independent sources.",
        queries=["evidence diversity synthesis"],
        required_terms=["independent sources", "evidence diversity"],
    )
    plan = ResearchPlan(
        question="Synthesize the evidence across independent sources.",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    cards = [
        EvidenceCard(
            id=index,
            source_id=1,
            branch_id=branch.id,
            claim="One source repeatedly reports the same evidence pattern.",
            supporting_excerpt="One source repeatedly reports the same evidence pattern.",
            source_url="https://example.com/source-1",
            source_title="Source 1",
            quality_score=1.0,
            relevance_score=1.0,
            confidence=1.0,
        )
        for index in range(1, 80)
    ]
    cards.extend(
        EvidenceCard(
            id=100 + source_id,
            source_id=source_id,
            branch_id=branch.id,
            claim=f"Independent source {source_id} contributes evidence diversity and a distinct context.",
            supporting_excerpt=f"Independent source {source_id} contributes evidence diversity and a distinct context.",
            source_url=f"https://example.com/source-{source_id}",
            source_title=f"Source {source_id}",
            quality_score=0.7,
            relevance_score=0.7,
            confidence=0.7,
        )
        for source_id in range(2, 8)
    )

    selected = _cards_for_synthesis(plan, cards)

    assert len({card.source_id for card in selected[:12]}) >= 6


def test_report_normalizer_repairs_heading_sources_and_uncited_paragraphs() -> None:
    source = SourceRecordV2(
        id=1,
        branch_id="branch_1",
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="academic",
        relevance_score=0.9,
    )
    raw = (
        "**Key Answer**\n\n"
        "This long factual paragraph discusses need for closure and misinformation acceptance in enough detail "
        "that it should not remain uncited after normalization.\n\n"
        "**References**\n\n"
        "[1] Old Source: https://example.com/old\n"
        "[21] Bad Source: https://example.com/bad\n"
    )

    normalized = _normalize_report_markdown(raw, [source])

    assert "## Key Answer" in normalized
    assert "## Sources" in normalized
    assert "[21]" not in normalized
    assert "should not remain uncited after normalization. [1]" in normalized
    assert "[1] Need for Closure Source: https://example.com/nfc" in normalized


def test_semantic_repair_focus_targets_missing_branches_only() -> None:
    state = {
        "plan": {
            "branches": [
                {"id": "branch_1", "required_terms": ["covered"]},
                {"id": "branch_2", "required_terms": ["missing"]},
            ]
        },
        "coverage_matrix": {
            "missing_branches": ["branch_2"],
            "branches": [
                {"branch_id": "branch_1", "complete": True, "missing_points": [], "required_points": []},
                {"branch_id": "branch_2", "complete": False, "missing_points": [], "required_points": ["usable sources >= 3"]},
            ],
        },
        "verification": {
            "semantic_verification": {
                "missing_context": ["direct evidence for the missing branch"],
                "search_focus": ["targeted follow-up"],
            }
        },
    }

    focus = _focus_terms_from_state(state)

    assert "branch_1" not in focus
    assert "branch_2" in focus
    assert "targeted follow-up" in focus["branch_2"]


def test_repair_focus_strips_internal_coverage_labels() -> None:
    state = {
        "plan": {
            "branches": [
                {
                    "id": "branch_2",
                    "required_terms": ["domain-specific misinformation", "information environment"],
                }
            ]
        },
        "coverage_matrix": {
            "missing_branches": ["branch_2"],
            "branches": [
                {
                    "branch_id": "branch_2",
                    "complete": False,
                    "missing_points": [
                        "required term coverage >= 55% (actual 33%)",
                        "required term: domain-specific misinformation",
                        "required term: information environment",
                        "branch evidence cards",
                    ],
                    "required_points": [],
                }
            ],
        },
        "verification": {},
    }

    focus = _focus_terms_from_state(state)

    assert focus["branch_2"] == ["domain-specific misinformation", "information environment"]
    assert all("required term" not in term.lower() for term in focus["branch_2"])
    assert all(">=" not in term for term in focus["branch_2"])


def test_acquire_route_reuses_existing_evidence_when_no_new_sources_or_candidates() -> None:
    no_progress_state = {
        "metrics": {
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "last_acquire_searches": 9,
        },
        "evidence_cards": [{"id": 1, "source_id": 1}],
    }
    progress_state = {
        "metrics": {
            "last_acquire_added_sources": 1,
            "last_acquire_added_candidates": 2,
            "last_acquire_searches": 0,
        },
        "evidence_cards": [{"id": 1, "source_id": 1}],
    }

    assert _acquire_route(no_progress_state) == "reuse_evidence"
    assert _acquire_route(progress_state) == "read_sources"


def test_coverage_route_finishes_when_no_evidence_and_acquisition_plateaued() -> None:
    state = {
        "coverage_matrix": {"complete": False},
        "evidence_cards": [],
        "metrics": {
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "coverage_rounds": 1,
            "search_count": 30,
            "max_search_queries": 96,
        },
    }

    assert _coverage_route(state) == "finish"


def test_coverage_route_continues_when_resume_budget_expands_after_plateau() -> None:
    state = {
        "coverage_matrix": {"complete": False},
        "evidence_cards": [{"id": 1, "source_id": 1}],
        "metrics": {
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "candidate_count_total": 900,
            "max_candidates": 5000,
            "coverage_rounds": 2,
            "search_count": 0,
            "max_search_queries": 192,
            "max_rounds": 8,
        },
    }

    assert _coverage_route(state) == "more_sources"


def test_verification_route_rewrites_unsupported_claims_before_more_search() -> None:
    state = {
        "verification": {
            "valid": False,
            "failures": [
                "Semantic judge found unsupported claim: the report overstates a mixed finding.",
                "Cited evidence-backed source count below threshold: 10 < 17",
            ],
        },
        "metrics": {"verification_rounds": 1, "max_rounds": 4},
    }

    assert _verification_route(state) == "rewrite"


def test_verification_route_rewrites_writing_and_support_failures_without_more_search() -> None:
    state = {
        "coverage_matrix": {"complete": True},
        "verification": {
            "valid": False,
            "failures": [
                "Weakly supported cited paragraph: the report overgeneralizes a mechanism.",
                "Acceptance criteria coverage below threshold: 0.59",
                "Report depth below threshold: 0.76",
                "Semantic report judge returned invalid structured output: bad json",
            ],
        },
        "metrics": {
            "verification_rounds": 1,
            "max_rounds": 4,
            "last_acquire_added_sources": 3,
            "last_acquire_added_candidates": 20,
        },
    }

    assert _verification_route(state) == "rewrite"


def test_coverage_route_synthesizes_after_zero_progress_followup() -> None:
    state = {
        "coverage_matrix": {"complete": False, "missing_branches": ["branch_1"]},
        "evidence_cards": [{"id": 1}],
        "metrics": {
            "coverage_rounds": 2,
            "max_rounds": 8,
            "search_count": 31,
            "max_search_queries": 80,
            "candidate_count_total": 132,
            "max_candidates": 750,
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "last_acquire_searches": 0,
        },
    }

    assert _coverage_route(state) == "synthesize"


def test_verification_route_rewrites_instead_of_researching_after_source_plateau() -> None:
    state = {
        "verification": {
            "valid": False,
            "failures": [
                "Branch coverage incomplete: branch_2",
                "Branch coverage below threshold: 0.64",
            ],
        },
        "metrics": {
            "verification_rounds": 1,
            "max_rounds": 4,
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "last_acquire_searches": 0,
        },
    }

    assert _verification_route(state) == "rewrite"


def test_ingest_mcp_manifest_reads_content(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "title": "Connector Doc",
                        "url": "mcp://docs/1",
                        "content": "Urban heat island public health connector evidence " * 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    docs = ingest_mcp_manifest(manifest)

    assert len(docs) == 1
    assert docs[0].provenance == "mcp"
    assert "Urban heat island public health connector evidence" in docs[0].content


def test_ingest_local_paths_reads_unknown_text_suffix_and_skips_binary_in_directory(tmp_path: Path) -> None:
    text_doc = tmp_path / "field-notes.research"
    text_doc.write_text("Community cooling center usage and heat-health planning evidence " * 12, encoding="utf-8")
    binary_doc = tmp_path / "image.bin"
    binary_doc.write_bytes(b"\x00\x01\x02\x03\x04")

    docs = ingest_local_paths([tmp_path])

    assert len(docs) == 1
    assert docs[0].title == "field-notes"
    assert "Community cooling center usage" in docs[0].content


def test_gemini_managed_interaction_lifecycle_is_converted_to_v2_artifacts(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        research_engine="gemini_managed",
        google_api_key="google",
        google_api_keys=("google",),
        tavily_api_key="",
    )
    artifacts = ResearchArtifactsV2.create(tmp_path, "managed")
    client = FakeGeminiClient()

    result = run_gemini_managed_research(
        question="managed question",
        settings=settings,
        artifacts=artifacts,
        client=client,
        poll_interval_seconds=0,
    )

    assert client.created["agent"] == "deep-research-pro-preview-12-2025"
    assert client.created["background"] is True
    assert result.verification.valid is True
    assert (artifacts.run_dir / "report.md").exists()
    assert (artifacts.run_dir / "sources.jsonl").exists()


class FakeGeminiClient:
    def __init__(self) -> None:
        self.created = {}
        self.interactions = self

    def create(self, **kwargs):
        self.created = kwargs
        return SimpleNamespace(id="interaction-1")

    def get(self, interaction_id):
        assert interaction_id == "interaction-1"
        return SimpleNamespace(
            status="completed",
            outputs=[
                SimpleNamespace(
                    text=(
                        "# Managed Report\n\n"
                        "Managed Gemini research returned a cited report. [1]\n\n"
                        "## Sources\n\n"
                        "[1] Managed Source: https://example.com/source\n"
                    )
                )
            ],
        )


class FakeSemanticJudge:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return SimpleNamespace(content=json.dumps(self.payload))


class InvalidSemanticJudge:
    def invoke(self, _messages):
        return SimpleNamespace(content='{"cards": [{"id": 1, "keep": true}')


class QuotaSemanticJudge:
    def invoke(self, _messages):
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 37s.")


class RaisingSemanticJudge:
    def invoke(self, _messages):
        raise AssertionError("large evidence decks should not invoke the semantic judge")

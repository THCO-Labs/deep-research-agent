import json
import time
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
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, ResearchState, SourceRecordV2, VerificationResultV2
from deep_research.semantic import (
    _load_json_object,
    apply_semantic_report_result,
    enrich_evidence_cards_with_semantics,
    verify_report_with_semantics,
)
from deep_research.semantic_planning import build_or_enrich_research_plan
from deep_research.semantic_planning import _loads_json_object
from deep_research.settings import Settings
from deep_research.scraper import ScrapeQualityError, ScrapeResult
from deep_research.source_validation import validate_source_content
from deep_research.synthesis import (
    _append_evidence_coverage_if_needed,
    _cards_for_synthesis,
    _coverage_repair_labels,
    _evidence_backed_sources,
    _normalize_report_markdown,
    _repair_weak_citation_support,
    _synthesis_model_spec,
    _synthesis_prompt,
    _synthesis_request_kwargs,
    _target_report_profile,
    build_claim_ledger,
    build_report_blueprint,
    build_sentence_plan,
    synthesize_report,
    synthesize_report_with_model,
)
from deep_research.verifier_v2 import _report_depth_score, _report_level_criteria, verify_report_v2
from deep_research.research_graph import (
    _acquire_route,
    _coverage_route,
    _focus_terms_from_state,
    _publish_best_draft,
    _select_best_draft,
    _selected_failed_draft,
    _semantic_gate_collapsed_coverage,
    _verification_route,
    _write_run_health,
)


from tests.test_v2_fakes import FakeGeminiClient, FakeSemanticJudge, InvalidSemanticJudge, QuotaSemanticJudge, RaisingSemanticJudge

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


def test_best_draft_selection_prefers_valid_then_fewest_failures() -> None:
    history = [
        {"draft_index": 1, "draft_path": "draft_report_1.md", "valid": False, "failure_count": 17, "quality_score": 0.61},
        {"draft_index": 2, "draft_path": "draft_report_2.md", "valid": False, "failure_count": 11, "quality_score": 0.72},
        {"draft_index": 3, "draft_path": "draft_report_3.md", "valid": False, "failure_count": 33, "quality_score": 0.55},
    ]

    assert _select_best_draft(history)["draft_index"] == 2

    history.append({"draft_index": 4, "draft_path": "draft_report_4.md", "valid": True, "failure_count": 2, "quality_score": 0.5})

    assert _select_best_draft(history)["draft_index"] == 4


def test_best_draft_selection_prefers_grounded_quality_over_issue_count_only() -> None:
    history = [
        {"draft_index": 2, "draft_path": "draft_report_2.md", "valid": False, "failure_count": 12, "quality_score": 0.72},
        {"draft_index": 3, "draft_path": "draft_report_3.md", "valid": False, "failure_count": 10, "quality_score": 0.68},
    ]

    assert _select_best_draft(history)["draft_index"] == 2


def test_research_state_tracks_current_draft_for_best_draft_history() -> None:
    assert "current_draft" in ResearchState.__annotations__


def test_best_failed_draft_is_published_for_final_artifacts(tmp_path: Path) -> None:
    artifacts = ResearchArtifactsV2.create(tmp_path, "best draft")
    artifacts.write_text("draft_report_1.md", "better draft\n")
    artifacts.write_text("draft_report_2.md", "regressed draft\n")
    artifacts.write_json("verification_1.json", {"valid": False, "failures": ["a"]})
    artifacts.write_json("verification_2.json", {"valid": False, "failures": ["a", "b", "c"]})
    metrics = {
        "verification_rounds": 2,
        "verification_failure_history": [1, 3],
        "draft_history": [
            {
                "draft_index": 1,
                "draft_path": "draft_report_1.md",
                "verification_path": "verification_1.json",
                "valid": False,
                "failure_count": 1,
            },
            {
                "draft_index": 2,
                "draft_path": "draft_report_2.md",
                "verification_path": "verification_2.json",
                "valid": False,
                "failure_count": 3,
            },
        ],
    }

    _publish_best_draft(artifacts, metrics)
    selected = _selected_failed_draft(artifacts, "regressed draft\n")
    _write_run_health(artifacts, metrics, {"valid": False, "failures": ["a", "b", "c"]})

    assert selected == "better draft\n"
    assert (artifacts.run_dir / "best_draft.md").read_text(encoding="utf-8") == "better draft\n"
    assert json.loads((artifacts.run_dir / "best_verification.json").read_text(encoding="utf-8"))["failures"] == ["a"]
    health = json.loads((artifacts.run_dir / "run_health.json").read_text(encoding="utf-8"))
    assert health["best_draft_index"] == 1
    assert health["best_draft_failure_count"] == 1
    assert health["verification_failures"] == 3


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


def test_acquisition_candidate_watchdog_rejects_stuck_scraper(tmp_path: Path) -> None:
    class FakeSearchClient:
        def search(self, query: str, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.com/stuck",
                        "title": "Stuck candidate",
                        "content": "short snippet without enough raw article content",
                        "score": 0.8,
                    }
                ]
            }

    class StuckScraper:
        def fetch(self, url: str):
            time.sleep(10)
            raise AssertionError("scraper watchdog did not fire")

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
        scrape_timeout_ms=20_000,
        candidate_scrape_timeout_seconds=0.05,
    )
    progress_events: list[tuple[str, dict]] = []

    started = time.perf_counter()
    result = acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "stuck scraper"),
        settings=settings,
        search_client=FakeSearchClient(),
        scraper=StuckScraper(),
        progress_callback=lambda message, data: progress_events.append((message, data)),
    )

    assert time.perf_counter() - started < 2
    assert result.sources == []
    assert any(
        message == "rejected source candidate"
        and "Candidate scrape hard timeout" in data.get("rejection_reason", "")
        for message, data in progress_events
    )


def test_acquisition_does_not_thread_wrap_playwright_scraper(tmp_path: Path, monkeypatch) -> None:
    class FakeSearchClient:
        def search(self, query: str, **kwargs):
            return {
                "results": [
                    {
                        "url": "https://example.com/playwright",
                        "title": "Playwright candidate",
                        "content": "short snippet without enough raw article content",
                        "score": 0.8,
                    }
                ]
            }

    class FakePlaywrightScraper:
        def __init__(self, *, timeout_ms: int, retries: int) -> None:
            self.timeout_ms = timeout_ms
            self.retries = retries

        def fetch(self, url: str):
            return ScrapeResult(
                url=url,
                title="Playwright candidate",
                markdown=("Need for closure misinformation acceptance evidence " * 40),
                extraction_method="fake_playwright",
            )

    monkeypatch.setattr("deep_research.acquisition.PlaywrightScraper", FakePlaywrightScraper)
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
        scrape_timeout_ms=20_000,
        candidate_scrape_timeout_seconds=0.001,
        max_browser_scrapes_per_query=1,
        scrape_retries=0,
    )

    result = acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "playwright scraper threading"),
        settings=settings,
        search_client=FakeSearchClient(),
    )

    assert len(result.sources) == 1
    assert result.sources[0].extraction_method == "fake_playwright"


def test_acquisition_time_budget_stops_hostile_candidate_scan(tmp_path: Path) -> None:
    class FakeSearchClient:
        def search(self, query: str, **kwargs):
            return {
                "results": [
                    {
                        "url": f"https://example.com/slow-{index}",
                        "title": f"Slow candidate {index}",
                        "content": "short snippet without enough raw article content",
                        "score": 0.8,
                    }
                    for index in range(20)
                ]
            }

    class SlowRejectingScraper:
        def fetch(self, url: str):
            time.sleep(0.03)
            raise ScrapeQualityError("slow unusable page")

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
        max_candidates=80,
        max_sources=17,
        min_usable_sources=17,
        search_depth="advanced",
        allow_raw_content=True,
        scrape_timeout_ms=20_000,
        candidate_scrape_timeout_seconds=1,
        acquisition_timeout_seconds=0.05,
        max_browser_scrapes_per_query=20,
    )
    progress_events: list[tuple[str, dict]] = []

    result = acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "acquisition budget"),
        settings=settings,
        search_client=FakeSearchClient(),
        scraper=SlowRejectingScraper(),
        progress_callback=lambda message, data: progress_events.append((message, data)),
    )

    assert result.metrics.time_budget_exhausted is True
    assert len(result.candidates) < 20
    assert any(message == "acquisition time budget exhausted" for message, _ in progress_events)


def test_acquisition_implicit_source_cap_does_not_use_candidate_budget(tmp_path: Path) -> None:
    class UnexpectedSearchClient:
        def search(self, query: str, **kwargs):  # pragma: no cover - cap should stop before search.
            raise AssertionError(f"unexpected search for {query}")

    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure shapes misinformation acceptance.",
        queries=["need for closure misinformation acceptance evidence"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    existing_sources = [
        SourceRecordV2(
            id=index,
            branch_id="branch_1",
            title=f"Existing source {index}",
            url=f"https://example.com/existing-{index}",
            canonical_url=f"https://example.com/existing-{index}",
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
        for index in range(1, 18)
    ]
    settings = SimpleNamespace(
        min_source_words=40,
        min_relevant_chunks=1,
        max_candidates=50,
        max_sources=0,
        min_usable_sources=17,
        search_depth="advanced",
        allow_raw_content=True,
    )

    result = acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "implicit source cap"),
        settings=settings,
        search_client=UnexpectedSearchClient(),
        existing_sources=existing_sources,
        existing_source_texts={source.id: "existing source text" for source in existing_sources},
        active_branch_ids={"branch_1"},
    )

    assert len(result.sources) == 17
    assert result.metrics.search_count == 0


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
                        "url": "https://example.com/no-raw-1",
                        "title": "Need for closure short snippet",
                        "content": "short snippet without enough raw article content",
                        "score": 0.8,
                    },
                    {
                        "url": "https://example.com/no-raw-2",
                        "title": "Need for closure second short snippet",
                        "content": "another short snippet without enough raw article content",
                        "score": 0.7,
                    }
                ]
            }

    class OneShotScraper:
        def __init__(self) -> None:
            self.urls: list[str] = []

        def fetch(self, url: str):
            self.urls.append(url)
            if len(self.urls) > 1:  # pragma: no cover - budget should skip the second candidate.
                raise AssertionError(f"unexpected browser scrape for {url}")
            raise ScrapeQualityError("first candidate unusable")

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
        max_browser_scrapes_per_query=1,
    )
    scraper = OneShotScraper()
    progress_events: list[tuple[str, dict]] = []

    result = acquire_sources(
        question="What is the role of need for closure on misinformation acceptance?",
        branches=[branch],
        artifacts=ResearchArtifactsV2.create(tmp_path, "browser budget"),
        settings=settings,
        search_client=FakeSearchClient(),
        scraper=scraper,
        progress_callback=lambda message, data: progress_events.append((message, data)),
    )

    assert result.sources == []
    assert [candidate.url for candidate in result.candidates] == ["https://example.com/no-raw-1"]
    assert scraper.urls == ["https://example.com/no-raw-1"]
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

from pathlib import Path

from deep_research.artifacts import RunArtifacts
from deep_research.scraper import ScrapeQualityError, ScrapeResult
from deep_research.settings import Settings
from deep_research.source_registry import SourceRegistry
from deep_research.tools import ToolContext, build_tools


class FakeSearchClient:
    def search(self, query: str, max_results: int) -> dict:
        return {
            "results": [
                {
                    "title": "Example Research",
                    "url": "https://example.com/research?utm_source=test",
                    "content": "A useful snippet",
                    "score": 0.9,
                }
            ]
        }


class MultiSearchClient:
    def search(self, query: str, max_results: int) -> dict:
        results = [
            {
                "title": "Blocked Source",
                "url": "https://example.com/blocked",
                "content": "blocked snippet",
                "score": 0.9,
            },
        ]
        results.extend(
            {
                "title": f"Useful Source {index}",
                "url": f"https://example.com/useful-{index}",
                "content": f"useful snippet {index}",
                "score": 0.8,
            }
            for index in range(1, 18)
        )
        return {"results": results[:max_results]}


class QualitySearchClient:
    def search(self, query: str, max_results: int) -> dict:
        results = [
            {
                "title": "What are urban heat islands?",
                "url": "https://medium.com/example/urban-heat-islands",
                "content": "blog snippet",
                "score": 1.0,
            },
            {
                "title": "Urban heat island public health documentation",
                "url": "https://docs.example.com/urban-heat",
                "content": "official documentation snippet",
                "score": 0.2,
            },
        ]
        return {"results": results[:max_results]}


class RelevanceSearchClient:
    def search(self, query: str, max_results: int) -> dict:
        results = [
            {
                "title": "Billing API documentation",
                "url": "https://docs.example.com/billing",
                "content": "invoice payment subscription documentation",
                "score": 1.0,
            },
            {
                "title": "Urban heat island health adaptation",
                "url": "https://example.com/urban-heat-health",
                "content": "urban heat islands affect public health and vulnerable populations",
                "score": 0.4,
            },
        ]
        return {"results": results[:max_results]}


class FakeScraper:
    def fetch(self, url: str) -> ScrapeResult:
        return ScrapeResult(
            url="https://example.com/research",
            title="Example Research",
            markdown="Research shows retrieval adds external context.",
        )


class SelectiveScraper:
    def fetch(self, url: str) -> ScrapeResult:
        if url.endswith("/blocked"):
            raise RuntimeError("403 Forbidden")
        return ScrapeResult(
            url=url,
            title=f"Title for {url.rsplit('/', 1)[-1]}",
            markdown=f"Useful source content from {url}.",
        )


class FakeChallengeScraper:
    def fetch(self, url: str) -> ScrapeResult:
        raise ScrapeQualityError("Fetched page appears to be a bot-protection page.")


class FakeFailingScraper:
    def fetch(self, url: str) -> ScrapeResult:
        raise RuntimeError("403 Forbidden")


class RecordingScraper:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def fetch(self, url: str) -> ScrapeResult:
        self.urls.append(url)
        return ScrapeResult(
            url=url,
            title=f"Title for {url}",
            markdown="Official documentation contains substantial urban heat island public health guidance. " * 20,
        )


def test_mocked_research_tools_generate_required_artifacts(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "mock integration")
    registry = SourceRegistry(artifacts)
    context = ToolContext(
        settings,
        artifacts,
        registry,
        search_client=FakeSearchClient(),
        scraper=FakeScraper(),
    )
    tools = build_tools(context)

    search_result = tools["web_search"].invoke({"query": "retrieval augmented generation", "max_results": 1})
    assert search_result["results"][0]["needs_scrape"] is True
    assert "snippet" not in search_result["results"][0]
    scrape_result = tools["deep_scrape"].invoke({"url": search_result["results"][0]["url"]})
    tools["write_file"].invoke(
        {
            "file_path": "report.md",
            "content": (
                "## Finding\n\n"
                "Retrieval adds external context for generation [1].\n\n"
                "## Sources\n"
                f"[1] {scrape_result['title']}: {scrape_result['url']}\n"
            ),
        }
    )
    verification = tools["verify_report_file"].invoke({"file_path": "report.md"})

    assert (artifacts.run_dir / "report.md").exists()
    assert (artifacts.run_dir / "sources.jsonl").exists()
    assert (artifacts.run_dir / "verification.json").exists()
    assert verification["valid"] is True


def test_collect_sources_skips_bad_candidates_and_returns_usable_sources(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "collect sources")
    registry = SourceRegistry(artifacts)
    context = ToolContext(
        settings,
        artifacts,
        registry,
        search_client=MultiSearchClient(),
        scraper=SelectiveScraper(),
    )
    tools = build_tools(context)

    result = tools["collect_sources"].invoke(
        {"query": "urban heat island definition", "target_count": 2}
    )

    assert result["usable_count"] == 17
    assert result["unusable_count"] == 1
    assert result["needs_more_sources"] is False
    assert [source["source_id"] for source in result["usable_sources"]] == list(range(2, 19))
    assert result["unusable_sources"][0]["source_id"] == 1
    assert registry.records[0].content_path is None
    assert registry.records[1].content_path == "source_docs/source_2.md"


def test_collect_sources_ranks_higher_quality_candidates_before_scraping(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "quality ranking")
    registry = SourceRegistry(artifacts)
    scraper = RecordingScraper()
    context = ToolContext(
        settings,
        artifacts,
        registry,
        search_client=QualitySearchClient(),
        scraper=scraper,
    )
    tools = build_tools(context)

    result = tools["collect_sources"].invoke({"query": "urban heat island docs", "target_count": 1})

    assert scraper.urls[0] == "https://docs.example.com/urban-heat"
    assert result["usable_sources"][0]["source_quality_type"] == "official_docs"
    assert result["candidate_ranking"][0]["source_id"] == 2
    assert result["candidate_ranking"][0]["source_rank_score"] >= result["candidate_ranking"][1]["source_rank_score"]


def test_collect_sources_uses_relevance_when_quality_points_to_wrong_topic(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "relevance ranking")
    registry = SourceRegistry(artifacts)
    scraper = RecordingScraper()
    context = ToolContext(
        settings,
        artifacts,
        registry,
        search_client=RelevanceSearchClient(),
        scraper=scraper,
    )
    tools = build_tools(context)

    result = tools["collect_sources"].invoke({"query": "urban heat island public health", "target_count": 1})

    assert scraper.urls[0] == "https://example.com/urban-heat-health"
    assert result["candidate_ranking"][0]["source_id"] == 2
    assert result["candidate_ranking"][0]["source_relevance_score"] > result["candidate_ranking"][1]["source_relevance_score"]


def test_deep_scrape_returns_unusable_source_for_quality_rejection(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "challenge rejection")
    registry = SourceRegistry(artifacts)
    context = ToolContext(
        settings,
        artifacts,
        registry,
        search_client=FakeSearchClient(),
        scraper=FakeChallengeScraper(),
    )
    tools = build_tools(context)

    result = tools["deep_scrape"].invoke({"url": "https://example.com/challenge"})

    assert result["source_usable"] is False
    assert result["needs_alternate_source"] is True
    assert registry.records == []


def test_deep_scrape_returns_unusable_source_for_fetch_failure(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "fetch failure")
    registry = SourceRegistry(artifacts)
    context = ToolContext(
        settings,
        artifacts,
        registry,
        search_client=FakeSearchClient(),
        scraper=FakeFailingScraper(),
    )
    tools = build_tools(context)
    tools["web_search"].invoke({"query": "rag", "max_results": 1})

    result = tools["deep_scrape"].invoke({"url": "https://example.com/research"})

    assert result["source_id"] == 1
    assert result["source_usable"] is False
    assert result["needs_alternate_source"] is True
    assert "403 Forbidden" in result["error"]
    assert registry.records[0].content_path is None


def test_read_file_missing_returns_explicit_error(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "missing read")
    registry = SourceRegistry(artifacts)
    tools = build_tools(ToolContext(settings, artifacts, registry))

    result = tools["read_file"].invoke({"file_path": "missing.md"})

    assert result.startswith("ERROR: file not found:")


def test_read_file_returns_bounded_preview(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
        tool_excerpt_char_limit=20,
    )
    artifacts = RunArtifacts.create(tmp_path, "bounded read")
    registry = SourceRegistry(artifacts)
    artifacts.write_text("source_docs/source_1.md", "A" * 100)
    tools = build_tools(ToolContext(settings, artifacts, registry))

    result = tools["read_file"].invoke({"file_path": "source_docs/source_1.md"})

    assert result.startswith("A" * 20)
    assert "TRUNCATED" in result
    assert (artifacts.run_dir / "source_docs" / "source_1.md").read_text(encoding="utf-8") == "A" * 100


def test_deep_scrape_recovers_registered_source_from_bad_url(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "scrape recovery")
    registry = SourceRegistry(artifacts)
    context = ToolContext(
        settings,
        artifacts,
        registry,
        search_client=FakeSearchClient(),
        scraper=FakeScraper(),
    )
    tools = build_tools(context)

    tools["web_search"].invoke({"query": "rag", "max_results": 1})
    scrape_result = tools["deep_scrape"].invoke({"url": "https://example.com/??"})

    assert scrape_result["url"] == "https://example.com/research"

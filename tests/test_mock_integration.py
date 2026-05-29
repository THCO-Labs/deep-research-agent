from pathlib import Path

from deep_research.artifacts import RunArtifacts
from deep_research.scraper import ScrapeResult
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


class FakeScraper:
    def fetch(self, url: str) -> ScrapeResult:
        return ScrapeResult(
            url="https://example.com/research",
            title="Example Research",
            markdown="Research shows retrieval adds external context.",
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

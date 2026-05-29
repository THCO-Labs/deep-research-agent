from pathlib import Path

from deep_research.artifacts import RunArtifacts
from deep_research.progress import summarize_stream_update
from deep_research.scraper import ScrapeResult
from deep_research.settings import Settings
from deep_research.source_registry import SourceRegistry
from deep_research.tools import ToolContext, build_tools


class FakeSearchClient:
    def search(self, query: str, max_results: int) -> dict:
        return {
            "results": [
                {
                    "title": "Progress Source",
                    "url": "https://example.com/progress",
                    "content": "snippet",
                    "score": 0.8,
                }
            ]
        }


class FakeScraper:
    def fetch(self, url: str) -> ScrapeResult:
        return ScrapeResult(
            url=url,
            title="Progress Source",
            markdown="Progress content.",
        )


def test_summarize_stream_update_suppresses_verbose_tool_payload() -> None:
    content = {"query": "rag", "results": [{"canonical_url": "https://example.com"}]}

    assert summarize_stream_update("tools", content) is None


def test_tool_context_emits_progress_events(tmp_path: Path) -> None:
    events: list[str] = []
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "progress")
    registry = SourceRegistry(artifacts)
    tools = build_tools(
        ToolContext(
            settings,
            artifacts,
            registry,
            search_client=FakeSearchClient(),
            scraper=FakeScraper(),
            on_progress=events.append,
        )
    )

    search_result = tools["web_search"].invoke({"query": "rag", "max_results": 1})
    tools["deep_scrape"].invoke({"url": search_result["results"][0]["url"]})

    assert any("search: rag" in event for event in events)
    assert any("search: registered 1 source" in event for event in events)
    assert any("scrape: source [1]" in event for event in events)

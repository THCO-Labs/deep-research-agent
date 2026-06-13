import json
from pathlib import Path

from deep_research.artifacts import RunArtifacts
from deep_research.progress import (
    ActivityLog,
    format_activity_summary,
    load_activity_events,
    render_activity_html,
    summarize_stream_event,
    summarize_stream_update,
)
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


def test_summarize_stream_event_returns_visible_agent_event() -> None:
    event = summarize_stream_event("model", "I will search primary sources first.")

    assert event == ("agent", "I will search primary sources first.")


def test_activity_log_persists_jsonl_and_markdown(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "activity")
    events: list[str] = []
    activity = ActivityLog(artifacts, on_update=events.append, progress_mode="live")

    activity.emit("search", "registered 2 source candidate(s)", data={"source_count": 2})

    assert any("search: registered 2 source" in event for event in events)
    assert '"stage": "search"' in (artifacts.run_dir / "activity.jsonl").read_text(encoding="utf-8")
    assert "**search**: registered 2 source candidate(s)" in (artifacts.run_dir / "activity.md").read_text(
        encoding="utf-8"
    )
    assert "Deep Research Activity" in (artifacts.run_dir / "activity.html").read_text(encoding="utf-8")


def test_activity_log_raw_progress_streams_json_events(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "raw activity")
    events: list[str] = []
    activity = ActivityLog(artifacts, on_update=events.append, progress_mode="raw")

    activity.emit("research_status", "searching source candidates", data={"searches": 1})

    assert len(events) == 1
    payload = json.loads(events[0])
    assert payload["stage"] == "research_status"
    assert payload["message"] == "searching source candidates"
    assert payload["data"]["searches"] == 1


def test_activity_log_live_and_markdown_include_event_details(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "detailed activity")
    events: list[str] = []
    activity = ActivityLog(artifacts, on_update=events.append, progress_mode="live")

    activity.emit(
        "research_status",
        "rejected source candidate",
        data={
            "phase": "acquire_sources",
            "url": "https://example.com/rejected",
            "rejection_reason": "source lacks a question-specific anchor phrase",
            "sources": 2,
            "candidates": 11,
            "searches": 1,
        },
    )

    markdown = (artifacts.run_dir / "activity.md").read_text(encoding="utf-8")
    assert "rejected source candidate |" in events[0]
    assert "rejection_reason=source lacks a question-specific anchor phrase" in events[0]
    assert "url=https://example.com/rejected" in markdown
    assert "sources=2" in markdown


def test_activity_views_include_verification_failure_details(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "verification details")
    activity = ActivityLog(artifacts, progress_mode="quiet")

    activity.emit(
        "research_status",
        "verification failed with 2 issue(s)",
        data={
            "phase": "verify",
            "valid": False,
            "failures": ["Weakly supported cited paragraph: claim A", "Branch coverage incomplete: branch_4"],
        },
    )

    events = load_activity_events(artifacts.run_dir)
    summary = format_activity_summary(events, run_name=artifacts.run_dir.name, limit=1)
    html = render_activity_html(events, run_name=artifacts.run_dir.name)

    assert "Weakly supported cited paragraph" in summary
    assert "Branch coverage incomplete" in summary
    assert "Weakly supported cited paragraph" in html


def test_activity_summary_and_html_are_safe_to_read(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "activity viewer")
    activity = ActivityLog(artifacts, progress_mode="quiet")

    activity.emit("model_fallback", "orchestrator failed <secret>; trying fallback")
    activity.emit("verify", "passed: score 1.00")

    events = load_activity_events(artifacts.run_dir)
    summary = format_activity_summary(events, run_name=artifacts.run_dir.name, limit=2)
    html = render_activity_html(events, run_name=artifacts.run_dir.name)

    assert "model_fallback=1" in summary
    assert "verify=1" in summary
    assert "&lt;secret&gt;" in html
    assert "hidden chain-of-thought" in html


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


def test_tool_context_persists_activity_events(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )
    artifacts = RunArtifacts.create(tmp_path, "progress activity")
    registry = SourceRegistry(artifacts)
    activity = ActivityLog(artifacts, progress_mode="quiet")
    tools = build_tools(
        ToolContext(
            settings,
            artifacts,
            registry,
            search_client=FakeSearchClient(),
            scraper=FakeScraper(),
            activity=activity,
        )
    )

    search_result = tools["web_search"].invoke({"query": "rag", "max_results": 1})
    tools["deep_scrape"].invoke({"url": search_result["results"][0]["url"]})

    activity_jsonl = (artifacts.run_dir / "activity.jsonl").read_text(encoding="utf-8")
    assert '"stage": "search"' in activity_jsonl
    assert '"stage": "scrape"' in activity_jsonl

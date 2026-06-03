import json
from pathlib import Path

import pytest

from deep_research import agent as agent_module
from deep_research.agent import ResearchRunError, run_research
from deep_research.settings import Settings


class EmptyAgent:
    def stream(self, *_args, **_kwargs):
        return iter(())


class FailingAgent:
    def stream(self, *_args, **_kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 42.156s.")


class ToolCallFailingAgent:
    def stream(self, *_args, **_kwargs):
        raise RuntimeError("tool_use_failed: Failed to call a function.")


def test_run_research_writes_model_route_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "create_agent", lambda _settings, _context: EmptyAgent())
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="hybrid",
        model="groq:main",
        fast_model="groq:fast",
        planner_model="google_genai:planner",
        researcher_model="groq:researcher",
        analyst_model="groq:analyst",
        verifier_model="google_genai:verifier",
        judge_model="google_genai:judge",
        google_api_keys=("google-a", "google-b"),
        groq_api_keys=("groq-a", "groq-b"),
        tavily_api_key="tavily",
        precollect_sources=False,
    )

    result = run_research("route manifest", settings, progress_mode="quiet")

    manifest = json.loads((result.run_dir / "model_routes.json").read_text(encoding="utf-8"))
    activity = (result.run_dir / "activity.md").read_text(encoding="utf-8")
    plan = (result.run_dir / "research_plan.md").read_text(encoding="utf-8")
    assert manifest["google_key_count"] == 2
    assert manifest["groq_key_count"] == 2
    assert "researcher=groq:researcher via GROQ_API_KEY1" in activity
    assert "Deterministic baseline plan" in plan
    assert "route manifest" in plan


def test_run_research_writes_redacted_run_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "create_agent", lambda _settings, _context: EmptyAgent())
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="hybrid",
        model="groq:main",
        fast_model="groq:fast",
        planner_model="google_genai:planner",
        researcher_model="groq:researcher",
        analyst_model="groq:analyst",
        verifier_model="google_genai:verifier",
        judge_model="google_genai:judge",
        google_api_keys=("google-secret-a", "google-secret-b"),
        groq_api_keys=("groq-secret-a", "groq-secret-b"),
        tavily_api_key="tavily-secret",
        precollect_sources=False,
    )

    result = run_research("manifest", settings, progress_mode="quiet")

    manifest_text = (result.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert manifest["schema_version"] == 1
    assert manifest["settings"]["provider"] == "hybrid"
    assert manifest["settings"]["google_key_count"] == 2
    assert manifest["settings"]["groq_key_count"] == 2
    assert manifest["settings"]["tavily_api_key_present"] is True
    assert manifest["model_routes"]["google_key_count"] == 2
    assert "python" in manifest["runtime"]
    assert "google-secret" not in manifest_text
    assert "groq-secret" not in manifest_text
    assert "tavily-secret" not in manifest_text


def test_run_research_writes_repair_checklist_when_verification_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(agent_module, "create_agent", lambda _settings, _context: EmptyAgent())
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        google_api_keys=("google",),
        tavily_api_key="tavily",
        precollect_sources=False,
    )

    result = run_research("missing report", settings, progress_mode="quiet")

    repair_path = result.run_dir / "findings" / "verification_repair.md"
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert repair_path.exists()
    assert "Create `report.md`" in repair_path.read_text(encoding="utf-8")
    assert metrics["repair_checklist_path"].replace("\\", "/") == "findings/verification_repair.md"


def test_run_research_writes_structured_failure_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(agent_module, "create_agent", lambda _settings, _context: FailingAgent())
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        google_api_keys=("google",),
        tavily_api_key="tavily",
        precollect_sources=False,
    )

    with pytest.raises(ResearchRunError) as raised:
        run_research("quota failure", settings, progress_mode="quiet")

    failure = json.loads((raised.value.result.run_dir / "failure.json").read_text(encoding="utf-8"))
    metrics = json.loads(raised.value.result.metrics_path.read_text(encoding="utf-8"))
    assert failure["category"] == "quota_or_rate_limit"
    assert failure["retry_after_seconds"] == 42
    assert metrics["error_category"] == "quota_or_rate_limit"


def test_run_research_precollects_sources_before_agent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeCollectSourcesTool:
        def invoke(self, args: dict[str, object]) -> dict[str, object]:
            calls.append(args)
            return {
                "usable_count": 1,
                "unusable_count": 0,
                "usable_sources": [
                    {
                        "source_id": 1,
                        "title": "Example Source",
                        "url": "https://example.com/source",
                        "content_path": "source_docs/source_1.md",
                        "excerpt": "Example source evidence.",
                        "source_quality_label": "usable",
                        "source_quality_score": 0.6,
                        "source_quality_type": "general_web",
                        "source_relevance_score": 1.0,
                    }
                ],
                "unusable_sources": [],
            }

    monkeypatch.setattr(agent_module, "create_agent", lambda _settings, _context: EmptyAgent())
    monkeypatch.setattr(
        agent_module,
        "build_tools",
        lambda _context: {"collect_sources": FakeCollectSourcesTool()},
    )
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        google_api_keys=("google",),
        tavily_api_key="tavily",
        max_sources=2,
        precollect_sources=True,
    )

    result = run_research("precollect question", settings, progress_mode="quiet")

    brief = (result.run_dir / "findings" / "precollected_sources.md").read_text(encoding="utf-8")
    plan = (result.run_dir / "research_plan.md").read_text(encoding="utf-8")
    activity = (result.run_dir / "activity.md").read_text(encoding="utf-8")
    assert calls[0]["query"] == "precollect question"
    assert calls[0]["target_count"] == 2
    assert "Pre-collected Source Brief" in brief
    assert "[1] Example Source" in brief
    assert "## Pre-Collection Result" in plan
    assert "[1] Example Source" in plan
    assert "pre-collected 1/2 usable source" in activity


def test_run_research_recovers_report_from_scraped_sources_after_tool_call_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_markdown = (
        "Multi-agent frameworks improve LLM reasoning by assigning specialized agents "
        "to planning, critique, validation, and answer refinement. Validator agents "
        "check reasoning paths, while critic agents provide feedback that helps iterative "
        "refinement. Multi-agent debate exposes alternative explanations before final synthesis."
    )

    def fake_build_tools(context):
        class FakeCollectSourcesTool:
            def invoke(self, args: dict[str, object]) -> dict[str, object]:
                record = context.registry.upsert_search_result(
                    url="https://example.com/multi-agent-reasoning",
                    title="Multi-Agent Reasoning Evidence",
                    query=str(args["query"]),
                    snippet=source_markdown,
                    search_score=0.95,
                )
                record = context.registry.record_scrape(
                    url=record.url,
                    title=record.title,
                    markdown=source_markdown,
                    extraction_method="test",
                )
                return {
                    "query": args["query"],
                    "target_count": args["target_count"],
                    "candidate_count": 1,
                    "usable_count": 1,
                    "unusable_count": 0,
                    "usable_sources": [
                        {
                            "source_id": record.id,
                            "title": record.title,
                            "url": record.url,
                            "content_path": record.content_path,
                            "excerpt": source_markdown,
                            "source_quality_label": record.source_quality_label,
                            "source_quality_score": record.source_quality_score,
                            "source_quality_type": record.source_quality_type,
                            "source_relevance_score": record.source_relevance_score,
                        }
                    ],
                    "unusable_sources": [],
                }

        return {"collect_sources": FakeCollectSourcesTool()}

    monkeypatch.setattr(agent_module, "create_agent", lambda _settings, _context: ToolCallFailingAgent())
    monkeypatch.setattr(agent_module, "build_tools", fake_build_tools)
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        google_api_keys=("google",),
        tavily_api_key="tavily",
        max_sources=2,
        precollect_sources=True,
    )

    result = run_research(
        "How do multi-agent frameworks improve LLM reasoning?",
        settings,
        progress_mode="quiet",
    )

    report = result.report_path.read_text(encoding="utf-8")
    verification = json.loads(result.verification_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    activity = (result.run_dir / "activity.md").read_text(encoding="utf-8")
    assert "Multi-agent frameworks improve LLM reasoning" in report
    assert verification["valid"] is True
    assert metrics["deterministic_report_recovery"] is True
    assert metrics["error_category"] == "tool_call_parse_error"
    assert "complete with deterministic recovery" in activity

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
    )

    result = run_research("route manifest", settings, progress_mode="quiet")

    manifest = json.loads((result.run_dir / "model_routes.json").read_text(encoding="utf-8"))
    activity = (result.run_dir / "activity.md").read_text(encoding="utf-8")
    assert manifest["google_key_count"] == 2
    assert manifest["groq_key_count"] == 2
    assert "researcher=groq:researcher via GROQ_API_KEY1" in activity


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
    )

    with pytest.raises(ResearchRunError) as raised:
        run_research("quota failure", settings, progress_mode="quiet")

    failure = json.loads((raised.value.result.run_dir / "failure.json").read_text(encoding="utf-8"))
    metrics = json.loads(raised.value.result.metrics_path.read_text(encoding="utf-8"))
    assert failure["category"] == "quota_or_rate_limit"
    assert failure["retry_after_seconds"] == 42
    assert metrics["error_category"] == "quota_or_rate_limit"

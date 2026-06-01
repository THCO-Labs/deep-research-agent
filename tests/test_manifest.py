from pathlib import Path

from deep_research.manifest import build_run_manifest, redacted_settings
from deep_research.model_router import describe_model_routes
from deep_research.settings import Settings


def test_redacted_settings_excludes_secret_values(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path / "runs",
        provider="hybrid",
        model="groq:main",
        fast_model="groq:fast",
        planner_model="google_genai:planner",
        researcher_model="groq:researcher",
        analyst_model="groq:analyst",
        verifier_model="google_genai:verifier",
        judge_model="google_genai:judge",
        google_api_keys=("google-secret",),
        groq_api_keys=("groq-secret",),
        tavily_api_key="tavily-secret",
    )

    payload = redacted_settings(settings)

    assert payload["google_key_count"] == 1
    assert payload["groq_key_count"] == 1
    assert payload["tavily_api_key_present"] is True
    assert "secret" not in str(payload)


def test_build_run_manifest_includes_routes_and_runtime(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path / "runs",
        google_api_key="google",
        tavily_api_key="tavily",
    )
    routes = describe_model_routes(settings)

    manifest = build_run_manifest(
        question="What is fine-tuning?",
        settings=settings,
        run_dir=tmp_path / "runs" / "run",
        model_routes=routes,
        progress_mode="quiet",
    )

    assert manifest["question"] == "What is fine-tuning?"
    assert manifest["progress_mode"] == "quiet"
    assert manifest["model_routes"] == routes
    assert "packages" in manifest["runtime"]

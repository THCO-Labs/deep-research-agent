from pathlib import Path

from deep_research import model_router
from deep_research.settings import Settings


class FakeChatModel:
    def __init__(self, *, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key


def test_model_router_distributes_groq_role_models(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_router, "ChatGroq", FakeChatModel)
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:main",
        fast_model="groq:fast",
        planner_model="groq:planner",
        researcher_model="groq:researcher",
        analyst_model="groq:analyst",
        verifier_model="groq:verifier",
        judge_model="groq:judge",
        groq_api_keys=("groq-a", "groq-b"),
        tavily_api_key="tavily",
    )

    models = model_router.build_agent_models(settings)

    assert models.orchestrator.api_key == "groq-a"
    assert models.researcher.api_key == "groq-b"
    assert models.planner.api_key == "groq-a"
    assert models.verifier.api_key == "groq-b"


def test_model_router_distributes_google_judge_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_router, "ChatGoogleGenerativeAI", FakeChatModel)
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="google",
        model="google_genai:main",
        fast_model="google_genai:fast",
        planner_model="google_genai:planner",
        researcher_model="google_genai:researcher",
        analyst_model="google_genai:analyst",
        verifier_model="google_genai:verifier",
        judge_model="google_genai:judge",
        google_api_keys=("google-a", "google-b"),
        tavily_api_key="tavily",
    )

    model = model_router.model_for_role(settings, "judge", settings.judge_model)

    assert model.model == "judge"
    assert model.api_key == "google-b"


def test_model_router_uses_four_keys_in_hybrid_defaults(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_router, "ChatGroq", FakeChatModel)
    monkeypatch.setattr(model_router, "ChatGoogleGenerativeAI", FakeChatModel)
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

    models = model_router.build_agent_models(settings)
    judge = model_router.model_for_role(settings, "judge", settings.judge_model)

    assert models.orchestrator.api_key == "groq-a"
    assert models.researcher.api_key == "groq-b"
    assert models.planner.api_key == "google-a"
    assert models.verifier.api_key == "google-b"
    assert judge.api_key == "google-b"


def test_model_route_manifest_exposes_key_slots_without_secret_values(tmp_path: Path) -> None:
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
        tavily_api_key="tavily",
    )

    manifest = model_router.describe_model_routes(settings)
    routes = {route["role"]: route for route in manifest["roles"]}

    assert routes["orchestrator"]["key_label"] == "GROQ_API_KEY"
    assert routes["researcher"]["key_label"] == "GROQ_API_KEY1"
    assert routes["planner"]["key_label"] == "GOOGLE_API_KEY"
    assert routes["verifier"]["key_label"] == "GOOGLE_API_KEY1"
    assert "secret" not in str(manifest)

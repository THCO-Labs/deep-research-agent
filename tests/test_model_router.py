from pathlib import Path

from deep_research import model_router
from deep_research.settings import Settings


class FakeChatModel:
    def __init__(self, *, model: str, api_key: str, **kwargs) -> None:
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

    def bind_tools(self, tools, **kwargs):
        return FakeRunnable(self)

    def _get_ls_params(self, stop=None, **kwargs):
        return {"ls_provider": "groq", "ls_model_name": self.model}


class FakeRunnable:
    def __init__(self, model: FakeChatModel) -> None:
        self.model = model
        self.calls = 0

    def invoke(self, input, config=None, **kwargs):
        self.calls += 1
        if self.model.model == "rate-limited" and self.model.api_key.endswith("a"):
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 42s.")
        if self.model.model == "timeout" and self.model.api_key.endswith("a"):
            raise RuntimeError("504 DEADLINE_EXCEEDED. The request timed out. Please try again.")
        if self.model.model == "retry-once" and self.calls == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 2s.")
        return self.model.model


def _primary(model):
    return getattr(model, "primary", model)


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

    assert _primary(models.orchestrator).api_key == "groq-a"
    assert _primary(models.researcher).api_key == "groq-b"
    assert _primary(models.planner).api_key == "groq-a"
    assert _primary(models.verifier).api_key == "groq-b"
    assert _primary(models.orchestrator).kwargs["timeout"] == 120
    assert _primary(models.orchestrator).kwargs["max_retries"] == 0


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

    assert _primary(model).model == "judge"
    assert _primary(model).api_key == "google-b"
    assert _primary(model).kwargs["request_timeout"] == 120
    assert _primary(model).kwargs["retries"] == 0


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

    assert _primary(models.orchestrator).api_key == "groq-a"
    assert _primary(models.researcher).api_key == "groq-b"
    assert _primary(models.planner).api_key == "google-a"
    assert _primary(models.verifier).api_key == "google-b"
    assert _primary(judge).api_key == "google-b"


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
    assert routes["researcher"]["fallback_routes"]
    assert "secret" not in str(manifest)


def test_model_route_manifest_describes_ollama_without_api_key(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="ollama",
        model="ollama:qwen2.5:7b",
        fast_model="ollama:qwen2.5:3b",
        planner_model="ollama:qwen2.5:3b",
        researcher_model="ollama:qwen2.5:3b",
        analyst_model="ollama:qwen2.5:3b",
        verifier_model="ollama:qwen2.5:3b",
        judge_model="ollama:qwen2.5:3b",
        tavily_api_key="tavily",
    )

    manifest = model_router.describe_model_routes(settings)
    routes = {route["role"]: route for route in manifest["roles"]}

    assert routes["orchestrator"]["provider"] == "ollama"
    assert routes["orchestrator"]["key_label"] == "OLLAMA_API_KEY"
    assert routes["orchestrator"]["key_count"] == 1


def test_model_router_supports_openrouter_free_provider(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model_router, "ChatOpenRouter", FakeChatModel)
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="openrouter",
        model="openrouter:openrouter/free",
        fast_model="openrouter:meta-llama/llama-3.2-3b-instruct:free",
        planner_model="openrouter:openrouter/free",
        researcher_model="openrouter:openrouter/free",
        analyst_model="openrouter:openrouter/free",
        verifier_model="openrouter:openrouter/free",
        judge_model="openrouter:openrouter/free",
        openrouter_api_keys=("openrouter-a", "openrouter-b"),
        openrouter_http_referer="https://example.com",
        openrouter_app_title="Research Test",
        tavily_api_key="tavily",
    )

    model = model_router.model_for_role(settings, "planner", settings.planner_model)
    manifest = model_router.describe_model_routes(settings)
    routes = {route["role"]: route for route in manifest["roles"]}

    assert _primary(model).model == "openrouter/free"
    assert _primary(model).api_key == "openrouter-a"
    assert _primary(model).kwargs["referer"] == "https://example.com"
    assert _primary(model).kwargs["app_title"] == "Research Test"
    assert _primary(model).kwargs["timeout_seconds"] == 120
    assert manifest["openrouter_key_count"] == 2
    assert manifest["model_request_timeout_seconds"] == 120
    assert routes["planner"]["provider"] == "openrouter"
    assert routes["planner"]["key_label"] == "OPENROUTER_API_KEY"


def test_openrouter_free_is_available_as_cross_provider_fallback(tmp_path: Path) -> None:
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
        google_api_keys=("google-a",),
        openrouter_api_keys=("openrouter-a",),
        tavily_api_key="tavily",
    )

    manifest = model_router.describe_model_routes(settings)
    routes = {route["role"]: route for route in manifest["roles"]}

    assert {
        "provider": "openrouter",
        "model": "openrouter/free",
        "key_slot": 0,
        "key_label": "OPENROUTER_API_KEY",
        "fallback_type": "free_provider",
    } in routes["orchestrator"]["fallback_routes"]


def test_model_router_can_disable_fallback_wrapping(tmp_path: Path, monkeypatch) -> None:
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
        model_fallbacks=False,
    )

    model = model_router.model_for_role(settings, "orchestrator", settings.model)
    manifest = model_router.describe_model_routes(settings)

    assert isinstance(model, FakeChatModel)
    assert manifest["model_fallbacks"] is False
    assert manifest["roles"][0]["fallback_routes"] == []


def test_fallback_chat_model_proxies_langsmith_provider_params() -> None:
    model = model_router.FallbackChatModel(
        primary=FakeChatModel(model="primary", api_key="key-a"),
        fallbacks=(FakeChatModel(model="fallback", api_key="key-b"),),
    )

    assert model._get_ls_params()["ls_provider"] == "groq"
    assert model._get_ls_params()["ls_model_name"] == "primary"


def test_bound_model_falls_back_on_rate_limit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model_router, "ChatGroq", FakeChatModel)
    events: list[str] = []
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:rate-limited",
        fast_model="groq:fast",
        planner_model="groq:planner",
        researcher_model="groq:researcher",
        analyst_model="groq:analyst",
        verifier_model="groq:verifier",
        judge_model="groq:judge",
        groq_api_keys=("groq-a", "groq-b"),
        tavily_api_key="tavily",
    )

    model = model_router.model_for_role(
        settings,
        "orchestrator",
        settings.model,
        on_fallback=events.append,
    )
    result = model.bind_tools([]).invoke("input")

    assert result == "rate-limited"
    assert events


def test_bound_model_falls_back_on_provider_timeout(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model_router, "ChatGroq", FakeChatModel)
    events: list[str] = []
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:timeout",
        fast_model="groq:fast",
        planner_model="groq:planner",
        researcher_model="groq:researcher",
        analyst_model="groq:analyst",
        verifier_model="groq:verifier",
        judge_model="groq:judge",
        groq_api_keys=("groq-a", "groq-b"),
        tavily_api_key="tavily",
    )

    model = model_router.model_for_role(
        settings,
        "orchestrator",
        settings.model,
        on_fallback=events.append,
    )
    result = model.bind_tools([]).invoke("input")

    assert result == "timeout"
    assert events


def test_bound_model_waits_and_retries_retryable_provider_window(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(model_router, "ChatGroq", FakeChatModel)
    sleeps: list[int] = []
    retry_events: list[str] = []
    monkeypatch.setattr(model_router.time, "sleep", lambda seconds: sleeps.append(seconds))
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:retry-once",
        fast_model="groq:fast",
        planner_model="groq:planner",
        researcher_model="groq:researcher",
        analyst_model="groq:analyst",
        verifier_model="groq:verifier",
        judge_model="groq:judge",
        groq_api_keys=("groq-a",),
        tavily_api_key="tavily",
        provider_retry_attempts=1,
        provider_retry_max_wait_seconds=5,
    )

    model = model_router.model_for_role(
        settings,
        "orchestrator",
        settings.model,
        on_retry=retry_events.append,
    )
    result = model.bind_tools([]).invoke("input")

    assert result == "retry-once"
    assert sleeps == [2]
    assert "waiting 2s before retry 1/1" in retry_events[0]


def test_bound_model_does_not_wait_past_retry_cap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(model_router, "ChatGroq", FakeChatModel)
    sleeps: list[int] = []
    monkeypatch.setattr(model_router.time, "sleep", lambda seconds: sleeps.append(seconds))
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:retry-once",
        fast_model="groq:fast",
        planner_model="groq:planner",
        researcher_model="groq:researcher",
        analyst_model="groq:analyst",
        verifier_model="groq:verifier",
        judge_model="groq:judge",
        groq_api_keys=("groq-a",),
        tavily_api_key="tavily",
        provider_retry_attempts=1,
        provider_retry_max_wait_seconds=1,
    )

    model = model_router.model_for_role(settings, "orchestrator", settings.model)

    try:
        model.bind_tools([]).invoke("input")
    except RuntimeError as exc:
        assert "RESOURCE_EXHAUSTED" in str(exc)
    else:
        raise AssertionError("Expected retry cap to preserve the provider error.")
    assert sleeps == []

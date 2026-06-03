from pathlib import Path

import pytest

from deep_research import deepagents_profiles
from deep_research.settings import Settings


@pytest.fixture(autouse=True)
def reset_profile_keys() -> None:
    deepagents_profiles._CONFIGURED_PROFILE_KEYS.clear()
    yield
    deepagents_profiles._CONFIGURED_PROFILE_KEYS.clear()


def _settings(tmp_path: Path, *, provider: str, model: str, role_model: str | None = None) -> Settings:
    role = role_model or model
    return Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider=provider,  # type: ignore[arg-type]
        model=model,
        fast_model=model,
        planner_model=role,
        researcher_model=role,
        analyst_model=role,
        verifier_model=role,
        judge_model=role,
        google_api_key="google",
        groq_api_key="groq",
        tavily_api_key="tavily",
    )


def test_configure_deepagents_profiles_registers_groq_tool_exclusion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def fake_register(key: str, profile: object) -> None:
        calls.append((key, profile))

    monkeypatch.setattr(deepagents_profiles, "register_harness_profile", fake_register)
    settings = _settings(tmp_path, provider="groq", model="groq:openai/gpt-oss-20b")

    deepagents_profiles.configure_deepagents_profiles(settings)
    deepagents_profiles.configure_deepagents_profiles(settings)

    assert [key for key, _ in calls] == ["fallbackchatmodel", "groq", "groq:openai/gpt-oss-20b"]
    assert all(deepagents_profiles.WRITE_TODOS_TOOL in profile.excluded_tools for _, profile in calls)


def test_configure_deepagents_profiles_registers_google_noop_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(deepagents_profiles, "register_harness_profile", lambda key, profile: calls.append((key, profile)))
    settings = _settings(tmp_path, provider="google", model="google_genai:gemini-2.5-flash")

    deepagents_profiles.configure_deepagents_profiles(settings)

    assert [key for key, _ in calls] == ["google_genai"]
    assert calls[0][1].excluded_tools == frozenset()


def test_configure_deepagents_profiles_handles_groq_role_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(deepagents_profiles, "register_harness_profile", lambda key, profile: calls.append((key, profile)))
    settings = _settings(
        tmp_path,
        provider="google",
        model="google_genai:gemini-2.5-flash",
        role_model="groq:openai/gpt-oss-20b",
    )

    deepagents_profiles.configure_deepagents_profiles(settings)

    assert [key for key, _ in calls] == ["fallbackchatmodel", "groq", "groq:openai/gpt-oss-20b", "google_genai"]

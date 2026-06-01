import os
from pathlib import Path

import pytest

from deep_research.settings import ConfigError, Settings


@pytest.fixture(autouse=True)
def clear_research_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith(("GOOGLE_API_KEY", "GROQ_API_KEY", "DEEP_RESEARCH_")) or name == "TAVILY_API_KEY":
            monkeypatch.delenv(name, raising=False)


def test_settings_loads_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GOOGLE_API_KEY=google-test\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.google_api_key == "google-test"
    assert settings.tavily_api_key == "tavily-test"
    assert settings.provider == "google"
    assert settings.max_sources == 12


def test_settings_auto_uses_hybrid_when_google_and_groq_are_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "GOOGLE_API_KEY=google-test\nGOOGLE_API_KEY1=google-test-1\n"
        "GROQ_API_KEY=groq-test\nGROQ_API_KEY1=groq-test-1\n"
        "TAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY1", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY1", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MODEL", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_FAST_MODEL", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.provider == "hybrid"
    assert settings.model == "groq:openai/gpt-oss-20b"
    assert settings.fast_model == "groq:openai/gpt-oss-20b"
    assert settings.planner_model == "google_genai:gemini-2.5-flash"
    assert settings.researcher_model == "groq:openai/gpt-oss-20b"
    assert settings.verifier_model == "google_genai:gemini-2.5-flash"
    assert settings.judge_model == "google_genai:gemini-2.5-flash"
    assert settings.scrape_char_limit == 6000
    assert settings.tool_excerpt_char_limit == 900
    assert settings.max_sources == 3
    assert settings.max_rounds == 1
    assert settings.google_key_pool == ("google-test", "google-test-1")
    assert settings.groq_key_pool == ("groq-test", "groq-test-1")


def test_settings_auto_uses_groq_when_only_groq_is_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.provider == "groq"
    assert settings.model == "groq:openai/gpt-oss-20b"
    assert settings.groq_key_pool == ("groq-test",)


def test_settings_loads_numbered_groq_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY1=groq-one\nGROQ_API_KEY2=groq-two\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY1", raising=False)
    monkeypatch.delenv("GROQ_API_KEY2", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.provider == "groq"
    assert settings.groq_api_key == "groq-one"
    assert settings.groq_key_pool == ("groq-one", "groq-two")


def test_settings_loads_numbered_google_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GOOGLE_API_KEY1=google-one\nGOOGLE_API_KEY2=google-two\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY1", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY2", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.provider == "google"
    assert settings.google_api_key == "google-one"
    assert settings.google_key_pool == ("google-one", "google-two")


def test_settings_supports_role_model_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_RESEARCHER_MODEL=groq:openai/gpt-oss-120b\n"
        "DEEP_RESEARCH_JUDGE_MODEL=groq:openai/gpt-oss-20b\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_RESEARCHER_MODEL", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_JUDGE_MODEL", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.researcher_model == "groq:openai/gpt-oss-120b"
    assert settings.judge_model == "groq:openai/gpt-oss-20b"


def test_settings_supports_disabling_model_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_MODEL_FALLBACKS=false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MODEL_FALLBACKS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.model_fallbacks is False


def test_settings_supports_provider_retry_window_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_PROVIDER_RETRY_ATTEMPTS=2\n"
        "DEEP_RESEARCH_PROVIDER_RETRY_MAX_WAIT_SECONDS=15\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER_RETRY_ATTEMPTS", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER_RETRY_MAX_WAIT_SECONDS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.provider_retry_attempts == 2
    assert settings.provider_retry_max_wait_seconds == 15


def test_settings_supports_explicit_provider_and_short_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    settings = Settings.from_env(
        project_root=tmp_path,
        provider="groq",
        model="openai/gpt-oss-120b",
    )

    assert settings.provider == "groq"
    assert settings.model == "groq:openai/gpt-oss-120b"


def test_settings_requires_api_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ConfigError):
        Settings.from_env(project_root=tmp_path)

from pathlib import Path

import pytest

from deep_research.settings import ConfigError, Settings


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


def test_settings_prefers_groq_when_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GOOGLE_API_KEY=google-test\nGROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MODEL", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_FAST_MODEL", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.provider == "groq"
    assert settings.model == "groq:openai/gpt-oss-20b"
    assert settings.fast_model == "groq:openai/gpt-oss-20b"
    assert settings.scrape_char_limit == 6000
    assert settings.tool_excerpt_char_limit == 1500


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

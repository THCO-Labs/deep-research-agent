import os
from pathlib import Path

import pytest

from deep_research.settings import ConfigError, Settings


@pytest.fixture(autouse=True)
def clear_research_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(os.environ):
        if name.startswith((
            "GOOGLE_API_KEY",
            "GROQ_API_KEY",
            "OPENROUTER_API_KEY",
            "TAVILY_API_KEY",
            "EXA_API_KEY",
            "BRAVE_SEARCH_API_KEY",
            "FIRECRAWL_API_KEY",
            "SERPER_API_KEY",
            "DEEP_RESEARCH_",
        )):
            monkeypatch.delenv(name, raising=False)
        elif "TAVILY" in name.upper() and "API" in name.upper() and "KEY" in name.upper():
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
    assert settings.mode == "max_quality"
    assert settings.max_sources == 0
    assert settings.min_usable_sources == 40


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
    assert settings.model == "google_genai:gemini-2.5-flash"
    assert settings.fast_model == "groq:openai/gpt-oss-20b"
    assert settings.planner_model == "google_genai:gemini-2.5-flash"
    assert settings.researcher_model == "groq:openai/gpt-oss-20b"
    assert settings.verifier_model == "google_genai:gemini-2.5-flash"
    assert settings.judge_model == "google_genai:gemini-2.5-flash"
    assert settings.scrape_char_limit == 6000
    assert settings.tool_excerpt_char_limit == 900
    assert settings.max_sources == 0
    assert settings.min_usable_sources == 40
    assert settings.max_rounds == 3
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


def test_settings_auto_uses_openrouter_when_only_openrouter_is_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "OPENROUTER_API_KEY=openrouter-test\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.provider == "openrouter"
    assert settings.model == "openrouter:meta-llama/llama-3.3-70b-instruct:free"
    assert settings.fast_model == "openrouter:meta-llama/llama-3.3-70b-instruct:free"
    assert settings.openrouter_key_pool == ("openrouter-test",)


def test_settings_loads_numbered_tavily_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GOOGLE_API_KEY=google-test\n"
        "TAVILY_API_KEY1=tavily-one\n"
        "TAVILY_API_KEY2=tavily-two\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY1", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY2", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.tavily_api_key == "tavily-one"
    assert settings.tavily_key_pool == ("tavily-one", "tavily-two")


def test_settings_loads_delimited_and_underscore_key_pools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GOOGLE_API_KEY=google-test\n"
        "TAVILY_API_KEY=tavily-one\n"
        "TAVILY_API_KEY_2=tavily-two\n"
        "TAVILY_API_KEYS=tavily-three,tavily-four;tavily-two\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY_2", raising=False)
    monkeypatch.delenv("TAVILY_API_KEYS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.tavily_api_key == "tavily-one"
    assert settings.tavily_key_pool == ("tavily-one", "tavily-two", "tavily-three", "tavily-four")


def test_settings_discovers_semantic_tavily_key_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GOOGLE_API_KEY=google-test\n"
        "TAVILY_SEARCH_API_KEY=tavily-search\n"
        "EXTRA_TAVILY_API_KEY=tavily-extra\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_SEARCH_API_KEY", raising=False)
    monkeypatch.delenv("EXTRA_TAVILY_API_KEY", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.tavily_api_key == "tavily-search"
    assert settings.tavily_key_pool == ("tavily-search", "tavily-extra")


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


def test_settings_supports_model_request_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_MODEL_REQUEST_TIMEOUT_SECONDS=45\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MODEL_REQUEST_TIMEOUT_SECONDS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.model_request_timeout_seconds == 45


def test_settings_supports_scrape_timeout_and_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_SCRAPE_TIMEOUT_MS=7000\n"
        "DEEP_RESEARCH_SCRAPE_RETRIES=2\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_SCRAPE_TIMEOUT_MS", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_SCRAPE_RETRIES", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.scrape_timeout_ms == 7000
    assert settings.scrape_retries == 2


def test_settings_supports_blocked_source_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        r"DEEP_RESEARCH_BLOCKED_SOURCE_PATTERNS=deep[_-]?research[_-]?bench;reference\.jsonl"
        "\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_BLOCKED_SOURCE_PATTERNS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.blocked_source_patterns == (r"deep[_-]?research[_-]?bench", r"reference\.jsonl")


def test_settings_supports_browser_fallback_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_MAX_BROWSER_SCRAPES_PER_QUERY=0\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MAX_BROWSER_SCRAPES_PER_QUERY", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.max_browser_scrapes_per_query == 0


def test_settings_supports_acquisition_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_ACQUISITION_TIMEOUT_SECONDS=300\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_ACQUISITION_TIMEOUT_SECONDS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.acquisition_timeout_seconds == 300


def test_settings_supports_followup_query_fairness_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_MAX_FOLLOWUP_QUERIES_PER_BRANCH=5\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MAX_FOLLOWUP_QUERIES_PER_BRANCH", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.max_followup_queries_per_branch == 5


def test_settings_supports_model_max_output_tokens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_MODEL_MAX_OUTPUT_TOKENS=12000\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MODEL_MAX_OUTPUT_TOKENS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.model_max_output_tokens == 12000


def test_settings_supports_disabling_precollection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_PRECOLLECT_SOURCES=false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PRECOLLECT_SOURCES", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.precollect_sources is False


def test_settings_supports_disabling_llm_planning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_LLM_PLANNING=false\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_LLM_PLANNING", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.llm_planning is False


def test_settings_supports_semantic_evidence_card_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\nTAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_SEMANTIC_EVIDENCE_MAX_LLM_CARDS=0\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_SEMANTIC_EVIDENCE_MAX_LLM_CARDS", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.semantic_evidence_max_llm_cards == 0


def test_settings_supports_ollama_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "DEEP_RESEARCH_PROVIDER=ollama\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    settings = Settings.from_env(project_root=tmp_path)

    assert settings.provider == "ollama"
    assert settings.model == "ollama:qwen2.5:7b"
    assert settings.fast_model == "ollama:qwen2.5:3b"


def test_settings_prefixes_ollama_tagged_model_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "DEEP_RESEARCH_PROVIDER=ollama\nTAVILY_API_KEY=tavily-test\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    settings = Settings.from_env(
        project_root=tmp_path,
        provider="ollama",
        model="qwen2.5:7b",
        fast_model="qwen2.5:3b",
    )

    assert settings.model == "ollama:qwen2.5:7b"
    assert settings.fast_model == "ollama:qwen2.5:3b"


def test_explicit_provider_ignores_env_model_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY=groq-test\n"
        "TAVILY_API_KEY=tavily-test\n"
        "DEEP_RESEARCH_PROVIDER=groq\n"
        "DEEP_RESEARCH_MODEL=groq:meta-llama/llama-4-scout-17b-16e-instruct\n"
        "DEEP_RESEARCH_FAST_MODEL=groq:meta-llama/llama-4-scout-17b-16e-instruct\n"
        "DEEP_RESEARCH_PLANNER_MODEL=groq:meta-llama/llama-4-scout-17b-16e-instruct\n"
        "DEEP_RESEARCH_RESEARCHER_MODEL=groq:meta-llama/llama-4-scout-17b-16e-instruct\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_MODEL", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_FAST_MODEL", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_PLANNER_MODEL", raising=False)
    monkeypatch.delenv("DEEP_RESEARCH_RESEARCHER_MODEL", raising=False)

    settings = Settings.from_env(project_root=tmp_path, provider="ollama")

    assert settings.provider == "ollama"
    assert settings.model == "ollama:qwen2.5:7b"
    assert settings.fast_model == "ollama:qwen2.5:3b"
    assert settings.planner_model == "ollama:qwen2.5:3b"
    assert settings.researcher_model == "ollama:qwen2.5:3b"


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

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from dotenv import load_dotenv

from deep_research.source_limits import MINIMUM_SOURCE_TARGET, source_floor

Mode = Literal["fast", "balanced", "max_quality"]
Provider = Literal["auto", "google", "groq", "hybrid", "ollama", "openrouter"]
ResolvedProvider = Literal["google", "groq", "hybrid", "ollama", "openrouter"]
ResearchEngineName = Literal["local_langgraph", "gemini_managed", "openai_managed"]

GOOGLE_DEFAULT_MODEL = "google_genai:gemini-2.5-flash"
GOOGLE_DEFAULT_FAST_MODEL = "google_genai:gemini-2.5-flash"
GROQ_DEFAULT_MODEL = "groq:openai/gpt-oss-20b"
GROQ_DEFAULT_FAST_MODEL = "groq:openai/gpt-oss-20b"
OLLAMA_DEFAULT_MODEL = "ollama:qwen2.5:7b"
OLLAMA_DEFAULT_FAST_MODEL = "ollama:qwen2.5:3b"
OPENROUTER_DEFAULT_MODEL = "openrouter:meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_DEFAULT_FAST_MODEL = "openrouter:meta-llama/llama-3.3-70b-instruct:free"
MODEL_PROVIDER_PREFIXES = frozenset({"google_genai", "groq", "ollama", "openrouter", "mistral_ai"})
HYBRID_DEFAULT_MODELS = {
    "orchestrator": GOOGLE_DEFAULT_MODEL,
    "fast": GROQ_DEFAULT_FAST_MODEL,
    "planner": GOOGLE_DEFAULT_FAST_MODEL,
    "researcher": GROQ_DEFAULT_FAST_MODEL,
    "analyst": GROQ_DEFAULT_FAST_MODEL,
    "verifier": GOOGLE_DEFAULT_FAST_MODEL,
    "judge": GOOGLE_DEFAULT_FAST_MODEL,
}


class ConfigError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    project_root: Path
    mode: Mode = "max_quality"
    out_dir: Path = Path("runs")
    max_sources: int = 80
    max_rounds: int = 6
    research_engine: ResearchEngineName = "local_langgraph"
    min_usable_sources: int = MINIMUM_SOURCE_TARGET
    max_search_queries: int = 12
    max_candidates: int = 80
    max_followup_queries_per_branch: int = 12
    min_source_words: int = 250
    min_relevant_chunks: int = 1
    search_depth: str = "advanced"
    allow_raw_content: bool = True
    semantic_verification: bool = True
    semantic_evidence_max_llm_cards: int = 120
    llm_planning: bool = True
    report_quality_gate: bool = True
    llm_synthesis: bool = True
    allow_failed_verification: bool = False
    strict_tool_models: bool = True
    local_input_paths: tuple[str, ...] = field(default_factory=tuple)
    mcp_manifest: str = ""
    provider: ResolvedProvider = "google"
    model: str = GOOGLE_DEFAULT_MODEL
    fast_model: str = GOOGLE_DEFAULT_FAST_MODEL
    planner_model: str = GOOGLE_DEFAULT_FAST_MODEL
    researcher_model: str = GOOGLE_DEFAULT_FAST_MODEL
    analyst_model: str = GOOGLE_DEFAULT_FAST_MODEL
    verifier_model: str = GOOGLE_DEFAULT_FAST_MODEL
    judge_model: str = GOOGLE_DEFAULT_FAST_MODEL
    synthesis_model: str = ""
    scrape_char_limit: int = 15_000
    scrape_timeout_ms: int = 20_000
    scrape_retries: int = 1
    max_browser_scrapes_per_query: int = 12
    blocked_source_patterns: tuple[str, ...] = field(default_factory=tuple)
    tool_excerpt_char_limit: int = 2_500
    precollect_sources: bool = True
    model_fallbacks: bool = True
    provider_retry_attempts: int = 1
    provider_retry_max_wait_seconds: int = 60
    model_request_timeout_seconds: int = 120
    model_max_output_tokens: int = 8192
    live: bool = False
    google_api_key: str = field(default="", repr=False)
    google_api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    groq_api_key: str = field(default="", repr=False)
    groq_api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    openrouter_api_key: str = field(default="", repr=False)
    openrouter_api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    openrouter_http_referer: str = ""
    openrouter_app_title: str = "Deep Research Agent"
    tavily_api_key: str = field(default="", repr=False)
    tavily_api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    mistral_api_key: str = field(default="", repr=False)
    mistral_api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | str | None = None,
        mode: Mode = "max_quality",
        out_dir: Path | str | None = None,
        max_sources: int | None = None,
        max_rounds: int | None = None,
        research_engine: ResearchEngineName | None = None,
        min_usable_sources: int | None = None,
        max_search_queries: int | None = None,
        max_candidates: int | None = None,
        max_followup_queries_per_branch: int | None = None,
        min_source_words: int | None = None,
        min_relevant_chunks: int | None = None,
        search_depth: str | None = None,
        allow_raw_content: bool | None = None,
        semantic_verification: bool | None = None,
        semantic_evidence_max_llm_cards: int | None = None,
        llm_planning: bool | None = None,
        report_quality_gate: bool | None = None,
        llm_synthesis: bool | None = None,
        allow_failed_verification: bool | None = None,
        strict_tool_models: bool | None = None,
        local_input_paths: tuple[str, ...] | None = None,
        mcp_manifest: str | None = None,
        provider: Provider | None = None,
        model: str | None = None,
        fast_model: str | None = None,
        planner_model: str | None = None,
        researcher_model: str | None = None,
        analyst_model: str | None = None,
        verifier_model: str | None = None,
        judge_model: str | None = None,
        synthesis_model: str | None = None,
        scrape_char_limit: int | None = None,
        scrape_timeout_ms: int | None = None,
        scrape_retries: int | None = None,
        max_browser_scrapes_per_query: int | None = None,
        blocked_source_patterns: tuple[str, ...] | None = None,
        precollect_sources: bool | None = None,
        model_fallbacks: bool | None = None,
        provider_retry_attempts: int | None = None,
        provider_retry_max_wait_seconds: int | None = None,
        model_request_timeout_seconds: int | None = None,
        model_max_output_tokens: int | None = None,
        live: bool = False,
    ) -> "Settings":
        root = Path(project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env", override=False)

        resolved_out = Path(out_dir) if out_dir is not None else Path("runs")
        if not resolved_out.is_absolute():
            resolved_out = root / resolved_out

        google_api_keys = _collect_numbered_env_values("GOOGLE_API_KEY")
        groq_api_keys = _collect_numbered_env_values("GROQ_API_KEY")
        openrouter_api_keys = _collect_numbered_env_values("OPENROUTER_API_KEY")
        mistral_api_keys = _collect_numbered_env_values("MISTRAL_API_KEY")
        google_api_key = google_api_keys[0] if google_api_keys else ""
        groq_api_key = groq_api_keys[0] if groq_api_keys else ""
        openrouter_api_key = openrouter_api_keys[0] if openrouter_api_keys else ""
        mistral_api_key = mistral_api_keys[0] if mistral_api_keys else ""
        tavily_api_keys = _collect_numbered_env_values(
            "TAVILY_API_KEY",
            extra_names=_semantic_api_key_env_names("TAVILY"),
        )
        tavily_api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not tavily_api_key and tavily_api_keys:
            tavily_api_key = tavily_api_keys[0]
        provider_explicit = provider is not None
        requested_provider = provider or os.environ.get("DEEP_RESEARCH_PROVIDER", "auto")
        resolved_provider = _resolve_provider(requested_provider, google_api_keys, groq_api_keys, openrouter_api_keys)
        mode_sources, mode_rounds = _mode_defaults(mode, resolved_provider)
        depth = _depth_defaults(mode)
        resolved_fast_model = _resolve_model(
            resolved_provider,
            fast_model or _env_model_override("DEEP_RESEARCH_FAST_MODEL", provider_explicit=provider_explicit),
            fast=True,
            role="fast",
        )

        resolved_min_usable_sources = source_floor(
            min_usable_sources
            if min_usable_sources is not None
            else int(os.environ.get("DEEP_RESEARCH_MIN_USABLE_SOURCES") or depth["min_usable_sources"])
        )
        resolved_max_sources = (
            0
            if max_sources == 0 or (max_sources is None and mode_sources == 0)
            else source_floor(max_sources if max_sources is not None else mode_sources)
        )
        resolved_max_candidates = max(
            resolved_min_usable_sources,
            int(
                max_candidates
                if max_candidates is not None
                else int(os.environ.get("DEEP_RESEARCH_MAX_CANDIDATES") or depth["max_candidates"])
            ),
        )

        settings = cls(
            project_root=root,
            mode=mode,
            out_dir=resolved_out.resolve(),
            max_sources=resolved_max_sources,
            max_rounds=max_rounds if max_rounds is not None else mode_rounds,
            research_engine=research_engine
            or os.environ.get("DEEP_RESEARCH_ENGINE", "local_langgraph"),  # type: ignore[arg-type]
            min_usable_sources=resolved_min_usable_sources,
            max_search_queries=max_search_queries
            if max_search_queries is not None
            else int(os.environ.get("DEEP_RESEARCH_MAX_SEARCH_QUERIES") or depth["max_search_queries"]),
            max_candidates=resolved_max_candidates,
            max_followup_queries_per_branch=max_followup_queries_per_branch
            if max_followup_queries_per_branch is not None
            else int(os.environ.get("DEEP_RESEARCH_MAX_FOLLOWUP_QUERIES_PER_BRANCH") or "12"),
            min_source_words=min_source_words
            if min_source_words is not None
            else int(os.environ.get("DEEP_RESEARCH_MIN_SOURCE_WORDS") or depth["min_source_words"]),
            min_relevant_chunks=min_relevant_chunks
            if min_relevant_chunks is not None
            else int(os.environ.get("DEEP_RESEARCH_MIN_RELEVANT_CHUNKS") or "1"),
            search_depth=search_depth or os.environ.get("DEEP_RESEARCH_SEARCH_DEPTH", "advanced"),
            allow_raw_content=_resolve_bool(
                allow_raw_content,
                os.environ.get("DEEP_RESEARCH_ALLOW_RAW_CONTENT"),
                default=True,
            ),
            semantic_verification=_resolve_bool(
                semantic_verification,
                os.environ.get("DEEP_RESEARCH_SEMANTIC_VERIFICATION"),
                default=True,
            ),
            semantic_evidence_max_llm_cards=semantic_evidence_max_llm_cards
            if semantic_evidence_max_llm_cards is not None
            else int(os.environ.get("DEEP_RESEARCH_SEMANTIC_EVIDENCE_MAX_LLM_CARDS") or "120"),
            llm_planning=_resolve_bool(
                llm_planning,
                os.environ.get("DEEP_RESEARCH_LLM_PLANNING"),
                default=True,
            ),
            report_quality_gate=_resolve_bool(
                report_quality_gate,
                os.environ.get("DEEP_RESEARCH_REPORT_QUALITY_GATE"),
                default=True,
            ),
            llm_synthesis=_resolve_bool(
                llm_synthesis,
                os.environ.get("DEEP_RESEARCH_LLM_SYNTHESIS"),
                default=True,
            ),
            allow_failed_verification=_resolve_bool(
                allow_failed_verification,
                os.environ.get("DEEP_RESEARCH_ALLOW_FAILED_VERIFICATION"),
                default=False,
            ),
            strict_tool_models=_resolve_bool(
                strict_tool_models,
                os.environ.get("DEEP_RESEARCH_STRICT_TOOL_MODELS"),
                default=True,
            ),
            local_input_paths=local_input_paths
            if local_input_paths is not None
            else _split_env_list(os.environ.get("DEEP_RESEARCH_LOCAL_INPUTS", "")),
            mcp_manifest=mcp_manifest if mcp_manifest is not None else os.environ.get("DEEP_RESEARCH_MCP_MANIFEST", ""),
            provider=resolved_provider,
            model=_resolve_model(
                resolved_provider,
                model or _env_model_override("DEEP_RESEARCH_MODEL", provider_explicit=provider_explicit),
                fast=False,
                role="orchestrator",
            ),
            fast_model=resolved_fast_model,
            planner_model=_resolve_role_model(
                resolved_provider,
                planner_model or _env_model_override("DEEP_RESEARCH_PLANNER_MODEL", provider_explicit=provider_explicit),
                fallback=resolved_fast_model,
                role="planner",
            ),
            researcher_model=_resolve_role_model(
                resolved_provider,
                researcher_model or _env_model_override("DEEP_RESEARCH_RESEARCHER_MODEL", provider_explicit=provider_explicit),
                fallback=resolved_fast_model,
                role="researcher",
            ),
            analyst_model=_resolve_role_model(
                resolved_provider,
                analyst_model or _env_model_override("DEEP_RESEARCH_ANALYST_MODEL", provider_explicit=provider_explicit),
                fallback=resolved_fast_model,
                role="analyst",
            ),
            verifier_model=_resolve_role_model(
                resolved_provider,
                verifier_model or _env_model_override("DEEP_RESEARCH_VERIFIER_MODEL", provider_explicit=provider_explicit),
                fallback=resolved_fast_model,
                role="verifier",
            ),
            judge_model=_resolve_role_model(
                resolved_provider,
                judge_model or _env_model_override("DEEP_RESEARCH_JUDGE_MODEL", provider_explicit=provider_explicit),
                fallback=resolved_fast_model,
                role="judge",
            ),
            synthesis_model=_resolve_role_model(
                resolved_provider,
                synthesis_model or _env_model_override("DEEP_RESEARCH_SYNTHESIS_MODEL", provider_explicit=provider_explicit),
                fallback="",
                role="synthesis",
            ),
            scrape_char_limit=scrape_char_limit
            or int(os.environ.get("DEEP_RESEARCH_SCRAPE_CHAR_LIMIT") or _default_scrape_limit(resolved_provider)),
            scrape_timeout_ms=scrape_timeout_ms
            if scrape_timeout_ms is not None
            else int(os.environ.get("DEEP_RESEARCH_SCRAPE_TIMEOUT_MS") or _default_scrape_timeout_ms(mode)),
            scrape_retries=scrape_retries
            if scrape_retries is not None
            else int(os.environ.get("DEEP_RESEARCH_SCRAPE_RETRIES") or _default_scrape_retries(mode)),
            max_browser_scrapes_per_query=max_browser_scrapes_per_query
            if max_browser_scrapes_per_query is not None
            else int(os.environ.get("DEEP_RESEARCH_MAX_BROWSER_SCRAPES_PER_QUERY") or "12"),
            blocked_source_patterns=blocked_source_patterns
            if blocked_source_patterns is not None
            else _split_env_list(os.environ.get("DEEP_RESEARCH_BLOCKED_SOURCE_PATTERNS", "")),
            tool_excerpt_char_limit=int(
                os.environ.get("DEEP_RESEARCH_TOOL_EXCERPT_CHAR_LIMIT") or _default_excerpt_limit(resolved_provider)
            ),
            precollect_sources=_resolve_bool(
                precollect_sources,
                os.environ.get("DEEP_RESEARCH_PRECOLLECT_SOURCES"),
                default=True,
            ),
            model_fallbacks=_resolve_bool(
                model_fallbacks,
                os.environ.get("DEEP_RESEARCH_MODEL_FALLBACKS"),
                default=True,
            ),
            provider_retry_attempts=provider_retry_attempts
            if provider_retry_attempts is not None
            else int(os.environ.get("DEEP_RESEARCH_PROVIDER_RETRY_ATTEMPTS") or "1"),
            provider_retry_max_wait_seconds=provider_retry_max_wait_seconds
            if provider_retry_max_wait_seconds is not None
            else int(os.environ.get("DEEP_RESEARCH_PROVIDER_RETRY_MAX_WAIT_SECONDS") or "60"),
            model_request_timeout_seconds=model_request_timeout_seconds
            if model_request_timeout_seconds is not None
            else int(os.environ.get("DEEP_RESEARCH_MODEL_REQUEST_TIMEOUT_SECONDS") or "120"),
            model_max_output_tokens=model_max_output_tokens
            if model_max_output_tokens is not None
            else int(os.environ.get("DEEP_RESEARCH_MODEL_MAX_OUTPUT_TOKENS") or "8192"),
            live=live,
            google_api_key=google_api_key,
            google_api_keys=google_api_keys,
            groq_api_key=groq_api_key,
            groq_api_keys=groq_api_keys,
            openrouter_api_key=openrouter_api_key,
            openrouter_api_keys=openrouter_api_keys,
            openrouter_http_referer=os.environ.get("OPENROUTER_HTTP_REFERER", "").strip(),
            openrouter_app_title=os.environ.get("OPENROUTER_APP_TITLE", "Deep Research Agent").strip()
            or "Deep Research Agent",
            tavily_api_key=tavily_api_key,
            tavily_api_keys=tavily_api_keys,
            mistral_api_key=mistral_api_key,
            mistral_api_keys=mistral_api_keys,
        )
        settings.validate()
        return settings

    @property
    def google_key_pool(self) -> tuple[str, ...]:
        return self.google_api_keys or ((self.google_api_key,) if self.google_api_key else ())

    @property
    def groq_key_pool(self) -> tuple[str, ...]:
        return self.groq_api_keys or ((self.groq_api_key,) if self.groq_api_key else ())

    @property
    def openrouter_key_pool(self) -> tuple[str, ...]:
        return self.openrouter_api_keys or ((self.openrouter_api_key,) if self.openrouter_api_key else ())

    @property
    def tavily_key_pool(self) -> tuple[str, ...]:
        return self.tavily_api_keys or ((self.tavily_api_key,) if self.tavily_api_key else ())

    @property
    def mistral_key_pool(self) -> tuple[str, ...]:
        return self.mistral_api_keys or ((self.mistral_api_key,) if self.mistral_api_key else ())

    def validate(self) -> None:
        missing = []
        if self.research_engine == "gemini_managed" and not self.google_key_pool:
            missing.append("GOOGLE_API_KEY")
        if self.research_engine == "local_langgraph":
            if self._uses_model_provider("google_genai") and not self.google_key_pool:
                missing.append("GOOGLE_API_KEY")
            if self._uses_model_provider("groq") and not self.groq_key_pool:
                missing.append("GROQ_API_KEY")
            if self._uses_model_provider("openrouter") and not self.openrouter_key_pool:
                missing.append("OPENROUTER_API_KEY")
            if not self.tavily_key_pool:
                missing.append("TAVILY_API_KEY")
        if missing:
            joined = ", ".join(missing)
            raise ConfigError(
                f"Missing required environment variable(s): {joined}. "
                "Set them in the shell or in .env at the project root."
            )
        if self.mode not in {"fast", "balanced", "max_quality"}:
            raise ConfigError(f"Unsupported mode: {self.mode}")
        if self.provider not in {"google", "groq", "hybrid", "ollama", "openrouter"}:
            raise ConfigError(f"Unsupported provider: {self.provider}")
        if self.research_engine not in {"local_langgraph", "gemini_managed", "openai_managed"}:
            raise ConfigError(f"Unsupported research engine: {self.research_engine}")
        if self.max_sources != 0 and self.max_sources < MINIMUM_SOURCE_TARGET:
            raise ConfigError(f"max_sources must be 0 for no explicit cap or at least {MINIMUM_SOURCE_TARGET}.")
        if self.max_rounds < 0:
            raise ConfigError("max_rounds must be zero or greater.")
        if self.provider_retry_attempts < 0:
            raise ConfigError("provider_retry_attempts must be zero or greater.")
        if self.provider_retry_max_wait_seconds < 0:
            raise ConfigError("provider_retry_max_wait_seconds must be zero or greater.")
        if self.model_request_timeout_seconds < 1:
            raise ConfigError("model_request_timeout_seconds must be at least 1.")
        if self.model_max_output_tokens < 512:
            raise ConfigError("model_max_output_tokens must be at least 512.")
        if self.scrape_char_limit < 1_000:
            raise ConfigError("scrape_char_limit must be at least 1000.")
        if self.scrape_timeout_ms < 1_000:
            raise ConfigError("scrape_timeout_ms must be at least 1000.")
        if self.scrape_retries < 1:
            raise ConfigError("scrape_retries must be at least 1.")
        if self.max_browser_scrapes_per_query < 0:
            raise ConfigError("max_browser_scrapes_per_query must be zero or greater.")
        for pattern in self.blocked_source_patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ConfigError(f"Invalid blocked source pattern {pattern!r}: {exc}") from exc
        if self.tool_excerpt_char_limit < 500:
            raise ConfigError("tool_excerpt_char_limit must be at least 500.")
        if self.min_usable_sources < MINIMUM_SOURCE_TARGET:
            raise ConfigError(f"min_usable_sources must be at least {MINIMUM_SOURCE_TARGET}.")
        if self.max_search_queries < 1:
            raise ConfigError("max_search_queries must be at least 1.")
        if self.max_candidates < self.min_usable_sources:
            raise ConfigError("max_candidates must be at least min_usable_sources.")
        if self.max_followup_queries_per_branch < 1:
            raise ConfigError("max_followup_queries_per_branch must be at least 1.")
        if self.min_source_words < 40:
            raise ConfigError("min_source_words must be at least 40.")
        if self.min_relevant_chunks < 1:
            raise ConfigError("min_relevant_chunks must be at least 1.")
        if self.semantic_evidence_max_llm_cards < 0:
            raise ConfigError("semantic_evidence_max_llm_cards must be zero or greater.")

    def _uses_model_provider(self, provider_prefix: str) -> bool:
        if self.provider == "google" and provider_prefix == "google_genai":
            return True
        if self.provider == "groq" and provider_prefix == "groq":
            return True
        if self.provider == "ollama" and provider_prefix == "ollama":
            return True
        if self.provider == "openrouter" and provider_prefix == "openrouter":
            return True
        models = (
            self.model,
            self.fast_model,
            self.planner_model,
            self.researcher_model,
            self.analyst_model,
            self.verifier_model,
            self.judge_model,
        )
        return any(model.startswith(f"{provider_prefix}:") for model in models)


def _mode_defaults(mode: Mode, provider: ResolvedProvider) -> tuple[int, int]:
    if provider in {"groq", "hybrid", "ollama", "openrouter"}:
        if mode == "fast":
            return 24, 2
        if mode == "max_quality":
            return 0, 6
        return 40, 4
    if mode == "fast":
        return 24, 2
    if mode == "max_quality":
        return 0, 6
    return 40, 4


def _depth_defaults(mode: Mode) -> dict[str, int]:
    if mode == "fast":
        return {
            "min_usable_sources": MINIMUM_SOURCE_TARGET,
            "max_search_queries": 16,
            "max_candidates": 160,
            "min_source_words": 180,
        }
    if mode == "max_quality":
        return {
            "min_usable_sources": 60,
            "max_search_queries": 192,
            "max_candidates": 5000,
            "min_source_words": 350,
        }
    return {
        "min_usable_sources": 24,
        "max_search_queries": 48,
        "max_candidates": 420,
        "min_source_words": 250,
    }


def _resolve_provider(
    provider: str,
    google_api_keys: Iterable[str],
    groq_api_keys: Iterable[str],
    openrouter_api_keys: Iterable[str],
) -> ResolvedProvider:
    normalized = provider.strip().lower()
    if normalized == "auto":
        has_google = bool(tuple(google_api_keys))
        has_groq = bool(tuple(groq_api_keys))
        has_openrouter = bool(tuple(openrouter_api_keys))
        if has_google and has_groq:
            return "hybrid"
        if has_groq:
            return "groq"
        if has_google:
            return "google"
        if has_openrouter:
            return "openrouter"
        return "google"
    if normalized in {"google", "groq", "hybrid", "ollama", "openrouter"}:
        return normalized  # type: ignore[return-value]
    raise ConfigError(f"Unsupported provider: {provider}")


def _resolve_model(
    provider: ResolvedProvider,
    model: str | None,
    *,
    fast: bool,
    role: str,
) -> str:
    chosen = model.strip() if model else _default_model(provider, fast=fast, role=role)
    if _has_provider_prefix(chosen):
        return chosen
    if provider == "hybrid":
        default_provider, _, _ = _default_model(provider, fast=fast, role=role).partition(":")
        prefix = default_provider
    elif provider == "ollama":
        prefix = "ollama"
    elif provider == "openrouter":
        prefix = "openrouter"
    else:
        prefix = "google_genai" if provider == "google" else "groq"
    return f"{prefix}:{chosen}"


def _resolve_role_model(
    provider: ResolvedProvider,
    model: str | None,
    *,
    fallback: str,
    role: str,
) -> str:
    if not model:
        if provider == "hybrid":
            return _default_model(provider, fast=True, role=role)
        return fallback
    return _resolve_model(provider, model, fast=True, role=role)


def _default_model(provider: ResolvedProvider, *, fast: bool, role: str) -> str:
    if provider == "ollama":
        return OLLAMA_DEFAULT_FAST_MODEL if fast else OLLAMA_DEFAULT_MODEL
    if provider == "openrouter":
        return OPENROUTER_DEFAULT_FAST_MODEL if fast else OPENROUTER_DEFAULT_MODEL
    if provider == "hybrid":
        return HYBRID_DEFAULT_MODELS.get(role, GROQ_DEFAULT_FAST_MODEL if fast else GROQ_DEFAULT_MODEL)
    if provider == "groq":
        return GROQ_DEFAULT_FAST_MODEL if fast else GROQ_DEFAULT_MODEL
    return GOOGLE_DEFAULT_FAST_MODEL if fast else GOOGLE_DEFAULT_MODEL


def _default_scrape_limit(provider: ResolvedProvider) -> int:
    return 6_000 if provider in {"groq", "hybrid", "ollama", "openrouter"} else 15_000


def _default_scrape_timeout_ms(mode: Mode) -> int:
    if mode == "fast":
        return 10_000
    if mode == "max_quality":
        return 20_000
    return 15_000


def _default_scrape_retries(mode: Mode) -> int:
    return 2 if mode == "max_quality" else 1


def _default_excerpt_limit(provider: ResolvedProvider) -> int:
    return 900 if provider in {"groq", "hybrid", "ollama", "openrouter"} else 2_500


def _has_provider_prefix(model_spec: str) -> bool:
    provider, separator, model_name = model_spec.partition(":")
    return bool(separator and model_name and provider in MODEL_PROVIDER_PREFIXES)


def _env_model_override(name: str, *, provider_explicit: bool) -> str | None:
    if provider_explicit:
        return None
    return os.environ.get(name)


def _collect_numbered_env_values(base_name: str, *, extra_names: Iterable[str] = ()) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    ordered_extra_names = []
    for name in extra_names:
        if name not in ordered_extra_names:
            ordered_extra_names.append(name)
    for name in [*_numbered_env_names(base_name), *_list_env_names(base_name), *ordered_extra_names]:
        raw_value = os.environ.get(name, "").strip()
        for value in _split_env_value_pool(raw_value):
            if value and value not in seen:
                values.append(value)
                seen.add(value)
    return tuple(values)


def _resolve_bool(value: bool | None, raw: str | None, *, default: bool) -> bool:
    if value is not None:
        return value
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"Unsupported boolean value: {raw}")


def _split_env_list(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(part.strip() for part in raw.split(";") if part.strip())


def _split_env_value_pool(raw: str) -> tuple[str, ...]:
    if not raw.strip():
        return ()
    return tuple(part.strip() for part in re.split(r"[;,\n]+", raw) if part.strip())


def _numbered_env_names(base_name: str) -> list[str]:
    indexed: list[tuple[int, str]] = []
    prefixes = (base_name, f"{base_name}_")
    for name in os.environ:
        if name == base_name:
            indexed.append((0, name))
            continue
        for prefix in prefixes:
            suffix = name.removeprefix(prefix)
            if suffix != name and suffix.isdigit():
                indexed.append((int(suffix), name))
                break
    return [name for _, name in sorted(indexed, key=lambda item: (item[0], item[1]))]


def _list_env_names(base_name: str) -> list[str]:
    names = []
    for candidate in (f"{base_name}S", f"{base_name}_POOL"):
        if candidate in os.environ:
            names.append(candidate)
    return names


def _semantic_api_key_env_names(service_name: str) -> tuple[str, ...]:
    service = service_name.upper()
    names = []
    for name in os.environ:
        normalized = name.upper()
        if service in normalized and "API" in normalized and "KEY" in normalized:
            names.append(name)
    return tuple(sorted(names, key=lambda name: (not name.upper().startswith(service), name.upper())))

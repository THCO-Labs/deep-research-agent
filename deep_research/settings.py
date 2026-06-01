from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

from dotenv import load_dotenv

Mode = Literal["fast", "balanced", "max_quality"]
Provider = Literal["auto", "google", "groq", "hybrid"]
ResolvedProvider = Literal["google", "groq", "hybrid"]

GOOGLE_DEFAULT_MODEL = "google_genai:gemini-2.5-flash"
GOOGLE_DEFAULT_FAST_MODEL = "google_genai:gemini-2.5-flash"
GROQ_DEFAULT_MODEL = "groq:openai/gpt-oss-20b"
GROQ_DEFAULT_FAST_MODEL = "groq:openai/gpt-oss-20b"
HYBRID_DEFAULT_MODELS = {
    "orchestrator": GROQ_DEFAULT_MODEL,
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
    mode: Mode = "balanced"
    out_dir: Path = Path("runs")
    max_sources: int = 12
    max_rounds: int = 2
    provider: ResolvedProvider = "google"
    model: str = GOOGLE_DEFAULT_MODEL
    fast_model: str = GOOGLE_DEFAULT_FAST_MODEL
    planner_model: str = GOOGLE_DEFAULT_FAST_MODEL
    researcher_model: str = GOOGLE_DEFAULT_FAST_MODEL
    analyst_model: str = GOOGLE_DEFAULT_FAST_MODEL
    verifier_model: str = GOOGLE_DEFAULT_FAST_MODEL
    judge_model: str = GOOGLE_DEFAULT_FAST_MODEL
    scrape_char_limit: int = 15_000
    tool_excerpt_char_limit: int = 2_500
    model_fallbacks: bool = True
    provider_retry_attempts: int = 1
    provider_retry_max_wait_seconds: int = 60
    live: bool = False
    google_api_key: str = field(default="", repr=False)
    google_api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    groq_api_key: str = field(default="", repr=False)
    groq_api_keys: tuple[str, ...] = field(default_factory=tuple, repr=False)
    tavily_api_key: str = field(default="", repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        project_root: Path | str | None = None,
        mode: Mode = "balanced",
        out_dir: Path | str | None = None,
        max_sources: int | None = None,
        max_rounds: int | None = None,
        provider: Provider | None = None,
        model: str | None = None,
        fast_model: str | None = None,
        planner_model: str | None = None,
        researcher_model: str | None = None,
        analyst_model: str | None = None,
        verifier_model: str | None = None,
        judge_model: str | None = None,
        scrape_char_limit: int | None = None,
        model_fallbacks: bool | None = None,
        provider_retry_attempts: int | None = None,
        provider_retry_max_wait_seconds: int | None = None,
        live: bool = False,
    ) -> "Settings":
        root = Path(project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env", override=False)

        resolved_out = Path(out_dir) if out_dir is not None else Path("runs")
        if not resolved_out.is_absolute():
            resolved_out = root / resolved_out

        google_api_keys = _collect_numbered_env_values("GOOGLE_API_KEY")
        groq_api_keys = _collect_numbered_env_values("GROQ_API_KEY")
        google_api_key = google_api_keys[0] if google_api_keys else ""
        groq_api_key = groq_api_keys[0] if groq_api_keys else ""
        tavily_api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        requested_provider = provider or os.environ.get("DEEP_RESEARCH_PROVIDER", "auto")
        resolved_provider = _resolve_provider(requested_provider, google_api_keys, groq_api_keys)
        mode_sources, mode_rounds = _mode_defaults(mode, resolved_provider)
        resolved_fast_model = _resolve_model(
            resolved_provider,
            fast_model or os.environ.get("DEEP_RESEARCH_FAST_MODEL"),
            fast=True,
            role="fast",
        )

        settings = cls(
            project_root=root,
            mode=mode,
            out_dir=resolved_out.resolve(),
            max_sources=max_sources if max_sources is not None else mode_sources,
            max_rounds=max_rounds if max_rounds is not None else mode_rounds,
            provider=resolved_provider,
            model=_resolve_model(
                resolved_provider,
                model or os.environ.get("DEEP_RESEARCH_MODEL"),
                fast=False,
                role="orchestrator",
            ),
            fast_model=resolved_fast_model,
            planner_model=_resolve_role_model(
                resolved_provider,
                planner_model or os.environ.get("DEEP_RESEARCH_PLANNER_MODEL"),
                fallback=resolved_fast_model,
                role="planner",
            ),
            researcher_model=_resolve_role_model(
                resolved_provider,
                researcher_model or os.environ.get("DEEP_RESEARCH_RESEARCHER_MODEL"),
                fallback=resolved_fast_model,
                role="researcher",
            ),
            analyst_model=_resolve_role_model(
                resolved_provider,
                analyst_model or os.environ.get("DEEP_RESEARCH_ANALYST_MODEL"),
                fallback=resolved_fast_model,
                role="analyst",
            ),
            verifier_model=_resolve_role_model(
                resolved_provider,
                verifier_model or os.environ.get("DEEP_RESEARCH_VERIFIER_MODEL"),
                fallback=resolved_fast_model,
                role="verifier",
            ),
            judge_model=_resolve_role_model(
                resolved_provider,
                judge_model or os.environ.get("DEEP_RESEARCH_JUDGE_MODEL"),
                fallback=resolved_fast_model,
                role="judge",
            ),
            scrape_char_limit=scrape_char_limit
            or int(os.environ.get("DEEP_RESEARCH_SCRAPE_CHAR_LIMIT") or _default_scrape_limit(resolved_provider)),
            tool_excerpt_char_limit=int(
                os.environ.get("DEEP_RESEARCH_TOOL_EXCERPT_CHAR_LIMIT") or _default_excerpt_limit(resolved_provider)
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
            live=live,
            google_api_key=google_api_key,
            google_api_keys=google_api_keys,
            groq_api_key=groq_api_key,
            groq_api_keys=groq_api_keys,
            tavily_api_key=tavily_api_key,
        )
        settings.validate()
        return settings

    @property
    def google_key_pool(self) -> tuple[str, ...]:
        return self.google_api_keys or ((self.google_api_key,) if self.google_api_key else ())

    @property
    def groq_key_pool(self) -> tuple[str, ...]:
        return self.groq_api_keys or ((self.groq_api_key,) if self.groq_api_key else ())

    def validate(self) -> None:
        missing = []
        if self._uses_model_provider("google_genai") and not self.google_key_pool:
            missing.append("GOOGLE_API_KEY")
        if self._uses_model_provider("groq") and not self.groq_key_pool:
            missing.append("GROQ_API_KEY")
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            joined = ", ".join(missing)
            raise ConfigError(
                f"Missing required environment variable(s): {joined}. "
                "Set them in the shell or in .env at the project root."
            )
        if self.mode not in {"fast", "balanced", "max_quality"}:
            raise ConfigError(f"Unsupported mode: {self.mode}")
        if self.provider not in {"google", "groq", "hybrid"}:
            raise ConfigError(f"Unsupported provider: {self.provider}")
        if self.max_sources < 1:
            raise ConfigError("max_sources must be at least 1.")
        if self.max_rounds < 0:
            raise ConfigError("max_rounds must be zero or greater.")
        if self.provider_retry_attempts < 0:
            raise ConfigError("provider_retry_attempts must be zero or greater.")
        if self.provider_retry_max_wait_seconds < 0:
            raise ConfigError("provider_retry_max_wait_seconds must be zero or greater.")
        if self.scrape_char_limit < 1_000:
            raise ConfigError("scrape_char_limit must be at least 1000.")
        if self.tool_excerpt_char_limit < 500:
            raise ConfigError("tool_excerpt_char_limit must be at least 500.")

    def _uses_model_provider(self, provider_prefix: str) -> bool:
        if self.provider == "google" and provider_prefix == "google_genai":
            return True
        if self.provider == "groq" and provider_prefix == "groq":
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
    if provider in {"groq", "hybrid"}:
        if mode == "fast":
            return 2, 1
        if mode == "max_quality":
            return 5, 2
        return 3, 1
    if mode == "fast":
        return 6, 1
    if mode == "max_quality":
        return 24, 3
    return 12, 2


def _resolve_provider(
    provider: str,
    google_api_keys: Iterable[str],
    groq_api_keys: Iterable[str],
) -> ResolvedProvider:
    normalized = provider.strip().lower()
    if normalized == "auto":
        has_google = bool(tuple(google_api_keys))
        has_groq = bool(tuple(groq_api_keys))
        if has_google and has_groq:
            return "hybrid"
        return "groq" if has_groq else "google"
    if normalized in {"google", "groq", "hybrid"}:
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
    if ":" in chosen:
        return chosen
    if provider == "hybrid":
        default_provider, _, _ = _default_model(provider, fast=fast, role=role).partition(":")
        prefix = default_provider
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
    if provider == "hybrid":
        return HYBRID_DEFAULT_MODELS.get(role, GROQ_DEFAULT_FAST_MODEL if fast else GROQ_DEFAULT_MODEL)
    if provider == "groq":
        return GROQ_DEFAULT_FAST_MODEL if fast else GROQ_DEFAULT_MODEL
    return GOOGLE_DEFAULT_FAST_MODEL if fast else GOOGLE_DEFAULT_MODEL


def _default_scrape_limit(provider: ResolvedProvider) -> int:
    return 6_000 if provider in {"groq", "hybrid"} else 15_000


def _default_excerpt_limit(provider: ResolvedProvider) -> int:
    return 900 if provider in {"groq", "hybrid"} else 2_500


def _collect_numbered_env_values(base_name: str) -> tuple[str, ...]:
    values: list[str] = []
    seen: set[str] = set()
    for name in _numbered_env_names(base_name):
        value = os.environ.get(name, "").strip()
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

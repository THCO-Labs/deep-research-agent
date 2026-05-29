from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Mode = Literal["fast", "balanced", "max_quality"]
Provider = Literal["auto", "google", "groq"]
ResolvedProvider = Literal["google", "groq"]

GOOGLE_DEFAULT_MODEL = "google_genai:gemini-2.5-flash"
GOOGLE_DEFAULT_FAST_MODEL = "google_genai:gemini-2.5-flash"
GROQ_DEFAULT_MODEL = "groq:openai/gpt-oss-20b"
GROQ_DEFAULT_FAST_MODEL = "groq:openai/gpt-oss-20b"


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
    scrape_char_limit: int = 15_000
    tool_excerpt_char_limit: int = 2_500
    live: bool = False
    google_api_key: str = field(default="", repr=False)
    groq_api_key: str = field(default="", repr=False)
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
        scrape_char_limit: int | None = None,
        live: bool = False,
    ) -> "Settings":
        root = Path(project_root or Path.cwd()).resolve()
        load_dotenv(root / ".env", override=False)

        mode_sources, mode_rounds = _mode_defaults(mode)
        resolved_out = Path(out_dir) if out_dir is not None else Path("runs")
        if not resolved_out.is_absolute():
            resolved_out = root / resolved_out

        google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
        tavily_api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        requested_provider = provider or os.environ.get("DEEP_RESEARCH_PROVIDER", "auto")
        resolved_provider = _resolve_provider(requested_provider, google_api_key, groq_api_key)

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
            ),
            fast_model=_resolve_model(
                resolved_provider,
                fast_model or os.environ.get("DEEP_RESEARCH_FAST_MODEL"),
                fast=True,
            ),
            scrape_char_limit=scrape_char_limit
            or int(os.environ.get("DEEP_RESEARCH_SCRAPE_CHAR_LIMIT") or _default_scrape_limit(resolved_provider)),
            tool_excerpt_char_limit=int(
                os.environ.get("DEEP_RESEARCH_TOOL_EXCERPT_CHAR_LIMIT") or _default_excerpt_limit(resolved_provider)
            ),
            live=live,
            google_api_key=google_api_key,
            groq_api_key=groq_api_key,
            tavily_api_key=tavily_api_key,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        missing = []
        if self.provider == "google" and not self.google_api_key:
            missing.append("GOOGLE_API_KEY")
        if self.provider == "groq" and not self.groq_api_key:
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
        if self.provider not in {"google", "groq"}:
            raise ConfigError(f"Unsupported provider: {self.provider}")
        if self.max_sources < 1:
            raise ConfigError("max_sources must be at least 1.")
        if self.max_rounds < 0:
            raise ConfigError("max_rounds must be zero or greater.")
        if self.scrape_char_limit < 1_000:
            raise ConfigError("scrape_char_limit must be at least 1000.")
        if self.tool_excerpt_char_limit < 500:
            raise ConfigError("tool_excerpt_char_limit must be at least 500.")


def _mode_defaults(mode: Mode) -> tuple[int, int]:
    if mode == "fast":
        return 6, 1
    if mode == "max_quality":
        return 24, 3
    return 12, 2


def _resolve_provider(
    provider: str,
    google_api_key: str,
    groq_api_key: str,
) -> ResolvedProvider:
    normalized = provider.strip().lower()
    if normalized == "auto":
        return "groq" if groq_api_key else "google"
    if normalized in {"google", "groq"}:
        return normalized  # type: ignore[return-value]
    raise ConfigError(f"Unsupported provider: {provider}")


def _resolve_model(provider: ResolvedProvider, model: str | None, *, fast: bool) -> str:
    chosen = model.strip() if model else _default_model(provider, fast=fast)
    if ":" in chosen:
        return chosen
    prefix = "google_genai" if provider == "google" else "groq"
    return f"{prefix}:{chosen}"


def _default_model(provider: ResolvedProvider, *, fast: bool) -> str:
    if provider == "groq":
        return GROQ_DEFAULT_FAST_MODEL if fast else GROQ_DEFAULT_MODEL
    return GOOGLE_DEFAULT_FAST_MODEL if fast else GOOGLE_DEFAULT_MODEL


def _default_scrape_limit(provider: ResolvedProvider) -> int:
    return 6_000 if provider == "groq" else 15_000


def _default_excerpt_limit(provider: ResolvedProvider) -> int:
    return 1_500 if provider == "groq" else 2_500

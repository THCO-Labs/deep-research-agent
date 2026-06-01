from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from deep_research.settings import Settings

MANIFEST_SCHEMA_VERSION = 1

PACKAGE_NAMES = (
    "deep-research-agent",
    "deepagents",
    "langchain",
    "langchain-google-genai",
    "langchain-groq",
    "playwright",
    "tavily-python",
)


def build_run_manifest(
    *,
    question: str,
    settings: Settings,
    run_dir: Path,
    model_routes: dict[str, object],
    progress_mode: str,
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "question": question.strip(),
        "progress_mode": progress_mode,
        "settings": redacted_settings(settings),
        "model_routes": model_routes,
        "runtime": runtime_metadata(),
    }


def redacted_settings(settings: Settings) -> dict[str, Any]:
    return {
        "project_root": str(settings.project_root),
        "out_dir": str(settings.out_dir),
        "mode": settings.mode,
        "provider": settings.provider,
        "model": settings.model,
        "fast_model": settings.fast_model,
        "planner_model": settings.planner_model,
        "researcher_model": settings.researcher_model,
        "analyst_model": settings.analyst_model,
        "verifier_model": settings.verifier_model,
        "judge_model": settings.judge_model,
        "max_sources": settings.max_sources,
        "max_rounds": settings.max_rounds,
        "scrape_char_limit": settings.scrape_char_limit,
        "tool_excerpt_char_limit": settings.tool_excerpt_char_limit,
        "model_fallbacks": settings.model_fallbacks,
        "provider_retry_attempts": settings.provider_retry_attempts,
        "provider_retry_max_wait_seconds": settings.provider_retry_max_wait_seconds,
        "live": settings.live,
        "google_key_count": len(settings.google_key_pool),
        "groq_key_count": len(settings.groq_key_pool),
        "tavily_api_key_present": bool(settings.tavily_api_key),
    }


def runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in PACKAGE_NAMES},
    }


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"

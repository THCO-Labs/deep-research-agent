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
    "langchain-ollama",
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
        "research_engine": settings.research_engine,
        "min_usable_sources": settings.min_usable_sources,
        "max_search_queries": settings.max_search_queries,
        "max_candidates": settings.max_candidates,
        "min_source_words": settings.min_source_words,
        "min_relevant_chunks": settings.min_relevant_chunks,
        "search_depth": settings.search_depth,
        "allow_raw_content": settings.allow_raw_content,
        "semantic_verification": settings.semantic_verification,
        "llm_planning": settings.llm_planning,
        "report_quality_gate": settings.report_quality_gate,
        "llm_synthesis": settings.llm_synthesis,
        "allow_failed_verification": settings.allow_failed_verification,
        "strict_tool_models": settings.strict_tool_models,
        "local_input_count": len(settings.local_input_paths),
        "mcp_manifest_present": bool(settings.mcp_manifest),
        "scrape_char_limit": settings.scrape_char_limit,
        "tool_excerpt_char_limit": settings.tool_excerpt_char_limit,
        "precollect_sources": settings.precollect_sources,
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

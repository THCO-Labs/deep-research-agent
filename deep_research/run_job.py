"""Container App Job entry point.

Reads job parameters from JOB_INPUT env var (JSON), calls run_research,
and writes the result to the shared file-system run directory so the
read-only API server can pick it up.

Expected JOB_INPUT JSON shape:
{
    "job_id": "job_abc123",
    "question": "...",
    "mode": "max_quality",
    "engine": "local_langgraph",
    ...  (all ResearchRequest fields)
}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from deep_research.agent import ResearchRunError, run_research
from deep_research.core.settings import Settings


def main() -> int:
    raw = os.environ.get("JOB_INPUT", "")
    if not raw:
        print("ERROR: JOB_INPUT environment variable is empty or not set.", file=sys.stderr)
        return 1

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: JOB_INPUT is not valid JSON: {exc}", file=sys.stderr)
        return 1

    job_id = payload.get("job_id", "")
    question = payload.get("question", "")
    if not job_id or not question:
        print("ERROR: JOB_INPUT must contain 'job_id' and 'question'.", file=sys.stderr)
        return 1

    runs_dir = Path(os.environ.get("RUNS_DIR", "/mnt/runs")).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_dir = runs_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the job input for the read API to discover.
    job_file = run_dir / "job.json"
    job_file.write_text(json.dumps({"job_id": job_id, "status": "running", **payload}, indent=2), encoding="utf-8")

    # Build Settings from payload — every field mirrors the ResearchRequest model.
    settings_kwargs: dict = {
        "out_dir": str(run_dir),
        "mode": payload.get("mode", "max_quality"),
        "research_engine": payload.get("engine", "local_langgraph"),
    }

    scalar_keys = [
        "max_sources", "max_rounds", "min_usable_sources", "max_search_queries",
        "max_candidates", "max_followup_queries_per_branch", "min_source_words",
        "mcp_manifest", "provider", "model", "fast_model", "planner_model",
        "researcher_model", "analyst_model", "verifier_model", "judge_model",
        "scrape_char_limit", "scrape_timeout_ms", "scrape_retries",
        "max_browser_scrapes_per_query", "provider_retry_attempts",
        "provider_retry_max_wait_seconds", "model_request_timeout_seconds",
        "model_max_output_tokens", "semantic_evidence_max_llm_cards",
        "allow_failed_verification",
    ]
    for key in scalar_keys:
        val = payload.get(key)
        if val is not None:
            settings_kwargs[key] = val

    if payload.get("synthesis_model"):
        settings_kwargs["model"] = payload["synthesis_model"]
    if payload.get("citation_model"):
        settings_kwargs["analyst_model"] = payload["citation_model"]
    if payload.get("input"):
        settings_kwargs["local_input_paths"] = tuple(payload["input"])

    bool_invert = {
        "no_model_fallbacks": "model_fallbacks",
        "no_llm_planning": "llm_planning",
        "no_llm_synthesis": "llm_synthesis",
        "no_semantic_verification": "semantic_verification",
    }
    for payload_key, settings_key in bool_invert.items():
        val = payload.get(payload_key)
        if val is not None:
            settings_kwargs[settings_key] = not val

    if payload.get("allow_weak_tool_models") is not None:
        settings_kwargs["allow_weak_tool_models"] = payload["allow_weak_tool_models"]

    settings = Settings(**settings_kwargs)

    writing_guidance = payload.get("writing_guidance", "")

    try:
        result = run_research(
            question=question,
            settings=settings,
            progress_mode="quiet",
            writing_guidance=writing_guidance,
            run_dir=run_dir,
        )
    except ResearchRunError as exc:
        result_payload = {
            "run_dir": str(run_dir),
            "report_path": str(exc.result.report_path) if exc.result else "N/A",
            "verification_path": str(exc.result.verification_path) if exc.result else "N/A",
            "metrics_path": str(exc.result.metrics_path) if exc.result else "N/A",
        }
        job_file.write_text(json.dumps({
            "job_id": job_id, "status": "failed", "error": str(exc),
            "result": result_payload, **payload,
        }, indent=2), encoding="utf-8")
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        job_file.write_text(json.dumps({
            "job_id": job_id, "status": "failed", "error": str(exc), **payload,
        }, indent=2), encoding="utf-8")
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    # Mark complete.
    result_payload = {
        "run_dir": str(result.run_dir),
        "report_path": str(result.report_path),
        "verification_path": str(result.verification_path),
        "metrics_path": str(result.metrics_path),
    }
    job_file.write_text(json.dumps({
        "job_id": job_id, "status": "completed", "result": result_payload, **payload,
    }, indent=2), encoding="utf-8")
    print(f"DONE: report written to {result.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

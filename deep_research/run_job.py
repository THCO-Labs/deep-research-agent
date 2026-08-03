"""Queue-triggered Container App worker.

Polls an Azure Storage Queue for research jobs, processes ONE message
to completion, and exits.  The Container App scale rule spawns a new
replica for each message; the replica dies after finishing, so failures
don't wedge a persistent process.

Required env vars:
    STORAGE_CONNECTION_STRING  –  Azure Storage account connection string
    QUEUE_NAME                 –  queue name (default: research-jobs)
    RUNS_DIR                   –  shared AzureFile mount (default: /mnt/runs)
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path

from azure.core.exceptions import ResourceNotFoundError
try:
    from azure.storage.queue import QueueClient, QueueMessage
except ImportError:
    print("ERROR: azure-storage-queue package is not installed.", file=sys.stderr)
    raise


# ── helpers ────────────────────────────────────────────────────────────────


def _decode_message(msg: QueueMessage) -> dict:
    """Decode a queue message, handling both base64 and plain-text."""
    raw = msg.content
    try:
        decoded = base64.b64decode(raw).decode("utf-8")
    except Exception:
        decoded = raw
    return json.loads(decoded)


# ── main ───────────────────────────────────────────────────────────────────


def main() -> int:
    conn_str = os.environ.get("STORAGE_CONNECTION_STRING", "")
    queue_name = os.environ.get("QUEUE_NAME", "research-jobs")
    runs_dir = Path(os.environ.get("RUNS_DIR", "/mnt/runs")).resolve()

    if not conn_str:
        print("ERROR: STORAGE_CONNECTION_STRING is not set.", file=sys.stderr)
        return 1

    # Idempotent — creates if not present, no-op otherwise.
    client: QueueClient = QueueClient.from_connection_string(conn_str, queue_name)
    try:
        client.create_queue()
    except Exception:
        pass  # may already exist

    # Visibility timeout: how long before an un-deleted message re-appears.
    # Set high enough that a full research run won't expire before completion.
    visibility_timeout = 3600  # 1 hour

    # Poll with a 30-second window.  If no message arrives within
    # ~90 seconds, the Container App scales this replica to zero.
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        messages = client.receive_messages(
            max_messages=1,
            visibility_timeout=visibility_timeout,
        )
        for msg in messages:
            try:
                payload = _decode_message(msg)
            except Exception as exc:
                print(f"WARNING: unreadable message — deleting. Error: {exc}", file=sys.stderr)
                client.delete_message(msg)
                continue

            job_id = payload.get("job_id", f"job_{uuid.uuid4().hex[:8]}")
            question = payload.get("question", "")
            if not question:
                print(f"WARNING: message missing 'question' — deleting.", file=sys.stderr)
                client.delete_message(msg)
                continue

            # ── process the job ────────────────────────────────────────
            runs_dir.mkdir(parents=True, exist_ok=True)
            run_dir = runs_dir / job_id
            run_dir.mkdir(parents=True, exist_ok=True)

            # Write job.json so the API can see it
            job_file = run_dir / "job.json"
            job_file.write_text(
                json.dumps({"job_id": job_id, "status": "running", **payload}, indent=2),
                encoding="utf-8",
            )

            success = _run_research(job_id, question, payload, run_dir, job_file)

            # Delete the message on success OR failure (we already persisted
            # the result to disk).  On a hard crash (SIGKILL), the message
            # becomes visible again after the visibility timeout.
            client.delete_message(msg)

            # One job per replica — exit so the next message gets a fresh
            # replica.
            return 0 if success else 1

        time.sleep(5)

    print("No messages received within poll window — exiting.", file=sys.stderr)
    return 0


def _run_research(
    job_id: str,
    question: str,
    payload: dict,
    run_dir: Path,
    job_file: Path,
) -> bool:
    """Run a single research job. Returns True on success."""
    try:
        from deep_research.agent import ResearchRunError, run_research
        from deep_research.core.settings import Settings
    except Exception as exc:
        job_file.write_text(json.dumps({
            "job_id": job_id, "status": "failed",
            "error": f"Import failed: {exc}", **payload,
        }, indent=2), encoding="utf-8")
        print(f"FATAL: cannot import deep_research — {exc}", file=sys.stderr)
        return False

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
        return False
    except Exception as exc:
        job_file.write_text(json.dumps({
            "job_id": job_id, "status": "failed", "error": str(exc), **payload,
        }, indent=2), encoding="utf-8")
        print(f"FAILED: {exc}", file=sys.stderr)
        return False

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
    return True


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only REST API for deep-research results.

Research execution is delegated to a queue-triggered Container App worker.
Both API and worker share the AzureFile mount at /mnt/runs.

POST /v1/research  -- push job to Azure Storage Queue (worker picks it up)
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from deep_research.core.settings import Mode, Provider, ResearchEngineName

app = FastAPI(
    title="Deep Research API Server",
    description="REST API server for Deep Research agent and Deep Research Bench evaluation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=***,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict[str, Any]] = {}

# --- Azure Container App Job trigger configuration ---
# ── Queue config (shared with the worker Container App) ──────────────
STORAGE_CONNECTION_STRING = os.environ.get("STORAGE_CONNECTION_STRING", "")
QUEUE_NAME = os.environ.get("QUEUE_NAME", "research-jobs")


class ResearchRequest(BaseModel):
    question: str = Field(..., description="Research query or prompt.")
    mode: Mode = Field("max_quality", description="Research mode: fast, balanced, max_quality.")
    engine: ResearchEngineName = Field("local_langgraph", description="Research engine: local_langgraph, gemini_managed, openai_managed.")
    max_sources: Optional[int] = Field(None, description="Maximum total sources.")
    max_rounds: Optional[int] = Field(None, description="Maximum search rounds.")
    min_usable_sources: Optional[int] = Field(None, description="Minimum usable sources target.")
    max_search_queries: Optional[int] = Field(None, description="Maximum search queries limit.")
    max_candidates: Optional[int] = Field(None, description="Maximum candidate pages considered.")
    max_followup_queries_per_branch: Optional[int] = Field(None, description="Maximum follow-up queries per branch.")
    min_source_words: Optional[int] = Field(None, description="Minimum words per source document.")
    input: list[str] = Field(default_factory=list, description="Local input files or directories to ingest.")
    mcp_manifest: Optional[str] = Field(None, description="Path to MCP manifest JSON.")
    provider: Optional[Provider] = Field(None, description="Model provider (auto, google, groq, hybrid, ollama, openrouter).")
    model: Optional[str] = Field(None, description="Primary model spec.")
    fast_model: Optional[str] = Field(None, description="Fast model spec.")
    planner_model: Optional[str] = Field(None, description="Planner model spec.")
    researcher_model: Optional[str] = Field(None, description="Researcher model spec.")
    analyst_model: Optional[str] = Field(None, description="Analyst model spec.")
    synthesis_model: Optional[str] = Field(None, description="Synthesis model spec for final report drafting.")
    citation_model: Optional[str] = Field(None, description="Citation model spec for attaching inline citations.")
    verifier_model: Optional[str] = Field(None, description="Verifier model spec.")
    judge_model: Optional[str] = Field(None, description="Judge model spec.")
    scrape_char_limit: Optional[int] = Field(None, description="Character limit per scrape.")
    scrape_timeout_ms: Optional[int] = Field(None, description="Scrape timeout in milliseconds.")
    scrape_retries: Optional[int] = Field(None, description="Retry count for failed scrapes.")
    max_browser_scrapes_per_query: Optional[int] = Field(None, description="Max browser fallback scrapes per query.")
    no_model_fallbacks: Optional[bool] = Field(None, description="Disable automatic model provider fallbacks.")
    provider_retry_attempts: Optional[int] = Field(None, description="Provider retry attempts.")
    provider_retry_max_wait_seconds: Optional[int] = Field(None, description="Max wait time between provider retries.")
    model_request_timeout_seconds: Optional[int] = Field(None, description="Individual LLM request timeout.")
    model_max_output_tokens: Optional[int] = Field(None, description="LLM max output tokens limit.")
    no_llm_planning: Optional[bool] = Field(None, description="Disable LLM planning stage.")
    no_llm_synthesis: Optional[bool] = Field(None, description="Disable LLM synthesis stage.")
    no_semantic_verification: Optional[bool] = Field(None, description="Disable semantic verification.")
    semantic_evidence_max_llm_cards: Optional[int] = Field(None, description="Max evidence cards evaluated by LLM verifier.")
    allow_failed_verification: Optional[bool] = Field(None, description="Allow report generation even if verification fails.")
    allow_weak_tool_models: Optional[bool] = Field(None, description="Opt out of strict tool-model policy.")
    writing_guidance: str = Field("", description="Optional guidance or checklist for synthesis/writing stage.")
    async_mode: bool = Field(True, description="Always async — the job runs externally.")


def _get_runs_dir() -> Path:
    env_dir = os.environ.get("RUNS_DIR", "/mnt/runs")
    path = Path(env_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_run_dir(job_id: str) -> Optional[Path]:
    if job_id in JOBS and JOBS[job_id].get("run_dir"):
        p = Path(JOBS[job_id]["run_dir"])
        if p.exists():
            return p
    runs_dir = _get_runs_dir()
    candidate = runs_dir / job_id
    if candidate.exists() and candidate.is_dir():
        return candidate
    # Scan runs_dir for any directory containing job_id in manifest.json or job.json
    dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        for fname in ("manifest.json", "job.json"):
            fpath = d / fname
            if fpath.exists():
                try:
                    ftext = fpath.read_text(encoding="utf-8")
                    if job_id in ftext:
                        return d
                except Exception:
                    pass
    return None


def _push_to_queue(job_id: str, payload: dict) -> bool:
    """Push a job payload to the Azure Storage Queue.

    The worker Container App picks it up and runs the research.
    This is fire-and-forget — the worker writes completion status to disk.
    """
    if not STORAGE_CONNECTION_STRING:
        print(
            f"WARNING: STORAGE_CONNECTION_STRING not set. "
            f"Job {job_id} queued to disk but not triggered.",
            flush=True,
        )
        return False

    try:
        from azure.storage.queue import QueueClient
        client = QueueClient.from_connection_string(STORAGE_CONNECTION_STRING, QUEUE_NAME)
        # Idempotent create — no-op if the queue already exists.
        try:
            client.create_queue()
        except Exception:
            pass
        msg_str = json.dumps(payload)
        client.send_message(msg_str)
        print(f"Queue message sent for {job_id}", flush=True)
        return True
    except Exception as exc:
        print(f"WARNING: Failed to push queue message: {exc}", flush=True)
        return False

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok", "service": "deep-research-api", "runs_dir": str(_get_runs_dir())}


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {"message": "Deep Research API Server is operational.", "docs": "/docs"}


@app.post("/v1/research", status_code=status.HTTP_202_ACCEPTED)
async def submit_research(req: ResearchRequest):
    """Queue a research job and trigger the Container App Job to execute it."""
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    runs_dir = _get_runs_dir()
    run_dir = runs_dir / job_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {"job_id": job_id, "question": req.question}
    for key in ResearchRequest.model_fields:
        if key in ("question",):
            continue
        val = getattr(req, key, None)
        if val is not None and val != [] and val != "":
            payload[key] = val

    # Write job.json so the read API can discover it even before the job starts.
    (run_dir / "job.json").write_text(json.dumps({"job_id": job_id, "status": "queued", **payload}, indent=2), encoding="utf-8")

    # Trigger the job (fire-and-forget — job writes its own completion status).
    triggered = _push_to_queue(job_id, payload)

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "question": req.question,
        "run_dir": str(run_dir),
        "error": None,
        "result": None,
        "triggered": triggered,
    }

    return {
        "job_id": job_id,
        "status": "queued",
        "triggered": triggered,
        "status_url": f"/v1/research/{job_id}",
        "report_url": f"/v1/research/{job_id}/report",
    }


@app.get("/v1/research/{job_id}")
def get_job_status(job_id: str):
    """Read job status and recent activity from the shared file system."""
    # In-memory fallback
    if job_id in JOBS:
        info = JOBS[job_id]
        run_dir_path = _resolve_run_dir(job_id)
        run_dir_str = str(run_dir_path) if run_dir_path else info.get("run_dir")
        report_exists = run_dir_path is not None and (run_dir_path / "report.md").exists()
        activity_events = _read_activity(run_dir_path)
        return {
            "job_id": job_id,
            "status": info["status"],
            "question": info["question"],
            "run_dir": run_dir_str,
            "error": info.get("error"),
            "report_available": report_exists,
            "recent_activity": activity_events,
            "result": info.get("result"),
        }

    # Fallback: read from disk (survives server restarts)
    run_dir_path = _resolve_run_dir(job_id)
    if run_dir_path:
        return _status_from_disk(job_id, run_dir_path)

    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


@app.get("/v1/research/{job_id}/activity")
def get_job_activity(job_id: str):
    run_dir_path = _resolve_run_dir(job_id)
    if not run_dir_path:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {"job_id": job_id, "events": _read_activity(run_dir_path)}


@app.get("/v1/research/{job_id}/reports")
def get_all_job_reports(job_id: str):
    run_dir_path = _resolve_run_dir(job_id)
    if not run_dir_path or not run_dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or run directory not yet initialized.")

    report_files = ["report.md", "best_draft.md", "failed_report.md", "draft_report.md"]
    for p in sorted(run_dir_path.glob("draft_report_*.md")):
        report_files.append(p.name)

    reports = {}
    for filename in report_files:
        p = run_dir_path / filename
        if p.exists():
            reports[filename] = {
                "size_bytes": p.stat().st_size,
                "url": f"/v1/research/{job_id}/report/{filename}",
                "content": p.read_text(encoding="utf-8"),
            }

    return {"job_id": job_id, "reports": reports}


@app.get("/v1/research/{job_id}/report", response_class=PlainTextResponse)
@app.get("/v1/research/{job_id}/report/{variant}", response_class=PlainTextResponse)
def get_job_report(job_id: str, variant: str = "report.md"):
    run_dir_path = _resolve_run_dir(job_id)
    if not run_dir_path:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    filename = variant if variant.endswith(".md") else f"{variant}.md"
    report_path = run_dir_path / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report variant '{filename}' for job '{job_id}' does not exist.")

    return report_path.read_text(encoding="utf-8")


@app.get("/v1/research/{job_id}/artifacts")
def get_job_artifacts(job_id: str):
    run_dir_path = _resolve_run_dir(job_id)
    if not run_dir_path:
        raise HTTPException(status_code=404, detail=f"Run artifacts directory for '{job_id}' not found.")

    artifacts = {}
    for item in run_dir_path.iterdir():
        if item.is_file():
            artifacts[item.name] = {
                "size_bytes": item.stat().st_size,
                "url": f"/v1/research/{job_id}/artifact/{item.name}",
            }
    return {"job_id": job_id, "run_dir": str(run_dir_path), "artifacts": artifacts}


@app.get("/v1/research/{job_id}/artifact/{filename}")
def download_artifact(job_id: str, filename: str):
    run_dir_path = _resolve_run_dir(job_id)
    if not run_dir_path:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    file_path = (run_dir_path / filename).resolve()
    if not file_path.exists() or not str(file_path).startswith(str(run_dir_path.resolve())):
        raise HTTPException(status_code=404, detail=f"Artifact '{filename}' not found.")

    return FileResponse(path=file_path, filename=filename)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_activity(run_dir_path: Optional[Path]) -> list[dict]:
    if not run_dir_path:
        return []
    act_file = run_dir_path / "activity.jsonl"
    if not act_file.exists():
        return []
    try:
        lines = act_file.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line) for line in lines[-20:] if line.strip()]
    except Exception:
        return []


def _status_from_disk(job_id: str, run_dir_path: Path) -> dict:
    report_file = run_dir_path / "report.md"
    act_file = run_dir_path / "activity.jsonl"
    job_file = run_dir_path / "job.json"
    manifest_file = run_dir_path / "manifest.json"

    question = ""
    status_str = "running" if act_file.exists() else "queued"
    error_msg = None
    result_payload = None

    if job_file.exists():
        try:
            jdata = json.loads(job_file.read_text(encoding="utf-8"))
            status_str = jdata.get("status", status_str)
            question = jdata.get("question", question)
            error_msg = jdata.get("error")
            result_payload = jdata.get("result")
        except Exception:
            pass

    if not question and manifest_file.exists():
        try:
            mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
            question = mdata.get("question", "")
        except Exception:
            pass

    if report_file.exists():
        status_str = "completed"

    return {
        "job_id": job_id,
        "status": status_str,
        "question": question,
        "run_dir": str(run_dir_path),
        "error": error_msg,
        "report_available": report_file.exists(),
        "recent_activity": _read_activity(run_dir_path),
        "result": result_payload,
    }

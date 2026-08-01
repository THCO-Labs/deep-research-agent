from __future__ import annotations

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from deep_research.agent import ResearchRunError, ResearchRunResult, run_research
from deep_research.bench.deepresearch_bench import _criteria_guidance
from deep_research.core.settings import Mode, Provider, ResearchEngineName, Settings

app = FastAPI(
    title="Deep Research API Server",
    description="REST API server for Deep Research agent and Deep Research Bench evaluation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=int(os.environ.get("MAX_WORKERS", "20")))
JOBS: Dict[str, Dict[str, Any]] = {}


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
    input: List[str] = Field(default_factory=list, description="Local input files or directories to ingest.")
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
    async_mode: bool = Field(True, description="If True, returns immediately with job_id. If False, waits for run completion.")


class ResearchJobStatus(BaseModel):
    job_id: str
    status: str  # queued, running, completed, failed
    question: str
    run_dir: Optional[str] = None
    error: Optional[str] = None
    report_available: bool = False


def _get_runs_dir() -> Path:
    env_dir = os.environ.get("RUNS_DIR", "runs")
    path = Path(env_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _enabled_unless_disabled(flag_value: Optional[bool]) -> Optional[bool]:
    return None if flag_value is None else not flag_value


def _execute_research_job(job_id: str, request_data: ResearchRequest):
    JOBS[job_id]["status"] = "running"
    base_runs_dir = _get_runs_dir()
    job_run_dir = base_runs_dir / job_id
    job_run_dir.mkdir(parents=True, exist_ok=True)
    JOBS[job_id]["run_dir"] = str(job_run_dir)

    settings_kwargs: Dict[str, Any] = {
        "out_dir": job_run_dir,
        "mode": request_data.mode,
        "research_engine": request_data.engine,
    }
    
    # Map all CLI options directly to Settings kwargs
    fields_to_map = [
        "max_sources", "max_rounds", "min_usable_sources", "max_search_queries",
        "max_candidates", "max_followup_queries_per_branch", "min_source_words",
        "mcp_manifest", "provider", "model", "fast_model", "planner_model",
        "researcher_model", "analyst_model", "verifier_model", "judge_model",
        "scrape_char_limit", "scrape_timeout_ms", "scrape_retries",
        "max_browser_scrapes_per_query", "provider_retry_attempts",
        "provider_retry_max_wait_seconds", "model_request_timeout_seconds",
        "model_max_output_tokens", "semantic_evidence_max_llm_cards",
        "allow_failed_verification"
    ]
    for key in fields_to_map:
        val = getattr(request_data, key, None)
        if val is not None:
            settings_kwargs[key] = val

    if request_data.synthesis_model:
        settings_kwargs["model"] = request_data.synthesis_model
    if request_data.citation_model:
        settings_kwargs["analyst_model"] = request_data.citation_model

    if request_data.input:
        settings_kwargs["local_input_paths"] = tuple(request_data.input)

    if request_data.no_model_fallbacks is not None:
        settings_kwargs["model_fallbacks"] = _enabled_unless_disabled(request_data.no_model_fallbacks)
    if request_data.no_llm_planning is not None:
        settings_kwargs["llm_planning"] = _enabled_unless_disabled(request_data.no_llm_planning)
    if request_data.no_llm_synthesis is not None:
        settings_kwargs["llm_synthesis"] = _enabled_unless_disabled(request_data.no_llm_synthesis)
    if request_data.no_semantic_verification is not None:
        settings_kwargs["semantic_verification"] = _enabled_unless_disabled(request_data.no_semantic_verification)
    if request_data.allow_weak_tool_models is not None:
        settings_kwargs["strict_tool_models"] = _enabled_unless_disabled(request_data.allow_weak_tool_models)

    settings = Settings.from_env(**settings_kwargs)

    try:
        result = run_research(
            question=request_data.question,
            settings=settings,
            writing_guidance=request_data.writing_guidance or "",
            progress_mode="quiet",
        )
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["run_dir"] = str(result.run_dir)
        JOBS[job_id]["result"] = {
            "run_dir": str(result.run_dir),
            "report_path": str(result.report_path),
            "verification_path": str(result.verification_path),
            "metrics_path": str(result.metrics_path),
        }
    except ResearchRunError as err:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(err)
        if getattr(err, "result", None) and getattr(err.result, "run_dir", None):
            JOBS[job_id]["run_dir"] = str(err.result.run_dir)
            JOBS[job_id]["result"] = {
                "run_dir": str(err.result.run_dir),
                "report_path": str(err.result.report_path),
                "verification_path": str(err.result.verification_path),
                "metrics_path": str(err.result.metrics_path),
            }
        if hasattr(err, "result") and err.result:
            JOBS[job_id]["run_dir"] = str(err.result.run_dir)
    except Exception as exc:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok", "service": "deep-research-api", "runs_dir": str(_get_runs_dir())}


@app.get("/", status_code=status.HTTP_200_OK)
def root():
    return {"message": "Deep Research API Server is operational.", "docs": "/docs"}


@app.post("/v1/research", status_code=status.HTTP_202_ACCEPTED)
async def submit_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    runs_dir = _get_runs_dir()
    job_run_dir = runs_dir / job_id
    job_run_dir.mkdir(parents=True, exist_ok=True)
    
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "question": req.question,
        "run_dir": str(job_run_dir),
        "error": None,
        "result": None,
    }

    if req.async_mode:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(executor, _execute_research_job, job_id, req)
        return {
            "job_id": job_id,
            "status": "queued",
            "status_url": f"/v1/research/{job_id}",
            "report_url": f"/v1/research/{job_id}/report",
        }
    else:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, _execute_research_job, job_id, req)
        job_info = JOBS[job_id]
        job_info["reports_url"] = f"/v1/research/{job_id}/reports"
        job_info["report_url"] = f"/v1/research/{job_id}/report"
        return job_info


@app.get("/v1/research/{job_id}")
def get_job_status(job_id: str):
    if job_id in JOBS:
        info = JOBS[job_id]
        run_dir_path = _resolve_run_dir(job_id)
        run_dir_str = str(run_dir_path) if run_dir_path else info.get("run_dir")
        report_exists = False
        activity_events = []
        if run_dir_path and run_dir_path.exists():
            if run_dir_path.joinpath("report.md").exists():
                report_exists = True
            act_file = run_dir_path.joinpath("activity.jsonl")
            if act_file.exists():
                try:
                    lines = act_file.read_text(encoding="utf-8").strip().splitlines()
                    activity_events = [json.loads(line) for line in lines[-20:] if line.strip()]
                except Exception:
                    pass

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
    
    # Fallback to checking disk for existing run dir
    run_dir_path = _resolve_run_dir(job_id)
    if run_dir_path and run_dir_path.exists():
        report_file = run_dir_path / "report.md"
        act_file = run_dir_path / "activity.jsonl"
        activity_events = []
        if act_file.exists():
            try:
                lines = act_file.read_text(encoding="utf-8").strip().splitlines()
                activity_events = [json.loads(line) for line in lines[-20:] if line.strip()]
            except Exception:
                pass
        return {
            "job_id": job_id,
            "status": "completed" if report_file.exists() else "unknown",
            "question": "",
            "run_dir": str(run_dir_path),
            "report_available": report_file.exists(),
            "recent_activity": activity_events,
        }

    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")


@app.get("/v1/research/{job_id}/activity")
def get_job_activity(job_id: str):
    """Returns the step-by-step progress events stream for the research job."""
    run_dir_path = _resolve_run_dir(job_id)
    if not run_dir_path or not run_dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    activity_path = run_dir_path / "activity.jsonl"
    if not activity_path.exists():
        return {"job_id": job_id, "events": []}

    events = []
    lines = activity_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        if line.strip():
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return {"job_id": job_id, "events": events}


def _resolve_run_dir(job_id: str) -> Optional[Path]:
    if job_id in JOBS and JOBS[job_id].get("run_dir"):
        p = Path(JOBS[job_id]["run_dir"])
        if p.exists():
            return p
    runs_dir = _get_runs_dir()
    candidate = runs_dir / job_id
    if candidate.exists() and candidate.is_dir():
        return candidate
    # Scan runs_dir for any directory containing job_id or timestamped folders
    dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        if (d / "manifest.json").exists():
            try:
                manifest_text = (d / "manifest.json").read_text(encoding="utf-8")
                if job_id in manifest_text:
                    return d
            except Exception:
                pass
    if dirs:
        return dirs[0]
    return None


@app.get("/v1/research/{job_id}/reports")
def get_all_job_reports(job_id: str):
    """Returns all generated report versions (final report, best draft, failed report, draft report)."""
    run_dir_path = _resolve_run_dir(job_id)
    if not run_dir_path or not run_dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found or run directory not yet initialized.")

    report_files = ["report.md", "best_draft.md", "failed_report.md", "draft_report.md"]
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
    """Fetch report or specific report variant (report.md, best_draft.md, draft_report.md, failed_report.md)."""
    run_dir_path = _resolve_run_dir(job_id)
    if not run_dir_path:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    # Standardize variant filename
    filename = variant if variant.endswith(".md") else f"{variant}.md"
    report_path = run_dir_path / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report variant '{filename}' for job '{job_id}' does not exist.")

    return report_path.read_text(encoding="utf-8")


@app.get("/v1/research/{job_id}/artifacts")
def get_job_artifacts(job_id: str):
    run_dir_path: Optional[Path] = None
    if job_id in JOBS and JOBS[job_id].get("run_dir"):
        run_dir_path = Path(JOBS[job_id]["run_dir"])
    else:
        runs_dir = _get_runs_dir()
        candidate = runs_dir / job_id
        if candidate.exists():
            run_dir_path = candidate

    if not run_dir_path or not run_dir_path.exists():
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
    run_dir_path: Optional[Path] = None
    if job_id in JOBS and JOBS[job_id].get("run_dir"):
        run_dir_path = Path(JOBS[job_id]["run_dir"])
    else:
        runs_dir = _get_runs_dir()
        candidate = runs_dir / job_id
        if candidate.exists():
            run_dir_path = candidate

    if not run_dir_path:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    file_path = (run_dir_path / filename).resolve()
    if not file_path.exists() or not str(file_path).startswith(str(run_dir_path.resolve())):
        raise HTTPException(status_code=404, detail=f"Artifact '{filename}' not found.")

    return FileResponse(path=file_path, filename=filename)

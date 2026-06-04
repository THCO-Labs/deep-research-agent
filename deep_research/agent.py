from __future__ import annotations

import re
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.errors import classify_exception
from deep_research.ingestion import ingest_local_paths, ingest_mcp_manifest
from deep_research.managed import run_gemini_managed_research
from deep_research.manifest import redacted_settings, runtime_metadata
from deep_research.model_policy import validate_strong_tool_models
from deep_research.model_router import describe_model_routes
from deep_research.progress import ActivityLog, ProgressCallback, ProgressMode
from deep_research.research_graph import (
    _card_from_dict,
    _coverage_from_dict,
    _load_source_texts,
    _plan_from_state,
    _source_from_dict,
    load_latest_checkpoint,
    run_local_research_graph,
)
from deep_research.schemas import RunManifestV2
from deep_research.settings import Settings
from deep_research.verifier_v2 import verify_report_v2


@dataclass(frozen=True)
class ResearchRunResult:
    run_dir: Path
    report_path: Path
    verification_path: Path
    metrics_path: Path


class ResearchRunError(RuntimeError):
    def __init__(self, message: str, result: ResearchRunResult):
        super().__init__(message)
        self.result = result


def create_agent(_settings: Settings, _context: Any) -> None:
    """Compatibility placeholder; the v2 runtime no longer uses DeepAgents."""
    raise RuntimeError("DeepAgents orchestration has been replaced by the v2 LangGraph research engine.")


def run_research(
    question: str,
    settings: Settings,
    *,
    on_update: ProgressCallback | None = None,
    progress_mode: ProgressMode = "live",
    writing_guidance: str = "",
) -> ResearchRunResult:
    if not question.strip():
        raise ValueError("Research question cannot be empty.")

    artifacts = ResearchArtifactsV2.create(settings.out_dir, question)
    activity = ActivityLog(artifacts, on_update=on_update, progress_mode=progress_mode)
    started = time.perf_counter()
    _emit(activity, "run", f"created {artifacts.run_dir}")
    _write_manifest(artifacts, question, settings, progress_mode)
    result = _result_for_artifacts(artifacts)

    try:
        if settings.research_engine == "gemini_managed":
            _emit(activity, "research_status", "starting Gemini managed Deep Research")
            run_gemini_managed_research(question=question, settings=settings, artifacts=artifacts)
            _emit(activity, "research_status", "Gemini managed Deep Research complete")
        elif settings.research_engine == "openai_managed":
            raise NotImplementedError("openai_managed is reserved for the next managed provider integration.")
        else:
            validate_strong_tool_models(settings)
            local_documents = ingest_local_paths(_resolve_input_paths(settings.local_input_paths, settings.project_root))
            mcp_documents = ingest_mcp_manifest(_resolve_optional_path(settings.mcp_manifest, settings.project_root))
            _emit(
                activity,
                "research_status",
                f"ingested {len(local_documents)} local document(s) and {len(mcp_documents)} MCP document(s)",
            )
            final_state = run_local_research_graph(
                question=question,
                settings=settings,
                artifacts=artifacts,
                activity=activity,
                local_documents=local_documents,
                mcp_documents=mcp_documents,
                writing_guidance=writing_guidance,
            )
            metrics = dict(final_state.get("metrics", {}))
            metrics["elapsed_seconds_total"] = round(time.perf_counter() - started, 3)
            artifacts.write_json("metrics.json", _public_metrics(metrics))
            verification = dict(final_state.get("verification", {}))
            if not settings.allow_failed_verification and not verification.get("valid", False):
                _write_verification_failure(artifacts, metrics, verification)
                _emit(activity, "error", "verification_failed: final report did not pass quality gates")
                raise ResearchRunError("Research verification failed; inspect verification.json and failure.json.", result)
        _emit(activity, "run", "complete")
    except ResearchRunError:
        raise
    except Exception as exc:
        failure = classify_exception(exc)
        artifacts.write_json("failure.json", failure.to_dict())
        artifacts.write_text("error.txt", f"{type(exc).__name__}: {exc}\n")
        metrics = {
            "engine": settings.research_engine,
            "error_category": failure.category,
            "verification_valid": False,
            "elapsed_seconds_total": round(time.perf_counter() - started, 3),
        }
        artifacts.write_json("metrics.json", metrics)
        if not artifacts.resolve_path("verification.json").exists():
            artifacts.write_json(
                "verification.json",
                {
                    "schema_version": 2,
                    "valid": False,
                    "failures": [failure.message],
                },
            )
        _emit(activity, "error", f"{failure.category}: {failure.message}")
        raise ResearchRunError(f"Research run failed: {exc}", result) from exc
    return result


def _write_verification_failure(
    artifacts: ResearchArtifactsV2,
    metrics: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    failures = list(verification.get("failures", []))
    message = "Final report failed verification."
    if failures:
        message += " " + "; ".join(str(failure) for failure in failures[:5])
    _archive_unaccepted_report(artifacts, verification)
    artifacts.write_json(
        "failure.json",
        {
            "category": "verification_failed",
            "retryable": True,
            "retry_after_seconds": None,
            "suggested_action": "Inspect verification.json, evidence_rejections.jsonl, coverage.json, and failed_report.md; rerun or resume after fixing synthesis/verification failures.",
            "error_type": "VerificationFailed",
            "message": message,
        },
    )
    metrics = dict(metrics)
    metrics["error_category"] = "verification_failed"
    metrics["verification_valid"] = False
    artifacts.write_json("metrics.json", _public_metrics(metrics))
    artifacts.write_text("error.txt", message + "\n")


def _archive_unaccepted_report(artifacts: ResearchArtifactsV2, verification: dict[str, Any]) -> None:
    report_path = artifacts.resolve_path("report.md")
    draft_path = artifacts.resolve_path("draft_report.md")
    failed_path = artifacts.resolve_path("failed_report.md")
    draft = ""
    if draft_path.exists():
        draft = draft_path.read_text(encoding="utf-8", errors="replace")
    elif report_path.exists():
        existing = report_path.read_text(encoding="utf-8", errors="replace")
        if "Research Run Failed Verification" not in existing[:200]:
            draft = existing
    if draft.strip():
        artifacts.write_text("failed_report.md", draft.rstrip() + "\n")
        if not draft_path.exists():
            artifacts.write_text("draft_report.md", draft.rstrip() + "\n")
    elif failed_path.exists():
        draft = failed_path.read_text(encoding="utf-8", errors="replace")
    artifacts.write_text("report.md", _failed_report_notice(verification))


def _failed_report_notice(verification: dict[str, Any]) -> str:
    failures = [str(failure) for failure in verification.get("failures", [])][:12]
    lines = [
        "# Research Run Failed Verification",
        "",
        "This run did not produce an accepted final report. The rejected draft is stored in `failed_report.md` and `draft_report.md` for debugging.",
        "",
        "## Verification Failures",
        "",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- Verification did not mark the draft as valid.")
    return "\n".join(lines).rstrip() + "\n"


def resume_research(
    run_id: str,
    settings: Settings,
    *,
    on_update: ProgressCallback | None = None,
    progress_mode: ProgressMode = "live",
) -> ResearchRunResult:
    started = time.perf_counter()
    run_dir = (settings.out_dir / run_id).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run does not exist: {run_dir}")
    artifacts = ResearchArtifactsV2.from_existing(run_dir)
    activity = ActivityLog(artifacts, on_update=on_update, progress_mode=progress_mode)
    state = load_latest_checkpoint(run_dir)
    question = str(state.get("request", {}).get("question") or "")
    result = _result_for_artifacts(artifacts)
    try:
        _emit(activity, "research_status", "resuming from latest graph checkpoint")
        final_state = run_local_research_graph(
            question=question,
            settings=settings,
            artifacts=artifacts,
            activity=activity,
            initial_state=state,
        )
        verification = dict(final_state.get("verification", {}))
        if not settings.allow_failed_verification and not verification.get("valid", False):
            _write_verification_failure(artifacts, dict(final_state.get("metrics", {})), verification)
            _emit(activity, "error", "verification_failed: resumed report did not pass quality gates")
            raise ResearchRunError("Research verification failed after resume; inspect verification.json and failure.json.", result)
        _emit(activity, "run", "resume complete")
        return result
    except ResearchRunError:
        raise
    except Exception as exc:
        failure = classify_exception(exc)
        artifacts.write_json("failure.json", failure.to_dict())
        artifacts.write_text("error.txt", f"{type(exc).__name__}: {exc}\n")
        metrics = artifacts.read_json("metrics.json") if artifacts.resolve_path("metrics.json").exists() else {}
        metrics.update(
            {
                "engine": settings.research_engine,
                "error_category": failure.category,
                "verification_valid": False,
                "elapsed_seconds_total": round(time.perf_counter() - started, 3),
            }
        )
        artifacts.write_json("metrics.json", _public_metrics(metrics))
        if not artifacts.resolve_path("verification.json").exists():
            artifacts.write_json(
                "verification.json",
                {
                    "schema_version": 2,
                    "valid": False,
                    "failures": [failure.message],
                },
            )
        _emit(activity, "error", f"{failure.category}: {failure.message}")
        raise ResearchRunError(f"Research resume failed: {exc}", result) from exc


def verify_research_run(run_id: str, settings: Settings) -> ResearchRunResult:
    run_dir = (settings.out_dir / run_id).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run does not exist: {run_dir}")
    artifacts = ResearchArtifactsV2.from_existing(run_dir)
    plan = artifacts.read_json("plan.json")
    report_path = artifacts.resolve_path("report.md")
    if not plan:
        raise FileNotFoundError(f"Run is missing plan.json: {run_dir}")
    if not report_path.exists():
        raise FileNotFoundError(f"Run is missing report.md: {run_dir}")

    sources = [_source_from_dict(row) for row in _read_jsonl(artifacts.resolve_path("sources.jsonl"))]
    cards = [_card_from_dict(row) for row in _read_jsonl(artifacts.resolve_path("evidence_cards.jsonl"))]
    coverage = _coverage_from_dict(artifacts.read_json("coverage.json"))
    source_texts = _load_source_texts(artifacts, sources)
    result = verify_report_v2(
        report_markdown=report_path.read_text(encoding="utf-8"),
        plan=_plan_from_state({"plan": plan}),
        sources=sources,
        evidence_cards=cards,
        coverage=coverage,
        source_texts=source_texts,
    )
    artifacts.write_json("verification.json", result.to_dict())
    metrics = artifacts.read_json("metrics.json")
    metrics["verification_valid"] = result.valid
    metrics["verification_failures"] = len(result.failures)
    artifacts.write_json("metrics.json", metrics)
    if not settings.allow_failed_verification and not result.valid:
        _write_verification_failure(artifacts, metrics, result.to_dict())
        raise ResearchRunError("Research verification failed; inspect verification.json and failure.json.", _result_for_artifacts(artifacts))
    return _result_for_artifacts(artifacts)


def _emit(activity: ActivityLog, stage: str, message: str) -> None:
    activity.emit(stage, message)


def _write_manifest(
    artifacts: ResearchArtifactsV2,
    question: str,
    settings: Settings,
    progress_mode: ProgressMode,
) -> None:
    model_routes = describe_model_routes(settings)
    manifest = RunManifestV2(
        run_id=artifacts.run_dir.name,
        question=question.strip(),
        engine=settings.research_engine,
        mode=settings.mode,
        managed_provider="gemini" if settings.research_engine == "gemini_managed" else None,
    ).to_dict()
    manifest.update(
        {
            "progress_mode": progress_mode,
            "settings": redacted_settings(settings),
            "runtime": runtime_metadata(),
            "model_routes": model_routes,
        }
    )
    artifacts.write_json("manifest.json", manifest)
    artifacts.write_json("model_routes.json", model_routes)


def _result_for_artifacts(artifacts: ResearchArtifactsV2) -> ResearchRunResult:
    return ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )


def _resolve_input_paths(paths: tuple[str, ...], root: Path) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        resolved.append(path)
    return resolved


def _resolve_optional_path(path_value: str, root: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"started_at_monotonic", "finished_at_monotonic"}
    }


def _ensure_source_section(report: str, registry: Any) -> str:
    """Compatibility helper for existing tests and reports."""
    if re.search(r"(?im)^##\s+sources\s*$", report):
        return report.rstrip() + "\n"
    source_lines = registry.source_lines() if hasattr(registry, "source_lines") else ""
    if not source_lines:
        return report.rstrip() + "\n"
    return report.rstrip() + "\n\n## Sources\n" + source_lines.rstrip() + "\n"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

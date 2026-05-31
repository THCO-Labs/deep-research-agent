from __future__ import annotations

import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from deepagents import create_deep_agent
from langchain.messages import HumanMessage

from deep_research.artifacts import RunArtifacts
from deep_research.progress import ProgressCallback, ProgressMode, progress_line, summarize_stream_update
from deep_research.prompts import orchestrator_prompt
from deep_research.settings import Settings
from deep_research.source_registry import SourceRegistry
from deep_research.subagents import load_subagents
from deep_research.tools import ToolContext, build_tools
from deep_research.verifier import verify_report


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


def create_agent(settings: Settings, context: ToolContext):
    tools = build_tools(context)
    root = settings.project_root
    return create_deep_agent(
        model=settings.model,
        tools=[
            tools["web_search"],
            tools["deep_scrape"],
            tools["write_file"],
            tools["read_file"],
            tools["verify_report_file"],
        ],
        system_prompt=orchestrator_prompt(settings),
        subagents=load_subagents(
            root / "subagents.yaml",
            tools,
            model=settings.fast_model,
            models_by_name={
                "planner": settings.planner_model,
                "researcher": settings.researcher_model,
                "analyst": settings.analyst_model,
                "verifier": settings.verifier_model,
            },
        ),
    )


def run_research(
    question: str,
    settings: Settings,
    *,
    on_update: ProgressCallback | None = None,
    progress_mode: ProgressMode = "live",
) -> ResearchRunResult:
    if not question.strip():
        raise ValueError("Research question cannot be empty.")

    started = time.perf_counter()
    artifacts = RunArtifacts.create(settings.out_dir, question)
    _emit(on_update, progress_mode, "run", f"created {artifacts.run_dir}")
    artifacts.write_text("request.md", question.strip() + "\n")
    artifacts.write_text(
        "research_plan.md",
        "# Research Plan\n\nThe orchestrator will replace this with a task-specific plan.\n",
    )
    registry = SourceRegistry(artifacts)
    context = ToolContext(
        settings=settings,
        artifacts=artifacts,
        registry=registry,
        on_progress=on_update if progress_mode == "live" else None,
    )
    _emit(on_update, progress_mode, "run", "building agent graph")
    agent = create_agent(settings, context)
    _emit(on_update, progress_mode, "run", "starting research stream")

    transcript: list[str] = []
    last_model_content = ""
    stream_error: Exception | None = None
    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            stream_mode="updates",
            config={"configurable": {"thread_id": artifacts.run_dir.name}},
        ):
            for node, update in chunk.items():
                messages = update.get("messages") if update else None
                if not messages:
                    continue
                msg_list = messages.value if hasattr(messages, "value") else messages
                for msg in msg_list:
                    content = getattr(msg, "content", None)
                    if not content:
                        continue
                    line = f"[{node}] {content}"
                    transcript.append(line)
                    if node == "model":
                        last_model_content = str(content)
                    if on_update and progress_mode == "raw":
                        on_update(line)
                    elif on_update and progress_mode == "live":
                        summary = summarize_stream_update(node, content)
                        if summary:
                            on_update(summary)
    except Exception as exc:
        stream_error = exc
        _emit(on_update, progress_mode, "error", f"{type(exc).__name__}: {exc}")
        artifacts.write_text("error.txt", f"{type(exc).__name__}: {exc}\n")

    _emit(on_update, progress_mode, "run", "finalizing artifacts")
    artifacts.write_text("transcript.log", "\n\n".join(transcript))
    _finalize_artifacts(artifacts, registry, context, started, last_model_content, stream_error)
    result = ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )
    if stream_error is not None:
        raise ResearchRunError(f"Research run failed: {stream_error}", result) from stream_error
    _emit(on_update, progress_mode, "run", "complete")
    return result


def _emit(
    on_update: ProgressCallback | None,
    progress_mode: ProgressMode,
    stage: str,
    message: str,
) -> None:
    if on_update and progress_mode == "live":
        on_update(progress_line(stage, message))


def _finalize_artifacts(
    artifacts: RunArtifacts,
    registry: SourceRegistry,
    context: ToolContext,
    started: float,
    last_model_content: str = "",
    stream_error: Exception | None = None,
) -> None:
    report_path = artifacts.resolve_path("report.md")
    report_reconstructed = False
    if not report_path.exists() and last_model_content.strip():
        reconstructed = _ensure_source_section(last_model_content.strip(), registry)
        artifacts.write_text(
            "report.md",
            reconstructed,
        )
        report_reconstructed = True

    if report_path.exists():
        result = verify_report(
            report_path.read_text(encoding="utf-8"),
            registry.records,
            verification_rounds=context.metrics.verification_rounds,
        )
    else:
        result = verify_report("", registry.records, verification_rounds=context.metrics.verification_rounds)
    artifacts.write_json("verification.json", result.to_dict())
    metrics = context.metrics.to_dict()
    metrics.update(
        {
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "source_count": len(registry.records),
            "report_exists": report_path.exists(),
            "report_reconstructed": report_reconstructed,
            "verification_valid": result.valid,
            "error": None if stream_error is None else f"{type(stream_error).__name__}: {stream_error}",
        }
    )
    artifacts.write_json("metrics.json", metrics)


def _ensure_source_section(report: str, registry: SourceRegistry) -> str:
    if re.search(r"(?ims)^#{2,3}\s+sources\s*$", report) or not registry.records:
        return report
    return report.rstrip() + "\n\n## Sources\n" + registry.source_lines() + "\n"

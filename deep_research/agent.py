from __future__ import annotations

import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from deepagents import create_deep_agent
from langchain.messages import HumanMessage

from deep_research.artifacts import RunArtifacts
from deep_research.deepagents_profiles import configure_deepagents_profiles
from deep_research.errors import classify_exception
from deep_research.manifest import build_run_manifest
from deep_research.model_router import build_agent_models, describe_model_routes, route_summary
from deep_research.progress import ActivityLog, ProgressCallback, ProgressMode, summarize_stream_event
from deep_research.prompts import orchestrator_prompt
from deep_research.repair import render_verification_repair_markdown
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


@dataclass(frozen=True)
class FinalizationResult:
    verification_valid: bool
    deterministic_report_recovery: bool


def create_agent(settings: Settings, context: ToolContext):
    configure_deepagents_profiles(settings)
    models = build_agent_models(
        settings,
        on_fallback=lambda message: context.emit("model_fallback", message),
        on_retry=lambda message: context.emit("model_retry", message),
    )
    tools = build_tools(context)
    root = settings.project_root
    return create_deep_agent(
        model=models.orchestrator,
        tools=[
            tools["web_search"],
            tools["deep_scrape"],
            tools["collect_sources"],
            tools["write_file"],
            tools["read_file"],
            tools["verify_report_file"],
        ],
        system_prompt=orchestrator_prompt(settings),
        subagents=load_subagents(
            root / "subagents.yaml",
            tools,
            model=models.fast,
            models_by_name={
                "planner": models.planner,
                "researcher": models.researcher,
                "analyst": models.analyst,
                "verifier": models.verifier,
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
    activity = ActivityLog(artifacts, on_update=on_update, progress_mode=progress_mode)
    _emit(activity, "run", f"created {artifacts.run_dir}")
    model_routes = describe_model_routes(settings)
    artifacts.write_json("model_routes.json", model_routes)
    artifacts.write_json(
        "run_manifest.json",
        build_run_manifest(
            question=question,
            settings=settings,
            run_dir=artifacts.run_dir,
            model_routes=model_routes,
            progress_mode=progress_mode,
        ),
    )
    _emit(activity, "model", route_summary(settings))
    artifacts.write_text("request.md", question.strip() + "\n")
    _write_research_plan(artifacts, question, settings)
    registry = SourceRegistry(artifacts)
    context = ToolContext(
        settings=settings,
        artifacts=artifacts,
        registry=registry,
        activity=activity,
        on_progress=on_update if progress_mode == "live" else None,
    )

    transcript: list[str] = []
    last_model_content = ""
    stream_error: Exception | None = None
    precollected_brief = ""
    try:
        if settings.precollect_sources:
            precollected_brief = _precollect_sources(question, context)
        _emit(activity, "run", "building agent graph")
        agent = create_agent(settings, context)
        _emit(activity, "run", "starting research stream")
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=_initial_user_message(question, precollected_brief))]},
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
                    event = summarize_stream_event(node, content)
                    if event:
                        stage, message = event
                        if progress_mode == "live":
                            # Tool-specific progress is emitted by ToolContext;
                            # this captures visible agent narration.
                            if stage != "tool":
                                activity.emit(stage, message, kind="agent_stream")
                        else:
                            activity.emit(stage, message, kind="agent_stream")
    except Exception as exc:
        stream_error = exc
        failure = classify_exception(exc)
        artifacts.write_json("failure.json", failure.to_dict())
        _emit(activity, "error", f"{failure.category}: {failure.message}")
        if failure.retry_after_seconds is not None:
            _emit(activity, "retry", f"provider suggested retry after {failure.retry_after_seconds}s")
        artifacts.write_text("error.txt", f"{type(exc).__name__}: {exc}\n")

    _emit(activity, "run", "finalizing artifacts")
    artifacts.write_text("transcript.log", "\n\n".join(transcript))
    finalization = _finalize_artifacts(
        artifacts,
        registry,
        context,
        started,
        last_model_content,
        stream_error,
        activity=activity,
    )
    result = ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )
    if stream_error is not None and not finalization.verification_valid:
        raise ResearchRunError(f"Research run failed: {stream_error}", result) from stream_error
    if stream_error is not None and finalization.deterministic_report_recovery:
        _emit(activity, "run", "complete with deterministic recovery")
    else:
        _emit(activity, "run", "complete")
    return result


def _emit(activity: ActivityLog, stage: str, message: str) -> None:
    activity.emit(stage, message)


def _finalize_artifacts(
    artifacts: RunArtifacts,
    registry: SourceRegistry,
    context: ToolContext,
    started: float,
    last_model_content: str = "",
    stream_error: Exception | None = None,
    activity: ActivityLog | None = None,
) -> FinalizationResult:
    report_path = artifacts.resolve_path("report.md")
    report_reconstructed = False
    deterministic_report_recovery = False
    if not report_path.exists() and stream_error is not None:
        deterministic_report_recovery = _write_deterministic_report_from_sources(
            artifacts=artifacts,
            registry=registry,
            activity=activity,
        )
        report_reconstructed = deterministic_report_recovery
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
            source_loader=lambda record: artifacts.read_text(record.content_path or ""),
        )
    else:
        result = verify_report(
            "",
            registry.records,
            verification_rounds=context.metrics.verification_rounds,
            source_loader=lambda record: artifacts.read_text(record.content_path or ""),
        )
    if stream_error is not None and not result.valid and not deterministic_report_recovery:
        deterministic_report_recovery = _write_deterministic_report_from_sources(
            artifacts=artifacts,
            registry=registry,
            activity=activity,
        )
        if deterministic_report_recovery:
            report_reconstructed = True
            result = verify_report(
                report_path.read_text(encoding="utf-8"),
                registry.records,
                verification_rounds=context.metrics.verification_rounds,
                source_loader=lambda record: artifacts.read_text(record.content_path or ""),
            )
    artifacts.write_json("verification.json", result.to_dict())
    repair_checklist_path = None
    if not result.valid:
        repair_path = artifacts.write_text(
            "findings/verification_repair.md",
            render_verification_repair_markdown(
                result,
                registry.records,
                report_exists=report_path.exists(),
            ),
        )
        repair_checklist_path = str(repair_path.relative_to(artifacts.run_dir))
        if activity is not None:
            activity.emit("repair", f"wrote {repair_checklist_path}")
    metrics = context.metrics.to_dict()
    failure = classify_exception(stream_error) if stream_error is not None else None
    scraped_quality_scores = [
        record.source_quality_score
        for record in registry.records
        if record.content_path and record.source_quality_score > 0
    ]
    scraped_relevance_scores = [
        record.source_relevance_score
        for record in registry.records
        if record.content_path and record.source_relevance_score > 0
    ]
    metrics.update(
        {
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "source_count": len(registry.records),
            "report_exists": report_path.exists(),
            "report_reconstructed": report_reconstructed,
            "deterministic_report_recovery": deterministic_report_recovery,
            "verification_valid": result.valid,
            "avg_source_quality_score": round(
                sum(scraped_quality_scores) / len(scraped_quality_scores),
                4,
            )
            if scraped_quality_scores
            else 0.0,
            "strong_source_count": sum(1 for score in scraped_quality_scores if score >= 0.70),
            "avg_source_relevance_score": round(
                sum(scraped_relevance_scores) / len(scraped_relevance_scores),
                4,
            )
            if scraped_relevance_scores
            else 0.0,
            "high_relevance_source_count": sum(1 for score in scraped_relevance_scores if score >= 0.70),
            "repair_checklist_path": repair_checklist_path,
            "google_key_count": len(context.settings.google_key_pool),
            "groq_key_count": len(context.settings.groq_key_pool),
            "error": None if stream_error is None else f"{type(stream_error).__name__}: {stream_error}",
            "error_category": None if failure is None else failure.category,
            "retryable": None if failure is None else failure.retryable,
            "retry_after_seconds": None if failure is None else failure.retry_after_seconds,
        }
    )
    if failure is not None:
        artifacts.write_json("failure.json", failure.to_dict())
    artifacts.write_json("metrics.json", metrics)
    return FinalizationResult(
        verification_valid=result.valid,
        deterministic_report_recovery=deterministic_report_recovery,
    )


def _ensure_source_section(report: str, registry: SourceRegistry) -> str:
    if re.search(r"(?ims)^#{2,3}\s+sources\s*$", report) or not registry.records:
        return report
    return report.rstrip() + "\n\n## Sources\n" + registry.source_lines() + "\n"


def _write_deterministic_report_from_sources(
    *,
    artifacts: RunArtifacts,
    registry: SourceRegistry,
    activity: ActivityLog | None,
) -> bool:
    deterministic_report = _deterministic_report_from_sources(
        question=artifacts.read_text("request.md").strip(),
        artifacts=artifacts,
        registry=registry,
    )
    if not deterministic_report:
        return False
    artifacts.write_text("report.md", deterministic_report)
    if activity is not None:
        activity.emit("recovery", "wrote report.md from scraped source docs")
    return True


def _deterministic_report_from_sources(
    *,
    question: str,
    artifacts: RunArtifacts,
    registry: SourceRegistry,
) -> str:
    scraped_records = [
        record
        for record in registry.records
        if record.content_path and record.content_hash
    ]
    if not scraped_records:
        return ""

    lines = [
        "# Research Report",
        "",
        "## Evidence From Scraped Sources",
        "",
    ]
    for record in scraped_records:
        body = _source_body(artifacts, record.content_path or "")
        sentences = _relevant_source_sentences(body, question, title=record.title, limit=2)
        if not sentences:
            continue
        lines.extend(
            [
                f"### Source [{record.id}]: {record.title}",
                "",
                " ".join(sentences).strip() + f" [{record.id}]",
                "",
            ]
        )

    if len(lines) <= 4:
        return ""

    lines.extend(["## Sources", ""])
    for record in scraped_records:
        lines.append(f"[{record.id}] {record.title}: {record.url}")
    return "\n".join(lines).rstrip() + "\n"


def _source_body(artifacts: RunArtifacts, content_path: str) -> str:
    if not content_path:
        return ""
    text = artifacts.read_text(content_path)
    parts = text.split("\n\n", 3)
    return parts[3] if len(parts) == 4 else text


def _relevant_source_sentences(
    text: str,
    query: str,
    *,
    title: str,
    limit: int,
) -> list[str]:
    terms = _significant_report_terms(query + " " + title)
    sentences = _source_sentences(text)
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            -len(_significant_report_terms(item[1]) & terms),
            item[0],
        ),
    )
    selected = [
        sentence
        for _, sentence in ranked
        if len(_significant_report_terms(sentence) & terms) > 0
    ][:limit]
    if selected:
        return selected
    return sentences[:limit]


def _source_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or _is_report_noise_block(block):
            continue
        cleaned = _clean_report_sentence(block)
        for candidate in re.split(r"(?<=[.!?])\s+", cleaned):
            sentence = candidate.strip()
            if 60 <= len(sentence) <= 420:
                sentences.append(sentence)
    return sentences


def _is_report_noise_block(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("#") or stripped.startswith("[!["):
        return True
    lower = stripped.lower()
    if lower.startswith(
        (
            "url:",
            "canonical url:",
            "source quality:",
            "quality reasons:",
            "source relevance:",
            "relevance matches:",
            "relevance missing:",
        )
    ):
        return True
    return len(stripped) < 80 and len(_significant_report_terms(stripped)) < 4


def _clean_report_sentence(text: str) -> str:
    cleaned = _repair_common_mojibake(text)
    cleaned = re.sub(r"!\[[^\]]*]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[*_`]+", "", cleaned)
    cleaned = cleaned.replace("#", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def _repair_common_mojibake(text: str) -> str:
    if not any(marker in text for marker in ("\u00e2", "\u00c2")):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return text


REPORT_STOPWORDS = {
    "about",
    "after",
    "also",
    "because",
    "before",
    "between",
    "could",
    "during",
    "from",
    "into",
    "more",
    "most",
    "only",
    "other",
    "over",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}


def _significant_report_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", text.lower())
        if token not in REPORT_STOPWORDS and not token.isdigit()
    }


def _precollect_sources(question: str, context: ToolContext) -> str:
    target_count = max(1, min(context.settings.max_sources, 3))
    tools = build_tools(context)
    context.emit("collect", f"pre-collecting up to {target_count} usable source(s)")
    result = tools["collect_sources"].invoke(
        {
            "query": question,
            "target_count": target_count,
            "max_results": max(context.settings.max_sources, target_count * 3),
        }
    )
    _write_research_plan(context.artifacts, question, context.settings, precollection_result=result)
    brief = _render_precollected_source_brief(result)
    context.artifacts.write_text("findings/precollected_sources.md", brief)
    context.emit(
        "collect",
        f"pre-collected {result.get('usable_count', 0)}/{target_count} usable source(s)",
    )
    return brief


def _write_research_plan(
    artifacts: RunArtifacts,
    question: str,
    settings: Settings,
    *,
    precollection_result: dict[str, Any] | None = None,
) -> None:
    artifacts.write_text(
        "research_plan.md",
        _render_research_plan(
            question,
            settings,
            precollection_result=precollection_result,
        ),
    )


def _render_research_plan(
    question: str,
    settings: Settings,
    *,
    precollection_result: dict[str, Any] | None = None,
) -> str:
    lines = [
        "# Research Plan",
        "",
        "Deterministic baseline plan. The model may refine this file, but the run should never depend on the model to create the first plan.",
        "",
        "## Core Question",
        "",
        question.strip(),
        "",
        "## Research Decomposition",
        "",
        "1. Clarify the main concept, scope, and terminology in the user question.",
        "2. Collect public-web sources with `collect_sources`, prioritizing usable scraped pages over search snippets.",
        "3. Prefer sources with strong quality and relevance scores, especially academic, official, standards, government, or primary sources.",
        "4. Synthesize the answer into `report.md` with factual paragraphs cited inline using scraped source IDs.",
        "5. Run deterministic verification and repair missing citations, unsupported claims, or source-list errors.",
        "",
        "## Source Strategy",
        "",
        f"- Initial query: `{question.strip()}`",
        f"- Target usable sources: `{max(1, min(settings.max_sources, 3))}`",
        f"- Search budget per collection pass: up to `{max(settings.max_sources, max(1, min(settings.max_sources, 3)) * 3)}` candidates",
        "- Do not cite search-only candidates or unusable scrape results.",
        "- If pre-collection does not find enough usable sources, run narrower follow-up queries.",
        "",
        "## Acceptance Criteria",
        "",
        "- `report.md` exists and directly answers the question.",
        "- Every factual paragraph has at least one inline citation.",
        "- Every cited source ID maps to a scraped source with `source_usable: true`.",
        "- `## Sources` lists the exact scraped source IDs used for citation.",
        "- `verification.json` is written; if invalid, `findings/verification_repair.md` explains the remaining repairs.",
    ]
    if precollection_result is not None:
        lines.extend(_precollection_plan_lines(precollection_result))
    return "\n".join(lines).rstrip() + "\n"


def _precollection_plan_lines(result: dict[str, Any]) -> list[str]:
    lines = [
        "",
        "## Pre-Collection Result",
        "",
        f"- Usable sources gathered: `{result.get('usable_count', 0)}` of `{result.get('target_count', 0)}`",
        f"- Candidate count: `{result.get('candidate_count', 0)}`",
        f"- Unusable sources skipped: `{result.get('unusable_count', 0)}`",
    ]
    usable_sources = result.get("usable_sources")
    if isinstance(usable_sources, list) and usable_sources:
        lines.extend(["", "### Usable Source IDs", ""])
        for source in usable_sources:
            if not isinstance(source, dict):
                continue
            lines.append(
                f"- [{source.get('source_id')}] {source.get('title')} - {source.get('url')}"
            )
    return lines


def _render_precollected_source_brief(result: dict[str, object]) -> str:
    usable_sources = result.get("usable_sources")
    unusable_sources = result.get("unusable_sources")
    lines = [
        "# Pre-collected Source Brief",
        "",
        "These sources were searched and scraped before the model graph started.",
        "Use only usable source IDs for citations. Collect more sources if coverage is insufficient.",
        "",
    ]
    if isinstance(usable_sources, list) and usable_sources:
        lines.append("## Usable Sources")
        for source in usable_sources:
            if not isinstance(source, dict):
                continue
            lines.extend(
                [
                    "",
                    f"### [{source.get('source_id')}] {source.get('title')}",
                    f"URL: {source.get('url')}",
                    f"Content path: {source.get('content_path')}",
                    (
                        "Quality: "
                        f"{source.get('source_quality_label')} "
                        f"({source.get('source_quality_score')}) - "
                        f"{source.get('source_quality_type')}"
                    ),
                    f"Relevance: {source.get('source_relevance_score')}",
                    "Excerpt:",
                    str(source.get("excerpt") or "").strip(),
                ]
            )
    else:
        lines.append("No usable sources were pre-collected.")

    if isinstance(unusable_sources, list) and unusable_sources:
        lines.extend(["", "## Unusable Sources"])
        for source in unusable_sources:
            if not isinstance(source, dict):
                continue
            lines.append(f"- {source.get('url')}: {source.get('error')}")

    return "\n".join(lines).rstrip() + "\n"


def _initial_user_message(question: str, precollected_brief: str) -> str:
    if not precollected_brief.strip():
        return question
    return (
        question.strip()
        + "\n\n"
        + "A deterministic pre-collection step already gathered public-web source evidence for this run. "
        + "Use the source IDs and excerpts in this brief before collecting more sources, and write the final "
        + "answer to report.md with inline citations.\n\n"
        + precollected_brief
    )

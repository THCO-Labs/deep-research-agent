from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from deep_research.settings import Settings


DEFAULT_DATASET = "perplexity-ai/draco"
DEFAULT_SPLIT = "test"


@dataclass(frozen=True)
class DracoTask:
    id: str
    problem: str
    domain: str
    rubric: dict[str, Any]


Runner = Callable[..., Any]


def load_draco_tasks(
    *,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    local_jsonl: Path | None = None,
    ids: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[DracoTask]:
    id_filter = {str(value) for value in ids or []}
    domain_filter = {str(value).casefold() for value in domains or []}
    rows = _load_local_rows(local_jsonl) if local_jsonl is not None else _load_dataset_rows(dataset_name, split)
    tasks: list[DracoTask] = []
    for row in rows:
        task = _task_from_row(row)
        if id_filter and task.id not in id_filter:
            continue
        if domain_filter and task.domain.casefold() not in domain_filter:
            continue
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def generate_draco_submission(
    *,
    model_name: str,
    settings: Settings,
    dataset_name: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    local_jsonl: Path | None = None,
    output_dir: Path | None = None,
    ids: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    limit: int | None = None,
    runner: Runner | None = None,
    resume_existing: bool = True,
) -> Path:
    tasks = load_draco_tasks(
        dataset_name=dataset_name,
        split=split,
        local_jsonl=local_jsonl,
        ids=ids,
        domains=domains,
        limit=limit,
    )
    if not tasks:
        raise ValueError("No DRACO tasks selected.")

    output_root = output_dir or Path("runs") / "draco" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "submission.jsonl"
    completed = _completed_ids(output_path) if resume_existing else set()
    benchmark_settings = _draco_settings(settings, output_root / "agent_runs")
    task_runner = runner or _default_runner()

    with output_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            if task.id in completed:
                continue
            row = _run_draco_task(task, benchmark_settings, task_runner)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return output_path


def _run_draco_task(task: DracoTask, settings: Settings, runner: Runner) -> dict[str, Any]:
    from deep_research.agent import ResearchRunError

    try:
        result = _call_runner(runner, task.problem, settings, writing_guidance=_rubric_guidance(task))
        article = _best_available_article(result)
        status = "completed" if article.strip() else "failed"
        error = None
        run_dir = result.run_dir
    except ResearchRunError as exc:
        article = _best_available_article(exc.result)
        status = "completed_with_internal_failure" if article.strip() else "failed"
        error = str(exc)
        run_dir = exc.result.run_dir
    except Exception as exc:  # noqa: BLE001 - benchmark runs must record task failures and continue.
        article = ""
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        run_dir = None

    return {
        "id": task.id,
        "domain": task.domain,
        "problem": task.problem,
        "model_answer": article.rstrip() + ("\n" if article.strip() else ""),
        "status": status,
        "run_dir": str(run_dir) if run_dir is not None else None,
        "error": error,
        "rubric_id": str(task.rubric.get("id") or task.id),
    }


def _call_runner(runner: Runner, problem: str, settings: Settings, *, writing_guidance: str) -> Any:
    signature = inspect.signature(runner)
    if "writing_guidance" in signature.parameters:
        return runner(problem, settings, writing_guidance=writing_guidance)
    return runner(problem, settings)


def _rubric_guidance(task: DracoTask) -> str:
    sections = _rubric_sections(task.rubric)
    return (
        "DRACO evaluation guidance for this task:\n"
        "Write a long-form deep research answer that directly satisfies the user problem. Use the rubric as a "
        "coverage, factual-accuracy, depth, presentation, and citation-quality checklist, while preserving "
        "source-grounded reasoning and inline citations. Do not mention DRACO, this guidance, or the rubric in "
        "the final answer. Do not paste criteria as headings or a checklist; convert them into natural analysis.\n\n"
        f"Task domain: {task.domain}\n\n"
        f"{sections}"
    ).strip()


def _rubric_sections(rubric: dict[str, Any]) -> str:
    lines: list[str] = []
    for section in rubric.get("sections", []) or []:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or section.get("id") or "Rubric Section").strip()
        lines.append(f"## {title}")
        for criterion in section.get("criteria", []) or []:
            if not isinstance(criterion, dict):
                continue
            requirement = str(criterion.get("requirement") or "").strip()
            if not requirement:
                continue
            weight = criterion.get("weight")
            suffix = f" (weight: {weight})" if weight is not None else ""
            lines.append(f"- {requirement}{suffix}")
    return "\n".join(lines).strip()


def _draco_settings(settings: Settings, out_dir: Path) -> Settings:
    return replace(
        settings,
        out_dir=out_dir.resolve(),
        allow_failed_verification=True,
    )


def _best_available_article(result: Any) -> str:
    for name in ("best_draft.md", "failed_report.md", "draft_report.md"):
        path = result.run_dir / name
        if path.exists():
            article = path.read_text(encoding="utf-8", errors="replace")
            if article.strip():
                return article
    if result.report_path.exists():
        article = result.report_path.read_text(encoding="utf-8", errors="replace")
        if article.strip():
            return article
    return ""


def _completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            completed.add(str(json.loads(line)["id"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return completed


def _load_dataset_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    try:
        if sys.modules.get("pyarrow") is None:
            sys.modules.pop("pyarrow", None)
        import pyarrow  # noqa: F401
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency.
        raise RuntimeError("Install the `datasets` and `pyarrow` packages to load DRACO from Hugging Face.") from exc
    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def _load_local_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"DRACO JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _task_from_row(row: dict[str, Any]) -> DracoTask:
    task_id = str(row.get("id") or "").strip()
    problem = str(row.get("problem") or "").strip()
    domain = str(row.get("domain") or "").strip()
    rubric = _parse_rubric(row.get("answer"))
    if not task_id:
        raise ValueError("DRACO row is missing `id`.")
    if not problem:
        raise ValueError(f"DRACO task {task_id} is missing `problem`.")
    if not rubric:
        raise ValueError(f"DRACO task {task_id} is missing a valid rubric in `answer`.")
    return DracoTask(id=task_id, problem=problem, domain=domain, rubric=rubric)


def _parse_rubric(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _default_runner() -> Runner:
    from deep_research.agent import run_research

    return run_research


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DRACO submissions with Deep Research Agent.")
    parser.add_argument("--model-name", default="deep_research_agent")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--local-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--ids", default="", help="Comma-separated DRACO task ids.")
    parser.add_argument("--domains", default="", help="Comma-separated DRACO domains.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list-tasks", action="store_true", help="List selected DRACO tasks without running the agent.")
    parser.add_argument("--no-resume-existing", action="store_true")
    parser.add_argument("--mode", choices=["fast", "balanced", "max_quality"], default="max_quality")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--provider", choices=["auto", "google", "groq", "hybrid", "ollama", "openrouter"], default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--min-usable-sources", type=int, default=None)
    parser.add_argument("--max-search-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-followup-queries-per-branch", type=int, default=None)
    parser.add_argument("--no-llm-planning", action="store_true", default=None)
    parser.add_argument("--no-llm-synthesis", action="store_true", default=None)
    parser.add_argument("--no-semantic-verification", action="store_true", default=None)
    parser.add_argument("--allow-failed-verification", action="store_true", default=None)
    parser.add_argument("--no-model-fallbacks", action="store_true", default=None)
    parser.add_argument("--provider-retry-attempts", type=int, default=None)
    parser.add_argument("--provider-retry-max-wait-seconds", type=int, default=None)
    parser.add_argument("--model-request-timeout-seconds", type=int, default=None)
    parser.add_argument("--model-max-output-tokens", type=int, default=None)
    parser.add_argument("--scrape-timeout-ms", type=int, default=None)
    parser.add_argument("--scrape-retries", type=int, default=None)
    parser.add_argument("--max-browser-scrapes-per-query", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_tasks:
        tasks = load_draco_tasks(
            dataset_name=args.dataset_name,
            split=args.split,
            local_jsonl=args.local_jsonl.resolve() if args.local_jsonl else None,
            ids=_parse_csv(args.ids),
            domains=_parse_csv(args.domains),
            limit=args.limit,
        )
        for task in tasks:
            print(f"{task.id}\t{task.domain}\t{task.problem[:160]}")
        print(f"Selected DRACO tasks: {len(tasks)}")
        return 0

    settings = _settings_from_args(args)
    output_path = generate_draco_submission(
        model_name=args.model_name,
        settings=settings,
        dataset_name=args.dataset_name,
        split=args.split,
        local_jsonl=args.local_jsonl.resolve() if args.local_jsonl else None,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        ids=_parse_csv(args.ids),
        domains=_parse_csv(args.domains),
        limit=args.limit,
        resume_existing=not args.no_resume_existing,
    )
    print(f"DRACO submission: {output_path}")
    return 0


def _settings_from_args(args: argparse.Namespace) -> Settings:
    return Settings.from_env(
        project_root=Path.cwd(),
        mode=args.mode,
        out_dir=args.out,
        provider=args.provider,
        max_sources=args.max_sources,
        max_rounds=args.max_rounds,
        min_usable_sources=args.min_usable_sources,
        max_search_queries=args.max_search_queries,
        max_candidates=args.max_candidates,
        max_followup_queries_per_branch=args.max_followup_queries_per_branch,
        llm_planning=_enabled_unless_disabled(args.no_llm_planning),
        llm_synthesis=_enabled_unless_disabled(args.no_llm_synthesis),
        semantic_verification=_enabled_unless_disabled(args.no_semantic_verification),
        allow_failed_verification=args.allow_failed_verification,
        model_fallbacks=_enabled_unless_disabled(args.no_model_fallbacks),
        provider_retry_attempts=args.provider_retry_attempts,
        provider_retry_max_wait_seconds=args.provider_retry_max_wait_seconds,
        model_request_timeout_seconds=args.model_request_timeout_seconds,
        model_max_output_tokens=args.model_max_output_tokens,
        scrape_timeout_ms=args.scrape_timeout_ms,
        scrape_retries=args.scrape_retries,
        max_browser_scrapes_per_query=args.max_browser_scrapes_per_query,
    )


def _enabled_unless_disabled(flag_value: bool | None) -> bool | None:
    return None if flag_value is None else not flag_value


def _parse_csv(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

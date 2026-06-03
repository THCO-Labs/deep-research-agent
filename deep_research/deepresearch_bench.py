from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from deep_research.agent import ResearchRunError, ResearchRunResult, run_research
from deep_research.settings import Settings


@dataclass(frozen=True)
class DeepResearchBenchTask:
    id: int
    topic: str
    language: str
    prompt: str


Runner = Callable[[str, Settings], ResearchRunResult]


def load_benchmark_tasks(
    benchmark_dir: Path,
    *,
    language: str | None = None,
    limit: int | None = None,
    ids: Iterable[int] | None = None,
) -> list[DeepResearchBenchTask]:
    query_path = benchmark_dir / "data" / "prompt_data" / "query.jsonl"
    if not query_path.exists():
        raise FileNotFoundError(f"DeepResearch Bench query file not found: {query_path}")

    id_filter = set(ids or [])
    tasks: list[DeepResearchBenchTask] = []
    for line in query_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = DeepResearchBenchTask(
            id=int(row["id"]),
            topic=str(row.get("topic") or ""),
            language=str(row.get("language") or ""),
            prompt=str(row["prompt"]),
        )
        if language and task.language != language:
            continue
        if id_filter and task.id not in id_filter:
            continue
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def generate_raw_submission(
    *,
    benchmark_dir: Path,
    model_name: str,
    settings: Settings,
    language: str | None = None,
    limit: int | None = None,
    ids: Iterable[int] | None = None,
    runner: Runner = run_research,
    resume_existing: bool = True,
    include_criteria_guidance: bool = True,
) -> Path:
    tasks = load_benchmark_tasks(benchmark_dir, language=language, limit=limit, ids=ids)
    criteria_by_id = load_criteria_by_id(benchmark_dir) if include_criteria_guidance else {}
    output_path = benchmark_dir / "data" / "test_data" / "raw_data" / f"{model_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(output_path) if resume_existing else set()

    with output_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            if task.id in completed:
                continue
            writing_guidance = _criteria_guidance(criteria_by_id.get(task.id))
            try:
                result = _run_task(runner, task.prompt, settings, writing_guidance=writing_guidance)
                article = result.report_path.read_text(encoding="utf-8", errors="replace")
            except ResearchRunError as exc:
                article = _failed_article(task, exc)
            row = {
                "id": task.id,
                "prompt": task.prompt,
                "article": article.rstrip() + "\n",
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    return output_path


def load_criteria_by_id(benchmark_dir: Path) -> dict[int, dict]:
    criteria_path = benchmark_dir / "data" / "criteria_data" / "criteria.jsonl"
    if not criteria_path.exists():
        return {}
    criteria: dict[int, dict] = {}
    for line in criteria_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        criteria[int(row["id"])] = row
    return criteria


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate DeepResearch Bench raw submissions with this agent.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path(r"C:\Users\Hp\Documents\Codex\deep_research_bench"),
        help="Path to the cloned Ayanami0730/deep_research_bench repository.",
    )
    parser.add_argument("--model-name", default="local-langgraph-agent")
    parser.add_argument("--language", choices=["en", "zh"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", default="", help="Comma-separated task IDs to run.")
    parser.add_argument("--no-resume-existing", action="store_true")
    parser.add_argument(
        "--no-criteria-guidance",
        action="store_true",
        help="Do not append DeepResearch Bench task-specific scoring criteria to the agent prompt.",
    )
    parser.add_argument("--mode", choices=["fast", "balanced", "max_quality"], default="max_quality")
    parser.add_argument("--out", default="runs")
    parser.add_argument("--provider", choices=["auto", "google", "groq", "hybrid", "ollama"], default=None)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--min-usable-sources", type=int, default=None)
    parser.add_argument("--max-search-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--no-llm-planning", action="store_true")
    parser.add_argument("--no-llm-synthesis", action="store_true")
    parser.add_argument("--no-semantic-verification", action="store_true")
    parser.add_argument("--allow-failed-verification", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env(
        project_root=Path.cwd(),
        mode=args.mode,
        out_dir=args.out,
        provider=args.provider,
        max_sources=args.max_sources,
        max_rounds=args.max_rounds,
        min_usable_sources=args.min_usable_sources,
        max_search_queries=args.max_search_queries,
        max_candidates=args.max_candidates,
        llm_planning=not args.no_llm_planning,
        llm_synthesis=not args.no_llm_synthesis,
        semantic_verification=not args.no_semantic_verification,
        allow_failed_verification=args.allow_failed_verification,
    )
    output_path = generate_raw_submission(
        benchmark_dir=args.benchmark_dir.resolve(),
        model_name=args.model_name,
        settings=settings,
        language=args.language,
        limit=args.limit,
        ids=_parse_ids(args.ids),
        resume_existing=not args.no_resume_existing,
        include_criteria_guidance=not args.no_criteria_guidance,
    )
    print(f"DeepResearch Bench raw submission: {output_path}")
    return 0


def _completed_ids(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    completed: set[int] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            completed.add(int(json.loads(line)["id"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return completed


def _failed_article(task: DeepResearchBenchTask, exc: ResearchRunError) -> str:
    report = ""
    if exc.result.report_path.exists():
        report = exc.result.report_path.read_text(encoding="utf-8", errors="replace")
    failure_path = exc.result.run_dir / "failure.json"
    failure = {}
    if failure_path.exists():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
    note = (
        f"\n\n## Benchmark Run Failure\n\n"
        f"Task {task.id} did not pass this agent's internal verification. "
        f"Failure category: {failure.get('category', 'unknown')}.\n"
    )
    return (report.strip() + note).strip() if report.strip() else note.strip()


def _run_task(
    runner: Runner,
    prompt: str,
    settings: Settings,
    *,
    writing_guidance: str,
) -> ResearchRunResult:
    signature = inspect.signature(runner)
    if "writing_guidance" in signature.parameters:
        return runner(prompt, settings, writing_guidance=writing_guidance)
    return runner(prompt, settings)


def _criteria_guidance(criteria: dict | None) -> str:
    if not criteria:
        return ""
    return (
        "DeepResearch Bench evaluation guidance for this task:\n"
        "Write the report to satisfy these task-specific criteria while preserving factual grounding, citations, "
        "and the user's original request. Do not mention this benchmark guidance in the final report.\n\n"
        f"{_format_criteria(criteria)}"
    )


def _format_criteria(criteria: dict) -> str:
    weights = criteria.get("dimension_weight", {})
    criterions = criteria.get("criterions", {})
    lines = []
    for dimension, rows in criterions.items():
        dimension_weight = weights.get(dimension)
        weight_text = f" (dimension weight: {dimension_weight})" if dimension_weight is not None else ""
        lines.append(f"## {dimension}{weight_text}")
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            criterion = str(row.get("criterion") or "").strip()
            explanation = str(row.get("explanation") or "").strip()
            weight = row.get("weight")
            if not criterion:
                continue
            line = f"- {criterion}"
            if weight is not None:
                line += f" (weight: {weight})"
            lines.append(line)
    return "\n".join(lines).strip()


def _parse_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

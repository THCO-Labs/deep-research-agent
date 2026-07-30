from __future__ import annotations

import argparse
import inspect
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

from langchain_core.messages import HumanMessage

from deep_research.agent import ResearchRunError, ResearchRunResult, run_research
from deep_research.planning.guidance import (
    criteria_acceptance_lines,
    extract_structured_criteria,
    format_criteria_guidance_block,
)
from deep_research.models.model_router import model_for_role
from deep_research.planning.semantic_planning import build_or_enrich_research_plan
from deep_research.core.settings import Settings
from deep_research.acquisition.source_limits import MINIMUM_SOURCE_TARGET
from deep_research.evidence.text_terms import ordered_terms, term_set
from deep_research.verification.verifier import parse_inline_citations, parse_source_list
from deep_research.bench.deepresearch_bench_formatting import _format_fact_proxy_summary, _format_proxy_summary
from deep_research.bench.deepresearch_bench_scoring import (
    _absolute_dimension_scores,
    _absolute_score,
    _avg,
    _canonical_dimension,
    _is_failed_submission,
    _load_fact_proxy_source_corpus,
    _missing_proxy_row,
    _raw_submission_path,
    _method_counts,
    _proxy_row_needs_attention,
    _relative_score,
    _score_fact_proxy_row,
    _score_proxy_row,
    _word_count,
    load_criteria_by_id,
    load_raw_submission_by_id,
    load_reference_by_id,
    summarize_fact_proxy_results,
    summarize_proxy_results,
)


@dataclass(frozen=True)
class DeepResearchBenchTask:
    id: int
    topic: str
    language: str
    prompt: str


Runner = Callable[[str, Settings], ResearchRunResult]
ProxyJudge = Callable[[dict[str, Any]], dict[str, Any]]

BENCHMARK_SOURCE_BLOCK_PATTERNS = (
    r"\bdeep[_-]?research[_-]?bench\b",
    r"\bdeepresearch[_-]?bench\b",
    r"\bdeepresearch[_-]?bench[_-]?reference\b",
    r"\bquery\.jsonl\b",
    r"\breference\.jsonl\b",
    r"\bcriteria\.jsonl\b",
)


@dataclass(frozen=True)
class DeepResearchBenchProxyPaths:
    raw_results_path: Path
    summary_path: Path
    result_path: Path


@dataclass(frozen=True)
class DeepResearchBenchFactProxyPaths:
    raw_results_path: Path
    summary_path: Path
    result_path: Path


@dataclass(frozen=True)
class DeepResearchBenchPlanningAuditPaths:
    raw_results_path: Path
    summary_path: Path
    result_path: Path


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
    benchmark_settings = _benchmark_safe_settings(settings)
    output_path = benchmark_dir / "data" / "test_data" / "raw_data" / f"{model_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_ids(output_path) if resume_existing else set()

    with output_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            if task.id in completed:
                continue
            writing_guidance = _criteria_guidance(criteria_by_id.get(task.id))
            try:
                result = _run_task(runner, task.prompt, benchmark_settings, writing_guidance=writing_guidance)
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


def audit_benchmark_planning(
    *,
    benchmark_dir: Path,
    model_name: str,
    settings: Settings | None = None,
    language: str | None = None,
    limit: int | None = None,
    ids: Iterable[int] | None = None,
    output_dir: Path | None = None,
    include_criteria_guidance: bool = True,
) -> DeepResearchBenchPlanningAuditPaths:
    tasks = load_benchmark_tasks(benchmark_dir, language=language, limit=limit, ids=ids)
    criteria_by_id = load_criteria_by_id(benchmark_dir) if include_criteria_guidance else {}
    audit_settings = settings or _default_planning_audit_settings(benchmark_dir)
    output_root = output_dir or benchmark_dir / "results" / "planning_audit" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_root / "planning_audit.jsonl"
    summary_path = output_root / "summary.json"
    result_path = output_root / "planning_audit_result.txt"

    rows: list[dict[str, Any]] = []
    with raw_results_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            criteria = criteria_by_id.get(task.id)
            guidance = _criteria_guidance(criteria)
            try:
                result = build_or_enrich_research_plan(
                    task.prompt,
                    settings=audit_settings,
                    planning_guidance=guidance,
                )
                row = _planning_audit_row(task, result, guidance)
            except Exception as exc:
                row = {
                    "id": task.id,
                    "prompt": task.prompt,
                    "topic": task.topic,
                    "language": task.language,
                    "error": str(exc),
                    "issues": ["planning_exception"],
                    "issue_count": 1,
                    "passed": False,
                }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    summary = _summarize_planning_audit(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    result_path.write_text(_format_planning_audit_summary(summary), encoding="utf-8")
    return DeepResearchBenchPlanningAuditPaths(
        raw_results_path=raw_results_path,
        summary_path=summary_path,
        result_path=result_path,
    )


def _default_planning_audit_settings(benchmark_dir: Path) -> Settings:
    output_root = benchmark_dir / "results" / "planning_audit_runs"
    return Settings(
        project_root=Path.cwd().resolve(),
        out_dir=output_root.resolve(),
        max_sources=0,
        max_rounds=1,
        min_usable_sources=MINIMUM_SOURCE_TARGET,
        max_search_queries=0,
        max_candidates=MINIMUM_SOURCE_TARGET,
        semantic_verification=False,
        llm_planning=False,
        llm_synthesis=False,
        allow_failed_verification=False,
    )


def _planning_audit_row(task: DeepResearchBenchTask, result: Any, guidance: str) -> dict[str, Any]:
    plan = result.plan
    branches = list(plan.branches)
    source_total = sum(max(int(branch.min_sources), 0) for branch in branches)
    criteria_rows = extract_structured_criteria(guidance)
    source_facing_criteria = _source_facing_criteria_rows(criteria_rows)
    acceptance_lines = criteria_acceptance_lines(guidance)
    branch_text = _planning_branch_text(branches)
    acceptance_text = " ".join(plan.acceptance_criteria)
    query_text = " ".join(query for branch in branches for query in branch.queries)

    criteria_acceptance_coverage = _criteria_text_coverage(criteria_rows, acceptance_text, include_explanation=True)
    criteria_branch_coverage = _criteria_text_coverage(source_facing_criteria, branch_text, include_explanation=False)
    criteria_query_coverage = _criteria_text_coverage(source_facing_criteria, query_text, include_explanation=False)
    duplicate_branch_titles = _duplicate_normalized_values(branch.title for branch in branches)
    empty_query_branches = [branch.id for branch in branches if not any(query.strip() for query in branch.queries)]
    drifted_branches = _request_drifted_branches(task.prompt, branches)

    issues: list[str] = []
    if not branches:
        issues.append("no_branches")
    if source_total < MINIMUM_SOURCE_TARGET:
        issues.append("minimum_source_total_below_floor")
    if duplicate_branch_titles:
        issues.append("duplicate_branch_titles")
    if empty_query_branches:
        issues.append("empty_branch_queries")
    if criteria_rows and criteria_acceptance_coverage < 1.0:
        issues.append("criteria_not_preserved_in_acceptance")
    if criteria_rows and criteria_branch_coverage < 0.55:
        issues.append("criteria_underrepresented_in_branches")
    if criteria_rows and criteria_query_coverage < 0.35:
        issues.append("criteria_underrepresented_in_queries")
    if drifted_branches:
        issues.append("branch_query_drift_from_request")

    return {
        "id": task.id,
        "prompt": task.prompt,
        "topic": task.topic,
        "language": task.language,
        "planner_used_model": bool(result.used_model),
        "planner_accepted": bool(result.accepted),
        "planner_failures": list(result.failures),
        "branch_count": len(branches),
        "branch_titles": [branch.title for branch in branches],
        "min_source_total": source_total,
        "minimum_source_ok": source_total >= MINIMUM_SOURCE_TARGET,
        "criteria_row_count": len(criteria_rows),
        "source_facing_criteria_row_count": len(source_facing_criteria),
        "acceptance_criteria_count": len(plan.acceptance_criteria),
        "criteria_guidance_line_count": len(acceptance_lines),
        "criteria_acceptance_coverage": criteria_acceptance_coverage,
        "criteria_branch_coverage": criteria_branch_coverage,
        "criteria_query_coverage": criteria_query_coverage,
        "duplicate_branch_title_count": len(duplicate_branch_titles),
        "duplicate_branch_titles": duplicate_branch_titles,
        "empty_query_branch_count": len(empty_query_branches),
        "empty_query_branches": empty_query_branches,
        "request_drifted_branch_count": len(drifted_branches),
        "request_drifted_branches": drifted_branches,
        "issues": issues,
        "issue_count": len(issues),
        "passed": not issues,
        "plan": plan.to_dict(),
    }


def _planning_branch_text(branches: list[Any]) -> str:
    parts: list[str] = []
    for branch in branches:
        parts.extend(
            [
                branch.title,
                branch.objective,
                " ".join(branch.queries),
                " ".join(branch.required_terms),
                " ".join(branch.completion_criteria),
            ]
        )
    return " ".join(parts)


def _criteria_text_coverage(criteria_rows: list[dict[str, Any]], text: str, *, include_explanation: bool) -> float:
    if not criteria_rows:
        return 1.0
    text_terms = term_set(text)
    text_lower = re.sub(r"\s+", " ", text.lower())
    scores: list[float] = []
    for row in criteria_rows:
        criterion = str(row.get("criterion") or "")
        explanation = str(row.get("explanation") or "")
        criterion_text = criterion if not include_explanation else f"{criterion} {explanation}".strip()
        criterion_text = re.sub(r"\s+", " ", criterion_text)
        criterion_terms = term_set(criterion_text)
        if not criterion_terms:
            scores.append(1.0)
            continue
        if criterion and re.sub(r"\s+", " ", criterion.lower()) in text_lower:
            scores.append(1.0)
            continue
        scores.append(len(criterion_terms & text_terms) / max(len(criterion_terms), 1))
    return round(sum(scores) / max(len(scores), 1), 4)


def _source_facing_criteria_rows(criteria_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in criteria_rows:
        dimension = str(row.get("dimension") or "").strip().lower()
        if dimension == "readability":
            continue
        rows.append(row)
    return rows


def _duplicate_normalized_values(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        key = re.sub(r"[^a-z0-9\u3400-\u9fff]+", " ", value.lower()).strip()
        if not key:
            continue
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen.add(key)
    return duplicates


def _request_drifted_branches(prompt: str, branches: list[Any]) -> list[str]:
    prompt_terms = term_set(prompt)
    if not prompt_terms:
        return []
    minimum_overlap = min(2, len(prompt_terms))
    drifted: list[str] = []
    for branch in branches:
        query_terms = term_set(" ".join(branch.queries) + " " + branch.title + " " + branch.objective)
        if len(prompt_terms & query_terms) < minimum_overlap:
            drifted.append(branch.id)
    return drifted


def _summarize_planning_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    issue_counts: dict[str, int] = {}
    for row in rows:
        for issue in row.get("issues", []):
            issue_counts[str(issue)] = issue_counts.get(str(issue), 0) + 1
    branch_counts = [int(row.get("branch_count", 0) or 0) for row in rows]
    source_totals = [int(row.get("min_source_total", 0) or 0) for row in rows]
    acceptance_scores = [float(row.get("criteria_acceptance_coverage", 1.0) or 0.0) for row in rows]
    branch_scores = [float(row.get("criteria_branch_coverage", 1.0) or 0.0) for row in rows]
    query_scores = [float(row.get("criteria_query_coverage", 1.0) or 0.0) for row in rows]
    passed_rows = [row for row in rows if row.get("passed")]
    return {
        "count": len(rows),
        "passed_count": len(passed_rows),
        "failed_count": len(rows) - len(passed_rows),
        "pass_rate": round(len(passed_rows) / max(len(rows), 1), 4),
        "issue_counts": dict(sorted(issue_counts.items())),
        "issue_task_ids": [row.get("id") for row in rows if row.get("issues")],
        "branch_count_min": min(branch_counts, default=0),
        "branch_count_avg": _avg(branch_counts),
        "branch_count_max": max(branch_counts, default=0),
        "min_source_total_min": min(source_totals, default=0),
        "min_source_total_avg": _avg(source_totals),
        "min_source_total_max": max(source_totals, default=0),
        "criteria_acceptance_coverage_min": min(acceptance_scores, default=1.0),
        "criteria_acceptance_coverage_avg": _avg(acceptance_scores),
        "criteria_branch_coverage_min": min(branch_scores, default=1.0),
        "criteria_branch_coverage_avg": _avg(branch_scores),
        "criteria_query_coverage_min": min(query_scores, default=1.0),
        "criteria_query_coverage_avg": _avg(query_scores),
    }


def _format_planning_audit_summary(summary: dict[str, Any]) -> str:
    lines = [
        "Planning Audit",
        f"Count: {summary.get('count', 0)}",
        f"Passed: {summary.get('passed_count', 0)}",
        f"Failed: {summary.get('failed_count', 0)}",
        f"Pass Rate: {float(summary.get('pass_rate', 0.0)):.4f}",
        f"Branch Count: min={summary.get('branch_count_min', 0)} avg={float(summary.get('branch_count_avg', 0.0)):.4f} max={summary.get('branch_count_max', 0)}",
        f"Min Source Total: min={summary.get('min_source_total_min', 0)} avg={float(summary.get('min_source_total_avg', 0.0)):.4f} max={summary.get('min_source_total_max', 0)}",
        f"Criteria Acceptance Coverage: min={float(summary.get('criteria_acceptance_coverage_min', 0.0)):.4f} avg={float(summary.get('criteria_acceptance_coverage_avg', 0.0)):.4f}",
        f"Criteria Branch Coverage: min={float(summary.get('criteria_branch_coverage_min', 0.0)):.4f} avg={float(summary.get('criteria_branch_coverage_avg', 0.0)):.4f}",
        f"Criteria Query Coverage: min={float(summary.get('criteria_query_coverage_min', 0.0)):.4f} avg={float(summary.get('criteria_query_coverage_avg', 0.0)):.4f}",
        f"Issue Counts: {summary.get('issue_counts', {})}",
        f"Issue Task IDs: {summary.get('issue_task_ids', [])}",
    ]
    return "\n".join(lines) + "\n"


def evaluate_raw_submission_proxy(
    *,
    benchmark_dir: Path,
    model_name: str,
    settings: Settings | None = None,
    language: str | None = None,
    limit: int | None = None,
    ids: Iterable[int] | None = None,
    raw_path: Path | None = None,
    output_dir: Path | None = None,
    use_llm: bool = True,
    allow_deterministic_fallback: bool = True,
    judge: ProxyJudge | None = None,
) -> DeepResearchBenchProxyPaths:
    tasks = load_benchmark_tasks(benchmark_dir, language=language, limit=limit, ids=ids)
    criteria_by_id = load_criteria_by_id(benchmark_dir)
    reference_by_id = load_reference_by_id(benchmark_dir)
    raw_rows = load_raw_submission_by_id(raw_path or _raw_submission_path(benchmark_dir, model_name))
    output_root = output_dir or benchmark_dir / "results" / "race_proxy" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_root / "raw_results.jsonl"
    summary_path = output_root / "summary.json"
    result_path = output_root / "race_proxy_result.txt"

    rows: list[dict[str, Any]] = []
    with raw_results_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            article_row = raw_rows.get(task.id)
            criteria = criteria_by_id.get(task.id)
            reference = reference_by_id.get(task.id)
            if article_row is None or criteria is None or reference is None:
                row = _missing_proxy_row(task, article_row, criteria, reference)
            else:
                row = _score_proxy_row(
                    task=task,
                    article=str(article_row.get("article") or ""),
                    reference=str(reference.get("article") or ""),
                    criteria=criteria,
                    settings=settings,
                    use_llm=use_llm,
                    allow_deterministic_fallback=allow_deterministic_fallback,
                    judge=judge,
                )
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    summary = summarize_proxy_results(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    result_path.write_text(_format_proxy_summary(summary), encoding="utf-8")
    return DeepResearchBenchProxyPaths(
        raw_results_path=raw_results_path,
        summary_path=summary_path,
        result_path=result_path,
    )


def evaluate_raw_submission_fact_proxy(
    *,
    benchmark_dir: Path,
    model_name: str,
    language: str | None = None,
    limit: int | None = None,
    ids: Iterable[int] | None = None,
    raw_path: Path | None = None,
    output_dir: Path | None = None,
    run_dir: Path | None = None,
    run_dirs_by_id: dict[int, Path] | None = None,
) -> DeepResearchBenchFactProxyPaths:
    tasks = load_benchmark_tasks(benchmark_dir, language=language, limit=limit, ids=ids)
    raw_rows = load_raw_submission_by_id(raw_path or _raw_submission_path(benchmark_dir, model_name))
    output_root = output_dir or benchmark_dir / "results" / "fact_proxy" / model_name
    output_root.mkdir(parents=True, exist_ok=True)
    raw_results_path = output_root / "raw_results.jsonl"
    summary_path = output_root / "summary.json"
    result_path = output_root / "fact_proxy_result.txt"

    rows: list[dict[str, Any]] = []
    with raw_results_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            article_row = raw_rows.get(task.id)
            task_run_dir = (run_dirs_by_id or {}).get(task.id) or run_dir
            if article_row is None:
                row = {
                    "id": task.id,
                    "prompt": task.prompt,
                    "topic": task.topic,
                    "language": task.language,
                    "error": "missing raw submission article",
                    "overall_score": 0.0,
                    "valid_rate": 0.0,
                }
            else:
                row = _score_fact_proxy_row(
                    task=task,
                    article=str(article_row.get("article") or ""),
                    run_dir=task_run_dir,
                )
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    summary = summarize_fact_proxy_results(rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    result_path.write_text(_format_fact_proxy_summary(summary), encoding="utf-8")
    return DeepResearchBenchFactProxyPaths(
        raw_results_path=raw_results_path,
        summary_path=summary_path,
        result_path=result_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate and proxy-score DeepResearch Bench submissions with this agent.")
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
    parser.add_argument(
        "--audit-planning",
        action="store_true",
        help="Run a no-search planning and criteria-coverage audit for the selected benchmark tasks.",
    )
    parser.add_argument(
        "--audit-with-llm-planning",
        action="store_true",
        help="Use the configured planner model during the planning audit. By default the audit is deterministic and no-cost.",
    )
    parser.add_argument("--audit-output-dir", type=Path, default=None, help="Directory for planning audit outputs.")
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
    parser.add_argument("--semantic-evidence-max-llm-cards", type=int, default=None)
    parser.add_argument("--allow-failed-verification", action="store_true", default=None)
    parser.add_argument("--no-model-fallbacks", action="store_true", default=None)
    parser.add_argument("--provider-retry-attempts", type=int, default=None)
    parser.add_argument("--provider-retry-max-wait-seconds", type=int, default=None)
    parser.add_argument("--model-request-timeout-seconds", type=int, default=None)
    parser.add_argument("--model-max-output-tokens", type=int, default=None)
    parser.add_argument("--scrape-timeout-ms", type=int, default=None)
    parser.add_argument("--scrape-retries", type=int, default=None)
    parser.add_argument("--max-browser-scrapes-per-query", type=int, default=None)
    parser.add_argument(
        "--score-proxy",
        action="store_true",
        help="Score an existing raw submission with a local RACE-style proxy evaluator instead of generating articles.",
    )
    parser.add_argument(
        "--proxy-deterministic",
        action="store_true",
        help="Use deterministic criteria/reference overlap scoring instead of an LLM proxy judge.",
    )
    parser.add_argument(
        "--no-proxy-fallback",
        action="store_true",
        help="Fail if the LLM proxy judge errors instead of falling back to deterministic scoring.",
    )
    parser.add_argument("--raw-path", type=Path, default=None, help="Raw submission JSONL path to score.")
    parser.add_argument("--proxy-output-dir", type=Path, default=None, help="Directory for proxy score outputs.")
    parser.add_argument(
        "--fact-proxy",
        action="store_true",
        help="Score citation faithfulness against a run directory's stored source text instead of generating articles.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory containing sources.jsonl and source_docs.")
    parser.add_argument(
        "--run-dir-map",
        type=Path,
        default=None,
        help="Optional JSON object mapping benchmark task id to run directory path.",
    )
    parser.add_argument("--fact-output-dir", type=Path, default=None, help="Directory for FACT proxy score outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.audit_planning:
        settings = _settings_from_args(args) if args.audit_with_llm_planning else _planning_audit_settings_from_args(args)
        paths = audit_benchmark_planning(
            benchmark_dir=args.benchmark_dir.resolve(),
            model_name=args.model_name,
            settings=settings,
            language=args.language,
            limit=args.limit,
            ids=_parse_ids(args.ids),
            output_dir=args.audit_output_dir.resolve() if args.audit_output_dir else None,
            include_criteria_guidance=not args.no_criteria_guidance,
        )
        print(f"DeepResearch Bench planning audit raw results: {paths.raw_results_path}")
        print(f"DeepResearch Bench planning audit summary: {paths.summary_path}")
        print(f"DeepResearch Bench planning audit result: {paths.result_path}")
        return 0
    settings = None if (args.score_proxy and args.proxy_deterministic) or args.fact_proxy else _settings_from_args(args)
    if args.fact_proxy:
        paths = evaluate_raw_submission_fact_proxy(
            benchmark_dir=args.benchmark_dir.resolve(),
            model_name=args.model_name,
            language=args.language,
            limit=args.limit,
            ids=_parse_ids(args.ids),
            raw_path=args.raw_path.resolve() if args.raw_path else None,
            output_dir=args.fact_output_dir.resolve() if args.fact_output_dir else None,
            run_dir=args.run_dir.resolve() if args.run_dir else None,
            run_dirs_by_id=_parse_run_dir_map(args.run_dir_map) if args.run_dir_map else None,
        )
        print(f"DeepResearch Bench FACT proxy raw results: {paths.raw_results_path}")
        print(f"DeepResearch Bench FACT proxy summary: {paths.summary_path}")
        print(f"DeepResearch Bench FACT proxy result: {paths.result_path}")
        return 0
    if args.score_proxy:
        paths = evaluate_raw_submission_proxy(
            benchmark_dir=args.benchmark_dir.resolve(),
            model_name=args.model_name,
            settings=settings,
            language=args.language,
            limit=args.limit,
            ids=_parse_ids(args.ids),
            raw_path=args.raw_path.resolve() if args.raw_path else None,
            output_dir=args.proxy_output_dir.resolve() if args.proxy_output_dir else None,
            use_llm=not args.proxy_deterministic,
            allow_deterministic_fallback=not args.no_proxy_fallback,
        )
        print(f"DeepResearch Bench proxy raw results: {paths.raw_results_path}")
        print(f"DeepResearch Bench proxy summary: {paths.summary_path}")
        print(f"DeepResearch Bench proxy result: {paths.result_path}")
        return 0
    assert settings is not None
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
        semantic_evidence_max_llm_cards=args.semantic_evidence_max_llm_cards,
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


def _planning_audit_settings_from_args(args: argparse.Namespace) -> Settings:
    root = Path.cwd().resolve()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    min_sources = max(args.min_usable_sources or MINIMUM_SOURCE_TARGET, MINIMUM_SOURCE_TARGET)
    max_candidates = max(args.max_candidates or min_sources, min_sources)
    max_sources = args.max_sources if args.max_sources is not None else 0
    return Settings(
        project_root=root,
        mode=args.mode,
        out_dir=out_dir.resolve(),
        max_sources=max_sources,
        max_rounds=args.max_rounds or 1,
        min_usable_sources=min_sources,
        max_search_queries=args.max_search_queries or 0,
        max_candidates=max_candidates,
        semantic_verification=False,
        llm_planning=False,
        llm_synthesis=False,
        allow_failed_verification=False,
    )


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
    report = _best_available_benchmark_article(exc.result)
    if report.strip():
        return report.strip()
    failure_path = exc.result.run_dir / "failure.json"
    failure = {}
    if failure_path.exists():
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
    note = (
        f"\n\n## Benchmark Run Failure\n\n"
        f"Task {task.id} did not pass this agent's internal verification. "
        f"Failure category: {failure.get('category', 'unknown')}.\n"
    )
    return note.strip()


def _best_available_benchmark_article(result: ResearchRunResult) -> str:
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


def _benchmark_safe_settings(settings: Settings) -> Settings:
    existing = tuple(settings.blocked_source_patterns)
    merged: list[str] = []
    for pattern in [*existing, *BENCHMARK_SOURCE_BLOCK_PATTERNS]:
        if pattern not in merged:
            merged.append(pattern)
    return replace(settings, blocked_source_patterns=tuple(merged))


def _criteria_guidance(criteria: dict | None) -> str:
    if not criteria:
        return ""
    structured = format_criteria_guidance_block(criteria)
    structured_section = f"\n\n{structured}" if structured else ""
    return (
        "DeepResearch Bench evaluation guidance for this task:\n"
        "Write the report to satisfy these task-specific criteria while preserving factual grounding, citations, "
        "and the user's original request. Treat the criteria as a coverage and depth checklist across "
        "comprehensiveness, insight, instruction-following, and readability. Do not mention this benchmark "
        "guidance in the final report. Convert each criterion into visible analysis, definitions, comparisons, "
        "limitations, implications, or evidence-strength discussion as appropriate; do not paste the criteria "
        "as headings or a checklist. Optimize readability with coherent section order, paragraph transitions, "
        "precise terminology, and question-specific tables only when they improve comprehension.\n\n"
        f"{_format_criteria(criteria)}"
        f"{structured_section}"
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
            if explanation:
                line += f": {explanation}"
            lines.append(line)
    return "\n".join(lines).strip()


def _parse_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_run_dir_map(path: Path) -> dict[int, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("--run-dir-map must point to a JSON object")
    return {int(key): Path(str(value)).resolve() for key, value in payload.items()}




if __name__ == "__main__":
    raise SystemExit(main())

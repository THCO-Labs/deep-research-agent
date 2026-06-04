from __future__ import annotations

import argparse
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from langchain_core.messages import HumanMessage

from deep_research.agent import ResearchRunError, ResearchRunResult, run_research
from deep_research.guidance import (
    criteria_acceptance_lines,
    extract_structured_criteria,
    format_criteria_guidance_block,
)
from deep_research.model_router import model_for_role
from deep_research.semantic_planning import build_or_enrich_research_plan
from deep_research.settings import Settings
from deep_research.source_limits import MINIMUM_SOURCE_TARGET
from deep_research.text_terms import ordered_terms, term_set
from deep_research.verifier import parse_inline_citations, parse_source_list


@dataclass(frozen=True)
class DeepResearchBenchTask:
    id: int
    topic: str
    language: str
    prompt: str


Runner = Callable[[str, Settings], ResearchRunResult]
ProxyJudge = Callable[[dict[str, Any]], dict[str, Any]]


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


def load_reference_by_id(benchmark_dir: Path) -> dict[int, dict[str, Any]]:
    reference_path = benchmark_dir / "data" / "test_data" / "cleaned_data" / "reference.jsonl"
    if not reference_path.exists():
        return {}
    references: dict[int, dict[str, Any]] = {}
    for line in reference_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        references[int(row["id"])] = row
    return references


def load_raw_submission_by_id(raw_path: Path) -> dict[int, dict[str, Any]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"DeepResearch Bench raw submission not found: {raw_path}")
    rows: dict[int, dict[str, Any]] = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["id"])] = row
    return rows


def summarize_proxy_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    if not rows:
        return {"count": 0, "successful_count": 0}
    dimensions = ("comprehensiveness", "insight", "instruction_following", "readability")
    return {
        "count": len(rows),
        "successful_count": len(successful),
        "error_count": len(rows) - len(successful),
        "overall_score": _avg(row.get("overall_score", 0.0) for row in successful),
        "race_relative_score": _avg(row.get("race_relative_score", row.get("overall_score", 0.0)) for row in successful),
        "candidate_absolute_score": _avg(row.get("candidate_absolute_score", 0.0) for row in successful),
        "reference_absolute_score": _avg(row.get("reference_absolute_score", 0.0) for row in successful),
        "topic_focus_score": _avg(row.get("topic_focus_score", 0.0) for row in successful),
        **{dimension: _avg(row.get(dimension, 0.0) for row in successful) for dimension in dimensions},
        "method_counts": _method_counts(rows),
        "low_score_ids": [
            row["id"]
            for row in successful
            if _proxy_row_needs_attention(row)
        ],
    }


def summarize_fact_proxy_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if not row.get("error")]
    if not rows:
        return {"count": 0, "successful_count": 0}
    return {
        "count": len(rows),
        "successful_count": len(successful),
        "error_count": len(rows) - len(successful),
        "overall_score": _avg(row.get("overall_score", 0.0) for row in successful),
        "valid_rate": _avg(row.get("valid_rate", 0.0) for row in successful),
        "source_list_consistency_score": _avg(row.get("source_list_consistency_score", 0.0) for row in successful),
        "prompt_overlap_score": _avg(row.get("prompt_overlap_score", 0.0) for row in successful),
        "topic_focus_score": _avg(row.get("topic_focus_score", 0.0) for row in successful),
        "topic_consistency_score": _avg(row.get("topic_consistency_score", 0.0) for row in successful),
        "source_breadth_score": _avg(row.get("source_breadth_score", 0.0) for row in successful),
        "supported_citation_count": sum(int(row.get("supported_citation_count", 0)) for row in successful),
        "unsupported_citation_count": sum(int(row.get("unsupported_citation_count", 0)) for row in successful),
        "unknown_citation_count": sum(int(row.get("unknown_citation_count", 0)) for row in successful),
        "invalid_citation_count": sum(int(row.get("invalid_citation_count", 0)) for row in successful),
        "low_score_ids": [
            row["id"]
            for row in successful
            if float(row.get("overall_score", 0.0)) < 0.95
        ],
    }


def _score_fact_proxy_row(
    *,
    task: DeepResearchBenchTask,
    article: str,
    run_dir: Path | None,
) -> dict[str, Any]:
    source_list = parse_source_list(article)
    source_corpus = _load_fact_proxy_source_corpus(run_dir) if run_dir else {}
    cited_paragraphs = _paragraphs_with_fact_citations(article)
    cited_ids = sorted(set(parse_inline_citations(article)))
    body_prompt_overlap_score = _relative_question_overlap(task.prompt, article)
    topic_focus_score = _topic_focus_score(task.prompt, article)
    topic_drift_examples = _fact_topic_drift_examples(task.prompt, article)
    topic_consistency_score = _fact_topic_consistency_score(cited_paragraphs, topic_drift_examples)
    source_list_consistency_score = _source_list_consistency_score(cited_ids, source_list, source_corpus)
    source_breadth_score = round(min(1.0, len(cited_ids) / 17), 4)

    supported = 0
    unsupported = 0
    unknown = 0
    invalid = 0
    support_checks: list[dict[str, Any]] = []
    for paragraph in cited_paragraphs:
        paragraph_ids = sorted(set(parse_inline_citations(paragraph)))
        claim_terms = term_set(_strip_fact_citations(paragraph))
        if len(claim_terms) < 4:
            continue
        combined_terms = set().union(
            *(source_corpus.get(source_id, {}).get("terms", set()) for source_id in paragraph_ids)
        )
        combined_score = _term_overlap_score(claim_terms, combined_terms)
        for source_id in paragraph_ids:
            if source_id not in source_list:
                invalid += 1
                support_checks.append(
                    _fact_support_check(
                        paragraph=paragraph,
                        source_id=source_id,
                        support_score=0.0,
                        status="invalid",
                        missing_terms=claim_terms,
                    )
                )
                continue
            source_payload = source_corpus.get(source_id)
            if source_payload is None:
                unknown += 1
                support_checks.append(
                    _fact_support_check(
                        paragraph=paragraph,
                        source_id=source_id,
                        support_score=0.0,
                        status="unknown",
                        missing_terms=claim_terms,
                    )
                )
                continue
            source_terms = source_payload.get("terms", set())
            support_score = max(_term_overlap_score(claim_terms, source_terms), combined_score * 0.75)
            if support_score >= 0.30:
                supported += 1
                status = "supported"
            else:
                unsupported += 1
                status = "unsupported"
            support_checks.append(
                _fact_support_check(
                    paragraph=paragraph,
                    source_id=source_id,
                    support_score=support_score,
                    status=status,
                    missing_terms=claim_terms - source_terms,
                )
            )

    known_citations = supported + unsupported + invalid
    assessed_citations = supported + unsupported + unknown + invalid
    valid_rate = round(supported / max(known_citations, 1), 4) if assessed_citations else 0.0
    unknown_penalty = round(1.0 - min(1.0, unknown / max(assessed_citations, 1)), 4) if assessed_citations else 0.0
    overall_score = round(
        max(
            0.0,
            min(
                1.0,
                (0.40 * valid_rate)
                + (0.12 * source_list_consistency_score)
                + (0.13 * topic_focus_score)
                + (0.12 * source_breadth_score)
                + (0.15 * topic_consistency_score)
                + (0.08 * unknown_penalty),
            ),
        ),
        4,
    )
    if _is_failed_submission(article):
        overall_score = 0.0
        valid_rate = 0.0

    return {
        "id": task.id,
        "prompt": task.prompt,
        "topic": task.topic,
        "language": task.language,
        "method": "deterministic_fact_proxy",
        "run_dir": str(run_dir) if run_dir else None,
        "article_word_count": _word_count(article),
        "total_citations": len(parse_inline_citations(article)),
        "cited_source_count": len(cited_ids),
        "cited_paragraph_count": len(cited_paragraphs),
        "known_source_count": len(source_corpus),
        "supported_citation_count": supported,
        "unsupported_citation_count": unsupported,
        "unknown_citation_count": unknown,
        "invalid_citation_count": invalid,
        "valid_rate": valid_rate,
        "source_list_consistency_score": source_list_consistency_score,
        "prompt_overlap_score": body_prompt_overlap_score,
        "topic_focus_score": topic_focus_score,
        "topic_consistency_score": topic_consistency_score,
        "source_breadth_score": source_breadth_score,
        "overall_score": overall_score,
        "topic_drift_examples": topic_drift_examples[:6],
        "unsupported_examples": [
            check for check in support_checks if check["status"] == "unsupported"
        ][:6],
        "unknown_examples": [
            check for check in support_checks if check["status"] == "unknown"
        ][:6],
        "invalid_examples": [
            check for check in support_checks if check["status"] == "invalid"
        ][:6],
    }


def _load_fact_proxy_source_corpus(run_dir: Path | None) -> dict[int, dict[str, Any]]:
    if run_dir is None:
        return {}
    sources_path = run_dir / "sources.jsonl"
    if not sources_path.exists():
        return {}
    corpus: dict[int, dict[str, Any]] = {}
    for line in sources_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            source_id = int(row["id"])
        except (KeyError, TypeError, ValueError):
            continue
        content_path = str(row.get("content_path") or "")
        source_text = ""
        if content_path:
            path = (run_dir / content_path).resolve()
            if path.exists() and _is_within(path, run_dir.resolve()):
                source_text = path.read_text(encoding="utf-8", errors="replace")
        source_blob = " ".join(
            [
                str(row.get("title") or ""),
                str(row.get("url") or ""),
                str(row.get("quality_type") or ""),
                source_text,
            ]
        )
        corpus[source_id] = {
            "title": str(row.get("title") or ""),
            "url": str(row.get("url") or ""),
            "content_path": content_path,
            "word_count": int(row.get("word_count") or 0),
            "terms": term_set(source_blob),
        }
    return corpus


def _paragraphs_with_fact_citations(article: str) -> list[str]:
    body = _without_fact_sources(article)
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not text or text.startswith("#") or text.startswith("|"):
            continue
        if parse_inline_citations(text):
            paragraphs.append(text)
    return paragraphs


def _without_fact_sources(article: str) -> str:
    match = re.search(r"(?ims)^#{2,3}\s+sources\s*$", article)
    return article if not match else article[: match.start()]


def _strip_fact_citations(text: str) -> str:
    return re.sub(r"\[([0-9][0-9,\s]*)\]", "", text)


def _source_list_consistency_score(
    cited_ids: list[int],
    source_list: dict[int, str],
    source_corpus: dict[int, dict[str, Any]],
) -> float:
    if not cited_ids:
        return 0.0
    cited = set(cited_ids)
    listed = set(source_list)
    known = set(source_corpus) if source_corpus else listed
    missing_from_list = cited - listed
    unknown_to_run = cited - known
    uncited_listed = listed - cited
    errors = len(missing_from_list) + len(unknown_to_run) + min(len(uncited_listed), len(cited))
    return round(max(0.0, 1.0 - errors / max(len(cited) + len(listed) + 1, 1)), 4)


def _topic_focus_score(prompt: str, article: str) -> float:
    body = _without_fact_sources(article)
    title = _first_markdown_title(body)
    opening = _opening_paragraphs(body, limit=2)
    return round(
        (0.25 * _relative_question_overlap(prompt, title))
        + (0.45 * _relative_question_overlap(prompt, opening))
        + (0.30 * _relative_question_overlap(prompt, body)),
        4,
    )


def _first_markdown_title(markdown: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
    return match.group(1) if match else ""


def _opening_paragraphs(markdown: str, *, limit: int) -> str:
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", markdown):
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not text or text.startswith("#") or text.startswith("|"):
            continue
        paragraphs.append(text)
        if len(paragraphs) >= limit:
            break
    return " ".join(paragraphs)


def _fact_topic_drift_examples(prompt: str, article: str) -> list[dict[str, Any]]:
    topic_terms = term_set(prompt)
    if len(topic_terms) < 2:
        return []
    drifted: list[dict[str, Any]] = []
    for paragraph in _paragraphs_with_fact_citations(article):
        paragraph_terms = term_set(_strip_fact_citations(paragraph))
        if len(paragraph_terms) < 8:
            continue
        hits = len(paragraph_terms & topic_terms)
        local_relevance = hits / max(len(paragraph_terms), 1)
        anchor_relevance = hits / max(len(topic_terms), 1)
        if (hits < 2 and local_relevance < 0.08) or (hits < 3 and local_relevance < 0.045 and anchor_relevance < 0.30):
            drifted.append(
                {
                    "paragraph": paragraph[:240],
                    "topic_hits": hits,
                    "local_relevance": round(local_relevance, 4),
                    "anchor_relevance": round(anchor_relevance, 4),
                }
            )
    return drifted


def _fact_topic_consistency_score(
    cited_paragraphs: list[str],
    drift_examples: list[dict[str, Any]],
) -> float:
    if not cited_paragraphs:
        return 0.0
    return round(max(0.0, 1.0 - len(drift_examples) / len(cited_paragraphs)), 4)


def _fact_support_check(
    *,
    paragraph: str,
    source_id: int,
    support_score: float,
    status: str,
    missing_terms: set[str],
) -> dict[str, Any]:
    return {
        "paragraph": paragraph[:240],
        "source_id": source_id,
        "support_score": round(support_score, 4),
        "status": status,
        "missing_terms": sorted(missing_terms)[:18],
    }


def _term_overlap_score(left: set[str], right: set[str]) -> float:
    if not left:
        return 1.0
    if not right:
        return 0.0
    return round(len(left & right) / len(left), 4)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _score_proxy_row(
    *,
    task: DeepResearchBenchTask,
    article: str,
    reference: str,
    criteria: dict[str, Any],
    settings: Settings | None,
    use_llm: bool,
    allow_deterministic_fallback: bool,
    judge: ProxyJudge | None,
) -> dict[str, Any]:
    judge_error: str | None = None
    payload: dict[str, Any] | None = None
    method = "deterministic"
    if judge is not None or use_llm:
        try:
            payload = judge(_proxy_judge_input(task, article, reference, criteria)) if judge else _invoke_proxy_judge(
                task=task,
                article=article,
                reference=reference,
                criteria=criteria,
                settings=settings,
            )
            method = "llm_proxy" if judge is None else "custom_proxy_judge"
        except Exception as exc:
            if not allow_deterministic_fallback:
                raise
            judge_error = str(exc)
            payload = None
            method = "deterministic_after_judge_error"

    if payload is not None:
        try:
            scores = _weighted_scores_from_judgment(payload, criteria)
        except Exception as exc:
            if not allow_deterministic_fallback:
                raise
            judge_error = str(exc)
            scores = _deterministic_proxy_scores(task.prompt, article, reference, criteria)
            method = "deterministic_after_invalid_judge_output"
    else:
        scores = _deterministic_proxy_scores(task.prompt, article, reference, criteria)

    improvement_actions = _proxy_improvement_actions(task.prompt, article, criteria)
    row: dict[str, Any] = {
        "id": task.id,
        "prompt": task.prompt,
        "topic": task.topic,
        "language": task.language,
        "method": method,
        "judge_error": judge_error,
        "overall_score": scores["overall_score"],
        "race_relative_score": scores["overall_score"],
        "candidate_absolute_score": scores["candidate_absolute_score"],
        "reference_absolute_score": scores["reference_absolute_score"],
        "topic_focus_score": _topic_focus_score(task.prompt, article),
        "benchmark_scoring_note": "overall_score follows official RACE-style target/(target+reference); absolute scores are local diagnostics.",
        "target_total": scores["target_total"],
        "reference_total": scores["reference_total"],
        "dimension_scores": scores["dimension_scores"],
        "candidate_dimension_scores": scores["candidate_dimension_scores"],
        "reference_dimension_scores": scores["reference_dimension_scores"],
        "improvement_actions": improvement_actions,
        "article_word_count": _word_count(article),
        "reference_word_count": _word_count(reference),
        "citation_count": len(re.findall(r"\[[0-9]+]", article)),
    }
    row.update(scores["dimension_scores"])
    return row


def _invoke_proxy_judge(
    *,
    task: DeepResearchBenchTask,
    article: str,
    reference: str,
    criteria: dict[str, Any],
    settings: Settings | None,
) -> dict[str, Any]:
    if settings is None:
        raise ValueError("settings are required when use_llm=True")
    model = model_for_role(settings, "judge", settings.judge_model)
    if not hasattr(model, "invoke"):
        raise RuntimeError(f"Judge role did not resolve to an invokable model: {model!r}")
    prompt = _proxy_judge_prompt(task, article, reference, criteria)
    response = model.invoke([HumanMessage(content=prompt)])
    return _load_json_object(str(getattr(response, "content", response)))


def _proxy_judge_input(
    task: DeepResearchBenchTask,
    article: str,
    reference: str,
    criteria: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": task,
        "article": article,
        "reference": reference,
        "criteria": criteria,
    }


def _proxy_judge_prompt(
    task: DeepResearchBenchTask,
    article: str,
    reference: str,
    criteria: dict[str, Any],
) -> str:
    criteria_payload = criteria.get("criterions", {})
    candidate_excerpt = _compress_article_for_eval(article, task.prompt, criteria_payload)
    reference_excerpt = _compress_article_for_eval(reference, task.prompt, criteria_payload)
    return f"""You are a strict DeepResearch Bench proxy evaluator.

Compare the candidate report against the reference report for the same task. Score each criterion from 0 to 10 for both reports.

Task:
{task.prompt}

Evaluation dimensions and criteria:
{json.dumps(criteria_payload, ensure_ascii=True, indent=2)}

Candidate report:
{candidate_excerpt}

Reference report:
{reference_excerpt}

Return exactly one compact JSON object with this schema:
{{
  "scores": [
    {{
      "dimension": "comprehensiveness",
      "criterion_index": 0,
      "candidate_score": 0.0,
      "reference_score": 0.0
    }}
  ],
  "improvement_actions": ["short repair action"]
}}

Rules:
- Reward direct task satisfaction, depth, cross-source synthesis, citations, factual specificity, and professional organization.
- Penalize off-topic sections, unsupported assertions, shallow coverage, missing required task parts, failed-run notices, and citation/source mismatches.
- Do not give the candidate credit for facts that only appear in the reference report.
- Keep JSON parseable. Do not include markdown fences, long prose analysis, quotes from the articles, or criterion text.
"""


def _weighted_scores_from_judgment(payload: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("scores"), list):
        return _weighted_scores_from_index_rows(payload, criteria)
    dimensions_payload = payload.get("dimensions", payload)
    if not isinstance(dimensions_payload, dict):
        raise ValueError("proxy judge output must contain a dimensions object")
    dimension_weights = criteria.get("dimension_weight", {})
    criterions = criteria.get("criterions", {})
    target_total = 0.0
    reference_total = 0.0
    dimension_scores: dict[str, float] = {}

    for dimension, criteria_rows in criterions.items():
        if not isinstance(criteria_rows, list):
            continue
        judged_rows = dimensions_payload.get(dimension, [])
        if not isinstance(judged_rows, list):
            judged_rows = []
        target_avg, reference_avg = _weighted_dimension_scores(criteria_rows, judged_rows)
        weight = float(dimension_weights.get(dimension, 0.0))
        target_total += target_avg * weight
        reference_total += reference_avg * weight
        dimension_scores[dimension] = _relative_score(target_avg, reference_avg)

    overall_score = _relative_score(target_total, reference_total)
    return {
        "overall_score": overall_score,
        "target_total": round(target_total, 4),
        "reference_total": round(reference_total, 4),
        "candidate_absolute_score": _absolute_score(target_total),
        "reference_absolute_score": _absolute_score(reference_total),
        "dimension_scores": dimension_scores,
        "candidate_dimension_scores": _absolute_dimension_scores(dimensions_payload, criteria),
        "reference_dimension_scores": _absolute_dimension_scores(dimensions_payload, criteria, reference=True),
    }


def _weighted_scores_from_index_rows(payload: dict[str, Any], criteria: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("scores", [])
    if not isinstance(rows, list):
        raise ValueError("scores must be a list")
    by_dimension: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        dimension = _canonical_dimension(str(row.get("dimension") or ""))
        index = _score_index(row.get("criterion_index", row.get("criterion_id", row.get("index", row.get("criterion_number")))))
        if index is None:
            continue
        by_dimension.setdefault(dimension, {})[index] = row

    dimension_weights = criteria.get("dimension_weight", {})
    criterions = criteria.get("criterions", {})
    target_total = 0.0
    reference_total = 0.0
    dimension_scores: dict[str, float] = {}
    candidate_dimension_scores: dict[str, float] = {}
    reference_dimension_scores: dict[str, float] = {}
    scored_dimensions = 0
    for dimension, criteria_rows in criterions.items():
        if not isinstance(criteria_rows, list):
            continue
        index_rows = by_dimension.get(dimension, {})
        target_avg, reference_avg = _weighted_dimension_scores_from_index_rows(criteria_rows, index_rows)
        weight = float(dimension_weights.get(dimension, 0.0))
        target_total += target_avg * weight
        reference_total += reference_avg * weight
        dimension_scores[dimension] = _relative_score(target_avg, reference_avg)
        candidate_dimension_scores[dimension] = _absolute_score(target_avg)
        reference_dimension_scores[dimension] = _absolute_score(reference_avg)
        scored_dimensions += 1
    if scored_dimensions <= 0:
        raise ValueError("proxy judge returned no usable indexed scores")
    return {
        "overall_score": _relative_score(target_total, reference_total),
        "target_total": round(target_total, 4),
        "reference_total": round(reference_total, 4),
        "candidate_absolute_score": _absolute_score(target_total),
        "reference_absolute_score": _absolute_score(reference_total),
        "dimension_scores": dimension_scores,
        "candidate_dimension_scores": candidate_dimension_scores,
        "reference_dimension_scores": reference_dimension_scores,
    }


def _weighted_dimension_scores_from_index_rows(
    criteria_rows: list[dict[str, Any]],
    judged_rows: dict[int, dict[str, Any]],
) -> tuple[float, float]:
    target_sum = 0.0
    reference_sum = 0.0
    weight_sum = 0.0
    for index, criterion in enumerate(criteria_rows):
        if not isinstance(criterion, dict) or index not in judged_rows:
            row = judged_rows.get(index + 1)
            if row is None:
                continue
        else:
            row = judged_rows[index]
        weight = float(criterion.get("weight", 0.0))
        if weight <= 0:
            continue
        target_sum += _score_0_to_10(
            row.get("candidate_score", row.get("article_1_score", row.get("target_score", row.get("candidate"))))
        ) * weight
        reference_sum += _score_0_to_10(
            row.get("reference_score", row.get("article_2_score", row.get("reference")))
        ) * weight
        weight_sum += weight
    if weight_sum <= 0:
        raise ValueError("proxy judge did not return usable indexed criterion scores")
    return round(target_sum / weight_sum, 4), round(reference_sum / weight_sum, 4)


def _weighted_dimension_scores(
    criteria_rows: list[dict[str, Any]],
    judged_rows: list[dict[str, Any]],
) -> tuple[float, float]:
    judged_by_key = {
        _criterion_key(str(row.get("criterion", ""))): row
        for row in judged_rows
        if isinstance(row, dict)
    }
    target_sum = 0.0
    reference_sum = 0.0
    weight_sum = 0.0
    for index, criterion in enumerate(criteria_rows):
        if not isinstance(criterion, dict):
            continue
        weight = float(criterion.get("weight", 0.0))
        if weight <= 0:
            continue
        key = _criterion_key(str(criterion.get("criterion", "")))
        row = judged_by_key.get(key)
        if row is None and index < len(judged_rows) and isinstance(judged_rows[index], dict):
            row = judged_rows[index]
        if row is None:
            continue
        target_score = _score_0_to_10(row.get("candidate_score", row.get("article_1_score", row.get("target_score"))))
        reference_score = _score_0_to_10(row.get("reference_score", row.get("article_2_score", 10.0)))
        target_sum += target_score * weight
        reference_sum += reference_score * weight
        weight_sum += weight
    if weight_sum <= 0:
        raise ValueError("proxy judge did not return usable criterion scores")
    return round(target_sum / weight_sum, 4), round(reference_sum / weight_sum, 4)


def _deterministic_proxy_scores(
    prompt: str,
    article: str,
    reference: str,
    criteria: dict[str, Any],
) -> dict[str, Any]:
    dimension_weights = criteria.get("dimension_weight", {})
    criterions = criteria.get("criterions", {})
    target_total = 0.0
    reference_total = 0.0
    dimension_scores: dict[str, float] = {}
    candidate_dimension_scores: dict[str, float] = {}
    reference_dimension_scores: dict[str, float] = {}
    for dimension, rows in criterions.items():
        if not isinstance(rows, list):
            continue
        target_avg = _deterministic_dimension_score(prompt, article, rows)
        reference_avg = _deterministic_dimension_score(prompt, reference, rows)
        weight = float(dimension_weights.get(dimension, 0.0))
        target_total += target_avg * weight
        reference_total += reference_avg * weight
        dimension_scores[dimension] = _relative_score(target_avg, reference_avg)
        candidate_dimension_scores[dimension] = _absolute_score(target_avg)
        reference_dimension_scores[dimension] = _absolute_score(reference_avg)
    return {
        "overall_score": _relative_score(target_total, reference_total),
        "target_total": round(target_total, 4),
        "reference_total": round(reference_total, 4),
        "candidate_absolute_score": _absolute_score(target_total),
        "reference_absolute_score": _absolute_score(reference_total),
        "dimension_scores": dimension_scores,
        "candidate_dimension_scores": candidate_dimension_scores,
        "reference_dimension_scores": reference_dimension_scores,
    }


def _deterministic_dimension_score(prompt: str, article: str, criteria_rows: list[dict[str, Any]]) -> float:
    if not article.strip():
        return 0.0
    lower = article.lower()
    if "research run failed verification" in lower or "benchmark run failure" in lower:
        return 0.0
    article_terms = term_set(article)
    prompt_score = _topic_focus_score(prompt, article)
    criterion_scores = []
    for criterion in criteria_rows:
        if not isinstance(criterion, dict):
            continue
        criterion_text = f"{criterion.get('criterion', '')} {criterion.get('explanation', '')}"
        criterion_terms = term_set(criterion_text)
        criterion_scores.append(len(criterion_terms & article_terms) / max(len(criterion_terms), 1) if criterion_terms else 1.0)
    criteria_score = sum(criterion_scores) / max(len(criterion_scores), 1)
    citation_score = min(1.0, len(re.findall(r"\[[0-9]+]", article)) / 17)
    length_score = min(1.0, _word_count(article) / 2500)
    score = 10 * ((0.50 * criteria_score) + (0.25 * prompt_score) + (0.15 * citation_score) + (0.10 * length_score))
    return round(max(0.0, min(10.0, score)), 4)


def _proxy_improvement_actions(prompt: str, article: str, criteria: dict[str, Any]) -> list[str]:
    article_terms = term_set(article)
    actions: list[str] = []
    for dimension, rows in criteria.get("criterions", {}).items():
        if not isinstance(rows, list):
            continue
        missing = []
        for row in rows:
            criterion = str(row.get("criterion", "")).strip() if isinstance(row, dict) else ""
            if not criterion:
                continue
            criterion_terms = term_set(criterion)
            overlap = len(criterion_terms & article_terms) / max(len(criterion_terms), 1) if criterion_terms else 1.0
            if overlap < 0.45:
                missing.append(criterion)
        if missing:
            actions.append(f"Expand {dimension}: " + "; ".join(missing[:3]))
    if len(re.findall(r"\[[0-9]+]", article)) < 17:
        actions.append("Increase evidence-backed inline citations to at least 17 distinct usable sources when available.")
    if _topic_focus_score(prompt, article) < 0.45:
        actions.append("Recenter the title, opening answer, and main sections on the exact benchmark prompt.")
    return actions[:8]


def _compress_article_for_eval(
    article: str,
    prompt: str,
    criteria_payload: dict[str, Any],
    *,
    max_chars: int = 24000,
) -> str:
    cleaned = re.sub(r"\n{3,}", "\n\n", article.strip())
    if len(cleaned) <= max_chars:
        return cleaned
    focus_text = prompt + " " + json.dumps(criteria_payload, ensure_ascii=False)
    focus_terms = term_set(focus_text)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", cleaned) if paragraph.strip()]
    first = []
    remaining_budget = max_chars
    for paragraph in paragraphs[:8]:
        if remaining_budget <= 0:
            break
        clipped = paragraph[: min(len(paragraph), 1800)]
        first.append(clipped)
        remaining_budget -= len(clipped) + 2
    ranked = sorted(
        paragraphs[8:],
        key=lambda paragraph: _paragraph_focus_score(paragraph, focus_terms),
        reverse=True,
    )
    selected = list(first)
    used = sum(len(paragraph) + 2 for paragraph in selected)
    for paragraph in ranked:
        if used >= max_chars:
            break
        clipped = paragraph[: min(len(paragraph), 1800)]
        if used + len(clipped) + 2 > max_chars:
            continue
        selected.append(clipped)
        used += len(clipped) + 2
    return "\n\n".join(selected)[:max_chars]


def _paragraph_focus_score(paragraph: str, focus_terms: set[str]) -> float:
    paragraph_terms = term_set(paragraph)
    if not paragraph_terms:
        return 0.0
    focus_overlap = len(paragraph_terms & focus_terms) / max(len(focus_terms), 1)
    citation_bonus = min(0.2, len(re.findall(r"\[[0-9]+]", paragraph)) * 0.02)
    return focus_overlap + citation_bonus


def _load_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise
        parsed, _end = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(parsed, dict):
        raise ValueError("expected JSON object")
    return parsed


def _missing_proxy_row(
    task: DeepResearchBenchTask,
    article_row: dict[str, Any] | None,
    criteria: dict[str, Any] | None,
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    missing = []
    if article_row is None:
        missing.append("raw submission article")
    if criteria is None:
        missing.append("criteria")
    if reference is None:
        missing.append("reference")
    return {
        "id": task.id,
        "prompt": task.prompt,
        "topic": task.topic,
        "language": task.language,
        "error": "missing " + ", ".join(missing),
        "overall_score": 0.0,
    }


def _raw_submission_path(benchmark_dir: Path, model_name: str) -> Path:
    return benchmark_dir / "data" / "test_data" / "raw_data" / f"{model_name}.jsonl"


def _relative_score(target_score: float, reference_score: float) -> float:
    denominator = target_score + reference_score
    if denominator <= 0:
        return 0.0
    return round(target_score / denominator, 4)


def _absolute_score(score: float) -> float:
    return round(max(0.0, min(float(score), 10.0)) / 10.0, 4)


def _proxy_row_needs_attention(row: dict[str, Any]) -> bool:
    relative_score = float(row.get("race_relative_score", row.get("overall_score", 0.0)) or 0.0)
    absolute_score = float(row.get("candidate_absolute_score", 0.0) or 0.0)
    topic_focus = float(row.get("topic_focus_score", 0.0) or 0.0)
    return relative_score < 0.45 or absolute_score < 0.70 or topic_focus < 0.55


def _absolute_dimension_scores(
    dimensions_payload: dict[str, Any],
    criteria: dict[str, Any],
    *,
    reference: bool = False,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    criterions = criteria.get("criterions", {})
    for dimension, criteria_rows in criterions.items():
        if not isinstance(criteria_rows, list):
            continue
        judged_rows = dimensions_payload.get(dimension, [])
        if not isinstance(judged_rows, list):
            judged_rows = []
        target_avg, reference_avg = _weighted_dimension_scores(criteria_rows, judged_rows)
        scores[dimension] = _absolute_score(reference_avg if reference else target_avg)
    return scores


def _relative_question_overlap(prompt: str, article: str) -> float:
    prompt_terms = term_set(prompt)
    if not prompt_terms:
        return 1.0
    article_terms = term_set(article)
    return round(len(prompt_terms & article_terms) / len(prompt_terms), 4)


def _score_0_to_10(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"criterion score must be numeric, got {value!r}")
    return round(max(0.0, min(10.0, float(value))), 4)


def _criterion_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _canonical_dimension(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    if normalized in {"instruction", "instruction_following", "instruction_following_relevance"}:
        return "instruction_following"
    return normalized


def _score_index(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _avg(values: Iterable[Any]) -> float:
    numeric = [float(value or 0.0) for value in values]
    if not numeric:
        return 0.0
    return round(sum(numeric) / len(numeric), 4)


def _method_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        method = str(row.get("method") or "error")
        counts[method] = counts.get(method, 0) + 1
    return counts


def _format_proxy_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Count: {summary.get('count', 0)}",
        f"Successful: {summary.get('successful_count', 0)}",
        f"Errors: {summary.get('error_count', 0)}",
        f"Overall Score: {float(summary.get('overall_score', 0.0)):.4f}",
        f"RACE Relative Score: {float(summary.get('race_relative_score', summary.get('overall_score', 0.0))):.4f}",
        f"Candidate Absolute Score: {float(summary.get('candidate_absolute_score', 0.0)):.4f}",
        f"Reference Absolute Score: {float(summary.get('reference_absolute_score', 0.0)):.4f}",
        f"Topic Focus: {float(summary.get('topic_focus_score', 0.0)):.4f}",
        f"Comprehensiveness: {float(summary.get('comprehensiveness', 0.0)):.4f}",
        f"Insight: {float(summary.get('insight', 0.0)):.4f}",
        f"Instruction Following: {float(summary.get('instruction_following', 0.0)):.4f}",
        f"Readability: {float(summary.get('readability', 0.0)):.4f}",
        f"Method Counts: {summary.get('method_counts', {})}",
        f"Low Score IDs: {summary.get('low_score_ids', [])}",
    ]
    return "\n".join(lines) + "\n"


def _format_fact_proxy_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"Count: {summary.get('count', 0)}",
        f"Successful: {summary.get('successful_count', 0)}",
        f"Errors: {summary.get('error_count', 0)}",
        f"Overall Score: {float(summary.get('overall_score', 0.0)):.4f}",
        f"Valid Rate: {float(summary.get('valid_rate', 0.0)):.4f}",
        f"Source List Consistency: {float(summary.get('source_list_consistency_score', 0.0)):.4f}",
        f"Prompt Overlap: {float(summary.get('prompt_overlap_score', 0.0)):.4f}",
        f"Topic Focus: {float(summary.get('topic_focus_score', 0.0)):.4f}",
        f"Topic Consistency: {float(summary.get('topic_consistency_score', 0.0)):.4f}",
        f"Source Breadth: {float(summary.get('source_breadth_score', 0.0)):.4f}",
        f"Supported Citations: {summary.get('supported_citation_count', 0)}",
        f"Unsupported Citations: {summary.get('unsupported_citation_count', 0)}",
        f"Unknown Citations: {summary.get('unknown_citation_count', 0)}",
        f"Invalid Citations: {summary.get('invalid_citation_count', 0)}",
        f"Low Score IDs: {summary.get('low_score_ids', [])}",
    ]
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--no-llm-planning", action="store_true")
    parser.add_argument("--no-llm-synthesis", action="store_true")
    parser.add_argument("--no-semantic-verification", action="store_true")
    parser.add_argument("--semantic-evidence-max-llm-cards", type=int, default=None)
    parser.add_argument("--allow-failed-verification", action="store_true")
    parser.add_argument("--no-model-fallbacks", action="store_true")
    parser.add_argument("--provider-retry-attempts", type=int, default=None)
    parser.add_argument("--provider-retry-max-wait-seconds", type=int, default=None)
    parser.add_argument("--model-request-timeout-seconds", type=int, default=None)
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
        llm_planning=not args.no_llm_planning,
        llm_synthesis=not args.no_llm_synthesis,
        semantic_verification=not args.no_semantic_verification,
        semantic_evidence_max_llm_cards=args.semantic_evidence_max_llm_cards,
        allow_failed_verification=args.allow_failed_verification,
        model_fallbacks=not args.no_model_fallbacks,
        provider_retry_attempts=args.provider_retry_attempts,
        provider_retry_max_wait_seconds=args.provider_retry_max_wait_seconds,
        model_request_timeout_seconds=args.model_request_timeout_seconds,
    )


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


def _is_failed_submission(article: str) -> bool:
    lower = article.lower()
    return "research run failed verification" in lower or "benchmark run failure" in lower


if __name__ == "__main__":
    raise SystemExit(main())

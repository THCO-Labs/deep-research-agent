from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

from deep_research.agent import ResearchRunError, ResearchRunResult, run_research
from deep_research.errors import classify_exception
from deep_research.model_router import model_for_role
from deep_research.settings import Settings

EXPECTED_ANSWER_RECALL_THRESHOLD = 0.65
EVAL_TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")
EVAL_STOPWORDS = {
    "and",
    "are",
    "but",
    "can",
    "for",
    "from",
    "has",
    "into",
    "its",
    "not",
    "that",
    "the",
    "their",
    "this",
    "while",
    "with",
}


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    expected_answer: str
    must_include: list[str]
    source_requirements: list[str]
    difficulty: str
    notes: str = ""


@dataclass(frozen=True)
class CoverageScore:
    score: float
    hits: list[str]
    missing: list[str]


def load_dataset(path: Path, limit: int | None = None) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            cases.append(BenchmarkCase(**data))
            if limit is not None and len(cases) >= limit:
                break
    return cases


def evaluate_dataset(
    cases: list[BenchmarkCase],
    settings: Settings,
    *,
    out_dir: Path,
    runner: Callable[[str, Settings], ResearchRunResult] = run_research,
    judge: Callable[[BenchmarkCase, str, Settings], float] | None = None,
) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    eval_dir = (out_dir / f"{run_id}-eval").resolve()
    eval_dir.mkdir(parents=True, exist_ok=False)
    results_path = eval_dir / "results.jsonl"

    rows = []
    for case in cases:
        started = time.perf_counter()
        result: ResearchRunResult | None = None
        run_error: BaseException | None = None
        try:
            result = runner(case.question, settings)
        except ResearchRunError as exc:
            result = exc.result
            run_error = exc
        except Exception as exc:
            run_error = exc

        report = _read_text(result.report_path) if result else ""
        verification = _read_json(result.verification_path) if result else {}
        metrics = _read_json(result.metrics_path) if result else {}
        failure = _failure_info(run_error, result, metrics)
        judge_score, judge_error = _safe_judge_score(
            case,
            report,
            settings,
            judge,
            skip=run_error is not None,
        )
        must_include = phrase_coverage(case.must_include, report)
        source_requirements = source_requirement_coverage(
            case,
            report,
            result.run_dir if result else None,
        )
        expected_answer_recall = expected_answer_recall_score(case.expected_answer, report)
        required_match = (
            expected_answer_recall >= EXPECTED_ANSWER_RECALL_THRESHOLD
            and must_include.score == 1.0
        )
        row = {
            "id": case.id,
            "question": case.question,
            "run_failed": run_error is not None,
            "error_message": None if run_error is None else str(run_error),
            "report_path": str(result.report_path) if result else None,
            "run_dir": str(result.run_dir) if result else None,
            "citation_verifier_score": verification.get("citation_validity_score", 0.0),
            "llm_judge_score": judge_score,
            "judge_error": judge_error,
            "required_answer_match": required_match,
            "expected_answer_recall": expected_answer_recall,
            "must_include_coverage": must_include.score,
            "missing_must_include": must_include.missing,
            "source_requirement_coverage": source_requirements.score,
            "missing_source_requirements": source_requirements.missing,
            "source_support_score": source_support_score(source_requirements.score, verification),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "source_count": metrics.get("source_count", 0),
            "avg_source_quality_score": metrics.get("avg_source_quality_score", 0.0),
            "strong_source_count": metrics.get("strong_source_count", 0),
            "search_count": metrics.get("search_count", 0),
            "scrape_count": metrics.get("scrape_count", 0),
            "verification_rounds": metrics.get("verification_rounds", 0),
            "verification_valid": verification.get("valid", False),
            "failure_category": failure.get("category"),
            "retryable": failure.get("retryable"),
            "retry_after_seconds": failure.get("retry_after_seconds"),
            "suggested_action": failure.get("suggested_action"),
            "repair_checklist_path": metrics.get("repair_checklist_path"),
            "report_reconstructed": metrics.get("report_reconstructed", False),
        }
        rows.append(row)
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    (eval_dir / "summary.json").write_text(
        json.dumps(summarize_results(rows), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return results_path


def required_answer_match(case: BenchmarkCase, report: str) -> bool:
    must_include = phrase_coverage(case.must_include, report)
    return (
        expected_answer_recall_score(case.expected_answer, report) >= EXPECTED_ANSWER_RECALL_THRESHOLD
        and must_include.score == 1.0
    )


def source_support_score(
    source_requirement_score: float,
    verification: dict[str, Any],
) -> float:
    verifier_support = float(verification.get("source_support_score", 0.0))
    return round((source_requirement_score + verifier_support) / 2, 4)


def phrase_coverage(requirements: list[str], text: str) -> CoverageScore:
    if not requirements:
        return CoverageScore(score=1.0, hits=[], missing=[])
    haystack = _normalize_for_phrase_match(text)
    hits = []
    missing = []
    for requirement in requirements:
        normalized = _normalize_for_phrase_match(requirement)
        if normalized and normalized in haystack:
            hits.append(requirement)
        else:
            missing.append(requirement)
    return CoverageScore(
        score=round(len(hits) / len(requirements), 4),
        hits=hits,
        missing=missing,
    )


def source_requirement_coverage(
    case: BenchmarkCase,
    report: str,
    run_dir: Path | None,
) -> CoverageScore:
    source_corpus = _load_source_corpus(run_dir)
    combined = report + "\n\n" + source_corpus
    return phrase_coverage(case.source_requirements, combined)


def expected_answer_recall_score(expected_answer: str, report: str) -> float:
    expected = expected_answer.strip()
    if not expected:
        return 1.0
    if _normalize_for_phrase_match(expected) in _normalize_for_phrase_match(report):
        return 1.0
    expected_terms = _significant_eval_terms(expected)
    if not expected_terms:
        return 0.0
    report_terms = _significant_eval_terms(report)
    return round(len(expected_terms & report_terms) / len(expected_terms), 4)


def llm_judge_score(case: BenchmarkCase, report: str, settings: Settings) -> float:
    routed_model = model_for_role(settings, "judge", settings.judge_model)
    model = routed_model if isinstance(routed_model, BaseChatModel) else init_chat_model(routed_model)
    prompt = f"""Grade this research report from 0.0 to 1.0.

Question: {case.question}
Expected answer: {case.expected_answer}
Must include: {case.must_include}

Report:
{report[:12000]}

Return only a decimal number between 0 and 1."""
    response = model.invoke([HumanMessage(content=prompt)])
    text = str(response.content).strip()
    try:
        value = float(text.split()[0])
    except ValueError:
        return 0.0
    return max(0.0, min(1.0, round(value, 4)))


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    return {
        "count": len(rows),
        "accuracy": round(sum(1 for row in rows if row["required_answer_match"]) / len(rows), 4),
        "citation_validity": round(
            sum(float(row["citation_verifier_score"]) for row in rows) / len(rows), 4
        ),
        "source_support": round(sum(float(row["source_support_score"]) for row in rows) / len(rows), 4),
        "must_include_coverage": round(
            sum(float(row.get("must_include_coverage", 0.0)) for row in rows) / len(rows),
            4,
        ),
        "source_requirement_coverage": round(
            sum(float(row.get("source_requirement_coverage", 0.0)) for row in rows) / len(rows),
            4,
        ),
        "avg_source_quality": round(
            sum(float(row.get("avg_source_quality_score", 0.0)) for row in rows) / len(rows),
            4,
        ),
        "llm_judge": round(sum(float(row["llm_judge_score"]) for row in rows) / len(rows), 4),
        "avg_runtime_seconds": round(sum(float(row["runtime_seconds"]) for row in rows) / len(rows), 3),
        "failures": [row["id"] for row in rows if not row["verification_valid"]],
        "run_failure_count": sum(1 for row in rows if row.get("run_failed")),
        "failure_categories": _failure_categories(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark evals for the deep research agent.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", default="eval_runs", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["fast", "balanced", "max_quality"], default="balanced")
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--scrape-char-limit", type=int, default=None)
    parser.add_argument("--provider", choices=["auto", "google", "groq", "hybrid"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fast-model", default=None)
    parser.add_argument("--planner-model", default=None)
    parser.add_argument("--researcher-model", default=None)
    parser.add_argument("--analyst-model", default=None)
    parser.add_argument("--verifier-model", default=None)
    parser.add_argument("--judge-model", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env(
        project_root=Path.cwd(),
        mode=args.mode,
        out_dir="runs",
        max_sources=args.max_sources,
        max_rounds=args.max_rounds,
        provider=args.provider,
        model=args.model,
        fast_model=args.fast_model,
        planner_model=args.planner_model,
        researcher_model=args.researcher_model,
        analyst_model=args.analyst_model,
        verifier_model=args.verifier_model,
        judge_model=args.judge_model,
        scrape_char_limit=args.scrape_char_limit,
        live=True,
    )
    results_path = evaluate_dataset(
        load_dataset(args.dataset, args.limit),
        settings,
        out_dir=args.out,
    )
    print(f"Results: {results_path}")
    print(json.dumps(_read_json(results_path.parent / "summary.json"), indent=2, sort_keys=True))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _safe_judge_score(
    case: BenchmarkCase,
    report: str,
    settings: Settings,
    judge: Callable[[BenchmarkCase, str, Settings], float] | None,
    *,
    skip: bool,
) -> tuple[float, str | None]:
    if skip:
        return 0.0, "skipped because research run failed"
    if not report.strip():
        return 0.0, "skipped because report is empty"
    try:
        score = judge(case, report, settings) if judge else llm_judge_score(case, report, settings)
    except Exception as exc:
        failure = classify_exception(exc)
        return 0.0, f"{failure.category}: {failure.message}"
    return score, None


def _failure_info(
    run_error: BaseException | None,
    result: ResearchRunResult | None,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    if result is not None:
        failure = _read_json(result.run_dir / "failure.json")
        if failure:
            return failure
    if run_error is not None:
        return classify_exception(run_error).to_dict()
    if metrics.get("error_category"):
        return {
            "category": metrics.get("error_category"),
            "retryable": metrics.get("retryable"),
            "retry_after_seconds": metrics.get("retry_after_seconds"),
            "suggested_action": None,
        }
    return {
        "category": None,
        "retryable": None,
        "retry_after_seconds": None,
        "suggested_action": None,
    }


def _normalize_for_phrase_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _significant_eval_terms(text: str) -> set[str]:
    return {
        token
        for token in EVAL_TOKEN_RE.findall(text.lower())
        if token not in EVAL_STOPWORDS and not token.isdigit()
    }


def _load_source_corpus(run_dir: Path | None) -> str:
    if run_dir is None:
        return ""
    source_dir = run_dir / "source_docs"
    if not source_dir.exists():
        return ""
    return "\n\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(source_dir.glob("*.md"))
    )


def _failure_categories(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        category = row.get("failure_category")
        if not category:
            continue
        counts[str(category)] = counts.get(str(category), 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())

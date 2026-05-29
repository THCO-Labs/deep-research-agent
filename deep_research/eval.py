from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from deep_research.agent import ResearchRunResult, run_research
from deep_research.settings import Settings


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    question: str
    expected_answer: str
    must_include: list[str]
    source_requirements: list[str]
    difficulty: str
    notes: str = ""


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
        result = runner(case.question, settings)
        report = result.report_path.read_text(encoding="utf-8") if result.report_path.exists() else ""
        verification = _read_json(result.verification_path)
        metrics = _read_json(result.metrics_path)
        judge_score = judge(case, report, settings) if judge else llm_judge_score(case, report, settings)
        row = {
            "id": case.id,
            "question": case.question,
            "report_path": str(result.report_path),
            "run_dir": str(result.run_dir),
            "citation_verifier_score": verification.get("citation_validity_score", 0.0),
            "llm_judge_score": judge_score,
            "required_answer_match": required_answer_match(case, report),
            "source_support_score": source_support_score(case, report, verification),
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "source_count": metrics.get("source_count", 0),
            "search_count": metrics.get("search_count", 0),
            "scrape_count": metrics.get("scrape_count", 0),
            "verification_rounds": metrics.get("verification_rounds", 0),
            "verification_valid": verification.get("valid", False),
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
    haystack = report.lower()
    checks = [case.expected_answer, *case.must_include]
    return all(item.lower() in haystack for item in checks if item)


def source_support_score(case: BenchmarkCase, report: str, verification: dict[str, Any]) -> float:
    requirement_hits = 0
    for requirement in case.source_requirements:
        if requirement.lower() in report.lower():
            requirement_hits += 1
    requirement_score = 1.0
    if case.source_requirements:
        requirement_score = requirement_hits / len(case.source_requirements)
    citation_score = float(verification.get("citation_validity_score", 0.0))
    return round((requirement_score + citation_score) / 2, 4)


def llm_judge_score(case: BenchmarkCase, report: str, settings: Settings) -> float:
    model = init_chat_model(settings.fast_model)
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
        "llm_judge": round(sum(float(row["llm_judge_score"]) for row in rows) / len(rows), 4),
        "avg_runtime_seconds": round(sum(float(row["runtime_seconds"]) for row in rows) / len(rows), 3),
        "failures": [row["id"] for row in rows if not row["verification_valid"]],
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
    parser.add_argument("--provider", choices=["auto", "google", "groq"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--fast-model", default=None)
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


if __name__ == "__main__":
    raise SystemExit(main())

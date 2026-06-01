import json
from pathlib import Path

from deep_research.agent import ResearchRunError, ResearchRunResult
from deep_research.artifacts import RunArtifacts
from deep_research.eval import (
    BenchmarkCase,
    evaluate_dataset,
    expected_answer_recall_score,
    phrase_coverage,
    required_answer_match,
)
from deep_research.settings import Settings


def test_evaluate_dataset_writes_results_jsonl(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path / "runs",
        google_api_key="google",
        tavily_api_key="tavily",
    )
    case = BenchmarkCase(
        id="case-1",
        question="What is RAG?",
        expected_answer="retrieval",
        must_include=["external context"],
        source_requirements=["retrieval"],
        difficulty="easy",
    )

    def fake_runner(question: str, run_settings: Settings) -> ResearchRunResult:
        artifacts = RunArtifacts.create(run_settings.out_dir, question)
        artifacts.write_text(
            "report.md",
            "RAG uses retrieval to add external context [1].\n\n"
            "## Sources\n[1] Example: https://example.com\n",
        )
        artifacts.write_text(
            "source_docs/source_1.md",
            "Retrieval augmented generation uses retrieval to add external context.",
        )
        artifacts.write_json(
            "verification.json",
            {"valid": True, "citation_validity_score": 1.0, "source_support_score": 1.0},
        )
        artifacts.write_json(
            "metrics.json",
            {
                "source_count": 1,
                "search_count": 1,
                "scrape_count": 1,
                "verification_rounds": 1,
                "avg_source_quality_score": 0.82,
                "strong_source_count": 1,
                "report_reconstructed": False,
                "repair_checklist_path": None,
            },
        )
        return ResearchRunResult(
            run_dir=artifacts.run_dir,
            report_path=artifacts.resolve_path("report.md"),
            verification_path=artifacts.resolve_path("verification.json"),
            metrics_path=artifacts.resolve_path("metrics.json"),
        )

    results_path = evaluate_dataset(
        [case],
        settings,
        out_dir=tmp_path / "evals",
        runner=fake_runner,
        judge=lambda _case, _report, _settings: 1.0,
    )

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["id"] == "case-1"
    assert rows[0]["required_answer_match"] is True
    assert rows[0]["citation_verifier_score"] == 1.0
    assert rows[0]["avg_source_quality_score"] == 0.82
    assert rows[0]["must_include_coverage"] == 1.0
    assert rows[0]["missing_must_include"] == []
    assert rows[0]["source_requirement_coverage"] == 1.0
    assert rows[0]["missing_source_requirements"] == []
    assert rows[0]["failure_category"] is None
    assert rows[0]["report_reconstructed"] is False

    summary = json.loads((results_path.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["must_include_coverage"] == 1.0
    assert summary["source_requirement_coverage"] == 1.0


def test_eval_coverage_reports_missing_terms() -> None:
    coverage = phrase_coverage(["retrieval", "model weights"], "RAG uses retrieval.")

    assert coverage.score == 0.5
    assert coverage.hits == ["retrieval"]
    assert coverage.missing == ["model weights"]


def test_required_answer_match_uses_recall_and_must_include() -> None:
    case = BenchmarkCase(
        id="case-2",
        question="Compare RAG and fine-tuning.",
        expected_answer="RAG retrieves external context while fine-tuning changes model weights.",
        must_include=["external context", "model weights"],
        source_requirements=[],
        difficulty="easy",
    )
    report = "RAG retrieves external context, while fine-tuning changes model weights."

    assert expected_answer_recall_score(case.expected_answer, report) == 1.0
    assert required_answer_match(case, report) is True


def test_evaluate_dataset_records_failed_cases_and_continues(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path / "runs",
        google_api_key="google",
        tavily_api_key="tavily",
    )
    cases = [
        BenchmarkCase(
            id="failed",
            question="quota failure case",
            expected_answer="answer",
            must_include=["answer"],
            source_requirements=["answer"],
            difficulty="easy",
        ),
        BenchmarkCase(
            id="passed",
            question="successful case",
            expected_answer="answer",
            must_include=["answer"],
            source_requirements=["answer"],
            difficulty="easy",
        ),
    ]

    def fake_runner(question: str, run_settings: Settings) -> ResearchRunResult:
        artifacts = RunArtifacts.create(run_settings.out_dir, question)
        result = ResearchRunResult(
            run_dir=artifacts.run_dir,
            report_path=artifacts.resolve_path("report.md"),
            verification_path=artifacts.resolve_path("verification.json"),
            metrics_path=artifacts.resolve_path("metrics.json"),
        )
        if "quota" in question:
            artifacts.write_json(
                "failure.json",
                {
                    "category": "quota_or_rate_limit",
                    "retryable": True,
                    "retry_after_seconds": 42,
                    "suggested_action": "Wait or switch provider.",
                },
            )
            artifacts.write_json("verification.json", {"valid": False})
            artifacts.write_json("metrics.json", {"error_category": "quota_or_rate_limit"})
            raise ResearchRunError("Research run failed: 429 RESOURCE_EXHAUSTED", result)

        artifacts.write_text("report.md", "The answer is present [1].\n\n## Sources\n[1] Example: https://example.com\n")
        artifacts.write_text("source_docs/source_1.md", "The answer is present.")
        artifacts.write_json(
            "verification.json",
            {"valid": True, "citation_validity_score": 1.0, "source_support_score": 1.0},
        )
        artifacts.write_json("metrics.json", {"source_count": 1})
        return result

    results_path = evaluate_dataset(
        cases,
        settings,
        out_dir=tmp_path / "evals",
        runner=fake_runner,
        judge=lambda _case, _report, _settings: 1.0,
    )

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads((results_path.parent / "summary.json").read_text(encoding="utf-8"))
    assert [row["id"] for row in rows] == ["failed", "passed"]
    assert rows[0]["run_failed"] is True
    assert rows[0]["failure_category"] == "quota_or_rate_limit"
    assert rows[0]["retry_after_seconds"] == 42
    assert rows[0]["llm_judge_score"] == 0.0
    assert rows[0]["judge_error"] == "skipped because research run failed"
    assert rows[1]["run_failed"] is False
    assert summary["run_failure_count"] == 1
    assert summary["failure_categories"] == {"quota_or_rate_limit": 1}


def test_evaluate_dataset_records_judge_failures(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path / "runs",
        google_api_key="google",
        tavily_api_key="tavily",
    )
    case = BenchmarkCase(
        id="judge-failure",
        question="What is the answer?",
        expected_answer="answer",
        must_include=["answer"],
        source_requirements=[],
        difficulty="easy",
    )

    def fake_runner(question: str, run_settings: Settings) -> ResearchRunResult:
        artifacts = RunArtifacts.create(run_settings.out_dir, question)
        artifacts.write_text("report.md", "The answer is present [1].\n\n## Sources\n[1] Example: https://example.com\n")
        artifacts.write_json(
            "verification.json",
            {"valid": True, "citation_validity_score": 1.0, "source_support_score": 1.0},
        )
        artifacts.write_json("metrics.json", {})
        return ResearchRunResult(
            run_dir=artifacts.run_dir,
            report_path=artifacts.resolve_path("report.md"),
            verification_path=artifacts.resolve_path("verification.json"),
            metrics_path=artifacts.resolve_path("metrics.json"),
        )

    results_path = evaluate_dataset(
        [case],
        settings,
        out_dir=tmp_path / "evals",
        runner=fake_runner,
        judge=lambda _case, _report, _settings: (_ for _ in ()).throw(RuntimeError("tool call failed")),
    )

    row = json.loads(results_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["llm_judge_score"] == 0.0
    assert row["judge_error"].startswith("tool_call_parse_error:")

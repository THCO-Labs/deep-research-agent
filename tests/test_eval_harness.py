import json
from pathlib import Path

from deep_research.agent import ResearchRunResult
from deep_research.artifacts import RunArtifacts
from deep_research.eval import BenchmarkCase, evaluate_dataset
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
        artifacts.write_json(
            "verification.json",
            {"valid": True, "citation_validity_score": 1.0},
        )
        artifacts.write_json(
            "metrics.json",
            {
                "source_count": 1,
                "search_count": 1,
                "scrape_count": 1,
                "verification_rounds": 1,
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

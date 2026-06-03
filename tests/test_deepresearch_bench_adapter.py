import json
from pathlib import Path

from deep_research.agent import ResearchRunError, ResearchRunResult
from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.deepresearch_bench import generate_raw_submission, load_benchmark_tasks
from deep_research.settings import Settings


def test_load_benchmark_tasks_filters_language_and_limit(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)

    tasks = load_benchmark_tasks(bench, language="en", limit=1)

    assert len(tasks) == 1
    assert tasks[0].id == 2
    assert tasks[0].language == "en"
    assert "English task" in tasks[0].prompt


def test_generate_raw_submission_writes_deepresearch_bench_format(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")

    output = generate_raw_submission(
        benchmark_dir=bench,
        model_name="local-test",
        settings=settings,
        language="en",
        runner=_fake_runner,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert output.name == "local-test.jsonl"
    assert rows == [
        {
            "article": "# Report\n\nAnswer for English task. [1]\n",
            "id": 2,
            "prompt": "English task",
        }
    ]


def test_generate_raw_submission_resumes_without_duplicate_rows(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")

    first = generate_raw_submission(
        benchmark_dir=bench,
        model_name="local-test",
        settings=settings,
        language="en",
        runner=_fake_runner,
    )
    second = generate_raw_submission(
        benchmark_dir=bench,
        model_name="local-test",
        settings=settings,
        language="en",
        runner=_fake_runner,
    )

    assert first == second
    assert len(second.read_text(encoding="utf-8").splitlines()) == 1


def test_generate_raw_submission_records_internal_failure(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")

    output = generate_raw_submission(
        benchmark_dir=bench,
        model_name="local-failure",
        settings=settings,
        ids=[2],
        runner=_failing_runner,
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["id"] == 2
    assert "Benchmark Run Failure" in row["article"]
    assert "verification_failed" in row["article"]


def test_generate_raw_submission_can_enrich_agent_prompt_with_criteria(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    criteria_dir = bench / "data" / "criteria_data"
    criteria_dir.mkdir(parents=True)
    (criteria_dir / "criteria.jsonl").write_text(
        json.dumps(
            {
                "id": 2,
                "prompt": "English task",
                "dimension_weight": {"comprehensiveness": 0.5},
                "criterions": {
                    "comprehensiveness": [
                        {
                            "criterion": "Cover the core mechanism",
                            "explanation": "Explain causes, evidence, and implications.",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen_questions: list[str] = []
    seen_guidance: list[str] = []

    def runner(question: str, settings: Settings, *, writing_guidance: str = "") -> ResearchRunResult:
        seen_questions.append(question)
        seen_guidance.append(writing_guidance)
        return _fake_runner("English task", settings)

    output = generate_raw_submission(
        benchmark_dir=bench,
        model_name="criteria-test",
        settings=Settings(project_root=tmp_path, out_dir=tmp_path / "runs"),
        ids=[2],
        runner=runner,
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["prompt"] == "English task"
    assert seen_questions[0] == "English task"
    assert "DeepResearch Bench evaluation guidance" in seen_guidance[0]
    assert "Cover the core mechanism" in seen_guidance[0]


def _bench_dir(tmp_path: Path) -> Path:
    bench = tmp_path / "deep_research_bench"
    query_dir = bench / "data" / "prompt_data"
    query_dir.mkdir(parents=True)
    raw_dir = bench / "data" / "test_data" / "raw_data"
    raw_dir.mkdir(parents=True)
    rows = [
        {"id": 1, "topic": "Finance", "language": "zh", "prompt": "Chinese task"},
        {"id": 2, "topic": "Science", "language": "en", "prompt": "English task"},
    ]
    (query_dir / "query.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return bench


def _fake_runner(question: str, settings: Settings) -> ResearchRunResult:
    artifacts = ResearchArtifactsV2.create(settings.out_dir, question)
    artifacts.write_text("report.md", f"# Report\n\nAnswer for {question}. [1]\n")
    artifacts.write_json("verification.json", {"valid": True})
    artifacts.write_json("metrics.json", {})
    return ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )


def _failing_runner(question: str, settings: Settings) -> ResearchRunResult:
    artifacts = ResearchArtifactsV2.create(settings.out_dir, question)
    artifacts.write_text("report.md", "# Failed Draft\n")
    artifacts.write_json("failure.json", {"category": "verification_failed"})
    result = ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )
    raise ResearchRunError("failed", result)

import json
from pathlib import Path

from deep_research.agent import ResearchRunError, ResearchRunResult
from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.draco import generate_draco_submission, load_draco_tasks
from deep_research.settings import Settings


def test_load_draco_tasks_filters_local_jsonl(tmp_path: Path) -> None:
    path = _write_draco_jsonl(
        tmp_path,
        [
            _row("task-1", "Academic", "Compare methods."),
            _row("task-2", "Finance", "Analyze filings."),
        ],
    )

    tasks = load_draco_tasks(local_jsonl=path, domains=["academic"], limit=1)

    assert len(tasks) == 1
    assert tasks[0].id == "task-1"
    assert tasks[0].domain == "Academic"
    assert tasks[0].rubric["sections"][0]["criteria"][0]["requirement"] == "Cover the central evidence"


def test_generate_draco_submission_writes_model_answer_with_rubric_guidance(tmp_path: Path) -> None:
    path = _write_draco_jsonl(tmp_path, [_row("task-1", "Academic", "Compare methods.")])
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")
    seen_guidance: list[str] = []

    def runner(problem: str, settings: Settings, *, writing_guidance: str) -> ResearchRunResult:
        seen_guidance.append(writing_guidance)
        artifacts = ResearchArtifactsV2.create(settings.out_dir, problem)
        artifacts.write_text("report.md", "# Report\n\nAnswer with sources. [1]\n")
        artifacts.write_json("verification.json", {"valid": True})
        artifacts.write_json("metrics.json", {})
        return ResearchRunResult(
            run_dir=artifacts.run_dir,
            report_path=artifacts.resolve_path("report.md"),
            verification_path=artifacts.resolve_path("verification.json"),
            metrics_path=artifacts.resolve_path("metrics.json"),
        )

    output = generate_draco_submission(
        model_name="local-test",
        settings=settings,
        local_jsonl=path,
        output_dir=tmp_path / "out",
        runner=runner,
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["id"] == "task-1"
    assert row["model_answer"] == "# Report\n\nAnswer with sources. [1]\n"
    assert row["status"] == "completed"
    assert row["rubric_id"] == "rubric-task-1"
    assert "DRACO evaluation guidance" in seen_guidance[0]
    assert "Cover the central evidence" in seen_guidance[0]


def test_generate_draco_submission_resumes_without_duplicate_rows(tmp_path: Path) -> None:
    path = _write_draco_jsonl(tmp_path, [_row("task-1", "Academic", "Compare methods.")])
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")

    first = generate_draco_submission(
        model_name="local-test",
        settings=settings,
        local_jsonl=path,
        output_dir=tmp_path / "out",
        runner=_fake_runner,
    )
    second = generate_draco_submission(
        model_name="local-test",
        settings=settings,
        local_jsonl=path,
        output_dir=tmp_path / "out",
        runner=_fake_runner,
    )

    assert first == second
    assert len(second.read_text(encoding="utf-8").splitlines()) == 1


def test_generate_draco_submission_records_internal_failure_best_draft(tmp_path: Path) -> None:
    path = _write_draco_jsonl(tmp_path, [_row("task-1", "Academic", "Compare methods.")])
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")

    output = generate_draco_submission(
        model_name="local-test",
        settings=settings,
        local_jsonl=path,
        output_dir=tmp_path / "out",
        runner=_failing_runner,
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "completed_with_internal_failure"
    assert "Best rejected answer" in row["model_answer"]
    assert row["error"]


def _fake_runner(problem: str, settings: Settings) -> ResearchRunResult:
    artifacts = ResearchArtifactsV2.create(settings.out_dir, problem)
    artifacts.write_text("report.md", "# Report\n\nAnswer.\n")
    artifacts.write_json("verification.json", {"valid": True})
    artifacts.write_json("metrics.json", {})
    return ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )


def _failing_runner(problem: str, settings: Settings) -> ResearchRunResult:
    artifacts = ResearchArtifactsV2.create(settings.out_dir, problem)
    artifacts.write_text("best_draft.md", "# Best rejected answer\n\nStill useful.\n")
    artifacts.write_json("verification.json", {"valid": False})
    artifacts.write_json("metrics.json", {})
    result = ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )
    raise ResearchRunError("verification failed", result)


def _write_draco_jsonl(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "draco.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(task_id: str, domain: str, problem: str) -> dict[str, str]:
    return {
        "id": task_id,
        "domain": domain,
        "problem": problem,
        "answer": json.dumps(
            {
                "id": f"rubric-{task_id}",
                "sections": [
                    {
                        "id": "factual-accuracy",
                        "title": "Factual Accuracy",
                        "criteria": [
                            {
                                "id": "central-evidence",
                                "weight": 10,
                                "requirement": "Cover the central evidence",
                            }
                        ],
                    }
                ],
            }
        ),
    }

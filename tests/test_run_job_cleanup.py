from pathlib import Path

from deep_research.run_job import _compact_completed_run_dir, _settings_kwargs_from_payload


def test_compact_completed_run_dir_keeps_reports_and_sources(tmp_path: Path) -> None:
    run_dir = tmp_path / "job_123"
    run_dir.mkdir()
    for name in (
        "report.md",
        "best_draft.md",
        "best_report.md",
        "draft_report.md",
        "draft_report_1.md",
        "draft_report_2.md",
        "failed_report.md",
        "assembled_best_report.md",
        "sources.jsonl",
        "metrics.json",
        "activity.jsonl",
        "job.json",
        "verification.json",
    ):
        (run_dir / name).write_text(name, encoding="utf-8")
    for directory in ("source_docs", "checkpoints", "findings", "documents"):
        path = run_dir / directory
        path.mkdir()
        (path / "large.txt").write_text("x", encoding="utf-8")

    removed = _compact_completed_run_dir(run_dir)

    assert sorted(path.name for path in run_dir.iterdir()) == [
        "assembled_best_report.md",
        "best_draft.md",
        "best_report.md",
        "draft_report.md",
        "draft_report_1.md",
        "draft_report_2.md",
        "failed_report.md",
        "report.md",
        "sources.jsonl",
    ]
    assert "source_docs" in removed
    assert "metrics.json" in removed
    assert "job.json" in removed


def test_compact_completed_run_dir_removes_non_numeric_draft_variants(tmp_path: Path) -> None:
    run_dir = tmp_path / "job_123"
    run_dir.mkdir()
    (run_dir / "draft_report_final.md").write_text("not canonical", encoding="utf-8")
    (run_dir / "draft_report_3.md").write_text("canonical", encoding="utf-8")

    _compact_completed_run_dir(run_dir)

    assert not (run_dir / "draft_report_final.md").exists()
    assert (run_dir / "draft_report_3.md").exists()


def test_settings_kwargs_preserve_synthesis_and_citation_roles(tmp_path: Path) -> None:
    payload = {
        "provider": "deepseek",
        "model": "deepseek:deepseek-chat",
        "fast_model": "deepseek:deepseek-chat",
        "planner_model": "deepseek:deepseek-chat",
        "researcher_model": "deepseek:deepseek-chat",
        "analyst_model": "deepseek:deepseek-chat",
        "synthesis_model": "deepseek:deepseek-chat",
        "verifier_model": "azure_openai:gpt-4o",
        "judge_model": "azure_openai:gpt-4o",
        "citation_model": "azure_openai:gpt-4o",
    }

    settings_kwargs = _settings_kwargs_from_payload(payload, tmp_path)

    assert settings_kwargs["model"] == "deepseek:deepseek-chat"
    assert settings_kwargs["analyst_model"] == "deepseek:deepseek-chat"
    assert settings_kwargs["synthesis_model"] == "deepseek:deepseek-chat"
    assert settings_kwargs["verifier_model"] == "azure_openai:gpt-4o"
    assert settings_kwargs["judge_model"] == "azure_openai:gpt-4o"
    assert settings_kwargs["citation_model"] == "azure_openai:gpt-4o"

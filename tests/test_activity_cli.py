from pathlib import Path

from deep_research import activity as activity_cli
from deep_research.artifacts import RunArtifacts
from deep_research.progress import ActivityLog


def test_activity_cli_prints_summary(tmp_path: Path, capsys) -> None:
    artifacts = RunArtifacts.create(tmp_path, "activity cli")
    activity = ActivityLog(artifacts, progress_mode="quiet")
    activity.emit("search", "registered 1 source candidate")

    status = activity_cli.main([str(artifacts.run_dir), "--limit", "1"])

    output = capsys.readouterr().out
    assert status == 0
    assert "Activity:" in output
    assert "search=1" in output


def test_activity_cli_regenerates_html(tmp_path: Path, capsys) -> None:
    artifacts = RunArtifacts.create(tmp_path, "activity cli html")
    activity = ActivityLog(artifacts, progress_mode="quiet")
    activity.emit("verify", "passed")
    (artifacts.run_dir / "activity.html").unlink()

    status = activity_cli.main([str(artifacts.run_dir), "--html"])

    output = capsys.readouterr().out
    assert status == 0
    assert "Activity dashboard:" in output
    assert (artifacts.run_dir / "activity.html").exists()


def test_activity_cli_uses_latest_run_when_run_dir_is_omitted(tmp_path: Path, capsys) -> None:
    first = RunArtifacts.create(tmp_path, "first activity")
    first_activity = ActivityLog(first, progress_mode="quiet")
    first_activity.emit("search", "first run")
    second = RunArtifacts.create(tmp_path, "second activity")
    second_activity = ActivityLog(second, progress_mode="quiet")
    second_activity.emit("verify", "second run")

    status = activity_cli.main(["--out", str(tmp_path), "--limit", "1"])

    output = capsys.readouterr().out
    assert status == 0
    assert f"Activity: {second.run_dir.name}" in output
    assert "verify=1" in output


def test_activity_cli_rejects_run_dir_with_latest(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "activity conflict")

    try:
        activity_cli.resolve_run_dir(artifacts.run_dir, out_dir=tmp_path, latest=True)
    except SystemExit as exc:
        assert "Pass either run_dir or --latest" in str(exc)
    else:
        raise AssertionError("Expected conflicting run selection to fail.")

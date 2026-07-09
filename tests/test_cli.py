from deep_research import cli


def test_config_command_prints_configuration_guide(capsys) -> None:
    status = cli.main(["config"])

    captured = capsys.readouterr()
    assert status == 0
    assert "# Deep Research Configuration Guide" in captured.out
    assert "DEEP_RESEARCH_PROVIDER" in captured.out


def test_top_level_help_mentions_config_command(capsys) -> None:
    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr()
    assert "config" in captured.out
    assert "full configuration guide" in captured.out

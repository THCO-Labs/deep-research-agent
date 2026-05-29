from pathlib import Path

import pytest

from deep_research.settings import Settings
from deep_research.subagents import load_subagents
from deep_research.tools import ToolContext, build_tools
from deep_research.artifacts import RunArtifacts
from deep_research.source_registry import SourceRegistry


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        google_api_key="google",
        tavily_api_key="tavily",
    )


def test_load_subagents_maps_tools(tmp_path: Path) -> None:
    config = tmp_path / "subagents.yaml"
    config.write_text(
        """
researcher:
  description: Research
  system_prompt: Prompt
  tools:
    - web_search
""",
        encoding="utf-8",
    )
    artifacts = RunArtifacts.create(tmp_path, "subagents")
    tools = build_tools(ToolContext(_settings(tmp_path), artifacts, SourceRegistry(artifacts)))

    subagents = load_subagents(config, tools)

    assert subagents[0]["name"] == "researcher"
    assert subagents[0]["tools"][0].name == "web_search"


def test_load_subagents_can_override_models(tmp_path: Path) -> None:
    config = tmp_path / "subagents.yaml"
    config.write_text(
        """
researcher:
  description: Research
  model: google_genai:gemini-2.5-flash
  system_prompt: Prompt
  tools:
    - web_search
""",
        encoding="utf-8",
    )
    artifacts = RunArtifacts.create(tmp_path, "subagent model")
    tools = build_tools(ToolContext(_settings(tmp_path), artifacts, SourceRegistry(artifacts)))

    subagents = load_subagents(config, tools, model="groq:llama-3.1-8b-instant")

    assert subagents[0]["model"] == "groq:llama-3.1-8b-instant"


def test_load_subagents_rejects_unknown_tools(tmp_path: Path) -> None:
    config = tmp_path / "subagents.yaml"
    config.write_text(
        """
bad:
  description: Bad
  system_prompt: Prompt
  tools:
    - missing_tool
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError):
        load_subagents(config, {})

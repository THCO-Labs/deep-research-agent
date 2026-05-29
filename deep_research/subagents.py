from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from langchain.tools import BaseTool


def load_subagents(
    config_path: Path,
    available_tools: dict[str, BaseTool],
    *,
    model: str | None = None,
) -> list[dict[str, Any]]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    subagents = []
    for name, spec in config.items():
        unknown_tools = [tool_name for tool_name in spec.get("tools", []) if tool_name not in available_tools]
        if unknown_tools:
            joined = ", ".join(unknown_tools)
            raise KeyError(f"Subagent {name!r} references unknown tool(s): {joined}")
        subagent = {
            "name": name,
            "description": spec["description"],
            "system_prompt": spec["system_prompt"],
        }
        if model:
            subagent["model"] = model
        elif "model" in spec:
            subagent["model"] = spec["model"]
        if "tools" in spec:
            subagent["tools"] = [available_tools[tool_name] for tool_name in spec["tools"]]
        subagents.append(subagent)
    return subagents

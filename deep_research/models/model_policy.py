from __future__ import annotations

from deep_research.core.settings import ConfigError, Settings

TOOL_HEAVY_ROLES = {
    "model": "orchestrator",
    "planner_model": "planner",
    "researcher_model": "researcher",
    "verifier_model": "verifier",
    "judge_model": "judge",
}


def validate_strong_tool_models(settings: Settings) -> None:
    if not settings.strict_tool_models or settings.research_engine != "local_langgraph":
        return
    weak = []
    for attr, role in TOOL_HEAVY_ROLES.items():
        model = str(getattr(settings, attr))
        if model.startswith("ollama:"):
            weak.append(f"{role}={model}")
    if weak:
        raise ConfigError(
            "Strict research mode requires proven cloud/tool-calling models for tool-heavy roles. "
            "Weak local models can be used for utility roles only. Offending routes: "
            + ", ".join(weak)
        )


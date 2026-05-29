"""Public-web deep research agent package."""

from deep_research.settings import Settings

__all__ = ["ResearchRunError", "ResearchRunResult", "Settings", "run_research"]


def __getattr__(name: str):
    if name in {"ResearchRunError", "ResearchRunResult", "run_research"}:
        from deep_research.agent import ResearchRunError, ResearchRunResult, run_research

        return {
            "ResearchRunError": ResearchRunError,
            "ResearchRunResult": ResearchRunResult,
            "run_research": run_research,
        }[name]
    raise AttributeError(name)

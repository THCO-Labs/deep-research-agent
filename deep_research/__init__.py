"""Public-web deep research agent package."""

from __future__ import annotations

import os
import sys

if sys.platform == "win32" and os.environ.get("DEEP_RESEARCH_ALLOW_OPTIONAL_PYARROW", "").lower() not in {"1", "true", "yes"}:
    # pandas imports optional pyarrow during sklearn/nltk startup. Some Windows
    # pyarrow wheels fail at native module import time, taking the process down
    # after successful runs. The agent does not require pyarrow, so make pandas
    # treat it as unavailable while preserving sklearn functionality.
    sys.modules.setdefault("pyarrow", None)

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

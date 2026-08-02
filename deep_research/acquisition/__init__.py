from __future__ import annotations

from typing import Any

__all__ = [
    "AcquisitionMetrics",
    "AcquisitionResult",
    "TavilySearchClientPool",
    "_branch_queries",
    "_trim_search_query",
    "acquire_sources",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from deep_research.acquisition.acquisition import (
            AcquisitionMetrics,
            AcquisitionResult,
            TavilySearchClientPool,
            _branch_queries,
            _trim_search_query,
            acquire_sources,
        )
        _globals = globals()
        for sym in __all__:
            _globals[sym] = locals()[sym]
        return _globals[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

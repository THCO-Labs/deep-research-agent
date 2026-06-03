from __future__ import annotations

MINIMUM_SOURCE_TARGET = 17
MAX_TAVILY_RESULTS_PER_QUERY = 20


def source_floor(value: int | None, *, default: int = MINIMUM_SOURCE_TARGET) -> int:
    return max(MINIMUM_SOURCE_TARGET, int(value if value is not None else default))


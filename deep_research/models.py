from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SourceRecord:
    id: int
    url: str
    canonical_url: str
    title: str
    fetched_at: str
    extraction_method: str
    content_hash: str = ""
    content_path: str | None = None
    query: str | None = None
    snippet: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Metrics:
    search_count: int = 0
    scrape_count: int = 0
    write_count: int = 0
    read_count: int = 0
    python_exec_count: int = 0
    verification_rounds: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class VerificationResult:
    valid: bool
    citation_validity_score: float
    source_support_score: float = 1.0
    missing_sources: list[str] = field(default_factory=list)
    unused_sources: list[int] = field(default_factory=list)
    unscraped_sources: list[int] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    weakly_supported_claims: list[dict[str, Any]] = field(default_factory=list)
    support_checks: list[dict[str, Any]] = field(default_factory=list)
    source_list_errors: list[str] = field(default_factory=list)
    cited_source_ids: list[int] = field(default_factory=list)
    total_citations: int = 0
    verification_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

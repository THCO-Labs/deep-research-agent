from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypedDict

ResearchIntent = Literal["general"]
ResearchEngine = Literal["local_langgraph", "gemini_managed", "openai_managed"]
SourceProvenance = Literal["web", "local_file", "mcp", "managed_gemini", "managed_openai"]


@dataclass(frozen=True)
class SourceRequirement:
    source_type: str
    min_count: int = 1
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchBranch:
    id: str
    title: str
    objective: str
    queries: list[str]
    source_types: list[str] = field(default_factory=list)
    min_sources: int = 1
    required_terms: list[str] = field(default_factory=list)
    completion_criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    intent: ResearchIntent
    audience: str
    report_outline: list[str]
    branches: list[ResearchBranch]
    source_requirements: list[SourceRequirement] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    writer_persona: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 2
        return payload


@dataclass(frozen=True)
class SourceCandidate:
    id: int
    branch_id: str
    title: str
    url: str
    query: str
    snippet: str = ""
    search_score: float | None = None
    raw_content: str | None = None
    provenance: SourceProvenance = "web"
    search_intent_id: str = ""
    search_intent_goal: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceRecordV2:
    id: int
    branch_id: str
    title: str
    url: str
    canonical_url: str
    provenance: SourceProvenance
    content_path: str
    content_hash: str
    extraction_method: str
    word_count: int
    quality_score: float
    quality_label: str
    quality_type: str
    relevance_score: float
    usable: bool = True
    rejection_reasons: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceCard:
    id: int
    source_id: int
    branch_id: str
    claim: str
    supporting_excerpt: str
    source_url: str
    source_title: str
    quality_score: float
    relevance_score: float
    confidence: float
    limitations: list[str] = field(default_factory=list)
    semantic_score: float | None = None
    semantic_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchCoverage:
    branch_id: str
    required_points: list[str]
    covered_points: list[str]
    missing_points: list[str]
    source_count: int
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageMatrix:
    branches: list[BranchCoverage]
    complete: bool
    coverage_score: float
    missing_branches: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "branches": [branch.to_dict() for branch in self.branches],
            "complete": self.complete,
            "coverage_score": self.coverage_score,
            "missing_branches": self.missing_branches,
        }


@dataclass(frozen=True)
class VerificationResultV2:
    valid: bool
    citation_validity_score: float
    source_support_score: float
    answer_coverage_score: float
    branch_coverage_score: float
    evidence_linkage_score: float
    source_quality_score: float
    report_structure_score: float
    request_alignment_score: float = 1.0
    source_breadth_score: float = 1.0
    report_cleanliness_score: float = 1.0
    criteria_coverage_score: float = 1.0
    language_alignment_score: float = 1.0
    report_depth_score: float = 1.0
    cited_source_alignment_score: float = 1.0
    semantic_verification_score: float = 1.0
    semantic_verification: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    cited_source_ids: list[int] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    weakly_supported_claims: list[dict[str, Any]] = field(default_factory=list)
    undercovered_criteria: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = 2
        return payload


@dataclass(frozen=True)
class RunManifestV2:
    run_id: str
    question: str
    engine: ResearchEngine
    mode: str
    schema_version: int = 2
    managed_provider: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchState(TypedDict, total=False):
    request: dict[str, Any]
    plan: dict[str, Any]
    source_candidates: list[dict[str, Any]]
    source_records: list[dict[str, Any]]
    evidence_cards: list[dict[str, Any]]
    semantic_judgments: dict[str, Any]
    source_policy: dict[str, Any]
    evidence_graph: dict[str, Any]
    reasoning_state: dict[str, Any]
    reasoning_decision: dict[str, Any]
    reasoning_focus_terms: dict[str, list[str]]
    search_intents: list[dict[str, Any]]
    search_intent_results: list[dict[str, Any]]
    plan_revisions: dict[str, Any]
    contradiction_queries: list[dict[str, Any]]
    contradictions: list[dict[str, Any]]
    coverage_matrix: dict[str, Any]
    section_plan: dict[str, Any]
    knowledge_base: dict[str, Any]
    draft_report: str
    current_draft: dict[str, Any]
    verification: dict[str, Any]
    metrics: dict[str, Any]
    failures: list[str]
    next_step: str

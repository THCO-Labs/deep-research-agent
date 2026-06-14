import json
import time
from pathlib import Path
from types import SimpleNamespace

from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.evidence import build_evidence_cards
from deep_research.evidence_hygiene import apply_evidence_hygiene, report_quality_issues
from deep_research.acquisition import TavilySearchClientPool, acquire_sources, _branch_queries, _trim_search_query
from deep_research.ingestion import ingest_local_paths, ingest_mcp_manifest
from deep_research.managed import run_gemini_managed_research
from deep_research.planning import build_research_plan
from deep_research.coverage import build_coverage_matrix
from deep_research.guidance import format_criteria_guidance_block
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, ResearchState, SourceRecordV2, VerificationResultV2
from deep_research.semantic import (
    _load_json_object,
    apply_semantic_report_result,
    enrich_evidence_cards_with_semantics,
    verify_report_with_semantics,
)
from deep_research.semantic_planning import build_or_enrich_research_plan
from deep_research.semantic_planning import _loads_json_object
from deep_research.settings import Settings
from deep_research.scraper import ScrapeQualityError
from deep_research.source_validation import validate_source_content
from deep_research.synthesis import (
    _append_evidence_coverage_if_needed,
    _cards_for_synthesis,
    _coverage_repair_labels,
    _evidence_backed_sources,
    _normalize_report_markdown,
    _repair_weak_citation_support,
    _synthesis_model_spec,
    _synthesis_prompt,
    _synthesis_request_kwargs,
    _target_report_profile,
    build_claim_ledger,
    build_report_blueprint,
    build_sentence_plan,
    synthesize_report,
    synthesize_report_with_model,
)
from deep_research.verifier_v2 import _report_depth_score, _report_level_criteria, verify_report_v2
from deep_research.research_graph import (
    _acquire_route,
    _coverage_route,
    _focus_terms_from_state,
    _publish_best_draft,
    _select_best_draft,
    _selected_failed_draft,
    _semantic_gate_collapsed_coverage,
    _verification_route,
    _write_run_health,
)


from tests.test_v2_fakes import FakeGeminiClient, FakeSemanticJudge, InvalidSemanticJudge, QuotaSemanticJudge, RaisingSemanticJudge

def test_semantic_repair_focus_targets_missing_branches_only() -> None:
    state = {
        "plan": {
            "branches": [
                {"id": "branch_1", "required_terms": ["covered"]},
                {"id": "branch_2", "required_terms": ["missing"]},
            ]
        },
        "coverage_matrix": {
            "missing_branches": ["branch_2"],
            "branches": [
                {"branch_id": "branch_1", "complete": True, "missing_points": [], "required_points": []},
                {"branch_id": "branch_2", "complete": False, "missing_points": [], "required_points": ["usable sources >= 3"]},
            ],
        },
        "verification": {
            "semantic_verification": {
                "missing_context": ["direct evidence for the missing branch"],
                "search_focus": ["targeted follow-up"],
            }
        },
    }

    focus = _focus_terms_from_state(state)

    assert "branch_1" not in focus
    assert "branch_2" in focus
    assert "targeted follow-up" in focus["branch_2"]


def test_repair_focus_strips_internal_coverage_labels() -> None:
    state = {
        "plan": {
            "branches": [
                {
                    "id": "branch_2",
                    "required_terms": ["domain-specific misinformation", "information environment"],
                }
            ]
        },
        "coverage_matrix": {
            "missing_branches": ["branch_2"],
            "branches": [
                {
                    "branch_id": "branch_2",
                    "complete": False,
                    "missing_points": [
                        "required term coverage >= 55% (actual 33%)",
                        "required term: domain-specific misinformation",
                        "required term: information environment",
                        "branch evidence cards",
                    ],
                    "required_points": [],
                }
            ],
        },
        "verification": {},
    }

    focus = _focus_terms_from_state(state)

    assert focus["branch_2"] == ["domain-specific misinformation", "information environment"]
    assert all("required term" not in term.lower() for term in focus["branch_2"])
    assert all(">=" not in term for term in focus["branch_2"])


def test_verification_route_allows_one_issue_count_regression() -> None:
    state = {
        "verification": {"valid": False, "failures": ["weakly supported cited paragraph"] * 47},
        "metrics": {
            "verification_rounds": 2,
            "max_rounds": 4,
            "verification_failure_history": [8, 47],
        },
    }

    assert _verification_route(state) == "rewrite"


def test_verification_route_stops_after_second_issue_count_regression() -> None:
    state = {
        "verification": {"valid": False, "failures": ["weakly supported cited paragraph"] * 42},
        "metrics": {
            "verification_rounds": 4,
            "max_rounds": 6,
            "verification_failure_history": [8, 47, 40, 42],
        },
    }

    assert _verification_route(state) == "finish"


def test_verification_route_allows_repair_when_issue_count_improves() -> None:
    state = {
        "verification": {"valid": False, "failures": ["weakly supported cited paragraph"] * 11},
        "metrics": {
            "verification_rounds": 2,
            "max_rounds": 4,
            "verification_failure_history": [17, 11],
        },
    }

    assert _verification_route(state) == "rewrite"


def _quality_history_entry(
    index: int,
    failures: int,
    source_support: float,
    request_alignment: float,
) -> dict[str, object]:
    quality_scores = {
        "source_support_score": source_support,
        "evidence_linkage_score": 0.90,
        "citation_validity_score": 0.95,
        "request_alignment_score": request_alignment,
        "criteria_coverage_score": 0.82,
        "answer_coverage_score": 0.88,
        "branch_coverage_score": 0.90,
        "report_depth_score": 0.86,
        "semantic_verification_score": 0.84,
    }
    return {
        "draft_index": index,
        "draft_path": f"draft_report_{index}.md",
        "valid": False,
        "failure_count": failures,
        "quality_score": round(source_support + request_alignment - failures * 0.001, 6),
        "quality_scores": quality_scores,
    }


def test_verification_route_continues_past_soft_round_limit_while_improving() -> None:
    state = {
        "verification": {"valid": False, "failures": ["weakly supported cited paragraph"] * 10},
        "metrics": {
            "verification_rounds": 3,
            "max_rounds": 3,
            "verification_failure_history": [20, 12, 10],
            "draft_history": [
                _quality_history_entry(1, 20, 0.70, 0.80),
                _quality_history_entry(2, 12, 0.72, 0.82),
                _quality_history_entry(3, 10, 0.73, 0.83),
            ],
        },
    }

    assert _verification_route(state) == "rewrite"


def test_verification_route_allows_one_source_support_regression_despite_fewer_issues() -> None:
    state = {
        "verification": {"valid": False, "failures": ["weakly supported cited paragraph"] * 10},
        "metrics": {
            "verification_rounds": 3,
            "max_rounds": 6,
            "verification_failure_history": [20, 12, 10],
            "draft_history": [
                _quality_history_entry(1, 20, 0.71, 0.80),
                _quality_history_entry(2, 12, 0.732, 0.82),
                _quality_history_entry(3, 10, 0.704, 0.84),
            ],
        },
    }

    assert _verification_route(state) == "rewrite"


def test_verification_route_stops_after_second_source_support_regression() -> None:
    state = {
        "verification": {"valid": False, "failures": ["weakly supported cited paragraph"] * 9},
        "metrics": {
            "verification_rounds": 4,
            "max_rounds": 6,
            "verification_failure_history": [20, 12, 10, 9],
            "draft_history": [
                _quality_history_entry(1, 20, 0.71, 0.80),
                _quality_history_entry(2, 12, 0.732, 0.82),
                _quality_history_entry(3, 10, 0.704, 0.84),
                _quality_history_entry(4, 9, 0.681, 0.85),
            ],
        },
    }

    assert _verification_route(state) == "finish"


def test_verification_route_stops_at_hard_cap_even_when_improving() -> None:
    state = {
        "verification": {"valid": False, "failures": ["weakly supported cited paragraph"] * 5},
        "metrics": {
            "verification_rounds": 8,
            "max_rounds": 3,
            "verification_failure_history": [20, 12, 10, 9, 8, 7, 6, 5],
            "draft_history": [
                _quality_history_entry(index, failures, 0.70 + index / 100, 0.80)
                for index, failures in enumerate([20, 12, 10, 9, 8, 7, 6, 5], start=1)
            ],
        },
    }

    assert _verification_route(state) == "finish"


def test_acquire_route_reuses_existing_evidence_when_no_new_sources_or_candidates() -> None:
    no_progress_state = {
        "metrics": {
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "last_acquire_searches": 9,
        },
        "evidence_cards": [{"id": 1, "source_id": 1}],
    }
    progress_state = {
        "metrics": {
            "last_acquire_added_sources": 1,
            "last_acquire_added_candidates": 2,
            "last_acquire_searches": 0,
        },
        "evidence_cards": [{"id": 1, "source_id": 1}],
    }

    assert _acquire_route(no_progress_state) == "reuse_evidence"
    assert _acquire_route(progress_state) == "read_sources"


def test_coverage_route_finishes_when_no_evidence_and_acquisition_plateaued() -> None:
    state = {
        "coverage_matrix": {"complete": False},
        "evidence_cards": [],
        "metrics": {
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "coverage_rounds": 1,
            "search_count": 30,
            "max_search_queries": 96,
        },
    }

    assert _coverage_route(state) == "finish"


def test_coverage_route_continues_when_resume_budget_expands_after_plateau() -> None:
    state = {
        "coverage_matrix": {"complete": False},
        "evidence_cards": [{"id": 1, "source_id": 1}],
        "metrics": {
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "candidate_count_total": 900,
            "max_candidates": 5000,
            "coverage_rounds": 2,
            "search_count": 0,
            "max_search_queries": 192,
            "max_rounds": 8,
        },
    }

    assert _coverage_route(state) == "more_sources"


def test_verification_route_rewrites_unsupported_claims_before_more_search() -> None:
    state = {
        "verification": {
            "valid": False,
            "failures": [
                "Semantic judge found unsupported claim: the report overstates a mixed finding.",
                "Cited evidence-backed source count below threshold: 10 < 17",
            ],
        },
        "metrics": {"verification_rounds": 1, "max_rounds": 4},
    }

    assert _verification_route(state) == "rewrite"


def test_verification_route_rewrites_writing_and_support_failures_without_more_search() -> None:
    state = {
        "coverage_matrix": {"complete": True},
        "verification": {
            "valid": False,
            "failures": [
                "Weakly supported cited paragraph: the report overgeneralizes a mechanism.",
                "Acceptance criteria coverage below threshold: 0.59",
                "Report depth below threshold: 0.76",
                "Semantic report judge returned invalid structured output: bad json",
            ],
        },
        "metrics": {
            "verification_rounds": 1,
            "max_rounds": 4,
            "last_acquire_added_sources": 3,
            "last_acquire_added_candidates": 20,
        },
    }

    assert _verification_route(state) == "rewrite"


def test_coverage_route_finishes_low_coverage_after_zero_progress_followup() -> None:
    state = {
        "coverage_matrix": {"complete": False, "coverage_score": 0.60, "missing_branches": ["branch_1"]},
        "evidence_cards": [{"id": 1}],
        "metrics": {
            "coverage_rounds": 2,
            "max_rounds": 8,
            "search_count": 31,
            "max_search_queries": 80,
            "candidate_count_total": 132,
            "max_candidates": 750,
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "last_acquire_searches": 0,
        },
    }

    assert _coverage_route(state) == "finish"


def test_coverage_route_synthesizes_strong_partial_coverage_after_zero_progress_followup() -> None:
    state = {
        "coverage_matrix": {"complete": False, "coverage_score": 0.90, "missing_branches": ["branch_5"]},
        "evidence_cards": [{"id": index} for index in range(12)],
        "metrics": {
            "coverage_rounds": 2,
            "max_rounds": 8,
            "search_count": 31,
            "max_search_queries": 80,
            "candidate_count_total": 132,
            "max_candidates": 750,
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "last_acquire_searches": 0,
        },
    }

    assert _coverage_route(state) == "synthesize"


def test_coverage_route_synthesizes_evidence_limited_report_after_source_cap() -> None:
    state = {
        "coverage_matrix": {
            "complete": False,
            "coverage_score": 0.70,
            "missing_branches": ["branch_3", "branch_4", "branch_5"],
        },
        "evidence_cards": [{"id": index} for index in range(15)],
        "metrics": {
            "coverage_rounds": 2,
            "max_rounds": 8,
            "search_count": 28,
            "max_search_queries": 80,
            "source_count": 40,
            "max_sources": 40,
            "candidate_count_total": 97,
            "max_candidates": 100,
            "last_acquire_added_sources": 6,
            "last_acquire_added_candidates": 23,
            "last_acquire_searches": 6,
        },
    }

    assert _coverage_route(state) == "synthesize"


def test_coverage_route_synthesizes_benchmark_ready_partial_deck_after_plateau() -> None:
    state = {
        "coverage_matrix": {
            "complete": False,
            "coverage_score": 0.80,
            "missing_branches": ["branch_4", "branch_5"],
        },
        "metrics": {
            "coverage_rounds": 2,
            "max_rounds": 4,
            "search_count": 7,
            "max_search_queries": 7,
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "last_acquire_searches": 0,
        },
        "evidence_cards": [{"id": index, "source_id": index} for index in range(1, 43)],
    }

    assert _coverage_route(state) == "synthesize"


def test_semantic_gate_collapse_detection_prefers_pre_gate_evidence() -> None:
    branches = [
        ResearchBranch(
            id=f"branch_{index}",
            title=f"Branch {index}",
            objective=f"Cover branch {index} evidence.",
            queries=[f"branch {index} evidence"],
            required_terms=[f"term{index}"],
        )
        for index in range(1, 6)
    ]
    plan = ResearchPlan(
        question="Benchmark question",
        intent="market_analysis",
        audience="analyst",
        report_outline=[],
        branches=branches,
    )
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=branch.id,
            title=f"Source {index}",
            url=f"https://example.com/{index}",
            canonical_url=f"https://example.com/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash=f"hash-{index}",
            extraction_method="test",
            word_count=500,
            quality_score=0.9,
            quality_label="high",
            quality_type="academic",
            relevance_score=0.9,
        )
        for index, branch in enumerate(branches, start=1)
    ]
    before_cards = [
        EvidenceCard(
            id=index,
            source_id=((index - 1) % 5) + 1,
            branch_id=f"branch_{((index - 1) % 5) + 1}",
            claim=f"term{((index - 1) % 5) + 1} evidence claim {index}",
            supporting_excerpt=f"term{((index - 1) % 5) + 1} evidence excerpt {index}",
            source_url=f"https://example.com/{((index - 1) % 5) + 1}",
            source_title=f"Source {((index - 1) % 5) + 1}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 16)
    ]

    assert _semantic_gate_collapsed_coverage(
        plan,
        sources,
        before_cards=before_cards,
        after_cards=[card for card in before_cards if card.branch_id == "branch_1"],
    )


def test_verification_route_rewrites_instead_of_researching_after_source_plateau() -> None:
    state = {
        "verification": {
            "valid": False,
            "failures": [
                "Branch coverage incomplete: branch_2",
                "Branch coverage below threshold: 0.64",
            ],
        },
        "metrics": {
            "verification_rounds": 1,
            "max_rounds": 4,
            "last_acquire_added_sources": 0,
            "last_acquire_added_candidates": 0,
            "last_acquire_searches": 0,
        },
    }

    assert _verification_route(state) == "rewrite"


def test_ingest_mcp_manifest_reads_content(tmp_path: Path) -> None:
    manifest = tmp_path / "mcp.json"
    manifest.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "title": "Connector Doc",
                        "url": "mcp://docs/1",
                        "content": "Urban heat island public health connector evidence " * 20,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    docs = ingest_mcp_manifest(manifest)

    assert len(docs) == 1
    assert docs[0].provenance == "mcp"
    assert "Urban heat island public health connector evidence" in docs[0].content


def test_ingest_local_paths_reads_unknown_text_suffix_and_skips_binary_in_directory(tmp_path: Path) -> None:
    text_doc = tmp_path / "field-notes.research"
    text_doc.write_text("Community cooling center usage and heat-health planning evidence " * 12, encoding="utf-8")
    binary_doc = tmp_path / "image.bin"
    binary_doc.write_bytes(b"\x00\x01\x02\x03\x04")

    docs = ingest_local_paths([tmp_path])

    assert len(docs) == 1
    assert docs[0].title == "field-notes"
    assert "Community cooling center usage" in docs[0].content


def test_gemini_managed_interaction_lifecycle_is_converted_to_v2_artifacts(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        research_engine="gemini_managed",
        google_api_key="google",
        google_api_keys=("google",),
        tavily_api_key="",
    )
    artifacts = ResearchArtifactsV2.create(tmp_path, "managed")
    client = FakeGeminiClient()

    result = run_gemini_managed_research(
        question="managed question",
        settings=settings,
        artifacts=artifacts,
        client=client,
        poll_interval_seconds=0,
    )

    assert client.created["agent"] == "deep-research-pro-preview-12-2025"
    assert client.created["background"] is True
    assert result.verification.valid is True
    assert (artifacts.run_dir / "report.md").exists()
    assert (artifacts.run_dir / "sources.jsonl").exists()

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

def test_synthesis_appends_evidence_coverage_for_missing_branches_and_sources() -> None:
    branches = [
        ResearchBranch(
            id="branch_1",
            title="Cooling center access",
            objective="Explain access to cooling centers during heat waves.",
            queries=["cooling center access heat waves"],
            required_terms=["cooling centers", "access"],
        ),
        ResearchBranch(
            id="branch_2",
            title="Transportation barriers",
            objective="Explain transportation barriers that limit cooling center use.",
            queries=["cooling center transportation barriers"],
            required_terms=["transportation barriers", "cooling center use"],
        ),
    ]
    plan = ResearchPlan(
        question="How do cooling centers reduce heat illness risk, and what limits their use?",
        intent="general",
        audience="general",
        report_outline=[branch.title for branch in branches],
        branches=branches,
    )
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id="branch_1",
            claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
            supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
            source_url="https://example.com/1",
            source_title="Source 1",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id="branch_2",
            claim="Transportation barriers can limit whether residents can use cooling centers during heat waves.",
            supporting_excerpt="Transportation barriers can limit whether residents can use cooling centers during heat waves.",
            source_url="https://example.com/2",
            source_title="Source 2",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Cooling Centers\n\n"
        "Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces. [1]\n\n"
        "## Sources\n\n"
        "[1] Source 1: https://example.com/1\n"
    )

    repaired = _append_evidence_coverage_if_needed(report, plan, cards)

    assert "Transportation barriers" in repaired
    assert "[2]" in repaired.split("## Sources")[0]


def test_synthesis_does_not_append_fragmented_coverage_for_criteria_rich_reports() -> None:
    branches = [
        ResearchBranch(
            id="branch_1",
            title="Cooling centers",
            objective="Explain how cooling centers reduce heat illness risk.",
            queries=["cooling centers heat illness risk"],
            required_terms=["cooling centers", "heat illness risk"],
        ),
        ResearchBranch(
            id="branch_2",
            title="Access barriers",
            objective="Explain barriers that limit access.",
            queries=["cooling center access barriers"],
            required_terms=["access barriers", "heat illness risk"],
        ),
    ]
    plan = ResearchPlan(
        question="How do cooling centers reduce heat illness risk, and what limits their use?",
        intent="general",
        audience="general",
        report_outline=[branch.title for branch in branches],
        branches=branches,
        acceptance_criteria=[
            f"Cover this task-specific insight criterion in synthesis: Criterion {index} requires evidence, mechanisms, limits, and implications."
            for index in range(1, 10)
        ],
    )
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id="branch_1",
            claim="Cooling centers reduce heat illness risk by giving residents cooler indoor spaces.",
            supporting_excerpt="Cooling centers reduce heat illness risk by giving residents cooler indoor spaces.",
            source_url="https://example.com/1",
            source_title="Source 1",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id="branch_2",
            claim="Access barriers limit whether residents can use cooling centers during heat waves.",
            supporting_excerpt="Access barriers limit whether residents can use cooling centers during heat waves.",
            source_url="https://example.com/2",
            source_title="Source 2",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Cooling Centers\n\n"
        "Cooling centers reduce heat illness risk by giving residents cooler indoor spaces. [1]\n\n"
        "## Sources\n\n"
        "[1] Source 1: https://example.com/1\n"
    )

    repaired = _append_evidence_coverage_if_needed(report, plan, cards)

    assert repaired == report
    assert "Additional Source-Backed Analysis" not in repaired
    assert "[2]" not in repaired.split("## Sources")[0]


def test_evidence_hygiene_rejects_boilerplate_and_metadata_cards() -> None:
    clean = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="Clean evidence states one supported claim in natural prose without copied page chrome.",
        supporting_excerpt="Clean evidence states one supported claim in natural prose without copied page chrome.",
        source_url="https://example.com/clean",
        source_title="Clean",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    noisy = EvidenceCard(
        id=2,
        source_id=2,
        branch_id="branch_1",
        claim="URL: https://example.com Canonical URL: https://example.com Branch: branch_1 Extraction method: raw",
        supporting_excerpt="URL: https://example.com Canonical URL: https://example.com Branch: branch_1 Extraction method: raw",
        source_url="https://example.com/noisy",
        source_title="Noisy",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    result = apply_evidence_hygiene([clean, noisy])

    assert [card.id for card in result.kept] == [1]
    assert result.rejected[0]["card"]["id"] == 2
    assert any("artifact" in reason or "metadata" in reason for reason in result.rejected[0]["reasons"])


def test_verifier_rejects_report_boilerplate_even_with_valid_citation() -> None:
    plan = build_research_plan("What are model adaptation methods?")
    branch_id = plan.branches[0].id
    source = SourceRecordV2(
        id=1,
        branch_id=branch_id,
        title="Model Adaptation Source",
        url="https://example.com/source",
        canonical_url="https://example.com/source",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="official_docs",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch_id,
        claim="Model adaptation changes pretrained systems so they perform better on specific tasks and formats.",
        supporting_excerpt="Model adaptation changes pretrained systems so they perform better on specific tasks and formats.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    coverage = CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[])
    report = (
        "# Research Report\n\n"
        "## Main Finding\n\n"
        "Model adaptation changes pretrained systems so they perform better on specific tasks and formats. [1]\n\n"
        "Share [](https://www.linkedin.com/shareArticle?url=https://example.com). [1]\n\n"
        "## Evidence Handling\n\n"
        "Scope note.\n\n"
        "## Adaptation Effects\n\n"
        "Model adaptation changes pretrained systems so they perform better on specific tasks and formats. [1]\n\n"
        "## Confidence Notes\n\n"
        "Confidence note.\n\n"
        "## Sources\n\n"
        "[1] Model Adaptation Source: https://example.com/source\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=coverage,
        source_texts={1: card.supporting_excerpt},
    )

    assert result.valid is False
    assert result.report_cleanliness_score < 1.0
    assert any("artifact" in failure or "URL-heavy" in failure for failure in result.failures)


def test_verifier_accepts_flexible_non_template_report_headings() -> None:
    plan = build_research_plan("How do urban heat islands affect public health?")
    all_plan_text = " ".join(
        branch.title + " " + branch.objective + " " + " ".join(branch.required_terms)
        for branch in plan.branches
    )
    source = SourceRecordV2(
        id=1,
        branch_id=plan.branches[0].id,
        title="Urban Heat Source",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=300,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=plan.branches[0].id,
        claim=(
            "Urban heat islands affect public health through heat exposure, vulnerable populations, "
            "health outcomes, mitigation choices, evidence tradeoffs, applications, risks, limitations, costs, safeguards, "
            "and decision guidance."
        ),
        supporting_excerpt=all_plan_text,
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    coverage = CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[])
    body = (
        "Urban heat islands affect public health through heat exposure, vulnerable populations, health outcomes, "
        "mitigation choices, evidence tradeoffs, applications, risks, limitations, costs, safeguards, and decision guidance. "
        f"{all_plan_text} [1]"
    )
    report = (
        "# Urban Heat and Public Health\n\n"
        "## The Core Relationship\n\n"
        f"{body}\n\n"
        "## How the Evidence Connects\n\n"
        f"Taken together, the evidence indicates that public health risk, methods, implementation, performance, "
        f"applications, examples, risks, limitations, costs, safeguards, best practices, decision criteria, and alternatives "
        f"must be read as connected factors rather than isolated findings. {all_plan_text} [1]\n\n"
        "## What Remains Uncertain\n\n"
        f"Confidence is limited by source scope and by evidence gaps around local context, but the available evidence "
        f"supports the main public-health pattern. {all_plan_text} [1]\n\n"
        "## Sources\n\n"
        "[1] Urban Heat Source: https://example.com/urban-heat\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=coverage,
        source_texts={1: body + " " + all_plan_text},
    )

    assert result.valid is True
    assert result.report_structure_score >= 0.60


def test_semantic_evidence_enrichment_filters_unentailed_cards(tmp_path: Path) -> None:
    plan = build_research_plan("How do cooling centers reduce heat illness risk during heat waves?")
    branch = plan.branches[0]
    strong = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
        supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves.",
        source_url="https://example.com/cooling",
        source_title="Cooling Centers",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    weak = EvidenceCard(
        id=2,
        source_id=2,
        branch_id=branch.id,
        claim="Invoice systems reconcile approvals and payments.",
        supporting_excerpt="Invoice systems reconcile approvals and payments.",
        source_url="https://example.com/invoices",
        source_title="Invoices",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    judge = FakeSemanticJudge(
        {
            "cards": [
                {
                    "id": 1,
                    "keep": True,
                    "branch_alignment_score": 0.94,
                    "entailment_score": 0.96,
                    "evidence_relevance_score": 0.95,
                    "normalized_claim": "Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
                    "key_points": ["cooler indoor spaces"],
                    "limitations": [],
                    "failure_reasons": [],
                },
                {
                    "id": 2,
                    "keep": False,
                    "branch_alignment_score": 0.05,
                    "entailment_score": 0.80,
                    "evidence_relevance_score": 0.02,
                    "normalized_claim": "Invoice systems reconcile approvals and payments.",
                    "key_points": [],
                    "limitations": [],
                    "failure_reasons": ["irrelevant to the research question"],
                },
            ]
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    result = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[strong, weak],
        settings=settings,
        model=judge,
    )

    assert [card.id for card in result.cards] == [1]
    assert result.cards[0].semantic_score == 0.95
    assert result.rejected[0]["card"]["id"] == 2
    assert result.failures == []
    assert "Return exactly one JSON object" in judge.prompts[0]


def test_semantic_evidence_enrichment_falls_back_on_invalid_json(tmp_path: Path) -> None:
    plan = build_research_plan("How does need for closure affect misinformation acceptance?")
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=plan.branches[0].id,
        claim="Need for closure can increase reliance on quick judgments when evaluating misinformation.",
        supporting_excerpt="Need for closure can increase reliance on quick judgments when evaluating misinformation.",
        source_url="https://example.com/nfc",
        source_title="Need for Closure Evidence",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    judge = InvalidSemanticJudge()
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    result = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=judge,
    )

    assert [kept.id for kept in result.cards] == [1]
    assert result.failures == []
    assert result.metrics["semantic_evidence_parse_failure_count"] == 1
    assert result.judgments[0]["fallback"] == "deterministic_after_invalid_judge_output"


def test_semantic_evidence_enrichment_uses_large_deck_fallback(tmp_path: Path) -> None:
    plan = build_research_plan("How does need for closure affect misinformation acceptance?")
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=plan.branches[0].id,
            claim=f"Need for closure evidence card {index} explains misinformation acceptance through quick judgments.",
            supporting_excerpt=f"Need for closure evidence card {index} explains misinformation acceptance through quick judgments.",
            source_url=f"https://example.com/{index}",
            source_title=f"Source {index}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 4)
    ]
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        semantic_verification=True,
        semantic_evidence_max_llm_cards=2,
    )

    result = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=cards,
        settings=settings,
        model=RaisingSemanticJudge(),
    )

    assert [card.id for card in result.cards] == [1, 2, 3]
    assert result.failures == []
    assert result.metrics["semantic_evidence_large_deck_fallback"] is True
    assert result.metrics["semantic_evidence_batches"] == 0


def test_semantic_report_verification_can_fail_valid_deterministic_result(tmp_path: Path) -> None:
    plan = build_research_plan("How do cooling centers reduce heat illness risk during heat waves?")
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=plan.branches[0].id,
        claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
        supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves.",
        source_url="https://example.com/cooling",
        source_title="Cooling Centers",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    deterministic = VerificationResultV2(
        valid=True,
        citation_validity_score=1.0,
        source_support_score=1.0,
        answer_coverage_score=1.0,
        branch_coverage_score=1.0,
        evidence_linkage_score=1.0,
        source_quality_score=1.0,
        report_structure_score=1.0,
    )
    judge = FakeSemanticJudge(
        {
            "answer_completeness_score": 0.4,
            "citation_entailment_score": 0.5,
            "evidence_use_score": 0.6,
            "contradiction_safety_score": 1.0,
            "overall_score": 0.5,
            "failures": ["missing context about who can access cooling centers"],
            "missing_context": ["access barriers"],
            "unsupported_claims": ["Cooling centers eliminate heat-wave risk."],
            "contradictions": [],
            "search_focus": ["cooling center access barriers"],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    semantic = verify_report_with_semantics(
        report_markdown="## Executive Summary\n\nCooling centers eliminate heat-wave risk. [1]",
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=judge,
    )
    merged = apply_semantic_report_result(deterministic, semantic)

    assert merged.valid is False
    assert merged.semantic_verification_score == 0.5
    assert any("unsupported claim" in failure.lower() for failure in merged.failures)
    assert merged.semantic_verification["search_focus"] == ["cooling center access barriers"]


def test_semantic_json_loader_accepts_python_literal_objects() -> None:
    payload = _load_json_object(
        "{'overall_score': 0.82, 'failures': [], 'unsupported_claims': [], "
        "'contradictions': [], 'missing_context': [], 'search_focus': []}"
    )

    assert payload["overall_score"] == 0.82
    assert payload["failures"] == []


def test_semantic_report_verification_accepts_json_like_failure_entries(tmp_path: Path) -> None:
    plan = build_research_plan("How do cooling centers reduce heat illness risk during heat waves?")
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=plan.branches[0].id,
        claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
        supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces.",
        source_url="https://example.com/cooling",
        source_title="Cooling Centers",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    judge = FakeSemanticJudge(
        {
            "answer_completeness_score": 0.8,
            "citation_entailment_score": 0.8,
            "evidence_use_score": 0.8,
            "contradiction_safety_score": 1.0,
            "overall_score": 0.8,
            "failures": [{"issue": "minor scope note"}],
            "missing_context": [],
            "unsupported_claims": [],
            "contradictions": [],
            "search_focus": [],
        }
    )

    semantic = verify_report_with_semantics(
        report_markdown="Cooling centers reduce heat illness risk. [1]",
        plan=plan,
        evidence_cards=[card],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True),
        model=judge,
    )

    assert semantic.score == 0.8
    assert '{"issue": "minor scope note"}' in semantic.failures


def test_normalize_report_sources_section_includes_multi_citation_ids() -> None:
    sources = [
        SourceRecordV2(
            id=index,
            branch_id="branch_1",
            title=f"Source {index}",
            url=f"https://example.com/{index}",
            canonical_url=f"https://example.com/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash="hash",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="high",
            quality_type="academic",
            relevance_score=0.9,
        )
        for index in (1, 10)
    ]
    report = "# Report\n\nThe evidence combines two source records. [1, 10]\n"

    normalized = _normalize_report_markdown(report, sources)

    assert "[1] Source 1: https://example.com/1" in normalized
    assert "[10] Source 10: https://example.com/10" in normalized


def test_normalize_report_removes_malformed_source_dump_before_canonical_sources() -> None:
    sources = [
        SourceRecordV2(
            id=1,
            branch_id="branch_1",
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="high",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id="branch_1",
            title="Misinformation Source",
            url="https://example.com/misinfo",
            canonical_url="https://example.com/misinfo",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="high",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]
    raw = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can shape misinformation acceptance through quick certainty and reduced scrutiny. [1]\n\n"
        "## Sources  [1] Old Source: https://old.example/source\n"
        "[2] Misinformation Source: https://example.com/misinfo\n"
        "Unnumbered Source: https://example.com/unlisted\n"
        "More copied metadata: https://example.com/metadata\n\n"
        "---\n\n"
        "*End of Report*\n"
    )

    normalized = _normalize_report_markdown(raw, sources)
    body = normalized.split("## Sources")[0]

    assert "https://old.example/source" not in normalized
    assert "Unnumbered Source" not in normalized
    assert "*End of Report*" not in body
    assert normalized.count("## Sources") == 1
    assert "[1] Need for Closure Source: https://example.com/nfc" in normalized
    assert "[2] Misinformation Source: https://example.com/misinfo" not in normalized


def test_report_blueprint_and_writer_use_professional_report_sections() -> None:
    plan = build_research_plan("What are urban heat islands and how do they affect public health?")
    branch = plan.branches[0]
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Urban Heat Evidence",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="official_docs",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Urban heat islands raise local temperatures and increase heat-related public health risks.",
        supporting_excerpt="Urban heat islands raise local temperatures and increase heat-related public health risks.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    coverage = CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[])

    blueprint = build_report_blueprint(plan=plan, evidence_cards=[card], coverage=coverage, sources=[source])
    report = synthesize_report(plan=plan, evidence_cards=[card], coverage=coverage, sources=[source])

    examples = [example["name"] for example in blueprint["style_examples"]]
    assert "Analytical explainer" in examples
    assert "Evidence review" in examples
    assert "quality_contract" in blueprint
    assert "depth_and_insight" in blueprint["quality_contract"]
    assert blueprint["source_summary"]["evidence_card_count"] == 1
    assert "section_contract" not in blueprint
    assert "## What the Sources Show Together" in report
    assert "## Implications" in report
    assert "## Sources" in report
    assert "[1] Urban Heat Evidence: https://example.com/urban-heat" in report


def test_claim_ledger_is_written_as_factual_boundary_for_synthesis() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Urban heat health effects",
        objective="Explain how urban heat affects public health.",
        queries=["urban heat health effects"],
        required_terms=["urban heat", "public health"],
    )
    plan = ResearchPlan(
        question="How does urban heat affect public health?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Urban Heat Evidence",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="official_docs",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Urban heat increases public health risks by raising local temperatures and heat exposure.",
        supporting_excerpt="Urban heat increases public health risks by raising local temperatures and heat exposure.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    ledger = build_claim_ledger(plan=plan, evidence_cards=[card], sources=[source])
    prompt = _synthesis_prompt(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
        previous_report="",
        verification_failures=[],
    )

    assert ledger["claim_count"] == 1
    assert ledger["claims"][0]["claim_id"] == "C001"
    assert ledger["claims"][0]["source_id"] == 1
    assert "Claim ledger - factual ground for the report" in prompt
    assert "Use the claim ledger as factual ground, not as a cage" in prompt
    assert "Cross-source interpretation is allowed and expected for RACE quality" in prompt
    assert "C001 (" in prompt
    assert "-> cite [1]" in prompt


def test_claim_ledger_preserves_breadth_for_literature_reviews() -> None:
    branches = [
        ResearchBranch(
            id=f"branch_{branch_index}",
            title=f"AI labor market theme {branch_index}",
            objective=f"Cover AI labor market restructuring theme {branch_index}.",
            queries=[f"AI labor market theme {branch_index}"],
            required_terms=["AI", "labor market", f"theme {branch_index}"],
        )
        for branch_index in range(1, 4)
    ]
    plan = ResearchPlan(
        question="Please write a literature review on AI and labor market restructuring.",
        intent="literature_review",
        audience="researcher",
        report_outline=[branch.title for branch in branches],
        branches=branches,
    )
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=branches[(index - 1) % len(branches)].id,
            title=f"Journal Article {index}",
            url=f"https://example.com/article-{index}",
            canonical_url=f"https://example.com/article-{index}",
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
        for index in range(1, 31)
    ]
    claim_templates = [
        "AI labor market restructuring evidence defines the task exposure concept for workers and firms.",
        "AI labor market restructuring evidence explains a mechanism through skills, jobs, and sectors.",
        "AI labor market restructuring evidence reports a 12 percent employment adjustment in exposed occupations.",
        "AI labor market restructuring evidence compares higher-exposure and lower-exposure sectors.",
        "AI labor market restructuring evidence notes uncertainty, limitations, and gaps in causal evidence.",
        "AI labor market restructuring evidence suggests policy and strategy implications for worker transitions.",
    ]
    cards = []
    for index in range(1, 31):
        claim = claim_templates[(index - 1) % len(claim_templates)]
        cards.append(
            EvidenceCard(
                id=index,
                source_id=index,
                branch_id=sources[index - 1].branch_id,
                claim=claim,
                supporting_excerpt=claim,
                source_url=sources[index - 1].url,
                source_title=sources[index - 1].title,
                quality_score=0.9,
                relevance_score=0.9,
                confidence=0.9,
            )
        )

    ledger = build_claim_ledger(plan=plan, evidence_cards=cards, sources=sources)
    sentence_plan = build_sentence_plan(
        plan=plan,
        evidence_cards=cards,
        sources=sources,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        claim_ledger=ledger,
    )
    prompt = _synthesis_prompt(
        plan=plan,
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=sources,
        previous_report="",
        verification_failures=[],
    )

    assert ledger["claim_count"] >= 28
    assert len(ledger["section_briefs"]) == 3
    assert ledger["safe_synthesis_frames"]
    assert sentence_plan["planned_claim_count"] >= 24
    assert sentence_plan["planned_source_count"] >= 24
    assert len(sentence_plan["sections"]) == 4
    assert "section_thesis" in sentence_plan["role_distribution"]
    assert any(
        spec["purpose"] == "implication"
        for section in sentence_plan["sections"]
        for spec in section["sentence_specs"]
    )
    assert "Section-level claim plan" in prompt
    assert "Safe synthesis moves" in prompt
    assert "Sentence-level content plan" in prompt
    assert "Follow the sentence-level content plan" in prompt
    assert "draw on at least 28 distinct ledger claims" in prompt


def test_coverage_repair_uses_neutral_expansion_heading() -> None:
    labels = _coverage_repair_labels("How does AI affect labor markets?")

    assert labels["heading"] == "Evidence Coverage Expansion:"
    assert "Additional Source-Backed Analysis" not in labels["heading"]

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

def test_synthesis_prompt_requires_user_request_language() -> None:
    question = "\u8bf7\u5206\u6790\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u5f71\u54cd\u516c\u5171\u5065\u5eb7"
    branch = ResearchBranch(
        id="branch_1",
        title="\u57ce\u5e02\u70ed\u5c9b\u4e0e\u516c\u5171\u5065\u5eb7",
        objective="\u8bf4\u660e\u57ce\u5e02\u70ed\u5c9b\u5bf9\u516c\u5171\u5065\u5eb7\u7684\u5f71\u54cd\u3002",
        queries=["\u57ce\u5e02\u70ed\u5c9b \u516c\u5171\u5065\u5eb7"],
        min_sources=1,
        required_terms=["\u57ce\u5e02\u70ed\u5c9b", "\u516c\u5171\u5065\u5eb7"],
    )
    plan = ResearchPlan(
        question=question,
        intent="general",
        audience="general",
        report_outline=[],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="\u57ce\u5e02\u70ed\u5c9b\u8bc1\u636e",
        url="https://example.com/urban-heat-zh",
        canonical_url="https://example.com/urban-heat-zh",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="\u57ce\u5e02\u70ed\u5c9b\u4f1a\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        supporting_excerpt="\u57ce\u5e02\u70ed\u5c9b\u4f1a\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    prompt = _synthesis_prompt(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
        previous_report="",
        verification_failures=[],
    )

    assert "Use Simplified Chinese prose" in prompt
    assert "Do not let a narrower context" in prompt
    assert "Report quality contract" in prompt
    assert "Open with the answer, not background" in prompt


def test_synthesis_filters_allowed_sources_to_evidence_backed_sources() -> None:
    source_with_card = SourceRecordV2(
        id=1,
        branch_id="branch_1",
        title="Evidence Source",
        url="https://example.com/evidence",
        canonical_url="https://example.com/evidence",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash-1",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    source_without_card = SourceRecordV2(
        id=2,
        branch_id="branch_1",
        title="No Evidence Source",
        url="https://example.com/no-evidence",
        canonical_url="https://example.com/no-evidence",
        provenance="web",
        content_path="source_docs/source_2.md",
        content_hash="hash-2",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="Evidence-backed sources are the only sources the report may cite.",
        supporting_excerpt="Evidence-backed sources are the only sources the report may cite.",
        source_url=source_with_card.url,
        source_title=source_with_card.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    result = _evidence_backed_sources([source_with_card, source_without_card], [card])

    assert [source.id for source in result] == [1]


def test_synthesis_repairs_weak_paragraph_citation_support() -> None:
    sources = [
        SourceRecordV2(
            id=1,
            branch_id="branch_1",
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash-1",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id="branch_1",
            title="Misinformation Source",
            url="https://example.com/misinformation",
            canonical_url="https://example.com/misinformation",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash-2",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id="branch_1",
            claim="Need for closure is a desire for quick certainty and firm answers.",
            supporting_excerpt="Need for closure is a desire for quick certainty and firm answers.",
            source_url=sources[0].url,
            source_title=sources[0].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id="branch_1",
            claim="Misinformation acceptance involves believing inaccurate information and can be shaped by heuristics.",
            supporting_excerpt="Misinformation acceptance involves believing inaccurate information and can be shaped by heuristics.",
            source_url=sources[1].url,
            source_title=sources[1].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Report\n\n"
        "Need for closure can shape misinformation acceptance because people seeking quick certainty may rely on "
        "heuristics and believe inaccurate information. [1]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
    )

    repaired = _repair_weak_citation_support(report, cards, sources, threshold=0.35)

    assert "[2]" in repaired.split("## Sources")[0]


def test_synthesis_repair_removes_individually_unsupported_citation() -> None:
    sources = [
        SourceRecordV2(
            id=1,
            branch_id="branch_1",
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash-1",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id="branch_1",
            title="Unrelated Source",
            url="https://example.com/unrelated",
            canonical_url="https://example.com/unrelated",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash-2",
            extraction_method="test",
            word_count=200,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id="branch_1",
            claim="Need for closure can shape misinformation acceptance through quick certainty and heuristic judgment.",
            supporting_excerpt="Need for closure can shape misinformation acceptance through quick certainty and heuristic judgment.",
            source_url=sources[0].url,
            source_title=sources[0].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id="branch_1",
            claim="Urban heat islands increase local temperature exposure.",
            supporting_excerpt="Urban heat islands increase local temperature exposure.",
            source_url=sources[1].url,
            source_title=sources[1].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Report\n\n"
        "Need for closure can shape misinformation acceptance through quick certainty and heuristic judgment. [1, 2]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
        "[2] Unrelated Source: https://example.com/unrelated\n"
    )

    repaired = _repair_weak_citation_support(report, cards, sources, threshold=0.35)

    assert "[1]" in repaired.split("## Sources")[0]
    assert "[2]" not in repaired.split("## Sources")[0]


def test_synthesis_card_selection_is_bounded_and_branch_balanced() -> None:
    plan = build_research_plan("Compare several approaches to reducing misinformation acceptance.")
    cards: list[EvidenceCard] = []
    card_id = 1
    for branch in plan.branches:
        for index in range(20):
            cards.append(
                EvidenceCard(
                    id=card_id,
                    source_id=card_id,
                    branch_id=branch.id,
                    claim=f"{branch.title} evidence item {index} explains misinformation acceptance.",
                    supporting_excerpt=f"{branch.title} evidence item {index} explains misinformation acceptance.",
                    source_url=f"https://example.com/{card_id}",
                    source_title=f"Source {card_id}",
                    quality_score=0.9,
                    relevance_score=0.9,
                    confidence=0.9,
                )
            )
            card_id += 1

    selected = _cards_for_synthesis(plan, cards)
    selected_branches = {card.branch_id for card in selected}

    assert len(selected) <= 96
    assert selected_branches == {branch.id for branch in plan.branches}


def test_synthesis_card_selection_prefers_question_phrase_overlap() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    broad_overlap = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for cognition can influence how people respond to misinformation.",
        supporting_excerpt="Need for cognition can influence how people respond to misinformation.",
        source_url="https://example.com/broad",
        source_title="Broad Overlap",
        quality_score=1.0,
        relevance_score=1.0,
        confidence=1.0,
    )
    phrase_overlap = EvidenceCard(
        id=2,
        source_id=2,
        branch_id=branch.id,
        claim="Need for closure is linked to misinformation acceptance when people seek certainty.",
        supporting_excerpt="Need for closure is linked to misinformation acceptance when people seek certainty.",
        source_url="https://example.com/phrase",
        source_title="Phrase Overlap",
        quality_score=0.8,
        relevance_score=0.8,
        confidence=0.8,
    )

    selected = _cards_for_synthesis(plan, [broad_overlap, phrase_overlap])

    assert selected[0].id == 2


def test_synthesis_card_selection_keeps_rare_required_term_under_budget() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mechanisms and boundary conditions",
        objective="Explain the evidence, mechanisms, and boundary conditions.",
        queries=["mechanisms evidence boundary conditions"],
        required_terms=["mechanisms", "evidence", "boundary conditions"],
    )
    plan = ResearchPlan(
        question="Explain the mechanisms and boundary conditions for misinformation acceptance.",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=["Cover mechanisms, evidence quality, and boundary conditions."],
    )
    cards = [
        EvidenceCard(
            id=index,
            source_id=1,
            branch_id=branch.id,
            claim="Mechanisms and evidence explain misinformation acceptance through repeated exposure.",
            supporting_excerpt="Mechanisms and evidence explain misinformation acceptance through repeated exposure.",
            source_url="https://example.com/repeated",
            source_title="Repeated Exposure",
            quality_score=1.0,
            relevance_score=1.0,
            confidence=1.0,
        )
        for index in range(1, 130)
    ]
    rare_card = EvidenceCard(
        id=500,
        source_id=1,
        branch_id=branch.id,
        claim="Boundary conditions include uncertainty, prior beliefs, and the information environment.",
        supporting_excerpt="Boundary conditions include uncertainty, prior beliefs, and the information environment.",
        source_url="https://example.com/boundary",
        source_title="Boundary Conditions",
        quality_score=0.45,
        relevance_score=0.45,
        confidence=0.45,
    )

    selected = _cards_for_synthesis(plan, cards + [rare_card])

    assert len(selected) <= 96
    assert rare_card.id in {card.id for card in selected}


def test_synthesis_card_selection_preserves_source_diversity() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Evidence diversity",
        objective="Synthesize findings across independent sources.",
        queries=["evidence diversity synthesis"],
        required_terms=["independent sources", "evidence diversity"],
    )
    plan = ResearchPlan(
        question="Synthesize the evidence across independent sources.",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    cards = [
        EvidenceCard(
            id=index,
            source_id=1,
            branch_id=branch.id,
            claim="One source repeatedly reports the same evidence pattern.",
            supporting_excerpt="One source repeatedly reports the same evidence pattern.",
            source_url="https://example.com/source-1",
            source_title="Source 1",
            quality_score=1.0,
            relevance_score=1.0,
            confidence=1.0,
        )
        for index in range(1, 80)
    ]
    cards.extend(
        EvidenceCard(
            id=100 + source_id,
            source_id=source_id,
            branch_id=branch.id,
            claim=f"Independent source {source_id} contributes evidence diversity and a distinct context.",
            supporting_excerpt=f"Independent source {source_id} contributes evidence diversity and a distinct context.",
            source_url=f"https://example.com/source-{source_id}",
            source_title=f"Source {source_id}",
            quality_score=0.7,
            relevance_score=0.7,
            confidence=0.7,
        )
        for source_id in range(2, 8)
    )

    selected = _cards_for_synthesis(plan, cards)

    assert len({card.source_id for card in selected[:12]}) >= 6


def test_report_normalizer_repairs_heading_sources_and_uncited_paragraphs() -> None:
    source = SourceRecordV2(
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
    )
    raw = (
        "**Key Answer**\n\n"
        "This long factual paragraph discusses need for closure and misinformation acceptance in enough detail "
        "that it should not remain uncited after normalization.\n\n"
        "**References**\n\n"
        "[1] Old Source: https://example.com/old\n"
        "[21] Bad Source: https://example.com/bad\n"
    )

    normalized = _normalize_report_markdown(raw, [source])

    assert "## Key Answer" in normalized
    assert "## Sources" in normalized
    assert "[21]" not in normalized
    assert "should not remain uncited after normalization. [1]" in normalized
    assert "[1] Need for Closure Source: https://example.com/nfc" in normalized

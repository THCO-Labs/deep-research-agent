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

def test_verifier_rejects_undercovered_acceptance_criteria() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            "Explain mediating variables such as heuristic processing and source credibility assessment.",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        supporting_excerpt="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty and stop evaluating alternatives. [1]\n\n"
        "## Evidence Strength and Limits\n\n"
        "The evidence is limited because it supports the broad relationship, not every pathway or boundary condition. [1]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt},
    )

    assert result.valid is False
    assert result.criteria_coverage_score < 0.65
    assert result.undercovered_criteria
    assert any("acceptance criteria coverage" in failure.lower() for failure in result.failures)


def test_verifier_does_not_treat_readability_rubrics_as_content_criteria() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            "Task-specific comprehensiveness criterion: Explain mechanisms linking need for closure and misinformation acceptance.",
            "Task-specific readability criterion: Effective Use of Formatting for Readability - Assesses headings, paragraphing, sentence variety, and navigation.",
            "Clarity, Precision, and Appropriate Use of Psychological Terminology - Assesses whether psychological terms (e.g., 'need for closure,' 'epistemic motivation,' 'heuristics,' 'misinformation susceptibility') are clearly defined, used accurately and consistently, and explained appropriately for an academic audience.",
            "Paragraph Cohesion, Clarity, and Transitions - Assesses if each paragraph focuses on a single, clear idea related to the topic, with well-structured sentences.",
            "Overall Textual Fluency and Engagement - Considers the overall quality of the writing style, including sentence variety and an engaging academic tone.",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure affects misinformation acceptance by encouraging quick certainty and reduced scrutiny.",
        supporting_excerpt="Need for closure affects misinformation acceptance by encouraging quick certainty and reduced scrutiny.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure affects misinformation acceptance by encouraging quick certainty, heuristic processing, and reduced scrutiny of false claims. [1]\n\n"
        "## Evidence Strength and Limits\n\n"
        "The evidence supports a mechanism-focused interpretation, while remaining cautious about causal strength and boundary conditions. [1]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: report},
    )

    assert result.criteria_coverage_score >= 0.50
    assert not any("readability" in row["criterion"].lower() for row in result.undercovered_criteria)
    assert not any("acceptance criteria coverage" in failure.lower() for failure in result.failures)


def test_verifier_rejects_shallow_report_even_with_supported_citation() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Cooling centers and heat illness risk",
        objective="Explain how cooling centers reduce heat illness risk during heat waves.",
        queries=["cooling centers heat illness risk"],
        min_sources=1,
        required_terms=["cooling centers", "heat illness", "heat waves"],
    )
    plan = ResearchPlan(
        question="How do cooling centers reduce heat illness risk during heat waves?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Cooling Center Source",
        url="https://example.com/cooling",
        canonical_url="https://example.com/cooling",
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
        claim="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves.",
        supporting_excerpt="Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Cooling Centers\n\n"
        "Cooling centers reduce heat illness risk by giving residents access to cooler indoor spaces during heat waves. [1]\n\n"
        "## Sources\n\n"
        "[1] Cooling Center Source: https://example.com/cooling\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt},
    )

    assert result.valid is False
    assert result.report_depth_score < 0.45
    assert any("report depth" in failure.lower() for failure in result.failures)


def test_verifier_accepts_covered_acceptance_criteria_gate() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance", "heuristic processing", "source credibility"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            "Explain mediating variables such as heuristic processing and source credibility assessment.",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure can increase misinformation acceptance through heuristic processing and source credibility assessment.",
        supporting_excerpt="Need for closure can increase misinformation acceptance through heuristic processing and source credibility assessment.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty, rely on heuristic processing, and use source credibility assessment as a shortcut. [1]\n\n"
        "## Evidence Strength and Limits\n\n"
        "The strongest supported mediating variables in this narrow evidence set are heuristic processing and source credibility assessment. [1]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt},
    )

    assert result.criteria_coverage_score >= 0.65
    assert not result.undercovered_criteria
    assert not any("acceptance criteria coverage" in failure.lower() for failure in result.failures)


def test_verifier_requires_broad_citation_use_when_evidence_deck_is_broad() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Cooling centers and heat illness risk",
        objective="Explain how cooling centers reduce heat illness risk during heat waves.",
        queries=["cooling centers heat illness risk"],
        min_sources=17,
        required_terms=["cooling centers", "heat illness risk", "heat waves", "cooler indoor spaces"],
    )
    plan = ResearchPlan(
        question="How do cooling centers reduce heat illness risk during heat waves?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    sources: list[SourceRecordV2] = []
    cards: list[EvidenceCard] = []
    for source_id in range(1, 18):
        source = SourceRecordV2(
            id=source_id,
            branch_id=branch.id,
            title=f"Cooling Center Evidence {source_id}",
            url=f"https://example.com/cooling/{source_id}",
            canonical_url=f"https://example.com/cooling/{source_id}",
            provenance="web",
            content_path=f"source_docs/source_{source_id}.md",
            content_hash=f"hash-{source_id}",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        )
        sources.append(source)
        cards.append(
            EvidenceCard(
                id=source_id,
                source_id=source_id,
                branch_id=branch.id,
                claim="Cooling centers reduce heat illness risk during heat waves by providing cooler indoor spaces.",
                supporting_excerpt="Cooling centers reduce heat illness risk during heat waves by providing cooler indoor spaces.",
                source_url=source.url,
                source_title=source.title,
                quality_score=0.9,
                relevance_score=0.9,
                confidence=0.9,
            )
        )
    body = (
        "Cooling centers reduce heat illness risk during heat waves by providing cooler indoor spaces. "
        "The evidence indicates that access to cooler indoor spaces matters for heat exposure and public health confidence."
    )
    report = (
        "# Cooling Centers and Heat Illness Risk\n\n"
        f"## Direct Answer\n\n{body} [1]\n\n"
        f"## Evidence Pattern\n\n{body} [1]\n\n"
        f"## Limits and Confidence\n\n{body} [1]\n\n"
        "## Sources\n\n"
        "[1] Cooling Center Evidence 1: https://example.com/cooling/1\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=sources,
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: cards[0].supporting_excerpt},
    )

    assert result.source_breadth_score < 1.0
    assert any("cited evidence-backed source count" in failure.lower() for failure in result.failures)


def test_verifier_rejects_incomplete_branch_coverage_even_when_average_score_is_high() -> None:
    branches = [
        ResearchBranch(
            id=f"branch_{index}",
            title=f"Coverage branch {index}",
            objective=f"Explain coverage branch {index}.",
            queries=[f"coverage branch {index}"],
            min_sources=1,
            required_terms=[f"coverage branch {index}"],
        )
        for index in range(1, 6)
    ]
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=f"branch_{index}",
            title=f"Coverage Source {index}",
            url=f"https://example.com/coverage/{index}",
            canonical_url=f"https://example.com/coverage/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash="hash",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        )
        for index in range(1, 5)
    ]
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=f"branch_{index}",
            claim=f"Coverage branch {index} is supported by evidence.",
            supporting_excerpt=f"Coverage branch {index} is supported by evidence.",
            source_url=f"https://example.com/coverage/{index}",
            source_title=f"Coverage Source {index}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 5)
    ]
    coverage = build_coverage_matrix(branches=branches, evidence_cards=cards, sources=sources)
    plan = ResearchPlan(
        question="How should all coverage branches be handled?",
        intent="general",
        audience="general",
        report_outline=[branch.title for branch in branches],
        branches=branches,
    )
    report = (
        "# Coverage Branches\n\n"
        "Coverage branch 1, coverage branch 2, coverage branch 3, and coverage branch 4 are supported by evidence. [1, 2, 3, 4]\n\n"
        "## Evidence Pattern\n\n"
        "The evidence indicates that the covered branches have source support, but one planned branch is not represented. [1, 2, 3, 4]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because one planned branch lacks evidence coverage. [1, 2, 3, 4]\n\n"
        "## Sources\n\n"
        + "\n".join(
            f"[{index}] Coverage Source {index}: https://example.com/coverage/{index}"
            for index in range(1, 5)
        )
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=sources,
        evidence_cards=cards,
        coverage=coverage,
        source_texts={card.source_id: card.supporting_excerpt for card in cards},
    )

    assert coverage.coverage_score >= 0.80
    assert coverage.complete is False
    assert result.valid is False
    assert any("coverage incomplete" in failure.lower() for failure in result.failures)

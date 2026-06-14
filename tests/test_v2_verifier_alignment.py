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

def test_report_quality_rejects_internal_research_artifact_language() -> None:
    report = (
        "# Report\n\n"
        "No verified evidence cards were available for branch_1, so the report cannot answer this section. [1]\n\n"
        "## Sources\n\n"
        "[1] Example: https://example.com\n"
    )

    issues = report_quality_issues(report)

    assert any("internal research artifact" in issue for issue in issues)


def test_irrelevant_recovery_fails_answer_coverage() -> None:
    plan = build_research_plan("What are urban heat islands and how do they affect public health?")
    branch_id = plan.branches[0].id
    source = SourceRecordV2(
        id=1,
        branch_id=branch_id,
        title="Invoice Processing Workflow",
        url="https://example.com/invoices",
        canonical_url="https://example.com/invoices",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=100,
        quality_score=0.9,
        quality_label="high",
        quality_type="academic",
        relevance_score=0.1,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch_id,
        claim="Invoice processing workflows reconcile purchase orders, receipts, approvals, and payment records.",
        supporting_excerpt="Invoice processing workflows reconcile purchase orders, receipts, approvals, and payment records.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.1,
        confidence=0.2,
    )
    coverage = CoverageMatrix(branches=[], complete=False, coverage_score=0.0, missing_branches=[branch_id])
    report = (
        "# Research Report\n\n"
        "## Mismatched Recovery Output\n\n"
        "Invoice processing workflows reconcile purchase orders, receipts, approvals, and payment records. [1]\n\n"
        "## Sources\n\n"
        "[1] Invoice Processing Workflow: https://example.com/invoices\n"
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
    assert result.answer_coverage_score < 0.5
    assert any("Answer coverage" in failure for failure in result.failures)


def test_verifier_rejects_report_that_drifts_from_original_question() -> None:
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
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="COVID Misinformation Source",
        url="https://example.com/covid-misinformation",
        canonical_url="https://example.com/covid-misinformation",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="COVID-19 misinformation can spread through social media during public health crises.",
        supporting_excerpt="COVID-19 misinformation can spread through social media during public health crises.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# COVID-19 Misinformation Report\n\n"
        "## Public Health Effects\n\n"
        "COVID-19 misinformation can spread through social media during public health crises and affect public behavior. [1]\n\n"
        "## Evidence Pattern\n\n"
        "The evidence indicates that public health communication can be challenged by viral false claims online. [1]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because this evidence does not resolve every public health context. [1]\n\n"
        "## Sources\n\n"
        "[1] COVID Misinformation Source: https://example.com/covid-misinformation\n"
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
    assert result.request_alignment_score < 0.45
    assert any("topic alignment" in failure.lower() for failure in result.failures)


def test_verifier_rejects_correct_title_with_contextual_case_as_opening_answer() -> None:
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
    )
    contextual_source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Contextual Conspiracy Belief Study",
        url="https://example.com/context",
        canonical_url="https://example.com/context",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash-1",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.8,
    )
    direct_source = SourceRecordV2(
        id=2,
        branch_id=branch.id,
        title="Need for Closure and Misinformation Acceptance",
        url="https://example.com/direct",
        canonical_url="https://example.com/direct",
        provenance="web",
        content_path="source_docs/source_2.md",
        content_hash="hash-2",
        extraction_method="test",
        word_count=160,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    contextual_card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="A study examined the role of need for cognitive closure for COVID-19 conspiracy beliefs.",
        supporting_excerpt="A study examined the role of need for cognitive closure for COVID-19 conspiracy beliefs.",
        source_url=contextual_source.url,
        source_title=contextual_source.title,
        quality_score=0.9,
        relevance_score=0.8,
        confidence=0.9,
    )
    direct_card = EvidenceCard(
        id=2,
        source_id=2,
        branch_id=branch.id,
        claim="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        supporting_excerpt="Need for closure can increase misinformation acceptance when people seek quick certainty.",
        source_url=direct_source.url,
        source_title=direct_source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "A study examined the role of need for cognitive closure for COVID-19 conspiracy beliefs. [1]\n\n"
        "## The Broader Relationship\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty and stop evaluating alternatives. [2]\n\n"
        "## Cross-Source Pattern\n\n"
        "Taken together, the evidence indicates that contextual conspiracy-belief evidence must be interpreted as one case within the broader relationship. [1, 2]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because contextual examples do not substitute for direct evidence about misinformation acceptance. [1, 2]\n\n"
        "## Sources\n\n"
        "[1] Contextual Conspiracy Belief Study: https://example.com/context\n"
        "[2] Need for Closure and Misinformation Acceptance: https://example.com/direct\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[contextual_source, direct_source],
        evidence_cards=[contextual_card, direct_card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={
            1: contextual_card.supporting_excerpt,
            2: direct_card.supporting_excerpt,
        },
    )

    assert result.valid is False
    assert any("opening answer topic alignment" in failure.lower() for failure in result.failures)


def test_verifier_rejects_cited_paragraphs_that_drift_inside_otherwise_aligned_report() -> None:
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
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Mixed Source",
        url="https://example.com/mixed",
        canonical_url="https://example.com/mixed",
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
    off_topic = (
        "An invoice approval workflow routes purchase orders, vendor onboarding, budget validation, "
        "exception handling, payment scheduling, and audit records through finance operations."
    )
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty and stop evaluating alternatives. [1]\n\n"
        f"{off_topic} [1]\n\n"
        "## Evidence Strength and Limits\n\n"
        "Confidence is limited because the cited evidence must be interpreted within the specific question about closure and misinformation. [1]\n\n"
        "## Sources\n\n"
        "[1] Mixed Source: https://example.com/mixed\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: card.supporting_excerpt + "\n\n" + off_topic},
    )

    assert result.valid is False
    assert any("topic-drift" in failure.lower() for failure in result.failures)


def test_verifier_rejects_stale_neighboring_concept_sources_even_if_cited() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how Need for Closure (NFC) affects misinformation acceptance.",
        queries=["NFC and misinformation acceptance studies"],
        min_sources=1,
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    direct_source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Need for Closure Source",
        url="https://example.com/nfc",
        canonical_url="https://example.com/nfc",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash-1",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    stale_source = SourceRecordV2(
        id=2,
        branch_id=branch.id,
        title="Need for cognition and misinformation acceptance",
        url="https://example.com/need-for-cognition",
        canonical_url="https://example.com/need-for-cognition",
        provenance="web",
        content_path="source_docs/source_2.md",
        content_hash="hash-2",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id=branch.id,
            claim="Need for closure can increase misinformation acceptance when people seek quick certainty.",
            supporting_excerpt="Need for closure can increase misinformation acceptance when people seek quick certainty.",
            source_url=direct_source.url,
            source_title=direct_source.title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id=branch.id,
            claim="Need for cognition concerns effortful thinking and appears in misinformation studies.",
            supporting_excerpt="Need for cognition concerns effortful thinking and appears in misinformation studies.",
            source_url=stale_source.url,
            source_title=stale_source.title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can increase misinformation acceptance when people seek quick certainty and stop evaluating alternatives. [1, 2]\n\n"
        "## Evidence Strength and Limits\n\n"
        "Taken together, the evidence indicates that neighboring constructs should not replace the specific need-for-closure pathway. [1, 2]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
        "[2] Need for cognition and misinformation acceptance: https://example.com/need-for-cognition\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[direct_source, stale_source],
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={
            1: "Need for closure can increase misinformation acceptance when people seek quick certainty.",
            2: (
                "Need for cognition is a motivation to engage in effortful thinking. "
                "Need for cognition appears in studies about false memories, cognitive effort, and misinformation. "
                "Some authors briefly mention need for closure as a related but different construct. "
            )
            * 4,
        },
    )

    assert result.valid is False
    assert result.cited_source_alignment_score < 1.0
    assert any("fails current branch/request alignment" in failure for failure in result.failures)


def test_verifier_rejects_individual_unsupported_citation_in_supported_group() -> None:
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
    )
    sources = [
        SourceRecordV2(
            id=1,
            branch_id=branch.id,
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
            branch_id=branch.id,
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
        SourceRecordV2(
            id=3,
            branch_id=branch.id,
            title="Unrelated Source",
            url="https://example.com/unrelated",
            canonical_url="https://example.com/unrelated",
            provenance="web",
            content_path="source_docs/source_3.md",
            content_hash="hash-3",
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
            branch_id=branch.id,
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
            branch_id=branch.id,
            claim="Misinformation acceptance involves believing inaccurate information.",
            supporting_excerpt="Misinformation acceptance involves believing inaccurate information.",
            source_url=sources[1].url,
            source_title=sources[1].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
        EvidenceCard(
            id=3,
            source_id=3,
            branch_id=branch.id,
            claim="Urban heat islands raise local temperature exposure.",
            supporting_excerpt="Urban heat islands raise local temperature exposure.",
            source_url=sources[2].url,
            source_title=sources[2].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        ),
    ]
    report = (
        "# Need for Closure and Misinformation Acceptance\n\n"
        "Need for closure can shape misinformation acceptance because people seeking quick certainty may believe inaccurate information. [1, 2, 3]\n\n"
        "## Synthesis\n\n"
        "Taken together, the evidence indicates that closure motivation and misinformation acceptance should be interpreted as a linked cognitive pattern. [1, 2]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because the evidence should be interpreted cautiously across information contexts. [1, 2]\n\n"
        "## Sources\n\n"
        "[1] Need for Closure Source: https://example.com/nfc\n"
        "[2] Misinformation Source: https://example.com/misinformation\n"
        "[3] Unrelated Source: https://example.com/unrelated\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=sources,
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={
            1: cards[0].supporting_excerpt,
            2: cards[1].supporting_excerpt,
            3: cards[2].supporting_excerpt,
        },
    )

    assert result.valid is False
    assert any(
        claim.get("support_kind") == "individual_citation" and claim.get("cited_source_ids") == [3]
        for claim in result.weakly_supported_claims
    )


def test_verifier_rejects_chinese_prompt_answered_in_english() -> None:
    question = "\u8bf7\u5206\u6790\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u5f71\u54cd\u516c\u5171\u5065\u5eb7"
    branch = ResearchBranch(
        id="branch_1",
        title="Urban heat and public health",
        objective="Explain how urban heat islands affect public health.",
        queries=["urban heat public health"],
        min_sources=1,
        required_terms=["urban heat", "public health"],
    )
    plan = ResearchPlan(
        question=question,
        intent="general",
        audience="general",
        report_outline=[branch.title],
        branches=[branch],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Urban Heat Source",
        url="https://example.com/urban-heat",
        canonical_url="https://example.com/urban-heat",
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
        claim="Urban heat islands increase public health risks by raising local heat exposure.",
        supporting_excerpt="Urban heat islands increase public health risks by raising local heat exposure.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# Urban Heat and Public Health\n\n"
        "Urban heat islands increase public health risks by raising local heat exposure, making heat illness and related health burdens more likely during extreme temperatures. [1]\n\n"
        "## Evidence Pattern\n\n"
        "Taken together, the evidence indicates that urban heat and public health risks are linked through exposure, vulnerability, and limits in local cooling access. [1]\n\n"
        "## Limits and Confidence\n\n"
        "Confidence is limited because the available evidence should be interpreted cautiously across neighborhoods, populations, and weather conditions. [1]\n\n"
        "## Sources\n\n"
        "[1] Urban Heat Source: https://example.com/urban-heat\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: report},
    )

    assert result.language_alignment_score < 0.80
    assert any("language alignment" in failure.lower() for failure in result.failures)


def test_verifier_accepts_chinese_report_structure_language_signals() -> None:
    question = "\u8bf7\u5206\u6790\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u5f71\u54cd\u516c\u5171\u5065\u5eb7"
    branch = ResearchBranch(
        id="branch_1",
        title="\u57ce\u5e02\u70ed\u5c9b\u4e0e\u516c\u5171\u5065\u5eb7",
        objective="\u8bf4\u660e\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u901a\u8fc7\u70ed\u66b4\u9732\u5f71\u54cd\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        queries=["\u57ce\u5e02\u70ed\u5c9b \u516c\u5171\u5065\u5eb7"],
        min_sources=1,
        required_terms=["\u57ce\u5e02\u70ed\u5c9b", "\u516c\u5171\u5065\u5eb7", "\u70ed\u66b4\u9732"],
    )
    plan = ResearchPlan(
        question=question,
        intent="general",
        audience="general",
        report_outline=[branch.title],
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
        claim="\u57ce\u5e02\u70ed\u5c9b\u4f1a\u63d0\u9ad8\u5c40\u5730\u6e29\u5ea6\uff0c\u5e76\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        supporting_excerpt="\u57ce\u5e02\u70ed\u5c9b\u4f1a\u63d0\u9ad8\u5c40\u5730\u6e29\u5ea6\uff0c\u5e76\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\u3002",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    report = (
        "# \u57ce\u5e02\u70ed\u5c9b\u4e0e\u516c\u5171\u5065\u5eb7\n\n"
        "\u57ce\u5e02\u70ed\u5c9b\u4f1a\u63d0\u9ad8\u5c40\u5730\u6e29\u5ea6\uff0c\u5e76\u901a\u8fc7\u70ed\u66b4\u9732\u589e\u52a0\u516c\u5171\u5065\u5eb7\u98ce\u9669\uff1b\u8bc1\u636e\u663e\u793a\uff0c\u8fd9\u4e00\u5f71\u54cd\u9700\u8981\u56f4\u7ed5\u70ed\u66b4\u9732\u3001\u6613\u611f\u4eba\u7fa4\u548c\u5730\u533a\u5dee\u5f02\u7efc\u5408\u7406\u89e3\u3002 [1]\n\n"
        "## \u7efc\u5408\u5206\u6790\n\n"
        "\u7efc\u5408\u6765\u770b\uff0c\u7814\u7a76\u8bc1\u636e\u8868\u660e\u57ce\u5e02\u70ed\u5c9b\u4e0d\u662f\u5b64\u7acb\u7684\u73af\u5883\u73b0\u8c61\uff0c\u800c\u662f\u4e0e\u516c\u5171\u5065\u5eb7\u98ce\u9669\u5171\u540c\u4f5c\u7528\u7684\u66b4\u9732\u673a\u5236\u3002 [1]\n\n"
        "## \u5c40\u9650\u4e0e\u4fe1\u5fc3\n\n"
        "\u5c40\u9650\u5728\u4e8e\u5355\u4e00\u8bc1\u636e\u6e90\u53ea\u80fd\u652f\u6301\u57fa\u672c\u5173\u7cfb\uff0c\u56e0\u6b64\u5bf9\u4e0d\u540c\u57ce\u5e02\u3001\u4eba\u7fa4\u548c\u5929\u6c14\u6761\u4ef6\u7684\u56e0\u679c\u63a8\u65ad\u5e94\u4fdd\u6301\u8c28\u614e\u3002 [1]\n\n"
        "## Sources\n\n"
        "[1] \u57ce\u5e02\u70ed\u5c9b\u8bc1\u636e: https://example.com/urban-heat-zh\n"
    )

    result = verify_report_v2(
        report_markdown=report,
        plan=plan,
        sources=[source],
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        source_texts={1: report},
    )

    assert result.language_alignment_score >= 0.80
    assert result.report_structure_score >= 0.60
    assert not any("language alignment" in failure.lower() for failure in result.failures)

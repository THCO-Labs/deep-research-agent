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
from deep_research.synthesis_refinement import _report_needs_depth_expansion
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


def test_refinement_depth_check_handles_numeric_citations() -> None:
    report = (
        "# Urban Heat\n\n"
        "Urban heat raises public health risk by increasing heat exposure in dense neighborhoods. [1]\n\n"
        "## Sources\n\n"
        "[1] Urban Heat Evidence: https://example.com/urban-heat\n"
    )
    target_profile = {
        "minimum_words": 80,
        "minimum_cited_paragraphs": 1,
        "minimum_major_sections_before_sources": 1,
    }

    assert _report_needs_depth_expansion(report, target_profile) is True


def test_model_synthesis_falls_back_when_model_returns_degenerate_report(tmp_path: Path, monkeypatch) -> None:
    class NullReportModel:
        def invoke(self, _messages):
            return SimpleNamespace(content="None")

    monkeypatch.setattr("deep_research.synthesis.model_for_role", lambda *_args, **_kwargs: NullReportModel())
    monkeypatch.setattr("deep_research.synthesis.BaseChatModel", object)
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

    report = synthesize_report_with_model(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path),
    )

    assert "None" not in report.split("## Sources")[0]
    assert "Urban heat increases public health risks" in report
    assert "[1]" in report.split("## Sources")[0]


def test_model_synthesis_preserves_previous_report_when_repair_response_is_empty(
    tmp_path: Path, monkeypatch
) -> None:
    class EmptyReportModel:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(content="")

    monkeypatch.setattr("deep_research.synthesis.model_for_role", lambda *_args, **_kwargs: EmptyReportModel())
    monkeypatch.setattr("deep_research.synthesis.BaseChatModel", object)
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
    previous = (
        "# Urban Heat\n\n"
        "Urban heat increases public health risks by raising local temperatures and heat exposure. [1]\n\n"
        "## Sources\n\n"
        "[1] Urban Heat Evidence: https://example.com/urban-heat\n"
    )

    report = synthesize_report_with_model(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path),
        previous_report=previous,
        verification_failures=["repair this draft"],
    )

    assert "Urban heat increases public health risks" in report
    assert "[1] Urban Heat Evidence: https://example.com/urban-heat" in report


def test_repair_prompt_tells_writer_to_preserve_stable_previous_content() -> None:
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
    previous = (
        "# Urban Heat\n\n"
        "Urban heat increases public health risks by raising local temperatures and heat exposure. [1]\n\n"
        "A different paragraph overstates an unsupported implementation claim. [1]\n\n"
        "## Sources\n\n"
        "[1] Urban Heat Evidence: https://example.com/urban-heat\n"
    )

    prompt = _synthesis_prompt(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
        previous_report=previous,
        verification_failures=["Weakly supported cited paragraph: A different paragraph overstates an unsupported implementation claim."],
    )

    assert "Repair preservation contract" in prompt
    assert "Treat this as an incremental repair, not a fresh report." in prompt
    assert "Preserve the previous draft's title, section order" in prompt
    assert "Urban heat increases public health risks" in prompt


def test_model_synthesis_excludes_sources_failed_by_alignment_verification(tmp_path: Path, monkeypatch) -> None:
    captured_prompts: list[str] = []

    class CapturingReportModel:
        def invoke(self, messages):
            captured_prompts.append(messages[0].content)
            return SimpleNamespace(
                content=(
                    "# Need for Closure and Misinformation Acceptance\n\n"
                    "Need for closure can shape misinformation acceptance when people seek quick certainty. [1]\n\n"
                    "## Sources\n\n"
                    "[1] Direct Source: https://example.com/direct\n"
                )
            )

    monkeypatch.setattr("deep_research.synthesis.model_for_role", lambda *_args, **_kwargs: CapturingReportModel())
    monkeypatch.setattr("deep_research.synthesis.BaseChatModel", object)
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how need for closure affects misinformation acceptance.",
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
    direct_source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Direct Source",
        url="https://example.com/direct",
        canonical_url="https://example.com/direct",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash-1",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="academic",
        relevance_score=0.9,
    )
    adjacent_source = SourceRecordV2(
        id=2,
        branch_id=branch.id,
        title="Adjacent Topic Source",
        url="https://example.com/adjacent",
        canonical_url="https://example.com/adjacent",
        provenance="web",
        content_path="source_docs/source_2.md",
        content_hash="hash-2",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="high",
        quality_type="academic",
        relevance_score=0.4,
    )
    direct_card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="Need for closure can shape misinformation acceptance when people seek quick certainty.",
        supporting_excerpt="Need for closure can shape misinformation acceptance when people seek quick certainty.",
        source_url=direct_source.url,
        source_title=direct_source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    adjacent_card = EvidenceCard(
        id=2,
        source_id=2,
        branch_id=branch.id,
        claim="Need for closure can shape a neighboring attitude outcome.",
        supporting_excerpt="Need for closure can shape a neighboring attitude outcome.",
        source_url=adjacent_source.url,
        source_title=adjacent_source.title,
        quality_score=0.9,
        relevance_score=0.4,
        confidence=0.9,
    )

    report = synthesize_report_with_model(
        plan=plan,
        evidence_cards=[direct_card, adjacent_card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[direct_source, adjacent_source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path),
        verification_failures=[
            "Cited source [2] fails current branch/request alignment: source main topic appears to be a neighboring concept rather than the requested concept",
        ],
    )

    assert captured_prompts
    assert "Adjacent Topic Source" not in captured_prompts[0]
    assert "https://example.com/adjacent" not in captured_prompts[0]
    assert "[2] Adjacent Topic Source" not in report


def test_report_level_criteria_ignores_traceability_quality_gate() -> None:
    criteria = _report_level_criteria(
        [
            "All data points must be traceable to at least one cited source",
            "Compare financing conditions across companies",
        ]
    )

    assert criteria == ["Compare financing conditions across companies"]


def test_synthesis_request_budget_caps_groq_completion_tokens(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:openai/gpt-oss-20b",
        groq_api_keys=("groq-a",),
        model_max_output_tokens=4000,
    )

    kwargs = _synthesis_request_kwargs(
        settings=settings,
        prompt="evidence " * 2500,
        model_spec=settings.model,
    )

    assert kwargs["max_tokens"] < 4000
    assert kwargs["max_tokens"] >= 768


def test_synthesis_request_budget_leaves_google_completion_tokens_unforced(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="google",
        model="google_genai:gemini-2.5-flash",
        google_api_keys=("google-a",),
        model_max_output_tokens=4000,
    )

    kwargs = _synthesis_request_kwargs(
        settings=settings,
        prompt="evidence " * 2500,
        model_spec=settings.model,
    )

    assert kwargs == {}


def test_criteria_rich_synthesis_profile_requires_reference_grade_depth() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain mechanisms, empirical evidence, mediators, moderators, and limitations.",
        queries=["need for closure misinformation acceptance empirical evidence"],
        min_sources=17,
        required_terms=["need for closure", "misinformation acceptance", "mechanisms"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="academic",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            f"Cover this task-specific insight criterion in synthesis: Criterion {index} explains the relationship in depth."
            for index in range(1, 18)
        ],
    )
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=branch.id,
            claim=f"Evidence item {index} links need for closure to misinformation acceptance through cognitive mechanisms.",
            supporting_excerpt=f"Evidence item {index} links need for closure to misinformation acceptance through cognitive mechanisms.",
            source_url=f"https://example.com/{index}",
            source_title=f"Source {index}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 18)
    ]

    profile = _target_report_profile(
        plan=plan,
        evidence_cards=cards,
        writing_guidance="DeepResearch Bench evaluation guidance",
    )

    assert profile["criteria_rich"] is True
    assert profile["minimum_words"] >= 3200
    assert profile["target_words"] >= 4500
    assert profile["target_words"] > profile["minimum_words"]
    assert profile["minimum_cited_paragraphs"] >= 28
    assert profile["minimum_major_sections_before_sources"] >= 16
    assert any(row["purpose"] == "Mechanisms and causal logic" for row in profile["section_plan"])


def test_criteria_rich_synthesis_respects_configured_provider(tmp_path: Path) -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain mechanisms and evidence.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    plan = ResearchPlan(
        question="What is the role of need for closure on misinformation acceptance?",
        intent="general",
        audience="academic",
        report_outline=[branch.title],
        branches=[branch],
        acceptance_criteria=[
            f"Cover this task-specific insight criterion in synthesis: Criterion {index} requires depth."
            for index in range(1, 10)
        ],
    )
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:openai/gpt-oss-20b",
        google_api_keys=("google-a",),
        groq_api_keys=("groq-a",),
    )

    assert _synthesis_model_spec(settings, plan) == "groq:openai/gpt-oss-20b"

    hybrid_settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="hybrid",
        model="groq:openai/gpt-oss-20b",
        google_api_keys=("google-a",),
        groq_api_keys=("groq-a",),
    )

    assert _synthesis_model_spec(hybrid_settings, plan) == "google_genai:gemini-2.5-flash"


def test_normal_synthesis_keeps_configured_model(tmp_path: Path) -> None:
    plan = build_research_plan("What are urban heat islands?")
    settings = Settings(
        project_root=tmp_path,
        out_dir=tmp_path,
        provider="groq",
        model="groq:openai/gpt-oss-20b",
        google_api_keys=("google-a",),
        groq_api_keys=("groq-a",),
    )

    assert _synthesis_model_spec(settings, plan) == "groq:openai/gpt-oss-20b"


def test_criteria_rich_depth_score_rewards_rich_natural_sectioning() -> None:
    branches = [
        ResearchBranch(
            id=f"branch_{index}",
            title=f"Analytical branch {index}",
            objective=f"Explain analytical branch {index} with evidence, mechanisms, limits, and implications.",
            queries=[f"analytical branch {index} evidence"],
            min_sources=1,
            required_terms=[f"analytical branch {index}", "evidence", "mechanisms"],
        )
        for index in range(1, 6)
    ]
    plan = ResearchPlan(
        question="How should this benchmark-style relationship be explained?",
        intent="general",
        audience="academic",
        report_outline=[branch.title for branch in branches],
        branches=branches,
        acceptance_criteria=[
            f"Cover this task-specific insight criterion in synthesis: Criterion {index} requires depth, evidence, mechanisms, limitations, and implications."
            for index in range(1, 18)
        ],
    )
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=branches[(index - 1) % len(branches)].id,
            claim=f"Evidence item {index} supports analytical branch mechanisms and implications.",
            supporting_excerpt=f"Evidence item {index} supports analytical branch mechanisms and implications.",
            source_url=f"https://example.com/{index}",
            source_title=f"Source {index}",
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
        )
        for index in range(1, 18)
    ]
    paragraph = (
        "This cited paragraph explains analytical branch evidence, mechanisms, limitations, implications, "
        "uncertainty, comparison, synthesis, and future research with enough terminology to count as a "
        "substantive report paragraph for a benchmark-style task. [1]"
    )
    thin_heading_report = (
        "# Benchmark Report\n\n"
        "## Direct Answer\n\n"
        + "\n\n".join(paragraph for _ in range(90))
        + "\n\n## Evidence\n\n"
        + "\n\n".join(paragraph for _ in range(20))
        + "\n\n## Sources\n\n[1] Source 1: https://example.com/1\n"
    )

    score = _report_depth_score(thin_heading_report, plan, cards)

    assert score < 0.90

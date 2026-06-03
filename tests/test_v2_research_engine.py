import json
from pathlib import Path
from types import SimpleNamespace

from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.evidence_hygiene import apply_evidence_hygiene
from deep_research.acquisition import _trim_search_query
from deep_research.ingestion import ingest_local_paths, ingest_mcp_manifest
from deep_research.managed import run_gemini_managed_research
from deep_research.planning import build_research_plan
from deep_research.schemas import CoverageMatrix, EvidenceCard, SourceRecordV2, VerificationResultV2
from deep_research.semantic import (
    apply_semantic_report_result,
    enrich_evidence_cards_with_semantics,
    verify_report_with_semantics,
)
from deep_research.semantic_planning import build_or_enrich_research_plan
from deep_research.settings import Settings
from deep_research.source_validation import validate_source_content
from deep_research.synthesis import (
    _evidence_backed_sources,
    _normalize_report_markdown,
    _repair_weak_citation_support,
    build_report_blueprint,
    synthesize_report,
)
from deep_research.verifier_v2 import verify_report_v2
from deep_research.research_graph import _focus_terms_from_state


def test_generic_plan_handles_paragraph_length_medical_question() -> None:
    question = (
        "I am trying to understand whether Mediterranean-style diets help adults with hypertension. "
        "Please cover what the diet means, how it may affect blood pressure, evidence from studies, "
        "who it may not be enough for, and practical decision guidance."
    )
    plan = build_research_plan(question)

    branch_ids = [branch.id for branch in plan.branches]
    all_queries = " ".join(query for branch in plan.branches for query in branch.queries)
    branch_text = " ".join(branch.title + " " + branch.objective + " " + " ".join(branch.required_terms) for branch in plan.branches)
    assert len(branch_ids) >= 4
    assert all(branch_id.startswith("branch_") for branch_id in branch_ids)
    assert "definition" not in branch_ids
    assert sum(branch.min_sources for branch in plan.branches) >= 17
    assert "mediterranean" in all_queries.lower()
    assert "hypertension" in all_queries.lower()
    assert "blood pressure" in branch_text.lower() or "pressure" in branch_text.lower()
    assert "urban heat" not in branch_text.lower()


def test_acquisition_trims_overlong_search_queries() -> None:
    long_query = " ".join(["criterion coverage phrase"] * 40)

    trimmed = _trim_search_query(long_query)

    assert len(trimmed) <= 380
    assert trimmed.split()[-1] in {"criterion", "coverage", "phrase"}


def test_generic_plan_handles_food_question_without_topic_specific_branches() -> None:
    plan = build_research_plan("What makes sourdough bread different from regular yeast bread?")

    branch_ids = [branch.id for branch in plan.branches]
    branch_text = " ".join(branch.title + " " + branch.objective + " " + " ".join(branch.required_terms) for branch in plan.branches)
    assert len(branch_ids) >= 1
    assert all(branch_id.startswith("branch_") for branch_id in branch_ids)
    assert "definition" not in branch_ids
    assert sum(branch.min_sources for branch in plan.branches) >= 17
    assert "sourdough" in " ".join(plan.branches[0].queries).lower()
    assert "bread" in branch_text.lower()
    assert "hypertension" not in branch_text.lower()


def test_llm_semantic_planning_accepts_valid_domain_specific_plan(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "public health policy analyst",
            "report_outline": [
                "Executive Summary",
                "Exposure Pathways",
                "Evidence Strength",
                "Intervention Choices",
                "Limitations",
                "Sources",
            ],
            "source_requirements": [
                {"source_type": "academic", "min_count": 3, "rationale": "study evidence"},
                {"source_type": "government", "min_count": 2, "rationale": "public health guidance"},
            ],
            "acceptance_criteria": ["Explain exposure, health outcomes, interventions, and uncertainty."],
            "branches": [
                {
                    "title": "Urban heat exposure pathways",
                    "objective": "Explain how urban heat islands change local heat exposure and who is exposed.",
                    "queries": ["urban heat island exposure pathways public health", "urban heat exposure vulnerable groups"],
                    "source_types": ["academic", "government"],
                    "min_sources": 2,
                    "required_terms": ["heat exposure", "vulnerable groups"],
                    "completion_criteria": ["Evidence explains exposure pathways."],
                },
                {
                    "title": "Health outcomes and burden",
                    "objective": "Assess heat illness, mortality, respiratory stress, and other public health outcomes.",
                    "queries": ["urban heat island health outcomes mortality", "urban heat respiratory cardiovascular risk"],
                    "source_types": ["academic"],
                    "min_sources": 3,
                    "required_terms": ["mortality", "heat illness"],
                    "completion_criteria": ["Evidence covers health outcomes."],
                },
                {
                    "title": "Measurement and mapping methods",
                    "objective": "Compare surface temperature, air temperature, satellite, and neighborhood mapping approaches.",
                    "queries": ["urban heat island measurement mapping methods", "satellite urban heat mapping air temperature"],
                    "source_types": ["academic", "government"],
                    "min_sources": 2,
                    "required_terms": ["temperature mapping", "satellite"],
                    "completion_criteria": ["Evidence covers measurement methods."],
                },
                {
                    "title": "Mitigation interventions",
                    "objective": "Evaluate tree canopy, cool roofs, reflective surfaces, shade, and urban design options.",
                    "queries": ["urban heat island mitigation tree canopy cool roofs", "urban heat reflective surfaces shade evidence"],
                    "source_types": ["academic", "government"],
                    "min_sources": 3,
                    "required_terms": ["tree canopy", "cool roofs"],
                    "completion_criteria": ["Evidence compares interventions."],
                },
                {
                    "title": "Emergency response and adaptation",
                    "objective": "Research warning systems, cooling centers, outreach, and emergency public health operations.",
                    "queries": ["heat wave cooling centers emergency response public health", "urban heat warning systems public health"],
                    "source_types": ["government", "general_web"],
                    "min_sources": 2,
                    "required_terms": ["cooling centers", "warnings"],
                    "completion_criteria": ["Evidence covers emergency response."],
                },
                {
                    "title": "Equity, costs, and implementation trade-offs",
                    "objective": "Analyze distributional effects, costs, maintenance, governance, and trade-offs.",
                    "queries": ["urban heat mitigation equity costs implementation", "urban heat island intervention tradeoffs equity"],
                    "source_types": ["academic", "government"],
                    "min_sources": 3,
                    "required_terms": ["equity", "costs"],
                    "completion_criteria": ["Evidence covers equity and implementation."],
                },
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "How do urban heat islands affect public health, and which interventions matter most?",
        settings=settings,
        model=planner,
    )

    assert result.accepted is True
    assert result.used_model is True
    assert len(result.plan.branches) == 6
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17
    assert "Exposure Pathways" in result.plan.report_outline
    assert "urban heat island exposure pathways public health" in result.plan.branches[0].queries


def test_llm_semantic_planning_rejects_invalid_output_without_losing_fallback(tmp_path: Path) -> None:
    planner = FakeSemanticJudge({"branches": []})
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What makes sourdough bread different from regular yeast bread?",
        settings=settings,
        model=planner,
    )

    assert result.accepted is False
    assert result.used_model is True
    assert result.failures
    assert len(result.plan.branches) >= 1
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17


def test_llm_semantic_planning_accepts_focused_single_branch_with_guidance(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "social psychology researcher",
            "report_outline": ["Direct Answer", "Evidence and Mechanisms", "Sources"],
            "branches": [
                {
                    "title": "Need for closure and misinformation acceptance",
                    "objective": "Explain how need for closure shapes misinformation acceptance using theory and empirical evidence.",
                    "queries": [
                        "need for closure misinformation acceptance empirical evidence",
                        "need for cognitive closure false information belief formation",
                    ],
                    "min_sources": 17,
                    "required_terms": ["need for closure", "misinformation acceptance", "mechanisms", "empirical evidence"],
                }
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What is the role of need for closure on misinformation acceptance?",
        settings=settings,
        model=planner,
        planning_guidance="Benchmark criterion: cover the psychological mechanisms and direct empirical evidence.",
    )

    assert result.accepted is True
    assert len(result.plan.branches) == 1
    assert result.plan.branches[0].min_sources >= 17
    assert "Benchmark criterion" in planner.prompts[0]


def test_llm_semantic_planning_augments_uncovered_guidance_criteria(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "social psychology researcher",
            "report_outline": ["Direct Answer", "Sources"],
            "branches": [
                {
                    "title": "Need for closure theory",
                    "objective": "Define need for closure and its theoretical foundations.",
                    "queries": ["need for closure theory lay epistemology"],
                    "min_sources": 17,
                    "required_terms": ["need for closure", "lay epistemology"],
                }
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What is the role of need for closure on misinformation acceptance?",
        settings=settings,
        model=planner,
        planning_guidance=(
            "DeepResearch Bench evaluation guidance for this task:\n"
            "- Consideration of Diverse Contexts and Misinformation Types (weight: 0.1)\n"
            "- Discussion of Research Limitations, Nuances, and Future Directions (weight: 0.05)\n"
        ),
    )

    titles = " ".join(branch.title for branch in result.plan.branches).lower()
    assert result.accepted is True
    assert "diverse contexts" in titles
    assert "limitations" in titles
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17


def test_llm_semantic_planning_does_not_use_lexical_scope_blacklists(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "social psychology researcher",
            "report_outline": ["Executive Summary", "Findings", "Sources"],
            "branches": [
                {
                    "title": "Need for closure definition",
                    "objective": "Define need for closure and its role in information processing.",
                    "queries": ["need for closure definition information processing", "need for closure epistemic motivation"],
                    "min_sources": 2,
                    "required_terms": ["need for closure", "information processing"],
                },
                {
                    "title": "Misinformation acceptance",
                    "objective": "Explain misinformation acceptance and belief formation.",
                    "queries": ["misinformation acceptance belief formation", "false information belief psychology"],
                    "min_sources": 2,
                    "required_terms": ["misinformation acceptance", "belief formation"],
                },
                {
                    "title": "Mechanisms linking closure and false beliefs",
                    "objective": "Analyze how closure motivation changes scrutiny, heuristics, and ambiguity tolerance.",
                    "queries": ["need for closure heuristics ambiguity misinformation", "closure motivation false beliefs"],
                    "min_sources": 2,
                    "required_terms": ["heuristics", "ambiguity tolerance"],
                },
                {
                    "title": "Empirical evidence for the relationship",
                    "objective": "Review studies linking need for closure to misinformation or conspiracy belief.",
                    "queries": ["need for closure misinformation empirical study", "need for closure conspiracy beliefs evidence"],
                    "min_sources": 2,
                    "required_terms": ["empirical evidence", "conspiracy beliefs"],
                },
                {
                    "title": "Interventions and mitigation strategies",
                    "objective": "Recommend interventions, mitigation, and media literacy strategies.",
                    "queries": ["need for closure misinformation intervention mitigation", "media literacy need for closure strategy"],
                    "min_sources": 2,
                    "required_terms": ["interventions", "mitigation"],
                },
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What is the role of need for closure on misinformation acceptance?",
        settings=settings,
        model=planner,
    )

    assert result.accepted is True
    assert any("Interventions" in branch.title for branch in result.plan.branches)
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17


def test_generic_source_validation_accepts_medical_topic() -> None:
    plan = build_research_plan(
        "Do Mediterranean diets help adults with hypertension? Cover blood pressure mechanisms and evidence."
    )
    branch = next(branch for branch in plan.branches if "pressure" in " ".join(branch.required_terms + branch.queries).lower())
    content = (
        "Mediterranean diets emphasize vegetables, fruits, legumes, whole grains, nuts, olive oil, "
        "and moderate fish intake. For adults with hypertension, the mechanism may involve lower "
        "sodium intake, higher potassium and fiber intake, improved endothelial function, and better "
        "weight control. Clinical nutrition studies often evaluate systolic and diastolic blood pressure, "
        "adherence, medication use, and cardiovascular risk. This dietary pattern can help some patients, "
        "but it does not replace medical evaluation, antihypertensive medication when indicated, or monitoring."
    )

    result = validate_source_content(
        title="Mediterranean diet and hypertension",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
    )

    assert result.usable is True
    assert result.relevance_score >= 0.30


def test_source_validation_rejects_generic_content_that_misses_original_question() -> None:
    plan = build_research_plan("What is the role of need for closure on misinformation acceptance?")
    branch = plan.branches[0]
    content = (
        "A purchase approval workflow starts with request submission, budget validation, manager approval, "
        "vendor onboarding, payment release, exception handling, and audit trails. The workflow improves "
        "budget control and operational efficiency for finance teams. "
    ) * 4

    result = validate_source_content(
        title="Operations workflow guide",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question=plan.question,
    )

    assert result.usable is False
    assert any("original question" in reason for reason in result.reasons)


def test_stanford_hai_bad_scrape_is_rejected() -> None:
    branch = build_research_plan("What are urban heat islands and how do they affect public health?").branches[0]
    bad_content = (
        "Explore Similar Terms. Stanford HAI. Your browser does not support the video tag. "
        "Subscribe to newsletter. Main navigation. Search. Related content. "
    ) * 8

    result = validate_source_content(
        title="Stanford HAI",
        content=bad_content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
    )

    assert result.usable is False
    assert any("boilerplate" in reason or "relevance" in reason for reason in result.reasons)


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


def test_evidence_hygiene_rejects_boilerplate_and_metadata_cards() -> None:
    clean = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="Fine-tuning adapts a pretrained model by training it further on task-specific examples.",
        supporting_excerpt="Fine-tuning adapts a pretrained model by training it further on task-specific examples.",
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
    assert blueprint["source_summary"]["evidence_card_count"] == 1
    assert "section_contract" not in blueprint
    assert "## What the Sources Show Together" in report
    assert "## Implications" in report
    assert "## Sources" in report
    assert "[1] Urban Heat Evidence: https://example.com/urban-heat" in report


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


class FakeGeminiClient:
    def __init__(self) -> None:
        self.created = {}
        self.interactions = self

    def create(self, **kwargs):
        self.created = kwargs
        return SimpleNamespace(id="interaction-1")

    def get(self, interaction_id):
        assert interaction_id == "interaction-1"
        return SimpleNamespace(
            status="completed",
            outputs=[
                SimpleNamespace(
                    text=(
                        "# Managed Report\n\n"
                        "Managed Gemini research returned a cited report. [1]\n\n"
                        "## Sources\n\n"
                        "[1] Managed Source: https://example.com/source\n"
                    )
                )
            ],
        )


class FakeSemanticJudge:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(messages[0].content)
        return SimpleNamespace(content=json.dumps(self.payload))


class InvalidSemanticJudge:
    def invoke(self, _messages):
        return SimpleNamespace(content='{"cards": [{"id": 1, "keep": true}')

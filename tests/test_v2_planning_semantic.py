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
from deep_research.semantic_planning import build_or_enrich_research_plan, _planning_prompt
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


def test_generic_plan_handles_chinese_question_without_empty_research_branch() -> None:
    question = "请为我整合近几年有关中性粒细胞在脑缺血急性期和慢性期的功能和发展变化的研究成果。"
    plan = build_research_plan(question)

    assert plan.branches
    assert sum(branch.min_sources for branch in plan.branches) >= 17
    assert plan.branches[0].title != "Research"
    assert "中性粒细胞" in " ".join(plan.branches[0].queries)


def test_semantic_planning_prompt_uses_delimited_context_and_keeps_baseline_terms() -> None:
    plan = build_research_plan("Compare open-source models by benchmark score, cost, and deployment constraints.")

    prompt = _planning_prompt(plan, planning_guidance="Prefer benchmark evidence.")

    assert "<CONTEXT>" in prompt
    assert "</CONTEXT>" in prompt
    assert "Deterministic baseline constraints:" in prompt
    assert '"baseline_terms"' in prompt
    assert "comparison dimensions and evidence requirements" in prompt


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


def test_planning_guidance_criteria_survive_without_llm_planner(tmp_path: Path) -> None:
    guidance = format_criteria_guidance_block(
        {
            "dimension_weight": {"insight": 0.4},
            "criterions": {
                "insight": [
                    {
                        "criterion": "Explain the central mechanism",
                        "explanation": "Describe the causal pathway, evidence strength, and uncertainty.",
                        "weight": 1.0,
                    }
                ]
            },
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=False)

    result = build_or_enrich_research_plan(
        "How do cooling centers reduce heat illness risk during heat waves?",
        settings=settings,
        planning_guidance=guidance,
    )

    criteria_text = "\n".join(result.plan.acceptance_criteria)
    assert "Task-specific insight criterion: Explain the central mechanism" in criteria_text
    assert "causal pathway" in criteria_text


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


def test_llm_semantic_planning_rejects_unresolved_placeholder_queries(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "financial analyst",
            "report_outline": ["Ranking", "Comparison", "Sources"],
            "branches": [
                {
                    "title": "Identify top global insurers",
                    "objective": "Determine the strongest global insurers by assets and financial strength.",
                    "queries": ["top global insurers by assets financial strength"],
                    "min_sources": 4,
                    "required_terms": ["global insurers", "financial strength"],
                },
                {
                    "title": "Company metrics",
                    "objective": "Collect financial metrics for each insurer.",
                    "queries": ["Company X 2022 annual report", "Company X credit rating 2023"],
                    "min_sources": 13,
                    "required_terms": ["financing", "credit rating"],
                },
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "收集整理目前国际综合实力前十的保险公司的相关资料，横向比较各公司的融资情况、信誉度、过往五年的增长幅度、实际分红、未来在中国发展潜力等维度。",
        settings=settings,
        model=planner,
    )

    assert result.accepted is False
    assert result.used_model is True
    assert any("unresolved placeholder" in failure.lower() for failure in result.failures)
    assert all("Company X" not in " ".join(branch.queries) for branch in result.plan.branches)


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


def test_llm_semantic_planning_does_not_append_title_words_to_required_terms(tmp_path: Path) -> None:
    planner = FakeSemanticJudge(
        {
            "audience": "social psychology researcher",
            "report_outline": ["Mechanisms", "Sources"],
            "branches": [
                {
                    "title": "Mediating variables in misinformation acceptance",
                    "objective": "Identify how need for closure can shape misinformation acceptance through intermediate variables.",
                    "queries": ["need for closure misinformation mediating variables"],
                    "min_sources": 17,
                    "required_terms": [
                        "mediating variables",
                        "depth of processing",
                        "source credibility assessment",
                        "trust in sources",
                        "emotional regulation",
                        "anxiety",
                        "cognitive load",
                        "mediating",
                        "variables",
                        "explaining",
                        "NFC",
                        "impacts",
                    ],
                }
            ],
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, llm_planning=True)

    result = build_or_enrich_research_plan(
        "What is the role of need for closure on misinformation acceptance?",
        settings=settings,
        model=planner,
    )

    terms = result.plan.branches[0].required_terms
    lowered = {term.lower() for term in terms}
    assert "depth of processing" in lowered
    assert "source credibility assessment" in lowered
    assert "nfc" in lowered
    assert "mediating" not in lowered
    assert "variables" not in lowered
    assert "explaining" not in lowered
    assert "impacts" not in lowered
    assert "role" not in lowered


def test_llm_semantic_planning_keeps_guidance_out_of_search_branches(tmp_path: Path) -> None:
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

    assert result.accepted is True
    assert len(result.plan.branches) == 1
    titles = " ".join(branch.title for branch in result.plan.branches).lower()
    queries = " ".join(query for branch in result.plan.branches for query in branch.queries).lower()
    assert "diverse contexts" not in titles
    assert "limitations" not in titles
    assert "weight:" not in queries
    assert sum(branch.min_sources for branch in result.plan.branches) >= 17
    assert any("diverse contexts" in criterion.lower() for criterion in result.plan.acceptance_criteria)
    assert any("limitations" in criterion.lower() for criterion in result.plan.acceptance_criteria)


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


def test_semantic_evidence_reuses_prior_judgments_without_recalling_model(tmp_path: Path) -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Direct relationship",
        objective="Explain the relationship.",
        queries=["relationship evidence"],
        required_terms=["relationship evidence"],
    )
    plan = ResearchPlan(
        question="How does one construct affect another?",
        intent="general",
        audience="general",
        report_outline=[],
        branches=[branch],
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="The evidence directly explains the relationship between the constructs.",
        supporting_excerpt="The evidence directly explains the relationship between the constructs.",
        source_url="https://example.com",
        source_title="Source",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    first_model = FakeSemanticJudge(
        {
            "cards": [
                {
                    "id": 1,
                    "keep": True,
                    "branch_alignment_score": 1.0,
                    "entailment_score": 1.0,
                    "evidence_relevance_score": 1.0,
                    "normalized_claim": "The evidence directly explains the relationship between the constructs.",
                    "key_points": ["direct relationship"],
                    "limitations": [],
                    "failure_reasons": [],
                }
            ]
        }
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    first = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=first_model,
    )
    second = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=RaisingSemanticJudge(),
        prior_judgments=first.judgments,
    )

    assert len(first_model.prompts) == 1
    assert len(second.cards) == 1
    assert second.metrics["semantic_evidence_cached_judgment_count"] == 1
    assert second.metrics["semantic_evidence_batches"] == 0


def test_semantic_evidence_uses_deterministic_fallback_on_provider_quota(tmp_path: Path) -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Direct relationship",
        objective="Explain the relationship.",
        queries=["relationship evidence"],
        required_terms=["relationship evidence"],
    )
    plan = ResearchPlan(
        question="How does one construct affect another?",
        intent="general",
        audience="general",
        report_outline=[],
        branches=[branch],
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="The evidence directly explains the relationship between the constructs.",
        supporting_excerpt="The evidence directly explains the relationship between the constructs.",
        source_url="https://example.com",
        source_title="Source",
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )
    settings = Settings(project_root=tmp_path, out_dir=tmp_path, semantic_verification=True)

    result = enrich_evidence_cards_with_semantics(
        plan=plan,
        evidence_cards=[card],
        settings=settings,
        model=QuotaSemanticJudge(),
    )

    assert len(result.cards) == 1
    assert result.metrics["semantic_evidence_provider_failure_count"] == 1
    assert any("deterministic fallback" in failure for failure in result.failures)

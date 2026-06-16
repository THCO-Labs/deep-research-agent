from deep_research.reasoning_state import build_research_state_artifact, decide_next_action
from deep_research.schemas import BranchCoverage, CoverageMatrix, ResearchBranch, ResearchPlan


def test_research_state_artifact_serializes_core_fields() -> None:
    plan = _plan()
    graph = {
        "claims": [
            {
                "id": "claim_1",
                "branch_id": "branch_1",
                "claim": "Spring Boot reduces configuration work.",
                "source_ids": [1],
                "evidence_card_ids": [1],
                "support_count": 1,
                "average_confidence": 0.8,
                "high_impact": True,
                "weak": True,
                "weakness_reasons": ["high-impact claim has fewer than two independent sources"],
            }
        ],
        "contradiction_edges": [],
        "metrics": {"support_edge_count": 1},
    }
    coverage = CoverageMatrix(
        branches=[
            BranchCoverage(
                branch_id="branch_1",
                required_points=["evidence"],
                covered_points=[],
                missing_points=["branch evidence cards"],
                source_count=1,
                complete=False,
            )
        ],
        complete=False,
        coverage_score=0.5,
        missing_branches=["branch_1"],
    )

    artifact = build_research_state_artifact(
        plan=plan,
        evidence_graph=graph,
        coverage=coverage,
        source_policy={"task_type": "technical_procurement", "label": "technical_procurement_source_policy"},
        metrics={"evidence_card_count": 1},
    ).to_dict()

    assert artifact["original_query"] == plan.question
    assert artifact["weak_claims"][0]["claim_id"] == "claim_1"
    assert artifact["summary"]["primary_action"] == "search_more"


def test_decide_next_action_returns_contradiction_search_for_supported_high_impact_claim() -> None:
    decision = decide_next_action(
        reasoning_state={"weak_claims": [], "unknowns": [], "contradictions": [], "readiness_status": "ready"},
        evidence_graph={
            "claims": [
                {
                    "id": "claim_1",
                    "branch_id": "branch_1",
                    "claim": "Spring Boot is effective for deployment.",
                    "high_impact": True,
                    "weak": False,
                    "support_count": 2,
                    "average_confidence": 0.8,
                }
            ],
            "metrics": {"support_edge_count": 2},
        },
        coverage={"complete": True, "coverage_score": 1.0, "missing_branches": []},
        metrics={"evidence_card_count": 2, "reasoning_iteration_count": 0, "max_reasoning_iterations": 2},
    )

    assert decision["action"] == "contradiction_search"


def test_decide_next_action_records_deferred_python_analysis() -> None:
    decision = decide_next_action(
        reasoning_state={"weak_claims": [], "unknowns": [], "contradictions": [], "readiness_status": "ready"},
        evidence_graph={
            "claims": [
                {
                    "id": "claim_1",
                    "branch_id": "branch_1",
                    "claim": "The benchmark score increased by 12 percent.",
                    "high_impact": False,
                    "weak": False,
                    "support_count": 2,
                    "average_confidence": 0.8,
                }
            ],
            "metrics": {"support_edge_count": 2},
        },
        coverage={"complete": True, "coverage_score": 1.0, "missing_branches": []},
        metrics={"evidence_card_count": 2},
    )

    assert decision["action"] == "analyze_with_python"
    assert decision["deferred"] is True


def test_decide_next_action_does_not_route_to_python_when_coverage_is_weak() -> None:
    decision = decide_next_action(
        reasoning_state={
            "weak_claims": [],
            "unknowns": [
                {
                    "id": "unknown_branch_3",
                    "branch_id": "branch_3",
                    "description": "Evidence is incomplete.",
                    "focus_terms": ["thermal compensation"],
                }
            ],
            "contradictions": [],
            "readiness_status": "needs_more_research",
        },
        evidence_graph={
            "claims": [
                {
                    "id": "claim_1",
                    "branch_id": "branch_3",
                    "claim": "The spindle reaches 12,000 rpm.",
                    "high_impact": False,
                    "weak": False,
                    "support_count": 2,
                    "average_confidence": 0.8,
                }
            ],
            "metrics": {"support_edge_count": 2},
        },
        coverage={"complete": False, "coverage_score": 0.61, "missing_branches": ["branch_3"]},
        metrics={
            "evidence_card_count": 2,
            "reasoning_iteration_count": 2,
            "max_reasoning_iterations": 2,
            "last_acquire_added_sources": 0,
            "last_acquire_searches": 0,
            "candidate_count_total": 90,
            "max_candidates": 90,
        },
    )

    assert decision["action"] == "synthesize"
    assert decision["deferred"] is False


def test_decide_next_action_synthesizes_when_reasoning_search_budget_is_exhausted() -> None:
    decision = decide_next_action(
        reasoning_state={
            "weak_claims": [{"claim_id": "claim_1", "branch_id": "branch_3"}],
            "unknowns": [{"id": "unknown_branch_3", "branch_id": "branch_3"}],
            "contradictions": [],
            "readiness_status": "needs_more_research",
        },
        evidence_graph={
            "claims": [
                {
                    "id": "claim_1",
                    "branch_id": "branch_3",
                    "claim": "Tool-use ability remains weakly evidenced.",
                    "high_impact": False,
                    "weak": True,
                    "support_count": 1,
                    "average_confidence": 0.5,
                }
            ],
            "metrics": {"support_edge_count": 22},
        },
        coverage={"complete": False, "coverage_score": 0.75, "missing_branches": ["branch_3", "branch_4"]},
        metrics={
            "evidence_card_count": 22,
            "reasoning_iteration_count": 2,
            "max_reasoning_iterations": 2,
            "search_count": 15,
            "max_search_queries": 12,
            "candidate_count_total": 40,
            "max_candidates": 60,
        },
    )

    assert decision["action"] == "synthesize"
    assert "budget is exhausted" in decision["rationale"]


def test_research_state_adds_comparative_benchmark_unknown_when_basis_is_missing() -> None:
    plan = ResearchPlan(
        question="Compare the strongest open-source models by benchmark score, cost, context length, and tool use.",
        intent="general",
        audience="technical generalist",
        report_outline=[],
        branches=[
            ResearchBranch(
                id="branch_1",
                title="Model comparison",
                objective="Rank options by comparable evidence.",
                queries=["open-source model benchmark comparison"],
                required_terms=["benchmark score", "cost", "context length"],
            )
        ],
    )
    graph = {
        "claims": [
            {
                "id": "claim_1",
                "branch_id": "branch_1",
                "claim": "Model A is a strong option for agentic workflows.",
                "source_ids": [1],
                "evidence_card_ids": [1],
                "support_count": 1,
                "average_confidence": 0.8,
                "high_impact": True,
                "weak": False,
                "weakness_reasons": [],
            }
        ],
        "contradiction_edges": [],
        "metrics": {"support_edge_count": 1},
    }
    coverage = CoverageMatrix(
        branches=[
            BranchCoverage(
                branch_id="branch_1",
                required_points=["benchmark score"],
                covered_points=["model option"],
                missing_points=[],
                source_count=1,
                complete=True,
            )
        ],
        complete=True,
        coverage_score=1.0,
        missing_branches=[],
    )

    artifact = build_research_state_artifact(
        plan=plan,
        evidence_graph=graph,
        coverage=coverage,
        source_policy={"task_type": "comparative_benchmark", "label": "comparative_benchmark_source_policy"},
        metrics={"evidence_card_count": 1},
    ).to_dict()

    assert any(item["id"] == "unknown_comparative_benchmark_basis" for item in artifact["unknowns"])


def test_decide_next_action_can_use_clamped_model_recommendation() -> None:
    decision = decide_next_action(
        reasoning_state={
            "weak_claims": [],
            "unknowns": [],
            "contradictions": [],
            "model_recommended_action": {
                "action": "search_more",
                "rationale": "Need stronger opposition evidence.",
                "branch_ids": ["branch_1"],
                "focus_terms": ["opposing evidence"],
                "priority": "high",
            },
        },
        evidence_graph={"claims": [], "metrics": {"support_edge_count": 1}},
        coverage={"complete": True, "coverage_score": 1.0, "missing_branches": []},
        metrics={"evidence_card_count": 1, "reasoning_iteration_count": 0, "max_reasoning_iterations": 2},
    )

    assert decision["action"] == "search_more"
    assert decision["branch_ids"] == ["branch_1"]


def _plan() -> ResearchPlan:
    return ResearchPlan(
        question="Compare Java enterprise frameworks.",
        intent="general",
        audience="technical generalist",
        report_outline=[],
        branches=[ResearchBranch(id="branch_1", title="Boot", objective="Explain Boot.", queries=["boot"])],
    )

from deep_research.acquisition import _branch_queries
from deep_research.research_graph import _focus_terms_by_branch as _decision_focus_terms_by_branch
from deep_research.research_graph_helpers import _acquire_route, _coverage_route, _focus_terms_from_state, _reasoning_route
from deep_research.schemas import ResearchBranch


def test_reasoning_search_more_overrides_complete_coverage_when_budget_remains() -> None:
    state = {
        "coverage_matrix": {"complete": True, "coverage_score": 1.0, "missing_branches": []},
        "evidence_cards": [{"id": 1}],
        "reasoning_decision": {"action": "search_more", "branch_ids": ["branch_1"]},
        "metrics": {
            "reasoning_iteration_count": 1,
            "max_reasoning_iterations": 2,
            "last_acquire_added_sources": 1,
            "last_acquire_searches": 1,
        },
    }

    assert _coverage_route(state) == "more_sources"


def test_reasoning_search_more_can_override_source_cap_plateau_when_search_budget_remains() -> None:
    state = {
        "coverage_matrix": {"complete": False, "coverage_score": 0.6, "missing_branches": ["branch_2"]},
        "evidence_cards": [{"id": 1}],
        "reasoning_decision": {"action": "search_more", "branch_ids": ["branch_2"]},
        "metrics": {
            "reasoning_iteration_count": 1,
            "max_reasoning_iterations": 2,
            "source_count": 18,
            "max_sources": 12,
            "candidate_count_total": 22,
            "max_candidates": 70,
            "search_count": 11,
            "max_search_queries": 14,
            "last_acquire_added_sources": 18,
            "last_acquire_searches": 11,
        },
    }

    assert _coverage_route(state) == "more_sources"


def test_reasoning_route_triggers_single_contradiction_search() -> None:
    state = {
        "reasoning_decision": {"action": "contradiction_search"},
        "metrics": {"reasoning_iteration_count": 1, "max_reasoning_iterations": 2, "contradiction_search_iterations": 0},
    }

    assert _reasoning_route(state) == "contradiction_search"


def test_reasoning_route_generates_search_intents_for_search_more() -> None:
    state = {
        "reasoning_decision": {"action": "search_more", "branch_ids": ["branch_2"]},
        "metrics": {
            "reasoning_iteration_count": 1,
            "max_reasoning_iterations": 5,
            "search_count": 3,
            "max_search_queries": 20,
            "candidate_count_total": 5,
            "max_candidates": 80,
        },
    }

    assert _reasoning_route(state) == "search_intents"


def test_reasoning_synthesize_decision_produces_draft_even_with_partial_coverage() -> None:
    state = {
        "coverage_matrix": {"complete": False, "coverage_score": 0.75, "missing_branches": ["branch_2"]},
        "evidence_cards": [{"id": index} for index in range(7)],
        "reasoning_decision": {"action": "synthesize", "rationale": "Reasoning loop exhausted."},
        "metrics": {
            "reasoning_iteration_count": 2,
            "max_reasoning_iterations": 2,
            "search_count": 18,
            "max_search_queries": 18,
        },
    }

    assert _coverage_route(state) == "synthesize"


def test_reasoning_focus_terms_are_available_to_acquisition() -> None:
    state = {
        "reasoning_focus_terms": {"branch_1": ["search_query: exact contradiction query"]},
        "coverage_matrix": {"branches": [], "missing_branches": []},
    }

    assert _focus_terms_from_state(state)["branch_1"] == ["search_query: exact contradiction query"]


def test_branch_queries_pass_direct_search_queries_through() -> None:
    branch = ResearchBranch(id="branch_1", title="Boot", objective="Test", queries=["base query"])

    queries = _branch_queries(branch, ["search_query: exact contradiction query"], "question")

    assert "exact contradiction query" in queries


def test_reasoning_followup_queries_do_not_repeat_full_question() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Tool suitability",
        objective="Resolve tool limitations.",
        queries=["ceramic cutting tools titanium"],
    )
    question = "Are ceramic cutting tools recommended for machining titanium alloy Ti-6Al-4V?"

    queries = _branch_queries(branch, ["thermal shock resistance", "ceramic vs carbide"], question)

    assert all(question not in query for query in queries[1:])
    assert any(query == "thermal shock resistance ceramic vs carbide evidence" for query in queries)


def test_reasoning_focus_terms_use_model_and_unknown_focus_when_decision_is_sparse() -> None:
    focus = _decision_focus_terms_by_branch(
        {"action": "search_more", "branch_ids": []},
        {
            "model_recommended_action": {
                "branch_ids": ["branch_2"],
                "focus_terms": ["ceramic vs carbide tool life"],
            },
            "unknowns": [
                {
                    "branch_id": "branch_2",
                    "focus_terms": ["thermal shock resistance"],
                }
            ],
        },
    )

    assert focus == {"branch_2": ["ceramic vs carbide tool life", "thermal shock resistance"]}


def test_acquire_route_rebuilds_evidence_when_followup_added_sources() -> None:
    state = {
        "evidence_cards": [{"id": 1}],
        "metrics": {
            "last_acquire_added_sources": 5,
            "source_count": 32,
            "max_sources": 20,
            "last_acquire_added_candidates": 10,
            "last_acquire_searches": 6,
        },
    }

    assert _acquire_route(state) == "read_sources"

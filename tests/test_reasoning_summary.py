from deep_research.context_builder import format_knowledge_packets_for_prompt
from deep_research.reasoning_summary import reasoning_brief_for_prompt, render_reasoning_summary


def test_reasoning_summary_renders_human_readable_audit() -> None:
    text = render_reasoning_summary(
        query="Question?",
        source_policy={"policy": {"label": "technical_procurement_source_policy", "preferred_source_types": ["spec_sheet"]}},
        reasoning_state={
            "task_type": "technical_procurement",
            "source_policy_label": "technical_procurement_source_policy",
            "readiness_status": "needs_caveats_or_more_support",
            "summary": {"weak_claim_count": 1, "unknown_count": 1, "contradiction_count": 1},
            "weak_claims": [{"claim_id": "claim_1", "claim": "Unsupported claim", "source_ids": [1], "reasons": ["single source"]}],
            "unknowns": [{"branch_id": "branch_1", "description": "Missing torque data", "reason": "coverage incomplete"}],
            "contradictions": [{"claim_id": "claim_1", "description": "Tension", "source_ids": [1]}],
        },
        reasoning_decision={"action": "contradiction_search", "rationale": "Need opposition evidence."},
        contradiction_queries=[{"query": "Question claim limitation"}],
    )

    assert "## Top Weak Claims" in text
    assert "Unsupported claim" in text
    assert "delayed for contradiction_search" in text


def test_reasoning_brief_is_included_in_knowledge_prompt() -> None:
    brief = reasoning_brief_for_prompt(
        {
            "readiness_status": "needs_caveats_or_more_support",
            "weak_claims": [{"claim_id": "claim_1", "claim": "Weak claim", "source_ids": [1], "reasons": ["single source"]}],
            "unknowns": [{"branch_id": "branch_1", "description": "Missing context", "reason": "low confidence"}],
            "contradictions": [{"claim_id": "claim_1", "description": "Contradiction", "source_ids": [1]}],
        },
        {"action": "synthesize", "rationale": "Ready with caveats."},
    )

    text = format_knowledge_packets_for_prompt({"reasoning_brief": brief, "section_packets": []})

    assert "reasoning brief" in text
    assert "Weak claim" in text
    assert "Contradiction" in text

import json
from pathlib import Path
from types import SimpleNamespace

from deep_research.reasoning_runtime import refine_reasoning_state_with_model
from deep_research.schemas import EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2
from deep_research.settings import Settings


def test_refine_reasoning_state_with_model_clamps_payload_to_existing_ids(monkeypatch) -> None:
    class ReasoningModel:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "weak_claims": [
                            {"claim_id": "claim_1", "reasons": ["single-source support"], "recommended_action": "search_more"},
                            {"claim_id": "fake_claim", "reasons": ["must be ignored"]},
                        ],
                        "unknowns": [
                            {
                                "branch_id": "branch_1",
                                "description": "Need counter-evidence.",
                                "reason": "Important claim needs opposition check.",
                                "severity": "medium",
                                "focus_terms": ["counter evidence"],
                            },
                            {"branch_id": "fake_branch", "description": "ignored"},
                        ],
                        "contradictions": [
                            {
                                "claim_id": "claim_1",
                                "description": "Possible tension.",
                                "source_ids": [1, 999],
                                "confidence": 0.7,
                                "needs_caveat": True,
                            }
                        ],
                        "next_action": {
                            "action": "search_more",
                            "rationale": "Search for counter-evidence.",
                            "branch_ids": ["branch_1", "fake_branch"],
                            "focus_terms": ["counter evidence"],
                            "priority": "high",
                        },
                    }
                )
            )

    monkeypatch.setattr("deep_research.reasoning_runtime.model_for_role", lambda *_args, **_kwargs: ReasoningModel())
    monkeypatch.setattr("deep_research.reasoning_runtime.BaseChatModel", object)
    plan = _plan()

    refined = refine_reasoning_state_with_model(
        reasoning_state={"weak_claims": [], "unknowns": [], "contradictions": [], "summary": {}},
        evidence_graph={
            "claims": [
                {
                    "id": "claim_1",
                    "branch_id": "branch_1",
                    "claim": "Spring Boot improves deployment.",
                    "source_ids": [1],
                    "evidence_card_ids": [1],
                    "average_confidence": 0.8,
                }
            ]
        },
        plan=plan,
        evidence_cards=[_card()],
        sources=[_source()],
        settings=Settings(project_root=Path("."), llm_synthesis=True),
    )

    assert refined["model_refinement"]["applied"] is True
    assert refined["weak_claims"][0]["claim_id"] == "claim_1"
    assert len(refined["weak_claims"]) == 1
    assert refined["contradictions"][0]["source_ids"] == [1]
    assert refined["model_recommended_action"]["branch_ids"] == ["branch_1"]


def _plan() -> ResearchPlan:
    return ResearchPlan(
        question="Trace Java architecture evolution.",
        intent="general",
        audience="technical generalist",
        report_outline=[],
        branches=[ResearchBranch(id="branch_1", title="Boot", objective="Explain Boot.", queries=["boot"])],
    )


def _source() -> SourceRecordV2:
    return SourceRecordV2(
        id=1,
        branch_id="branch_1",
        title="Source",
        url="https://example.com/source",
        canonical_url="https://example.com/source",
        provenance="web",
        content_path="source_docs/source.md",
        content_hash="hash",
        extraction_method="httpx",
        word_count=500,
        quality_score=0.8,
        quality_label="high",
        quality_type="official_docs",
        relevance_score=0.9,
    )


def _card() -> EvidenceCard:
    return EvidenceCard(
        id=1,
        source_id=1,
        branch_id="branch_1",
        claim="Spring Boot improves deployment.",
        supporting_excerpt="Spring Boot improves deployment.",
        source_url="https://example.com/source",
        source_title="Source",
        quality_score=0.8,
        relevance_score=0.9,
        confidence=0.8,
    )

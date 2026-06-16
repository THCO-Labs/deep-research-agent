from __future__ import annotations

from deep_research.artifacts_v2 import ResearchArtifactsV2
from types import SimpleNamespace

from deep_research.context_builder import (
    build_knowledge_base,
    format_knowledge_packets_for_prompt,
    refine_knowledge_base_from_payload,
    write_knowledge_base,
)
from deep_research.context_builder_runtime import refine_knowledge_base_with_model
from deep_research.schemas import CoverageMatrix, ResearchBranch, ResearchPlan
from deep_research.section_writing import build_adaptive_section_plan
from deep_research.settings import Settings

from tests.test_section_writing import _card, _source


def test_build_knowledge_base_creates_branch_notes_and_section_packets() -> None:
    plan, source, cards = _context_fixture()
    section_plan = build_adaptive_section_plan(
        plan=plan,
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )

    kb = build_knowledge_base(
        plan=plan,
        evidence_cards=cards,
        sources=[source],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        section_plan=section_plan.to_dict(),
    )

    assert kb["schema_version"] == 1
    assert kb["branches"][0]["branch_id"] == "performance"
    assert kb["branches"][0]["source_ids"] == [1]
    assert kb["section_packets"]
    assert kb["section_packets"][0]["source_ids"] == [1]
    prompt = format_knowledge_packets_for_prompt(kb)
    assert "Knowledge" not in prompt
    assert "source_ids: 1" in prompt
    assert "productive spindle hours" in prompt


def test_write_knowledge_base_persists_index_branch_and_packet_files(tmp_path) -> None:
    plan, source, cards = _context_fixture()
    section_plan = build_adaptive_section_plan(
        plan=plan,
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )
    kb = build_knowledge_base(
        plan=plan,
        evidence_cards=cards,
        sources=[source],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        section_plan=section_plan.to_dict(),
    )
    artifacts = ResearchArtifactsV2.create(tmp_path, "knowledge base")

    write_knowledge_base(artifacts=artifacts, knowledge_base=kb)

    assert artifacts.resolve_path("knowledge_base/index.md").exists()
    assert artifacts.resolve_path("knowledge_base/manifest.json").exists()
    assert artifacts.resolve_path("knowledge_base/branches/performance.md").exists()
    assert artifacts.resolve_path("knowledge_base/section_packets/branch_1_performance.md").exists()


def test_refine_knowledge_base_clamps_model_notes_to_real_evidence() -> None:
    plan, source, cards = _context_fixture()
    kb = _knowledge_base(plan, source, cards)

    refined = refine_knowledge_base_from_payload(
        kb,
        {
            "branches": [
                {
                    "branch_id": "performance",
                    "source_ids": [1, 999],
                    "evidence_card_ids": [1, 999],
                    "model_focus": [
                        "Use productive spindle hours as the utilization anchor [1].",
                        "Invented unsupported supplier claim [999].",
                    ],
                    "open_questions": ["Ask whether maintenance assumptions are supported."],
                }
            ],
            "section_packets": [
                {
                    "section_id": "branch_1_performance",
                    "source_ids": [1],
                    "evidence_card_ids": [1],
                    "model_focus": ["Compare productive spindle hours with titanium performance [1]."],
                    "limitations": ["Stable spindle utilization is an explicit caveat [1]."],
                }
            ],
        },
        evidence_cards=cards,
        sources=[source],
    )

    branch = refined["branches"][0]
    packet = next(row for row in refined["section_packets"] if row["section_id"] == "branch_1_performance")
    assert branch["source_ids"] == [1]
    assert branch["evidence_card_ids"] == [1, 2]
    assert branch["model_evidence_card_ids"] == [1]
    assert branch["model_source_ids"] == [1]
    assert branch["model_focus"] == ["Use productive spindle hours as the utilization anchor [1]."]
    assert branch["open_questions"] == ["Ask whether maintenance assumptions are supported."]
    assert packet["model_focus"] == ["Compare productive spindle hours with titanium performance [1]."]
    prompt = format_knowledge_packets_for_prompt(refined)
    assert "model focus:" in prompt
    assert "Compare productive spindle hours" in prompt


def test_refine_knowledge_base_with_model_applies_clamped_payload(tmp_path, monkeypatch) -> None:
    class ContextModel:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(
                content="""
                {
                  "branches": [
                    {
                      "branch_id": "performance",
                      "source_ids": [1],
                      "evidence_card_ids": [1],
                      "model_focus": ["Anchor the argument in productive spindle hours [1]."],
                      "open_questions": ["Do not claim maintenance cost without evidence."]
                    }
                  ],
                  "section_packets": []
                }
                """
            )

    monkeypatch.setattr("deep_research.context_builder_runtime.model_for_role", lambda *_args, **_kwargs: ContextModel())
    monkeypatch.setattr("deep_research.context_builder_runtime.BaseChatModel", object)
    plan, source, cards = _context_fixture()
    kb = _knowledge_base(plan, source, cards)

    refined = refine_knowledge_base_with_model(
        knowledge_base=kb,
        plan=plan,
        evidence_cards=cards,
        sources=[source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path, model_request_timeout_seconds=1),
    )

    assert refined["model_refinement"]["applied"] is True
    assert refined["model_refinement"]["reason"] == "llm_refinement_clamped"
    assert refined["branches"][0]["model_focus"] == ["Anchor the argument in productive spindle hours [1]."]


def test_refine_knowledge_base_with_model_repairs_malformed_json(tmp_path, monkeypatch) -> None:
    class RepairingContextModel:
        calls = 0

        def invoke(self, _messages, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content='{"branches": [{"branch_id": "performance" "model_focus": ["broken"]}]}')
            return SimpleNamespace(
                content="""
                {
                  "branches": [
                    {
                      "branch_id": "performance",
                      "source_ids": [1],
                      "evidence_card_ids": [1],
                      "model_focus": ["Anchor the argument in productive spindle hours [1]."]
                    }
                  ],
                  "section_packets": []
                }
                """
            )

    model = RepairingContextModel()
    monkeypatch.setattr("deep_research.context_builder_runtime.model_for_role", lambda *_args, **_kwargs: model)
    monkeypatch.setattr("deep_research.context_builder_runtime.BaseChatModel", object)
    plan, source, cards = _context_fixture()
    kb = _knowledge_base(plan, source, cards)

    refined = refine_knowledge_base_with_model(
        knowledge_base=kb,
        plan=plan,
        evidence_cards=cards,
        sources=[source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path, model_request_timeout_seconds=1),
    )

    assert refined["model_refinement"]["applied"] is True
    assert refined["model_refinement"]["json_repair_applied"] is True
    assert refined["branches"][0]["model_focus"] == ["Anchor the argument in productive spindle hours [1]."]


def test_refine_knowledge_base_with_model_keeps_base_on_failure(tmp_path, monkeypatch) -> None:
    class FailingModel:
        def invoke(self, _messages, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr("deep_research.context_builder_runtime.model_for_role", lambda *_args, **_kwargs: FailingModel())
    monkeypatch.setattr("deep_research.context_builder_runtime.BaseChatModel", object)
    plan, source, cards = _context_fixture()
    kb = _knowledge_base(plan, source, cards)

    refined = refine_knowledge_base_with_model(
        knowledge_base=kb,
        plan=plan,
        evidence_cards=cards,
        sources=[source],
        settings=Settings(project_root=tmp_path, out_dir=tmp_path, model_request_timeout_seconds=1),
    )

    assert refined["branches"][0]["notes"] == kb["branches"][0]["notes"]
    assert refined["model_refinement"]["applied"] is False
    assert "provider unavailable" in refined["model_refinement"]["reason"]


def _context_fixture():
    branch = ResearchBranch(
        id="performance",
        title="Titanium performance",
        objective="Assess machining performance and utilization assumptions.",
        queries=["CNC titanium machining performance"],
        required_terms=["titanium", "machining", "performance"],
    )
    plan = ResearchPlan(
        question="Evaluate a CNC platform for aerospace titanium machining.",
        intent="general",
        audience="technical buyer",
        report_outline=[],
        branches=[branch],
    )
    source = _source(1, "performance", "Performance Source")
    cards = [
        _card(1, source, "The platform can sustain 5,000 productive spindle hours per year."),
        _card(2, source, "Titanium machining performance depends on stable spindle utilization."),
    ]
    return plan, source, cards


def _knowledge_base(plan, source, cards):
    section_plan = build_adaptive_section_plan(
        plan=plan,
        evidence_cards=cards,
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )
    return build_knowledge_base(
        plan=plan,
        evidence_cards=cards,
        sources=[source],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        section_plan=section_plan.to_dict(),
    )

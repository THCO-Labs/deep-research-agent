from __future__ import annotations

from types import SimpleNamespace

from deep_research.schemas import CoverageMatrix
from deep_research.section_audit_runtime import audit_report_sections_with_model, section_audit_failures
from deep_research.section_writing import build_adaptive_section_plan
from deep_research.settings import Settings

from tests.test_section_writing import _card, _single_branch_plan, _source


def test_llm_section_audit_review_adds_targeted_repair_guidance(tmp_path, monkeypatch) -> None:
    class ReviewModel:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(
                content="""
                {
                  "section_reviews": [
                    {
                      "section_id": "branch_1_performance",
                      "locked": false,
                      "failures": ["The section states a recommendation without reconciling utilization assumptions."],
                      "warnings": [],
                      "repair_guidance": "Add a sourced utilization assumption before making the recommendation."
                    }
                  ],
                  "global_repair_guidance": ["Check recommendation consistency against utilization evidence."]
                }
                """
            )

    monkeypatch.setattr("deep_research.section_audit_runtime.model_for_role", lambda *_args, **_kwargs: ReviewModel())
    monkeypatch.setattr("deep_research.section_audit_runtime.BaseChatModel", object)
    plan = _single_branch_plan()
    source = _source(1, "performance", "Performance Source")
    card = _card(1, source, "Titanium machining performance depends on stable spindle utilization.")
    section_plan = build_adaptive_section_plan(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )
    report = (
        "# Report\n\n"
        "## Titanium performance\n\n"
        "Titanium machining performance depends on stable spindle utilization [1].\n\n"
        "## Sources\n\n"
        "[1] Performance Source: https://example.com/1\n"
    )

    audit = audit_report_sections_with_model(
        section_plan=section_plan.to_dict(),
        report_markdown=report,
        plan=plan,
        evidence_cards=[card],
        sources=[source],
        source_texts={1: card.supporting_excerpt},
        settings=Settings(project_root=tmp_path, out_dir=tmp_path, model_request_timeout_seconds=1),
    )

    failures = section_audit_failures(audit)
    assert audit["llm_review_applied"] is True
    assert any("reconciling utilization assumptions" in failure for failure in failures)
    assert any("Add a sourced utilization assumption" in failure for failure in failures)


def test_section_audit_falls_back_when_llm_review_fails(tmp_path, monkeypatch) -> None:
    class FailingModel:
        def invoke(self, _messages, **_kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr("deep_research.section_audit_runtime.model_for_role", lambda *_args, **_kwargs: FailingModel())
    monkeypatch.setattr("deep_research.section_audit_runtime.BaseChatModel", object)
    plan = _single_branch_plan()
    source = _source(1, "performance", "Performance Source")
    card = _card(1, source, "Titanium machining performance depends on stable spindle utilization.")
    section_plan = build_adaptive_section_plan(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )

    audit = audit_report_sections_with_model(
        section_plan=section_plan.to_dict(),
        report_markdown="# Report\n\n## Titanium performance\n\nNo citation here.",
        plan=plan,
        evidence_cards=[card],
        sources=[source],
        source_texts={1: card.supporting_excerpt},
        settings=Settings(project_root=tmp_path, out_dir=tmp_path, model_request_timeout_seconds=1),
    )

    assert audit["llm_review_applied"] is False
    assert "provider unavailable" in audit["llm_review_error"]

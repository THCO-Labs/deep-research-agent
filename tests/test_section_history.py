from __future__ import annotations

from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.section_history import assemble_best_section_report, publish_section_versions, select_best_section_versions


def test_select_best_section_versions_prefers_locked_supported_section() -> None:
    best = select_best_section_versions(
        [
            {
                "section_id": "decision",
                "draft_index": 1,
                "section_path": "section_drafts/draft_1_decision.md",
                "locked": False,
                "failure_count": 1,
                "citation_support_score": 0.9,
                "evidence_linkage_score": 0.9,
            },
            {
                "section_id": "decision",
                "draft_index": 2,
                "section_path": "section_drafts/draft_2_decision.md",
                "locked": True,
                "failure_count": 0,
                "citation_support_score": 0.6,
                "evidence_linkage_score": 0.6,
            },
        ]
    )

    assert best["decision"]["draft_index"] == 2


def test_publish_section_versions_writes_best_section_artifacts(tmp_path) -> None:
    artifacts = ResearchArtifactsV2.create(tmp_path, "section history")
    metrics = {}
    section_audit = {
        "audits": [
            {
                "section_id": "opening_answer",
                "heading": "Bottom line",
                "matched_report_section_markdown": "## Bottom line\n\nSupported answer [1].",
                "locked": True,
                "failures": [],
                "citation_support_score": 0.9,
                "evidence_linkage_score": 1.0,
                "cited_source_ids": [1],
            }
        ]
    }

    publish_section_versions(
        artifacts=artifacts,
        metrics=metrics,
        section_audit=section_audit,
        draft_index=1,
    )

    assert metrics["best_section_count"] == 1
    assert metrics["locked_best_section_count"] == 1
    assert artifacts.resolve_path("section_drafts/draft_1_opening_answer.md").exists()
    assert artifacts.resolve_path("best_sections/opening_answer.md").exists()
    assert artifacts.read_json("best_sections.json")["sections"]["opening_answer"]["locked"] is True


def test_assemble_best_section_report_uses_complete_locked_sections(tmp_path) -> None:
    artifacts = ResearchArtifactsV2.create(tmp_path, "section assembly")
    artifacts.write_json(
        "best_sections.json",
        {
            "schema_version": 1,
            "sections": {
                "opening_answer": {
                    "section_path": "best_sections/opening_answer.md",
                    "locked": True,
                },
                "decision_logic": {
                    "section_path": "best_sections/decision_logic.md",
                    "locked": True,
                },
            },
        },
    )
    artifacts.write_text("best_sections/opening_answer.md", "## Bottom line\n\nSupported answer [1].\n")
    artifacts.write_text("best_sections/decision_logic.md", "## Decision logic\n\nSupported trade-off [2].\n")
    latest_report = (
        "# Original Report\n\n"
        "## Old section\n\n"
        "Old text [1].\n\n"
        "## Sources\n\n"
        "[1] Source One: https://example.com/1\n"
        "[2] Source Two: https://example.com/2\n"
        "[3] Unused Source: https://example.com/3\n"
    )
    section_plan = {
        "sections": [
            {"id": "opening_answer"},
            {"id": "decision_logic"},
        ]
    }

    assembly = assemble_best_section_report(
        artifacts=artifacts,
        latest_report=latest_report,
        section_plan=section_plan,
    )

    assert assembly["usable_for_final"] is True
    assert assembly["locked_section_count"] == 2
    assert "# Original Report" in assembly["report"]
    assert "Supported answer [1]." in assembly["report"]
    assert "[2] Source Two: https://example.com/2" in assembly["report"]
    assert "Unused Source" not in assembly["report"]


def test_assemble_best_section_report_does_not_promote_partial_sections(tmp_path) -> None:
    artifacts = ResearchArtifactsV2.create(tmp_path, "partial section assembly")
    artifacts.write_json(
        "best_sections.json",
        {
            "schema_version": 1,
            "sections": {
                "opening_answer": {
                    "section_path": "best_sections/opening_answer.md",
                    "locked": True,
                },
            },
        },
    )
    artifacts.write_text("best_sections/opening_answer.md", "## Bottom line\n\nSupported answer [1].\n")
    section_plan = {
        "sections": [
            {"id": "opening_answer"},
            {"id": "decision_logic"},
        ]
    }

    assembly = assemble_best_section_report(
        artifacts=artifacts,
        latest_report="# Report\n\n## Sources\n\n[1] Source One: https://example.com/1\n",
        section_plan=section_plan,
    )

    assert assembly["usable_for_final"] is False
    assert assembly["assembled_section_count"] == 1

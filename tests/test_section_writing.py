from __future__ import annotations

from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2
from deep_research.section_writing import (
    audit_section_draft,
    audit_report_sections,
    build_adaptive_section_plan,
    extract_report_sections,
    format_section_plan_for_prompt,
    refine_section_audits_from_payload,
    refine_section_plan_from_payload,
)


def test_adaptive_section_plan_uses_task_mode_without_fixed_report_template() -> None:
    plan = ResearchPlan(
        question="Compare CNC platforms for titanium aerospace parts and recommend the best option.",
        intent="general",
        audience="technical buyer",
        report_outline=[],
        branches=[
            ResearchBranch(
                id="architecture",
                title="Machine architecture",
                objective="Compare turret turn-mill and B-axis mill-turn architectures.",
                queries=["CNC mill turn architecture comparison"],
                required_terms=["turret", "B-axis", "mill-turn"],
            ),
            ResearchBranch(
                id="service",
                title="Regional support",
                objective="Assess service, support, and spare-parts risk.",
                queries=["CNC vendor service Mexico"],
                required_terms=["service", "support", "spare parts"],
            ),
        ],
        acceptance_criteria=["Recommendation follows from machine architecture, titanium performance, and service risk."],
    )
    source = _source(1, "architecture", "Machine Architecture")
    card = _card(1, source, "B-axis mill-turn platforms support complex five-axis turning and milling.")

    section_plan = build_adaptive_section_plan(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )

    assert section_plan.report_mode == "comparative_decision"
    assert section_plan.structure_style == "decision-first, trade-off driven"
    title_hints = [section.title_hint for section in section_plan.sections]
    assert "Machine architecture" in title_hints
    assert "Executive Summary" not in title_hints
    prompt_text = format_section_plan_for_prompt(section_plan)
    assert "internal controls" not in prompt_text
    assert "Machine architecture" in prompt_text


def test_section_audit_flags_exact_number_without_nearby_citation() -> None:
    source = _source(1, "performance", "Performance Source")
    card = _card(1, source, "The platform can sustain 5,000 productive spindle hours per year.")
    section_plan = build_adaptive_section_plan(
        plan=_single_branch_plan(),
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )

    audit = audit_section_draft(
        section=section_plan.sections[0],
        markdown="The platform can sustain 5,000 productive spindle hours per year.",
        evidence_cards=[card],
        sources=[source],
        source_texts={1: card.supporting_excerpt},
    )

    assert not audit.locked
    assert any("Exact factual claim lacks" in failure for failure in audit.failures)


def test_section_audit_locks_supported_cited_section() -> None:
    source = _source(1, "performance", "Performance Source")
    card = _card(1, source, "The platform can sustain 5,000 productive spindle hours per year.")
    section_plan = build_adaptive_section_plan(
        plan=_single_branch_plan(),
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )

    audit = audit_section_draft(
        section=section_plan.sections[0],
        markdown="The platform can sustain 5,000 productive spindle hours per year [1].",
        evidence_cards=[card],
        sources=[source],
        source_texts={1: "The platform can sustain 5,000 productive spindle hours per year."},
    )

    assert audit.locked
    assert audit.citation_support_score >= 0.9
    assert audit.evidence_linkage_score == 1.0


def test_section_audit_rejects_citation_without_evidence_card() -> None:
    source = _source(1, "performance", "Performance Source")
    uncited_source = _source(2, "performance", "Unsupported Source")
    card = _card(1, source, "Supported spindle-hour evidence.")
    section_plan = build_adaptive_section_plan(
        plan=_single_branch_plan(),
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source, uncited_source],
    )

    audit = audit_section_draft(
        section=section_plan.sections[0],
        markdown="The machine is best for this workload [2].",
        evidence_cards=[card],
        sources=[source, uncited_source],
        source_texts={2: "This page discusses a related topic."},
    )

    assert not audit.locked
    assert "Cited source [2] has no evidence card." in audit.failures


def test_llm_section_plan_refinement_is_clamped_to_real_evidence() -> None:
    source = _source(1, "performance", "Performance Source")
    card = _card(1, source, "Supported spindle-hour evidence.")
    base_plan = build_adaptive_section_plan(
        plan=_single_branch_plan(),
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )

    refined = refine_section_plan_from_payload(
        base_plan,
        {
            "report_mode": "consulting memo",
            "structure_style": "decision narrative",
            "sections": [
                {
                    "id": "decision_logic",
                    "title_hint": "What the workload really requires",
                    "role": "decision_dimension",
                    "purpose": "Separate machine utilization assumptions from sourced constraints.",
                    "branch_ids": ["performance"],
                    "evidence_card_ids": [card.id, 999],
                    "source_ids": [source.id, 999],
                    "writing_task": "Use a natural heading and keep unsupported numbers out.",
                }
            ],
        },
        evidence_cards=[card],
        sources=[source],
    )

    assert refined.report_mode == "consulting_memo"
    assert refined.structure_style == "decision narrative"
    assert [section.title_hint for section in refined.sections] == ["What the workload really requires"]
    assert refined.sections[0].evidence_card_ids == [card.id]
    assert refined.sections[0].source_ids == [source.id]


def test_report_section_extraction_matches_natural_headings_to_contracts() -> None:
    source = _source(1, "performance", "Performance Source")
    card = _card(1, source, "Titanium machining performance depends on stable spindle utilization.")
    section_plan = build_adaptive_section_plan(
        plan=_single_branch_plan(),
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )
    report = (
        "# Report\n\n"
        "## What the machine can actually sustain\n\n"
        "Titanium machining performance depends on stable spindle utilization [1].\n\n"
        "## Sources\n\n"
        "[1] Performance Source: https://example.com/1\n"
    )

    sections = extract_report_sections(report, section_plan=section_plan)

    assert any(section.section_id == "branch_1_performance" for section in sections)
    assert all("Sources" not in section.markdown for section in sections)


def test_report_section_audit_summarizes_locked_sections() -> None:
    source = _source(1, "performance", "Performance Source")
    card = _card(1, source, "Titanium machining performance depends on stable spindle utilization.")
    section_plan = build_adaptive_section_plan(
        plan=_single_branch_plan(),
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

    audit = audit_report_sections(
        section_plan=section_plan,
        report_markdown=report,
        evidence_cards=[card],
        sources=[source],
        source_texts={1: card.supporting_excerpt},
    )

    assert audit["section_count"] == len(section_plan.sections)
    assert audit["locked_count"] >= 1
    assert any(row["heading"] == "Titanium performance" for row in audit["audits"])


def test_llm_section_audit_review_can_add_but_not_remove_failures() -> None:
    base = {
        "schema_version": 1,
        "section_count": 1,
        "locked_count": 0,
        "all_locked": False,
        "audits": [
            {
                "section_id": "opening_answer",
                "locked": False,
                "failures": ["Section has no citations."],
                "warnings": [],
            }
        ],
    }

    refined = refine_section_audits_from_payload(
        base,
        {
            "section_reviews": [
                {
                    "section_id": "opening_answer",
                    "locked": True,
                    "failures": ["Recommendation contradicts the cited trade-off table."],
                    "repair_guidance": "Reconcile the recommendation with the table before finalizing.",
                }
            ],
            "global_repair_guidance": ["Check final ranking against section-level evidence."],
        },
    )

    failures = refined["audits"][0]["failures"]
    assert not refined["audits"][0]["locked"]
    assert "Section has no citations." in failures
    assert "Recommendation contradicts the cited trade-off table." in failures
    assert refined["global_repair_guidance"] == ["Check final ranking against section-level evidence."]


def _single_branch_plan() -> ResearchPlan:
    return ResearchPlan(
        question="Evaluate a CNC platform for aerospace titanium machining.",
        intent="general",
        audience="technical buyer",
        report_outline=[],
        branches=[
            ResearchBranch(
                id="performance",
                title="Titanium performance",
                objective="Assess machining performance and utilization assumptions.",
                queries=["CNC titanium machining performance"],
                required_terms=["titanium", "machining", "performance"],
            )
        ],
    )


def _source(source_id: int, branch_id: str, title: str) -> SourceRecordV2:
    return SourceRecordV2(
        id=source_id,
        branch_id=branch_id,
        title=title,
        url=f"https://example.com/{source_id}",
        canonical_url=f"https://example.com/{source_id}",
        provenance="web",
        content_path=f"source_docs/source_{source_id}.md",
        content_hash=f"hash-{source_id}",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="official_docs",
        relevance_score=0.9,
    )


def _card(card_id: int, source: SourceRecordV2, claim: str) -> EvidenceCard:
    return EvidenceCard(
        id=card_id,
        source_id=source.id,
        branch_id=source.branch_id,
        claim=claim,
        supporting_excerpt=claim,
        source_url=source.url,
        source_title=source.title,
        quality_score=source.quality_score,
        relevance_score=source.relevance_score,
        confidence=0.9,
    )


def test_preface_skipping_and_heading_matching() -> None:
    plan = _single_branch_plan()
    source = _source(1, "performance", "Performance Source")
    card = _card(1, source, "The platform can sustain 5,000 spindle hours.")
    section_plan = build_adaptive_section_plan(
        plan=plan,
        evidence_cards=[card],
        coverage=CoverageMatrix(branches=[], complete=True, coverage_score=1.0, missing_branches=[]),
        sources=[source],
    )
    # preface is only title
    report = (
        "# Research Report: Titanium Machining\n\n"
        "## Titanium performance\n\n"
        "Titanium machining performance depends on stable spindle utilization [1].\n\n"
        "## Sources\n\n"
        "[1] Performance Source: https://example.com/1\n"
    )
    drafts = extract_report_sections(report, section_plan=section_plan)
    # Preface should be skipped, so only 1 section (Titanium performance)
    assert len(drafts) == 1
    assert drafts[0].section_id == "branch_1_performance"


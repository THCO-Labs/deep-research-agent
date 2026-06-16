from deep_research.schemas import ResearchBranch, ResearchPlan, SourceRecordV2
from deep_research.source_policy import infer_source_policy, score_sources_against_policy


def test_source_policy_detects_technical_procurement() -> None:
    plan = _plan("Compare CNC machine models, vendor specs, manuals, integration, and cost.")

    policy = infer_source_policy(plan)

    assert policy.task_type == "technical_procurement"
    assert "spec_sheet" in policy.preferred_source_types
    assert "manual_pdf" in policy.preferred_source_types


def test_source_policy_scores_preferred_independent_sources() -> None:
    policy = infer_source_policy(_plan("Compare hardware vendor specs and datasheets."))
    sources = [
        _source(1, "https://vendor-a.example/spec.pdf", "spec_sheet", 0.9),
        _source(2, "https://vendor-b.example/manual.pdf", "manual_pdf", 0.85),
    ]

    result = score_sources_against_policy(policy, sources)

    assert result["preferred_source_count"] == 2
    assert result["independent_domain_count"] == 2
    assert result["score"] > 0.4


def test_source_policy_detects_comparative_benchmark_before_procurement() -> None:
    plan = _plan("What are the strongest open-source models in 2026 by benchmark score, cost, context, and tool use?")

    policy = infer_source_policy(plan)

    assert policy.task_type == "comparative_benchmark"
    assert "software_repository" in policy.preferred_source_types
    assert "academic" in policy.preferred_source_types


def _plan(question: str) -> ResearchPlan:
    return ResearchPlan(
        question=question,
        intent="general",
        audience="technical generalist",
        report_outline=[],
        branches=[
            ResearchBranch(
                id="branch_1",
                title="Specs",
                objective="Compare technical specifications.",
                queries=[question],
                required_terms=["specifications"],
            )
        ],
    )


def _source(source_id: int, url: str, quality_type: str, quality: float) -> SourceRecordV2:
    return SourceRecordV2(
        id=source_id,
        branch_id="branch_1",
        title=f"Source {source_id}",
        url=url,
        canonical_url=url,
        provenance="web",
        content_path=f"source_docs/source_{source_id}.md",
        content_hash="hash",
        extraction_method="httpx",
        word_count=500,
        quality_score=quality,
        quality_label="high",
        quality_type=quality_type,
        relevance_score=0.9,
    )

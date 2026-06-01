from deep_research.models import SourceRecord, VerificationResult
from deep_research.repair import render_verification_repair_markdown


def test_render_verification_repair_markdown_lists_failed_invariants() -> None:
    result = VerificationResult(
        valid=False,
        citation_validity_score=0.25,
        source_support_score=0.1,
        missing_sources=["[2] is not in sources.jsonl."],
        unsupported_claims=["This paragraph has no citation."],
        weakly_supported_claims=[
            {
                "paragraph": "Fine-tuning always solves every deployment problem [1].",
                "cited_source_ids": [1],
                "missing_terms": ["always", "deployment", "problem"],
            }
        ],
        source_list_errors=["Report is missing a parseable Sources section."],
        cited_source_ids=[1, 2],
    )
    record = SourceRecord(
        id=1,
        url="https://example.com",
        canonical_url="https://example.com/",
        title="Example",
        fetched_at="2026-01-01T00:00:00+00:00",
        extraction_method="search",
    )

    markdown = render_verification_repair_markdown(result, [record], report_exists=True)

    assert "Fix the `## Sources` section" in markdown
    assert "Weakly Supported Claims" in markdown
    assert "This paragraph has no citation." in markdown
    assert "[2] is not in sources.jsonl." in markdown


def test_render_verification_repair_markdown_handles_passed_result() -> None:
    result = VerificationResult(valid=True, citation_validity_score=1.0, source_support_score=1.0)

    markdown = render_verification_repair_markdown(result, [], report_exists=True)

    assert "None. Verification passed." in markdown

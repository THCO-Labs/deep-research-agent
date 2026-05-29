from deep_research.models import SourceRecord
from deep_research.verifier import parse_inline_citations, parse_source_list, verify_report


def test_parse_citations_excludes_sources_section() -> None:
    report = "Claim [1].\n\n## Sources\n[1] Example: https://example.com"

    assert parse_inline_citations(report) == [1]
    assert parse_source_list(report) == {1: "https://example.com"}


def test_verify_report_detects_valid_report() -> None:
    record = SourceRecord(
        id=1,
        url="https://example.com",
        canonical_url="https://example.com/",
        title="Example",
        fetched_at="2026-01-01T00:00:00+00:00",
        extraction_method="playwright",
    )
    report = "## Finding\n\nA factual claim is cited [1].\n\n## Sources\n[1] Example: https://example.com"

    result = verify_report(report, [record])

    assert result.valid is False
    assert result.unscraped_sources == [1]


def test_verify_report_flags_uncited_paragraph() -> None:
    report = "## Finding\n\nThis factual paragraph has no citation.\n\n## Sources\n"

    result = verify_report(report, [])

    assert result.valid is False
    assert result.unsupported_claims


def test_verify_report_accepts_scraped_source() -> None:
    record = SourceRecord(
        id=1,
        url="https://example.com",
        canonical_url="https://example.com/",
        title="Example",
        fetched_at="2026-01-01T00:00:00+00:00",
        extraction_method="playwright",
        content_hash="abc",
        content_path="source_docs/source_1.md",
    )
    report = "## Finding\n\nA factual claim is cited [1].\n\n## Sources\n[1] Example: https://example.com"

    result = verify_report(report, [record])

    assert result.valid is True
    assert result.citation_validity_score == 1.0

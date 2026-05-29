from deep_research.agent import _ensure_source_section
from deep_research.models import SourceRecord


class FakeRegistry:
    records = [
        SourceRecord(
            id=1,
            url="https://example.com",
            canonical_url="https://example.com/",
            title="Example",
            fetched_at="2026-01-01T00:00:00+00:00",
            extraction_method="playwright",
            content_hash="abc",
            content_path="source_docs/source_1.md",
        )
    ]

    def source_lines(self) -> str:
        return "[1] Example: https://example.com"


def test_reconstructed_report_gets_source_section() -> None:
    report = "A cited claim [1]."

    reconstructed = _ensure_source_section(report, FakeRegistry())

    assert "## Sources" in reconstructed
    assert "[1] Example: https://example.com" in reconstructed

from __future__ import annotations

import re
from collections import Counter

from deep_research.models import SourceRecord, VerificationResult
from deep_research.urls import canonicalize_url

CITATION_RE = re.compile(r"\[(\d+)\]")
SOURCE_LINE_RE = re.compile(r"^\[(\d+)\]\s+(.+?):\s+(https?://\S+)\s*$")


def parse_inline_citations(markdown: str) -> list[int]:
    return [int(match.group(1)) for match in CITATION_RE.finditer(_without_sources(markdown))]


def parse_source_list(markdown: str) -> dict[int, str]:
    sources_text = _sources_section(markdown)
    parsed: dict[int, str] = {}
    for line in sources_text.splitlines():
        match = SOURCE_LINE_RE.match(line.strip())
        if match:
            parsed[int(match.group(1))] = match.group(3)
    return parsed


def verify_report(
    report_markdown: str,
    source_records: list[SourceRecord],
    *,
    verification_rounds: int = 0,
) -> VerificationResult:
    registry_by_id = {record.id: record for record in source_records}
    source_list = parse_source_list(report_markdown)
    citations = parse_inline_citations(report_markdown)
    cited_ids = sorted(set(citations))
    missing: list[str] = []
    source_errors: list[str] = []
    unscraped: list[int] = []

    if not source_list:
        source_errors.append("Report is missing a parseable Sources section.")

    for source_id in cited_ids:
        if source_id not in registry_by_id:
            missing.append(f"[{source_id}] is not in sources.jsonl.")
        if source_id not in source_list:
            missing.append(f"[{source_id}] is cited inline but missing from Sources.")
        record = registry_by_id.get(source_id)
        if record and (not record.content_path or not record.content_hash):
            unscraped.append(source_id)

    for source_id, listed_url in source_list.items():
        record = registry_by_id.get(source_id)
        if record is None:
            source_errors.append(f"Sources entry [{source_id}] is not in sources.jsonl.")
            continue
        try:
            listed_canonical = canonicalize_url(listed_url)
        except ValueError as exc:
            source_errors.append(f"Sources entry [{source_id}] has invalid URL: {exc}")
            continue
        if listed_canonical != record.canonical_url:
            source_errors.append(
                f"Sources entry [{source_id}] URL does not match registry canonical URL."
            )

    unused = sorted(record.id for record in source_records if record.id not in cited_ids)
    unsupported = _paragraphs_without_citations(report_markdown)
    source_sequence_errors = _source_sequence_errors(source_list)
    source_errors.extend(source_sequence_errors)

    denominator = max(len(cited_ids) + len(source_list) + len(unsupported), 1)
    failures = len(set(missing)) + len(source_errors) + len(unsupported) + len(set(unscraped))
    score = max(0.0, 1.0 - failures / denominator)
    valid = not missing and not source_errors and not unsupported and not unscraped and bool(cited_ids)
    return VerificationResult(
        valid=valid,
        citation_validity_score=round(score, 4),
        missing_sources=sorted(set(missing)),
        unused_sources=unused,
        unscraped_sources=sorted(set(unscraped)),
        unsupported_claims=unsupported,
        source_list_errors=source_errors,
        cited_source_ids=cited_ids,
        total_citations=sum(Counter(citations).values()),
        verification_rounds=verification_rounds,
    )


def _sources_section(markdown: str) -> str:
    match = re.search(r"(?ims)^#{2,3}\s+sources\s*$", markdown)
    if not match:
        return ""
    return markdown[match.end() :]


def _without_sources(markdown: str) -> str:
    match = re.search(r"(?ims)^#{2,3}\s+sources\s*$", markdown)
    return markdown if not match else markdown[: match.start()]


def _paragraphs_without_citations(markdown: str) -> list[str]:
    body = _without_sources(markdown)
    unsupported: list[str] = []
    paragraphs = re.split(r"\n\s*\n", body)
    for paragraph in paragraphs:
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not text or text.startswith("#") or text.startswith("|"):
            continue
        if text.startswith(("-", "*")) and len(text) < 80:
            continue
        if not CITATION_RE.search(text):
            unsupported.append(text[:240])
    return unsupported


def _source_sequence_errors(source_list: dict[int, str]) -> list[str]:
    if not source_list:
        return []
    expected = list(range(1, max(source_list) + 1))
    actual = sorted(source_list)
    if actual != expected:
        return [f"Sources must be sequential without gaps; found {actual}."]
    return []

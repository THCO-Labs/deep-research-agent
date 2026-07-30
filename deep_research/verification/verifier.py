from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from deep_research.core.models import SourceRecord, VerificationResult
from deep_research.evidence.text_terms import term_set
from deep_research.acquisition.urls import canonicalize_url

CITATION_RE = re.compile(r"\[([0-9][0-9,;\s]*)\]")
SOURCE_LINE_RE = re.compile(r"^\[(\d+)\]\s+(.+?):\s+(https?://\S+)\s*$")
SUPPORT_THRESHOLD = 0.28
MIN_SUPPORT_TOKENS = 6

SourceLoader = Callable[[SourceRecord], str]


def parse_inline_citations(markdown: str) -> list[int]:
    citations: list[int] = []
    for match in CITATION_RE.finditer(_without_sources(markdown)):
        citations.extend(_citation_ids_from_match(match.group(1)))
    return citations


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
    source_loader: SourceLoader | None = None,
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
    support_checks, source_support_errors = _source_support_checks(
        report_markdown,
        registry_by_id,
        source_loader,
    )
    source_errors.extend(source_support_errors)
    weak_claims = [check for check in support_checks if not check["supported"]]
    source_support_score = _average_support_score(support_checks)
    source_sequence_errors = _source_sequence_errors(source_list, registry_by_id)
    source_errors.extend(source_sequence_errors)

    denominator = max(len(cited_ids) + len(source_list) + len(unsupported) + len(support_checks), 1)
    failures = (
        len(set(missing))
        + len(source_errors)
        + len(unsupported)
        + len(weak_claims)
        + len(set(unscraped))
    )
    score = max(0.0, 1.0 - failures / denominator)
    valid = (
        not missing
        and not source_errors
        and not unsupported
        and not weak_claims
        and not unscraped
        and bool(cited_ids)
    )
    return VerificationResult(
        valid=valid,
        citation_validity_score=round(score, 4),
        source_support_score=source_support_score,
        missing_sources=sorted(set(missing)),
        unused_sources=unused,
        unscraped_sources=sorted(set(unscraped)),
        unsupported_claims=unsupported,
        weakly_supported_claims=weak_claims,
        support_checks=support_checks,
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
        if not _has_inline_citation(text):
            unsupported.append(text[:240])
    return unsupported


def _source_support_checks(
    markdown: str,
    registry_by_id: dict[int, SourceRecord],
    source_loader: SourceLoader | None,
) -> tuple[list[dict[str, object]], list[str]]:
    if source_loader is None:
        return [], []

    checks: list[dict[str, object]] = []
    errors: list[str] = []
    source_text_cache: dict[int, str] = {}

    for paragraph in _paragraphs_with_citations(markdown):
        cited_ids = sorted(set(parse_inline_citations(paragraph)))
        source_text_parts: list[str] = []
        for source_id in cited_ids:
            record = registry_by_id.get(source_id)
            if record is None or not record.content_path:
                continue
            if source_id not in source_text_cache:
                try:
                    source_text_cache[source_id] = source_loader(record)
                except Exception as exc:  # verifier surfaces the exact read failure.
                    errors.append(f"Source content for [{source_id}] could not be read: {exc}")
                    source_text_cache[source_id] = ""
            source_text_parts.append(source_text_cache[source_id])

        claim_tokens = _significant_tokens(_strip_citations(paragraph))
        if len(claim_tokens) < MIN_SUPPORT_TOKENS:
            continue

        source_tokens = _significant_tokens(" ".join(source_text_parts))
        support_score = _token_support_score(claim_tokens, source_tokens)
        supported = support_score >= SUPPORT_THRESHOLD
        checks.append(
            {
                "paragraph": paragraph[:240],
                "cited_source_ids": cited_ids,
                "support_score": round(support_score, 4),
                "supported": supported,
                "matched_terms": sorted(claim_tokens & source_tokens)[:20],
                "missing_terms": sorted(claim_tokens - source_tokens)[:20],
            }
        )
    return checks, errors


def _paragraphs_with_citations(markdown: str) -> list[str]:
    body = _without_sources(markdown)
    cited: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not text or text.startswith("#") or text.startswith("|"):
            continue
        if _has_inline_citation(text):
            cited.append(text)
    return cited


def _strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def _has_inline_citation(text: str) -> bool:
    return bool(parse_inline_citations(text))


def _citation_ids_from_match(raw_ids: str) -> list[int]:
    ids: list[int] = []
    for part in re.split(r"[,;\s]+", raw_ids):
        stripped = part.strip()
        if stripped:
            ids.append(int(stripped))
    return ids


def _significant_tokens(text: str) -> set[str]:
    return term_set(text)


def _token_support_score(claim_tokens: set[str], source_tokens: set[str]) -> float:
    if not claim_tokens:
        return 1.0
    if not source_tokens:
        return 0.0
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def _average_support_score(checks: list[dict[str, object]]) -> float:
    if not checks:
        return 1.0
    return round(sum(float(check["support_score"]) for check in checks) / len(checks), 4)


def _source_sequence_errors(
    source_list: dict[int, str],
    registry_by_id: dict[int, SourceRecord],
) -> list[str]:
    if not source_list:
        return []
    actual = sorted(source_list)
    scraped_ids = sorted(
        source_id
        for source_id, record in registry_by_id.items()
        if record.content_path and record.content_hash
    )
    if scraped_ids and actual != scraped_ids:
        return [
            "Sources should list all scraped source IDs used for citation; "
            f"found {actual}, scraped source IDs are {scraped_ids}."
        ]
    return []

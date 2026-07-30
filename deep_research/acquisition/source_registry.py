from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from deep_research.runtime.artifacts import RunArtifacts
from deep_research.core.models import SourceRecord
from deep_research.evidence.source_quality import score_source
from deep_research.evidence.source_relevance import score_source_relevance
from deep_research.acquisition.urls import canonicalize_url


class SourceRegistry:
    def __init__(self, artifacts: RunArtifacts):
        self.artifacts = artifacts
        self.records: list[SourceRecord] = []
        self._by_url: dict[str, SourceRecord] = {}
        self._by_hash: dict[str, SourceRecord] = {}
        self._persist()

    def upsert_search_result(
        self,
        *,
        url: str,
        title: str,
        query: str,
        snippet: str | None = None,
        search_score: float | None = None,
    ) -> SourceRecord:
        canonical_url = canonicalize_url(url)
        existing = self._by_url.get(canonical_url)
        if existing:
            if not existing.title and title:
                existing.title = title
            existing.query = query or existing.query
            existing.snippet = existing.snippet or snippet
            existing.search_score = existing.search_score if existing.search_score is not None else search_score
            self._apply_quality(existing)
            self._apply_relevance(existing)
            self._persist()
            return existing

        quality = score_source(
            url=url,
            title=title,
            snippet=snippet,
            search_score=search_score,
        )
        relevance = score_source_relevance(
            query=query,
            url=url,
            title=title,
            snippet=snippet,
        )
        record = SourceRecord(
            id=len(self.records) + 1,
            url=url,
            canonical_url=canonical_url,
            title=title or canonical_url,
            fetched_at=_now_iso(),
            extraction_method="search",
            query=query,
            snippet=snippet,
            search_score=search_score,
            source_quality_score=quality.score,
            source_quality_label=quality.label,
            source_quality_type=quality.source_type,
            source_quality_reasons=quality.reasons,
            source_relevance_score=relevance.score,
            source_relevance_matched_terms=relevance.matched_terms,
            source_relevance_missing_terms=relevance.missing_terms,
        )
        self._add(record)
        return record

    def record_scrape(
        self,
        *,
        url: str,
        title: str,
        markdown: str,
        extraction_method: str,
    ) -> SourceRecord:
        canonical_url = canonicalize_url(url)
        content_hash = _content_hash(markdown)
        existing_by_url = self._by_url.get(canonical_url)
        existing_by_hash = self._by_hash.get(content_hash)
        if existing_by_hash is not None and existing_by_url is None:
            return existing_by_hash

        existing = existing_by_url
        record = existing or SourceRecord(
            id=len(self.records) + 1,
            url=url,
            canonical_url=canonical_url,
            title=title or canonical_url,
            fetched_at=_now_iso(),
            extraction_method=extraction_method,
        )
        record.url = url
        record.canonical_url = canonical_url
        record.title = title or record.title
        record.fetched_at = _now_iso()
        record.extraction_method = extraction_method
        record.content_hash = content_hash
        record.content_path = f"source_docs/source_{record.id}.md"
        self._apply_quality(record, markdown=markdown)
        self._apply_relevance(record, markdown=markdown)
        self.artifacts.write_text(
            record.content_path,
            f"# {record.title}\n\n"
            f"URL: {record.url}\n"
            f"Canonical URL: {record.canonical_url}\n"
            f"Source quality: {record.source_quality_label} "
            f"({record.source_quality_score}) - {record.source_quality_type}\n"
            f"Quality reasons: {', '.join(record.source_quality_reasons)}\n\n"
            f"Source relevance: {record.source_relevance_score}\n"
            f"Relevance matches: {', '.join(record.source_relevance_matched_terms)}\n"
            f"Relevance missing: {', '.join(record.source_relevance_missing_terms)}\n\n"
            f"{markdown}",
        )
        if existing is None:
            self._add(record)
        else:
            self._by_url[canonical_url] = record
            self._by_hash[content_hash] = record
            self._persist()
        return record

    def source_lines(self) -> str:
        lines = []
        for record in self.records:
            lines.append(f"[{record.id}] {record.title}: {record.url}")
        return "\n".join(lines)

    def to_rows(self) -> list[dict[str, object]]:
        return [record.to_dict() for record in self.records]

    def _add(self, record: SourceRecord) -> None:
        self.records.append(record)
        self._by_url[record.canonical_url] = record
        if record.content_hash:
            self._by_hash[record.content_hash] = record
        self._persist()

    def _persist(self) -> None:
        self.artifacts.write_jsonl("sources.jsonl", self.to_rows())

    def _apply_quality(self, record: SourceRecord, markdown: str | None = None) -> None:
        quality = score_source(
            url=record.url,
            title=record.title,
            snippet=record.snippet,
            markdown=markdown,
            search_score=record.search_score,
        )
        record.source_quality_score = quality.score
        record.source_quality_label = quality.label
        record.source_quality_type = quality.source_type
        record.source_quality_reasons = quality.reasons

    def _apply_relevance(self, record: SourceRecord, markdown: str | None = None) -> None:
        relevance = score_source_relevance(
            query=record.query or "",
            url=record.url,
            title=record.title,
            snippet=record.snippet,
            markdown=markdown,
        )
        record.source_relevance_score = relevance.score
        record.source_relevance_matched_terms = relevance.matched_terms
        record.source_relevance_missing_terms = relevance.missing_terms


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")

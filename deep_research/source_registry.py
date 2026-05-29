from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from deep_research.artifacts import RunArtifacts
from deep_research.models import SourceRecord
from deep_research.urls import canonicalize_url


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
    ) -> SourceRecord:
        canonical_url = canonicalize_url(url)
        existing = self._by_url.get(canonical_url)
        if existing:
            if not existing.title and title:
                existing.title = title
                self._persist()
            return existing

        record = SourceRecord(
            id=len(self.records) + 1,
            url=url,
            canonical_url=canonical_url,
            title=title or canonical_url,
            fetched_at=_now_iso(),
            extraction_method="search",
            query=query,
            snippet=snippet,
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
        self.artifacts.write_text(
            record.content_path,
            f"# {record.title}\n\nURL: {record.url}\nCanonical URL: {record.canonical_url}\n\n{markdown}",
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


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")

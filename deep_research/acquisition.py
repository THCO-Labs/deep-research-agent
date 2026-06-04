from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from tavily import TavilyClient

from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.errors import classify_exception
from deep_research.ingestion import IngestedDocument
from deep_research.schemas import ResearchBranch, SourceCandidate, SourceRecordV2
from deep_research.scraper import PlaywrightScraper, ScrapeQualityError, ScrapeResult
from deep_research.source_limits import MAX_TAVILY_RESULTS_PER_QUERY, MINIMUM_SOURCE_TARGET, source_floor
from deep_research.source_quality import score_source
from deep_research.source_validation import branch_terms, content_terms, validate_source_content
from deep_research.urls import canonicalize_url


@dataclass
class AcquisitionMetrics:
    search_count: int = 0
    candidate_count: int = 0
    scrape_count: int = 0
    usable_source_count: int = 0
    rejected_source_count: int = 0
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "search_count": self.search_count,
            "candidate_count": self.candidate_count,
            "scrape_count": self.scrape_count,
            "usable_source_count": self.usable_source_count,
            "rejected_source_count": self.rejected_source_count,
            "failures": self.failures,
        }


@dataclass
class AcquisitionResult:
    candidates: list[SourceCandidate]
    sources: list[SourceRecordV2]
    source_texts: dict[int, str]
    metrics: AcquisitionMetrics


class TavilySearchClientPool:
    def __init__(self, settings: Any) -> None:
        keys = tuple(getattr(settings, "tavily_key_pool", ()) or ())
        if not keys:
            single = str(getattr(settings, "tavily_api_key", "") or "").strip()
            keys = (single,) if single else ()
        if not keys:
            raise ValueError("Tavily API key pool cannot be empty.")
        self._clients = tuple(TavilyClient(api_key=key) for key in keys)
        self.key_count = len(self._clients)
        self._cursor = 0

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        for offset in range(self.key_count):
            index = (self._cursor + offset) % self.key_count
            try:
                response = self._clients[index].search(query, **kwargs)
                self._cursor = (index + 1) % self.key_count
                return response
            except Exception as exc:
                last_error = exc
                if classify_exception(exc).category != "quota_or_rate_limit":
                    raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("No Tavily search clients were available.")


def acquire_sources(
    *,
    question: str,
    branches: list[ResearchBranch],
    artifacts: ResearchArtifactsV2,
    settings: Any,
    search_client: Any | None = None,
    scraper: Any | None = None,
    local_documents: list[IngestedDocument] | None = None,
    mcp_documents: list[IngestedDocument] | None = None,
    existing_candidates: list[SourceCandidate] | None = None,
    existing_sources: list[SourceRecordV2] | None = None,
    existing_source_texts: dict[int, str] | None = None,
    searched_queries: set[str] | None = None,
    focus_terms_by_branch: dict[str, list[str]] | None = None,
) -> AcquisitionResult:
    metrics = AcquisitionMetrics()
    client = search_client or TavilySearchClientPool(settings)
    page_scraper = scraper or PlaywrightScraper()
    min_words = int(getattr(settings, "min_source_words", 250))
    min_chunks = int(getattr(settings, "min_relevant_chunks", 1))
    max_candidates = int(getattr(settings, "max_candidates", 80))
    required_branch_sources = sum(branch.min_sources for branch in branches)
    explicit_max_sources = int(getattr(settings, "max_sources", MINIMUM_SOURCE_TARGET) or 0)
    if explicit_max_sources <= 0:
        max_sources = max(max_candidates, required_branch_sources, int(getattr(settings, "min_usable_sources", MINIMUM_SOURCE_TARGET)))
    else:
        max_sources = max(
            source_floor(int(getattr(settings, "min_usable_sources", MINIMUM_SOURCE_TARGET) or MINIMUM_SOURCE_TARGET)),
            source_floor(explicit_max_sources),
            required_branch_sources,
        )
    max_candidates = max(max_candidates, max_sources)
    candidates: list[SourceCandidate] = list(existing_candidates or [])
    sources: list[SourceRecordV2] = list(existing_sources or [])
    source_texts: dict[int, str] = dict(existing_source_texts or {})
    seen_urls: set[str] = {source.canonical_url for source in sources}
    searched = set(searched_queries or ())

    ingested_documents = list(local_documents or []) + list(mcp_documents or [])
    if not existing_sources:
        _record_ingested_documents(
            ingested_documents,
            branches,
            artifacts,
            sources,
            source_texts,
            question=question,
            min_words=min_words,
            min_chunks=min_chunks,
            metrics=metrics,
        )

    next_candidate_id = max((candidate.id for candidate in candidates), default=0) + 1
    focus_terms_by_branch = focus_terms_by_branch or {}
    for branch in branches:
        if len(sources) >= max_sources:
            break
        branch_source_count = sum(1 for source in sources if source.branch_id == branch.id)
        forced_terms = focus_terms_by_branch.get(branch.id, [])
        if branch_source_count >= branch.min_sources and not forced_terms:
            continue
        branch_queries = _branch_queries(branch, forced_terms, question)
        for query in branch_queries:
            if len(candidates) >= max_candidates or len(sources) >= max_sources:
                break
            query = _trim_search_query(query)
            if query in searched:
                continue
            searched.add(query)
            metrics.search_count += 1
            try:
                search_results = _search(client, query, settings)
            except Exception as exc:
                metrics.failures.append(f"Search failed for {query!r}: {exc}")
                continue
            for item in search_results:
                url = str(item.get("url") or "")
                if not url:
                    continue
                canonical = _safe_canonical(url)
                if canonical in seen_urls:
                    continue
                seen_urls.add(canonical)
                candidate = SourceCandidate(
                    id=next_candidate_id,
                    branch_id=branch.id,
                    title=str(item.get("title") or url),
                    url=url,
                    query=query,
                    snippet=str(item.get("content") or item.get("snippet") or ""),
                    search_score=_float_or_none(item.get("score")),
                    raw_content=_raw_content(item),
                    provenance="web",
                )
                next_candidate_id += 1
                candidates.append(candidate)
                metrics.candidate_count += 1
                record = _candidate_to_source(
                    candidate,
                    branch,
                    artifacts,
                    page_scraper,
                    min_words=min_words,
                    min_chunks=min_chunks,
                    metrics=metrics,
                    question=question,
                    source_id=len(sources) + 1,
                )
                if record is None:
                    continue
                sources.append(record.source)
                source_texts[record.source.id] = record.text
                if sum(1 for source in sources if source.branch_id == branch.id) >= branch.min_sources:
                    break

    artifacts.write_jsonl("sources.jsonl", [source.to_dict() for source in sources])
    return AcquisitionResult(
        candidates=candidates,
        sources=sources,
        source_texts=source_texts,
        metrics=metrics,
    )


@dataclass(frozen=True)
class _RecordedSource:
    source: SourceRecordV2
    text: str


def _record_ingested_documents(
    documents: list[IngestedDocument],
    branches: list[ResearchBranch],
    artifacts: ResearchArtifactsV2,
    sources: list[SourceRecordV2],
    source_texts: dict[int, str],
    *,
    min_words: int,
    min_chunks: int,
    metrics: AcquisitionMetrics,
    question: str,
) -> None:
    for document in documents:
        branch = _best_branch_for_text(document.title + "\n" + document.content, branches)
        source_id = len(sources) + 1
        validation = validate_source_content(
            title=document.title,
            content=document.content,
            branch=branch,
            min_words=max(40, min_words // 2),
            min_relevant_chunks=max(1, min_chunks),
            question=question,
        )
        if not validation.usable:
            metrics.rejected_source_count += 1
            metrics.failures.append(f"Rejected {document.url}: {', '.join(validation.reasons)}")
            continue
        quality = score_source(url=document.url, title=document.title, markdown=document.content)
        source = _write_source(
            source_id=source_id,
            branch=branch,
            title=document.title,
            url=document.url,
            text=document.content,
            extraction_method=str(document.metadata.get("suffix") or document.provenance),
            provenance=document.provenance,
            artifacts=artifacts,
            quality_score=quality.score,
            quality_label=quality.label,
            quality_type=quality.source_type,
            relevance_score=validation.relevance_score,
            word_count=validation.word_count,
            metadata=document.metadata,
        )
        sources.append(source)
        source_texts[source.id] = document.content
        metrics.usable_source_count += 1


def _candidate_to_source(
    candidate: SourceCandidate,
    branch: ResearchBranch,
    artifacts: ResearchArtifactsV2,
    scraper: Any,
    *,
    min_words: int,
    min_chunks: int,
    metrics: AcquisitionMetrics,
    question: str,
    source_id: int,
) -> _RecordedSource | None:
    try:
        scraped = _scrape_candidate(candidate, scraper, metrics)
    except Exception as exc:
        metrics.rejected_source_count += 1
        metrics.failures.append(f"Rejected {candidate.url}: {exc}")
        return None

    validation = validate_source_content(
        title=scraped.title,
        content=scraped.markdown,
        branch=branch,
        min_words=min_words,
        min_relevant_chunks=min_chunks,
        question=question,
    )
    if not validation.usable:
        metrics.rejected_source_count += 1
        metrics.failures.append(f"Rejected {candidate.url}: {', '.join(validation.reasons)}")
        return None

    quality = score_source(
        url=scraped.url,
        title=scraped.title,
        snippet=candidate.snippet,
        markdown=scraped.markdown,
        search_score=candidate.search_score,
    )
    source = _write_source(
        source_id=source_id,
        branch=branch,
        title=scraped.title,
        url=scraped.url,
        text=scraped.markdown,
        extraction_method=scraped.extraction_method,
        provenance="web",
        artifacts=artifacts,
        quality_score=quality.score,
        quality_label=quality.label,
        quality_type=quality.source_type,
        relevance_score=validation.relevance_score,
        word_count=validation.word_count,
        metadata={
            "query": candidate.query,
            "candidate_id": candidate.id,
            "search_score": candidate.search_score,
            "relevant_chunk_count": validation.relevant_chunk_count,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    metrics.usable_source_count += 1
    return _RecordedSource(source=source, text=scraped.markdown)


def _scrape_candidate(candidate: SourceCandidate, scraper: Any, metrics: AcquisitionMetrics) -> ScrapeResult:
    if candidate.raw_content and len(candidate.raw_content.split()) >= 80:
        return ScrapeResult(
            url=candidate.url,
            title=candidate.title,
            markdown=candidate.raw_content,
            extraction_method="tavily_raw_content",
        )
    metrics.scrape_count += 1
    try:
        return scraper.fetch(candidate.url)
    except ScrapeQualityError:
        raise


def _write_source(
    *,
    source_id: int,
    branch: ResearchBranch,
    title: str,
    url: str,
    text: str,
    extraction_method: str,
    provenance: str,
    artifacts: ResearchArtifactsV2,
    quality_score: float,
    quality_label: str,
    quality_type: str,
    relevance_score: float,
    word_count: int,
    metadata: dict[str, Any],
) -> SourceRecordV2:
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    content_path = f"source_docs/source_{source_id}.md"
    canonical = _safe_canonical(url)
    artifacts.write_text(
        content_path,
        f"# {title}\n\n"
        f"URL: {url}\n"
        f"Canonical URL: {canonical}\n"
        f"Branch: {branch.id}\n"
        f"Extraction method: {extraction_method}\n"
        f"Word count: {word_count}\n\n"
        f"{text.strip()}\n",
    )
    return SourceRecordV2(
        id=source_id,
        branch_id=branch.id,
        title=title,
        url=url,
        canonical_url=canonical,
        provenance=provenance,  # type: ignore[arg-type]
        content_path=content_path,
        content_hash=content_hash,
        extraction_method=extraction_method,
        word_count=word_count,
        quality_score=quality_score,
        quality_label=quality_label,
        quality_type=quality_type,
        relevance_score=relevance_score,
        metadata=metadata,
    )


def _search(client: Any, query: str, settings: Any) -> list[dict[str, Any]]:
    query = _trim_search_query(query)
    explicit_max_sources = int(getattr(settings, "max_sources", MINIMUM_SOURCE_TARGET) or 0)
    max_results = min(
        MAX_TAVILY_RESULTS_PER_QUERY,
        MAX_TAVILY_RESULTS_PER_QUERY if explicit_max_sources <= 0 else source_floor(explicit_max_sources),
    )
    kwargs = {
        "max_results": max_results,
        "search_depth": getattr(settings, "search_depth", "advanced"),
        "chunks_per_source": 3,
        "include_raw_content": bool(getattr(settings, "allow_raw_content", True)),
    }
    try:
        response = client.search(query, **kwargs)
    except TypeError:
        response = client.search(query, max_results=max_results)
    return list(response.get("results", []))


def _branch_queries(branch: ResearchBranch, forced_terms: list[str], question: str) -> list[str]:
    if not forced_terms:
        return branch.queries
    focus = " ".join(term for term in forced_terms if term).strip()
    followups = _dedupe(
        branch.queries
        + [
            f"{question} {branch.title} {focus}",
            f"{branch.title} {focus} evidence review findings",
            f"{branch.title} {focus} mechanisms context limitations",
        ]
    )
    return followups


def _trim_search_query(query: str, *, max_chars: int = 380) -> str:
    cleaned = " ".join(str(query).split()).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    boundary = cleaned.rfind(" ", 0, max_chars)
    return cleaned[: boundary if boundary > 120 else max_chars].strip()


def _best_branch_for_text(text: str, branches: list[ResearchBranch]) -> ResearchBranch:
    terms = content_terms(text)
    return max(
        branches,
        key=lambda branch: len(branch_terms(branch) & terms),
    )


def _safe_canonical(url: str) -> str:
    try:
        return canonicalize_url(url)
    except ValueError:
        return url.strip()


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _raw_content(item: dict[str, Any]) -> str | None:
    for key in ("raw_content", "rawContent", "content"):
        value = item.get(key)
        if isinstance(value, str) and len(value.split()) >= 120:
            return value
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = " ".join(value.split()).strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result

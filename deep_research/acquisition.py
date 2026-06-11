from __future__ import annotations

import hashlib
import logging
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable

from tavily import TavilyClient

try:
    from ddgs import DDGS
except Exception:
    try:
        from duckduckgo_search import DDGS
    except Exception:
        DDGS = None

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
        # Once all Tavily keys return quota errors, stop probing them — go straight
        # to DuckDuckGo for the rest of the run. Avoids logging the same warning
        # 149 times per run.
        self._tavily_dead = False
        # Track DuckDuckGo failures so we can short-circuit when DDG starts
        # rate-limiting us (which happens after ~50-100 rapid queries).
        self._ddg_consecutive_failures = 0
        self._ddg_dead = False

    def search(self, query: str, **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        if not self._tavily_dead:
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
                    if not _duckduckgo_available():
                        raise
                    continue  # try next key
            # All Tavily keys returned quota errors. Mark as dead and log once.
            if last_error is not None:
                self._tavily_dead = True
                logging.getLogger(__name__).warning(
                    "All Tavily keys exhausted (%s); switching to DuckDuckGo for remainder of run.",
                    last_error,
                )
        if self._ddg_dead:
            raise RuntimeError("All search providers exhausted (Tavily quota + DuckDuckGo unavailable).")
        try:
            fallback_results = _search_with_duckduckgo(
                query,
                max_results=int(kwargs.get("max_results", 5)),
            )
        except Exception as exc:
            self._ddg_consecutive_failures += 1
            if self._ddg_consecutive_failures >= 5:
                self._ddg_dead = True
                logging.getLogger(__name__).error(
                    "DuckDuckGo failed %d times in a row (%s); marking unavailable.",
                    self._ddg_consecutive_failures, exc,
                )
            return {"results": [], "_provider": "duckduckgo"}
        if fallback_results:
            self._ddg_consecutive_failures = 0
            return {"results": fallback_results, "_provider": "duckduckgo"}
        # Empty results from DDG can mean rate-limited. Count as failure.
        self._ddg_consecutive_failures += 1
        if self._ddg_consecutive_failures >= 10:
            self._ddg_dead = True
            logging.getLogger(__name__).error(
                "DuckDuckGo returned empty results %d times in a row; marking unavailable.",
                self._ddg_consecutive_failures,
            )
        return {"results": [], "_provider": "duckduckgo"}


MID_ACQUISITION_CHECKPOINT_INTERVAL = 5  # save state every N accepted sources


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
    active_branch_ids: set[str] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
    mid_checkpoint_callback: Callable[[list[SourceCandidate], list[SourceRecordV2]], None] | None = None,
) -> AcquisitionResult:
    metrics = AcquisitionMetrics()
    client = search_client or TavilySearchClientPool(settings)
    page_scraper = scraper or PlaywrightScraper(
        timeout_ms=int(getattr(settings, "scrape_timeout_ms", 20_000)),
        retries=int(getattr(settings, "scrape_retries", 1)),
    )
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
    searched.update(candidate.query for candidate in candidates if candidate.query)

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
        if active_branch_ids is not None and branch.id not in active_branch_ids:
            continue
        if len(sources) >= max_sources:
            break
        branch_source_count = sum(1 for source in sources if source.branch_id == branch.id)
        forced_terms = focus_terms_by_branch.get(branch.id, [])
        coverage_followup = active_branch_ids is not None and branch.id in active_branch_ids
        if branch_source_count >= branch.min_sources and not forced_terms and not coverage_followup:
            continue
        branch_queries = _branch_queries(
            branch,
            forced_terms or (branch.required_terms if coverage_followup else []),
            question,
        )
        branch_query_limit = int(getattr(settings, "max_followup_queries_per_branch", 12) or 12)
        searched_for_branch = 0
        for query in branch_queries:
            if coverage_followup and searched_for_branch >= branch_query_limit:
                break
            if len(candidates) >= max_candidates or len(sources) >= max_sources:
                break
            query = _trim_search_query(query)
            if query in searched:
                continue
            searched.add(query)
            searched_for_branch += 1
            metrics.search_count += 1
            _emit_progress(
                progress_callback,
                "searching source candidates",
                branch_id=branch.id,
                query=query,
                sources=len(sources),
                candidates=len(candidates),
                searches=metrics.search_count,
            )
            try:
                search_response = _search_raw(client, query, settings)
                search_results = list(search_response.get("results", []))
                search_provider = str(search_response.get("_provider", "tavily"))
            except Exception as exc:
                metrics.failures.append(f"Search failed for {query!r}: {exc}")
                _emit_progress(
                    progress_callback,
                    "source search failed",
                    branch_id=branch.id,
                    query=query,
                    error=str(exc),
                    sources=len(sources),
                    candidates=len(candidates),
                    searches=metrics.search_count,
                )
                continue
            _emit_progress(
                progress_callback,
                f"search returned {len(search_results)} candidate(s)",
                branch_id=branch.id,
                query=query,
                search_provider=search_provider,
                sources=len(sources),
                candidates=len(candidates),
                searches=metrics.search_count,
            )
            browser_scrapes_for_query = 0
            browser_scrape_limit = int(getattr(settings, "max_browser_scrapes_per_query", 4) or 0)
            for item in search_results:
                url = str(item.get("url") or "")
                if not url:
                    continue
                title = str(item.get("title") or url)
                snippet = str(item.get("content") or item.get("snippet") or "")
                block_reason = _blocked_source_reason(
                    url=url,
                    title=title,
                    snippet=snippet,
                    settings=settings,
                )
                if block_reason:
                    metrics.rejected_source_count += 1
                    metrics.failures.append(f"Blocked {url}: {block_reason}")
                    _emit_progress(
                        progress_callback,
                        "blocked source candidate",
                        branch_id=branch.id,
                        url=url,
                        reason=block_reason,
                        sources=len(sources),
                        candidates=len(candidates),
                        searches=metrics.search_count,
                    )
                    continue
                requires_browser = not _candidate_has_raw_content(candidate_raw := _raw_content(item))
                if requires_browser and browser_scrapes_for_query >= browser_scrape_limit and browser_scrape_limit > 0:
                    metrics.rejected_source_count += 1
                    reason = f"browser fallback budget exhausted for query ({browser_scrape_limit})"
                    metrics.failures.append(f"Skipped {url}: {reason}")
                    _emit_progress(
                        progress_callback,
                        "skipped source candidate",
                        branch_id=branch.id,
                        url=url,
                        reason=reason,
                        sources=len(sources),
                        candidates=len(candidates),
                        searches=metrics.search_count,
                    )
                    continue
                # When browser scraping is fully disabled (limit=0), fall through to httpx scraping
                # rather than skipping — DuckDuckGo results have no raw content but httpx can still
                # fetch many of them directly.
                if requires_browser and browser_scrape_limit == 0:
                    requires_browser = False
                canonical = _safe_canonical(url)
                if canonical in seen_urls:
                    continue
                seen_urls.add(canonical)
                candidate = SourceCandidate(
                    id=next_candidate_id,
                    branch_id=branch.id,
                    title=title,
                    url=url,
                    query=query,
                    snippet=snippet,
                    search_score=_float_or_none(item.get("score")),
                    raw_content=candidate_raw,
                    provenance="web",
                )
                next_candidate_id += 1
                candidates.append(candidate)
                metrics.candidate_count += 1
                if requires_browser:
                    browser_scrapes_for_query += 1
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
                    # Surface the rejection reason from the metrics failures log so
                    # the activity file records WHY a candidate was rejected.
                    rejection_reason = ""
                    if metrics.failures:
                        last_failure = metrics.failures[-1]
                        if url in last_failure:
                            # Strip the "Rejected <url>: " prefix for brevity
                            rejection_reason = last_failure.split(": ", 2)[-1].split(";")[0].strip()[:200]
                    _emit_progress(
                        progress_callback,
                        "rejected source candidate",
                        branch_id=branch.id,
                        url=url,
                        rejection_reason=rejection_reason,
                        sources=len(sources),
                        candidates=len(candidates),
                        searches=metrics.search_count,
                    )
                    continue
                sources.append(record.source)
                source_texts[record.source.id] = record.text
                _emit_progress(
                    progress_callback,
                    f"accepted source [{record.source.id}]",
                    branch_id=branch.id,
                    source_id=record.source.id,
                    sources=len(sources),
                    candidates=len(candidates),
                    searches=metrics.search_count,
                )
                # Mid-acquisition checkpoint: save progress every N sources so a
                # crash/resume doesn't redo the entire acquisition from scratch.
                if (
                    mid_checkpoint_callback is not None
                    and len(sources) % MID_ACQUISITION_CHECKPOINT_INTERVAL == 0
                ):
                    try:
                        mid_checkpoint_callback(candidates, sources)
                    except Exception:
                        pass  # never let checkpoint failure abort acquisition
                if sum(1 for source in sources if source.branch_id == branch.id) >= branch.min_sources:
                    break

    artifacts.write_jsonl("sources.jsonl", [source.to_dict() for source in sources])
    return AcquisitionResult(
        candidates=candidates,
        sources=sources,
        source_texts=source_texts,
        metrics=metrics,
    )


def _emit_progress(
    callback: Callable[[str, dict[str, Any]], None] | None,
    message: str,
    **data: Any,
) -> None:
    if callback is None:
        return
    callback(message, data)


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


def _search_raw(client: Any, query: str, settings: Any) -> dict[str, Any]:
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
    return response


def _duckduckgo_available() -> bool:
    return DDGS is not None


_DDG_QUERY_TIMEOUT_SECONDS = 30.0  # hard wall-clock per DDG query


class _DuckDuckGoTimeoutError(RuntimeError):
    """Raised when a DDG query exceeds the wall-clock budget."""


def _ddg_query_with_timeout(query: str, *, max_results: int) -> list[dict[str, Any]]:
    """Run DDGS().text() in a daemon thread with a hard wall-clock timeout.

    The duckduckgo-search library has no timeout parameter and can block on
    a hung TCP socket indefinitely. We wrap it the same way we wrap Playwright
    and synthesis: daemon thread + thread.join(timeout). On timeout the thread
    is abandoned (dies with the process); main thread raises and the caller
    treats it like any DDG failure.
    """
    if DDGS is None:
        return []
    result_holder: list[list[dict[str, Any]]] = []
    error_holder: list[BaseException] = []

    def _runner() -> None:
        rows: list[dict[str, Any]] = []
        try:
            with DDGS() as ddgs:
                for rank, row in enumerate(ddgs.text(query, max_results=max_results)):
                    url = str((row or {}).get("href") or "")
                    if not url:
                        continue
                    title = str((row or {}).get("title") or url)
                    snippet = str((row or {}).get("body") or (row or {}).get("snippet") or "")
                    if not snippet:
                        snippet = title
                    # Position-decay score so search_score is informative even
                    # without Tavily relevance data.
                    positional_score = round(0.9 * (0.95 ** rank), 4)
                    rows.append(
                        {
                            "url": url,
                            "title": title,
                            "content": snippet,
                            "score": positional_score,
                        }
                    )
            result_holder.append(rows)
        except BaseException as exc:  # noqa: BLE001
            error_holder.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=_DDG_QUERY_TIMEOUT_SECONDS)

    if thread.is_alive():
        raise _DuckDuckGoTimeoutError(
            f"DuckDuckGo query did not return within {_DDG_QUERY_TIMEOUT_SECONDS:.0f}s; "
            f"abandoning thread."
        )
    if error_holder:
        raise error_holder[0]
    return result_holder[0] if result_holder else []


def _search_with_duckduckgo(query: str, *, max_results: int) -> list[dict[str, Any]]:
    if DDGS is None:
        return []
    try:
        return _ddg_query_with_timeout(query, max_results=max_results)
    except _DuckDuckGoTimeoutError as exc:
        # Re-raise as a regular exception so the caller's failure counter
        # treats this as a hard failure (5 consecutive → DDG marked dead).
        # Otherwise we wait 10 empty results which is twice as slow.
        logging.getLogger(__name__).warning(
            "DuckDuckGo query timed out (%s); counting as hard failure.", exc,
        )
        raise RuntimeError(str(exc)) from exc
    except Exception:
        return []


def _branch_queries(branch: ResearchBranch, forced_terms: list[str], question: str) -> list[str]:
    if not forced_terms:
        return branch.queries
    terms = _dedupe([term for term in forced_terms if term])[:12]
    focus = " ".join(terms).strip()
    followups = list(branch.queries)
    if focus:
        followups.extend(
            [
                f"{question} {branch.title} {focus}",
                f"{branch.title} {focus} evidence review findings",
                f"{branch.title} {focus} mechanisms context limitations",
                f"{branch.objective} {focus} empirical evidence",
            ]
        )
    for term in terms:
        followups.extend(
            [
                f"{question} {term}",
                f"{branch.title} {term} evidence",
                f"{branch.objective} {term}",
            ]
        )
    for index in range(0, max(0, len(terms) - 1), 2):
        followups.append(f"{question} {terms[index]} {terms[index + 1]}")
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


def _candidate_has_raw_content(raw_content: str | None) -> bool:
    return bool(raw_content and len(raw_content.split()) >= 80)


def _blocked_source_reason(*, url: str, title: str, snippet: str, settings: Any) -> str:
    patterns = tuple(getattr(settings, "blocked_source_patterns", ()) or ())
    if not patterns:
        return ""
    haystack = "\n".join([url, title, snippet])
    for pattern in patterns:
        if re.search(pattern, haystack, flags=re.I):
            return f"matched blocked source pattern: {pattern}"
    return ""


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

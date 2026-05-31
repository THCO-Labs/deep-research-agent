from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from langchain.tools import BaseTool, tool
from langchain_experimental.utilities import PythonREPL
from tavily import TavilyClient

from deep_research.artifacts import RunArtifacts
from deep_research.models import Metrics
from deep_research.progress import ActivityLog, ProgressCallback, progress_line
from deep_research.scraper import PlaywrightScraper, ScrapeQualityError, ScrapeResult
from deep_research.settings import Settings
from deep_research.source_registry import SourceRegistry
from deep_research.urls import canonicalize_url
from deep_research.verifier import verify_report


class ResearchToolError(RuntimeError):
    """Raised when a research tool cannot complete its requested operation."""


@dataclass
class ToolContext:
    settings: Settings
    artifacts: RunArtifacts
    registry: SourceRegistry
    search_client: Any | None = None
    scraper: Any | None = None
    activity: ActivityLog | None = None
    on_progress: ProgressCallback | None = None
    repl: PythonREPL = field(default_factory=PythonREPL)
    metrics: Metrics = field(default_factory=Metrics)

    def __post_init__(self) -> None:
        if self.search_client is None:
            self.search_client = TavilyClient(api_key=self.settings.tavily_api_key)
        if self.scraper is None:
            self.scraper = PlaywrightScraper()

    def emit(self, stage: str, message: str) -> None:
        if self.activity:
            self.activity.emit(stage, message)
            return
        if self.on_progress:
            self.on_progress(progress_line(stage, message))


def build_tools(context: ToolContext) -> dict[str, BaseTool]:
    def search_candidates(query: str, max_results: int) -> tuple[str, list[dict[str, Any]]]:
        cleaned = query.strip()
        if not cleaned:
            raise ResearchToolError("web_search query cannot be empty.")
        bounded_results = max(1, max_results)
        context.metrics.search_count += 1
        context.emit("search", f"{cleaned} (top {bounded_results})")
        try:
            response = context.search_client.search(cleaned, max_results=bounded_results)
        except Exception as exc:
            raise ResearchToolError(f"Search failed for query {cleaned!r}: {exc}") from exc

        results = []
        for item in response.get("results", []):
            url = item.get("url")
            if not url:
                continue
            record = context.registry.upsert_search_result(
                url=url,
                title=item.get("title") or url,
                query=cleaned,
                snippet=item.get("content"),
            )
            results.append(
                {
                    "source_id": record.id,
                    "title": record.title,
                    "url": record.url,
                    "canonical_url": record.canonical_url,
                    "score": item.get("score"),
                    "needs_scrape": True,
                }
            )
        source_ids = ", ".join(f"[{item['source_id']}]" for item in results) or "none"
        context.emit("search", f"registered {len(results)} source candidate(s): {source_ids}")
        return cleaned, results

    def scrape_candidate(url: str) -> dict[str, Any]:
        cleaned = _resolve_scrape_target(url, context)
        if not cleaned:
            raise ResearchToolError("deep_scrape URL cannot be empty.")
        context.metrics.scrape_count += 1
        context.emit("scrape", cleaned)
        try:
            result: ScrapeResult = context.scraper.fetch(cleaned)
        except ScrapeQualityError as exc:
            context.emit("scrape", f"rejected unusable source: {cleaned}")
            return _unusable_source_payload(cleaned, context, str(exc))
        except Exception as exc:
            context.emit("scrape", f"failed unusable source: {cleaned}")
            return _unusable_source_payload(cleaned, context, f"Scrape failed: {exc}")

        markdown = result.markdown[: context.settings.scrape_char_limit]
        excerpt = markdown[: context.settings.tool_excerpt_char_limit]
        record = context.registry.record_scrape(
            url=result.url,
            title=result.title,
            markdown=markdown,
            extraction_method=result.extraction_method,
        )
        payload = {
            "source_id": record.id,
            "title": record.title,
            "url": record.url,
            "canonical_url": record.canonical_url,
            "content_path": record.content_path,
            "excerpt": excerpt,
            "saved_chars": len(markdown),
            "source_usable": True,
        }
        context.emit(
            "scrape",
            f"source [{record.id}] {record.title} ({len(markdown):,} chars)",
        )
        return payload

    @tool
    def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the public web and register source candidates.

        This returns candidates only. Call deep_scrape or collect_sources before
        relying on sources or citing source IDs.
        """
        bounded_results = max(1, min(max_results, context.settings.max_sources))
        cleaned, results = search_candidates(query, bounded_results)
        return {"query": cleaned, "results": results}

    @tool
    def deep_scrape(url: str) -> dict[str, Any]:
        """Render a public URL, register usable markdown, or return an unusable-source result."""
        return scrape_candidate(url)

    @tool
    def collect_sources(query: str, target_count: int = 3, max_results: int = 0) -> dict[str, Any]:
        """Search and scrape candidates until enough usable, citeable sources are collected.

        Prefer this over manually pairing web_search and deep_scrape for normal
        research. It skips failed, blocked, and low-quality pages and returns
        only scraped sources as usable.
        """
        target = max(1, min(target_count, context.settings.max_sources))
        candidate_limit = max_results if max_results > 0 else max(target * 3, context.settings.max_sources)
        candidate_limit = max(target, min(candidate_limit, 10))
        cleaned, candidates = search_candidates(query, candidate_limit)
        usable_sources = []
        unusable_sources = []
        seen_urls: set[str] = set()

        for candidate in candidates:
            if len(usable_sources) >= target:
                break
            url = str(candidate.get("url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            scrape_result = scrape_candidate(url)
            if scrape_result.get("source_usable") is True:
                usable_sources.append(scrape_result)
            else:
                unusable_sources.append(scrape_result)

        context.emit(
            "collect",
            f"gathered {len(usable_sources)}/{target} usable source(s), skipped {len(unusable_sources)}",
        )
        return {
            "query": cleaned,
            "target_count": target,
            "candidate_count": len(candidates),
            "usable_count": len(usable_sources),
            "unusable_count": len(unusable_sources),
            "usable_sources": usable_sources,
            "unusable_sources": unusable_sources,
            "needs_more_sources": len(usable_sources) < target,
            "instruction": "Use only usable_sources for citations. If needs_more_sources is true, run another query.",
        }

    @tool
    def write_file(file_path: str, content: str) -> str:
        """Write a UTF-8 text file inside the current research run directory."""
        context.metrics.write_count += 1
        path = context.artifacts.write_text(file_path, content)
        relative = path.relative_to(context.artifacts.run_dir)
        context.emit("write", str(relative))
        return f"Wrote {relative}"

    @tool
    def read_file(file_path: str) -> str:
        """Read a bounded UTF-8 text preview from inside the current research run directory."""
        context.metrics.read_count += 1
        path = context.artifacts.resolve_path(file_path)
        if not path.exists():
            context.emit("read", f"missing {file_path}")
            return f"ERROR: file not found: {file_path}"
        content = path.read_text(encoding="utf-8")
        preview = _bounded_preview(content, context.settings.tool_excerpt_char_limit)
        if len(preview) < len(content):
            context.emit("read", f"{file_path} (preview {len(preview):,}/{len(content):,} chars)")
            return (
                preview
                + "\n\n"
                + f"[TRUNCATED: returned {len(preview):,} of {len(content):,} chars. "
                + "The complete file remains in the run directory for verification.]"
            )
        context.emit("read", file_path)
        return preview

    @tool
    def verify_report_file(file_path: str = "report.md") -> dict[str, Any]:
        """Run deterministic citation and source-support checks on a report file."""
        context.metrics.verification_rounds += 1
        context.emit("verify", f"checking {file_path}")
        path = context.artifacts.resolve_path(file_path)
        if not path.exists():
            result = {
                "valid": False,
                "citation_validity_score": 0.0,
                "source_support_score": 0.0,
                "missing_sources": [f"Report file not found: {file_path}"],
                "unused_sources": [record.id for record in context.registry.records],
                "unsupported_claims": [],
                "weakly_supported_claims": [],
                "support_checks": [],
                "source_list_errors": [],
                "unscraped_sources": [record.id for record in context.registry.records if not record.content_path],
                "cited_source_ids": [],
                "total_citations": 0,
                "verification_rounds": context.metrics.verification_rounds,
            }
            context.artifacts.write_json("verification.json", result)
            context.emit("verify", "failed: report file missing")
            return result
        report = path.read_text(encoding="utf-8")
        result = verify_report(
            report,
            context.registry.records,
            verification_rounds=context.metrics.verification_rounds,
            source_loader=lambda record: context.artifacts.read_text(record.content_path or ""),
        )
        context.artifacts.write_json("verification.json", result.to_dict())
        status = "passed" if result.valid else "failed"
        context.emit(
            "verify",
            f"{status}: score {result.citation_validity_score:.2f}, "
            f"{len(result.unsupported_claims)} uncited paragraph(s), "
            f"{len(result.weakly_supported_claims)} weakly supported claim(s)",
        )
        return result.to_dict()

    @tool
    def python_repl(code: str) -> str:
        """Execute Python code for numeric/data analysis and return stdout/stderr."""
        if not code.strip():
            raise ResearchToolError("python_repl code cannot be empty.")
        context.metrics.python_exec_count += 1
        context.emit("analysis", "running Python code")
        return context.repl.run(code)

    return {
        "web_search": web_search,
        "deep_scrape": deep_scrape,
        "collect_sources": collect_sources,
        "write_file": write_file,
        "read_file": read_file,
        "verify_report_file": verify_report_file,
        "python_repl": python_repl,
    }


def _resolve_scrape_target(url: str, context: ToolContext) -> str:
    cleaned = url.strip()
    if not cleaned:
        return cleaned
    source_id_text = cleaned.strip("[]# sourceSOURCE")
    if source_id_text.isdigit():
        source_id = int(source_id_text)
        for record in context.registry.records:
            if record.id == source_id:
                context.emit("scrape", f"resolved source [{source_id}] to {record.url}")
                return record.url

    try:
        canonical = canonicalize_url(cleaned)
    except ValueError:
        canonical = ""
    if canonical:
        for record in context.registry.records:
            if record.canonical_url == canonical:
                return record.url

        host = urlsplit(canonical).hostname
        same_host = [
            record
            for record in context.registry.records
            if urlsplit(record.canonical_url).hostname == host
        ]
        if len(same_host) == 1:
            context.emit("scrape", f"corrected URL to registered source [{same_host[0].id}]")
            return same_host[0].url

    unscraped = [record for record in context.registry.records if not record.content_path]
    if len(unscraped) == 1:
        context.emit("scrape", f"using only unscraped source candidate [{unscraped[0].id}]")
        return unscraped[0].url
    return cleaned


def _unusable_source_payload(url: str, context: ToolContext, error: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "url": url,
        "error": error,
        "source_usable": False,
        "needs_alternate_source": True,
        "instruction": "Do not cite this source. Search for or scrape an alternate source.",
    }
    source_id = _source_id_for_url(url, context)
    if source_id is not None:
        payload["source_id"] = source_id
    return payload


def _source_id_for_url(url: str, context: ToolContext) -> int | None:
    try:
        canonical = canonicalize_url(url)
    except ValueError:
        return None
    for record in context.registry.records:
        if record.canonical_url == canonical or record.url == url:
            return record.id
    return None


def _bounded_preview(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    return content[:limit].rstrip()

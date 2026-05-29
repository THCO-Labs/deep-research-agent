from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from langchain.tools import BaseTool, tool
from langchain_experimental.utilities import PythonREPL
from tavily import TavilyClient

from deep_research.artifacts import RunArtifacts
from deep_research.models import Metrics
from deep_research.progress import ProgressCallback, progress_line
from deep_research.scraper import PlaywrightScraper, ScrapeResult
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
    on_progress: ProgressCallback | None = None
    repl: PythonREPL = field(default_factory=PythonREPL)
    metrics: Metrics = field(default_factory=Metrics)

    def __post_init__(self) -> None:
        if self.search_client is None:
            self.search_client = TavilyClient(api_key=self.settings.tavily_api_key)
        if self.scraper is None:
            self.scraper = PlaywrightScraper()

    def emit(self, stage: str, message: str) -> None:
        if self.on_progress:
            self.on_progress(progress_line(stage, message))


def build_tools(context: ToolContext) -> dict[str, BaseTool]:
    @tool
    def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
        """Search the public web and register source candidates.

        This returns candidates only. Call deep_scrape on a result URL before
        relying on it or citing its source_id.
        """
        cleaned = query.strip()
        if not cleaned:
            raise ResearchToolError("web_search query cannot be empty.")
        bounded_results = max(1, min(max_results, context.settings.max_sources))
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
        return {"query": cleaned, "results": results}

    @tool
    def deep_scrape(url: str) -> dict[str, Any]:
        """Render a public URL with Playwright, convert it to markdown, and register it."""
        cleaned = _resolve_scrape_target(url, context)
        if not cleaned:
            raise ResearchToolError("deep_scrape URL cannot be empty.")
        context.metrics.scrape_count += 1
        context.emit("scrape", cleaned)
        try:
            result: ScrapeResult = context.scraper.fetch(cleaned)
        except Exception as exc:
            raise ResearchToolError(f"Scrape failed for URL {cleaned!r}: {exc}") from exc

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
        }
        context.emit(
            "scrape",
            f"source [{record.id}] {record.title} ({len(markdown):,} chars)",
        )
        return payload

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
        """Read a UTF-8 text file from inside the current research run directory."""
        context.metrics.read_count += 1
        path = context.artifacts.resolve_path(file_path)
        if not path.exists():
            context.emit("read", f"missing {file_path}")
            return f"ERROR: file not found: {file_path}"
        context.emit("read", file_path)
        return path.read_text(encoding="utf-8")

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
                "missing_sources": [f"Report file not found: {file_path}"],
                "unused_sources": [record.id for record in context.registry.records],
                "unsupported_claims": [],
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
        )
        context.artifacts.write_json("verification.json", result.to_dict())
        status = "passed" if result.valid else "failed"
        context.emit(
            "verify",
            f"{status}: score {result.citation_validity_score:.2f}, "
            f"{len(result.unsupported_claims)} unsupported paragraph(s)",
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

    unsraped = [record for record in context.registry.records if not record.content_path]
    if len(unsraped) == 1:
        context.emit("scrape", f"using only unsraped source candidate [{unsraped[0].id}]")
        return unsraped[0].url
    return cleaned

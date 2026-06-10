from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from io import BytesIO
import json
import os
import re
import subprocess
import sys
from time import sleep
import threading
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup, FeatureNotFound, Tag
from bs4.builder import builder_registry
from markdownify import markdownify
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

RETRY_HTTP_STATUS = {
    HTTPStatus.REQUEST_TIMEOUT,
    HTTPStatus.TOO_EARLY,
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
}
ACCESS_CONTROL_TEXT_RE = re.compile(
    r"\b(?:access\s+denied|forbidden|request\s+blocked|temporarily\s+blocked|security\s+verification|"
    r"verify\s+(?:you|that\s+you)\s+are\s+human|captcha|javascript\s+and\s+cookies|"
    r"checking\s+if\s+the\s+site\s+connection\s+is\s+secure|"
    r"subscription\s+required|paywall)\b",
    flags=re.I,
)
NON_CONTENT_TAG_RE = re.compile(r"^(?:script|style|noscript|svg|canvas|iframe|nav|header|footer|aside|form|button)$", re.I)
CONTENT_TAG_RE = re.compile(r"^(?:article|main|section|div|body)$", re.I)
HEADING_KEY_RE = re.compile(r"(?:^|[_-])(?:headline|title|name|label)(?:$|[_-])", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]+")
URL_RE = re.compile(r"https?://\S+", re.I)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]{0,160}]\((?:https?://)?[^)]+\)", re.I)
SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?$")


@dataclass(frozen=True)
class ScrapeResult:
    url: str
    title: str
    markdown: str
    extraction_method: str = "playwright"


class ScrapeQualityError(RuntimeError):
    """Raised when a fetched page is not usable research content."""


@dataclass(frozen=True)
class ExtractionCandidate:
    method: str
    markdown: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@lru_cache(maxsize=1)
def _available_bs4_parsers() -> tuple[str, ...]:
    parser_names: list[str] = []
    for builder in builder_registry.builders:
        parser_name = str(getattr(builder, "NAME", "") or "").strip()
        if parser_name and parser_name not in parser_names:
            parser_names.append(parser_name)
    return tuple(parser_names) or ("html.parser",)


@lru_cache(maxsize=1)
def _user_agent_pool() -> tuple[str, ...]:
    try:
        from fake_useragent import UserAgent

        user_agent = UserAgent()
        generated = [str(user_agent.random) for _ in range(6)]
        unique = tuple(dict.fromkeys(value for value in generated if value and len(value) > 20))
        if unique:
            return unique
    except Exception:
        pass
    return ("Mozilla/5.0 AppleWebKit/537.36 Chrome Safari",)


def _request_headers(attempt: int) -> dict[str, str]:
    user_agents = _user_agent_pool()
    return {
        "User-Agent": user_agents[attempt % len(user_agents)],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,application/pdf;q=0.7,*/*;q=0.6",
        "Accept-Language": "en-US,en;q=0.9",
    }


def _kill_chrome_process(pid_holder: list[int]) -> None:
    """Force-kill a Chromium process and its entire child tree.

    Called after a browser-scrape hard timeout to ensure no orphan Chrome
    processes accumulate.  Uses 'taskkill /T /F' on Windows (kills the whole
    process tree) and SIGKILL on POSIX.  Silently ignores all errors so a
    failed kill never blocks the caller.
    """
    for pid in pid_holder:
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    timeout=5,
                )
            else:
                os.kill(pid, 9)
        except Exception:
            pass


class PlaywrightScraper:
    def __init__(self, *, timeout_ms: int = 30_000, retries: int = 2):
        self.timeout_ms = timeout_ms
        self.retries = max(1, retries)

    def fetch(self, url: str) -> ScrapeResult:
        errors: list[str] = []
        try:
            return self._fetch_with_httpx(url)
        except Exception as exc:
            errors.append(f"httpx: {exc}")
        try:
            return self._fetch_with_playwright(url)
        except Exception as exc:
            errors.append(f"playwright: {exc}")
        raise ScrapeQualityError("; ".join(errors) or f"Could not fetch usable content from {url}")

    def _fetch_with_playwright(self, url: str) -> ScrapeResult:
        """Run the browser fetch in a daemon thread so we can enforce a hard wall-clock
        timeout independent of Playwright's internal timeouts.

        Background: sync_playwright blocks the calling thread entirely.  page.evaluate()
        has no reliable per-call timeout in the sync API, so a hung JS Promise (infinite
        scroll, service workers, etc.) can freeze the process indefinitely.  Running in a
        daemon thread lets us join() with a timeout and raise instead of hanging forever.

        Chrome process cleanup: we track the Chromium subprocess PID and force-kill it
        (plus its entire process tree) on timeout, so no orphan Chrome processes are left
        behind even when the Playwright context never exits cleanly.
        """
        hard_timeout_s = (self.timeout_ms + 15_000) / 1000  # page timeout + 15 s grace

        result_holder: list[ScrapeResult] = []
        error_holder: list[BaseException] = []
        browser_holder: list[Any] = []
        chrome_pid_holder: list[int] = []

        def _run() -> None:
            try:
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True)
                    browser_holder.append(browser)
                    # Record the Chromium PID so we can force-kill it on timeout.
                    try:
                        pid = browser._impl_obj._channel._connection._transport._proc.pid
                        if pid:
                            chrome_pid_holder.append(pid)
                    except Exception:
                        pass
                    try:
                        page = browser.new_page()
                        response = page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                        if response is not None and response.status >= 400:
                            raise RuntimeError(f"HTTP {response.status} while fetching {url}")
                        if response is not None and _is_pdf_response(response.headers, response.url):
                            markdown = _extract_pdf_text(response.body())
                            result_holder.append(ScrapeResult(
                                url=response.url,
                                title=_title_from_url(response.url),
                                markdown=markdown,
                                extraction_method="playwright_pdf",
                            ))
                            return
                        try:
                            page.wait_for_load_state("networkidle", timeout=5_000)
                        except PlaywrightTimeoutError:
                            pass
                        _settle_dynamic_page(page)
                        title = page.title()
                        html = page.content()
                        refresh_url = _meta_refresh_url(html, page.url)
                        if refresh_url and refresh_url != page.url:
                            page.goto(refresh_url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                            try:
                                page.wait_for_load_state("networkidle", timeout=5_000)
                            except PlaywrightTimeoutError:
                                pass
                            _settle_dynamic_page(page)
                            title = page.title()
                            html = page.content()
                        result_holder.append(ScrapeResult(
                            url=page.url,
                            title=title,
                            markdown=html_to_markdown(html, title=title),
                        ))
                    finally:
                        try:
                            browser.close()
                        except Exception:
                            pass
            except Exception as exc:
                error_holder.append(exc)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=hard_timeout_s)

        if thread.is_alive():
            # Thread is still hung. Force-kill the Chrome process tree so no orphan
            # processes are left behind, then raise a timeout error.
            _kill_chrome_process(chrome_pid_holder)
            if browser_holder:
                try:
                    browser_holder[0].close()
                except Exception:
                    pass
            raise ScrapeQualityError(
                f"Browser scrape hard timeout ({hard_timeout_s:.0f}s) exceeded for {url}"
            )

        if error_holder:
            raise error_holder[0]
        if result_holder:
            return result_holder[0]
        raise ScrapeQualityError(f"Browser scrape returned no result for {url}")

    def _fetch_with_httpx(self, url: str) -> ScrapeResult:
        response = self._httpx_get(url)
        response.raise_for_status()
        if _is_pdf_response(response.headers, str(response.url)):
            return ScrapeResult(
                url=str(response.url),
                title=_title_from_url(str(response.url)),
                markdown=_extract_pdf_text(response.content),
                extraction_method="httpx_pdf",
            )
        content_type = response.headers.get("content-type", "").lower()
        if "text/plain" in content_type:
            markdown = _clean_markdown(response.text)
            if _is_low_content(markdown):
                raise ScrapeQualityError("Fetched text/plain response did not contain enough research text.")
            return ScrapeResult(
                url=str(response.url),
                title=_title_from_url(str(response.url)),
                markdown=markdown,
                extraction_method="httpx_text",
            )
        refresh_url = _meta_refresh_url(response.text, str(response.url))
        if refresh_url and refresh_url != str(response.url):
            response = self._httpx_get(refresh_url)
            response.raise_for_status()
        title = _extract_title(response.text) or url
        return ScrapeResult(
            url=str(response.url),
            title=title,
            markdown=html_to_markdown(response.text, title=title),
            extraction_method="httpx",
        )

    def _httpx_get(self, url: str) -> httpx.Response:
        # Separate connect/read/write/pool timeouts so a hostile server that drips
        # bytes forever (or a stalled TCP read) can't pin the worker thread.
        total = self.timeout_ms / 1000
        timeout = httpx.Timeout(
            connect=min(10.0, total),
            read=total,
            write=total,
            pool=10.0,
        )
        last_error: Exception | None = None
        attempts = self.retries
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            for attempt in range(attempts):
                headers = _request_headers(attempt)
                try:
                    response = client.get(url, headers=headers)
                    if response.status_code not in {status.value for status in RETRY_HTTP_STATUS} or attempt == attempts - 1:
                        return response
                    last_error = RuntimeError(f"HTTP {response.status_code} while fetching {url}")
                except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                    last_error = exc
                    if attempt == attempts - 1:
                        raise
                sleep(min(0.2 * (attempt + 1), 1.0))
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"HTTP fetch failed for {url}")


def html_to_markdown(html: str, *, title: str = "") -> str:
    soup = _parse_with_available_parser(html)
    if _is_challenge_page(soup, title=title):
        raise ScrapeQualityError("Fetched page appears to be a bot-protection or JavaScript challenge page.")

    candidates = extract_markdown_candidates(html, title=title)
    if candidates:
        return candidates[0].markdown

    raise ScrapeQualityError("Fetched page did not contain enough article text after parser fallback extraction.")


def extract_markdown_candidates(html: str, *, title: str = "") -> list[ExtractionCandidate]:
    candidates: list[ExtractionCandidate] = []
    for method, markdown in _iter_extraction_attempts(html, title=title):
        cleaned = _clean_markdown(markdown)
        if not cleaned or _is_low_content(cleaned):
            continue
        candidates.append(ExtractionCandidate(method=method, markdown=cleaned, score=_markdown_score(cleaned)))
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


def _iter_extraction_attempts(html: str, *, title: str) -> list[tuple[str, str]]:
    attempts: list[tuple[str, str]] = []
    sanitized_html = _html_without_unusable_nodes(html)
    attempts.append(("structured_json", _extract_structured_article_text(html)))
    attempts.extend(_extract_with_trafilatura_variants(sanitized_html))
    attempts.append(("readability", _extract_with_readability(sanitized_html)))
    attempts.append(("newspaper4k", _extract_with_newspaper(sanitized_html)))
    attempts.append(("goose3", _extract_with_goose(sanitized_html)))
    attempts.append(("justext", _extract_with_justext(sanitized_html)))
    attempts.append(("selectolax", _extract_with_selectolax(sanitized_html)))
    attempts.append(("inscriptis", _extract_with_inscriptis(sanitized_html)))
    attempts.append(("beautifulsoup", _extract_with_beautifulsoup_parsers(sanitized_html, title=title)))
    attempts.append(("lxml_text", _extract_with_lxml_text(sanitized_html)))
    return attempts


def _parse_with_available_parser(html: str) -> BeautifulSoup:
    for parser in _available_bs4_parsers():
        try:
            return BeautifulSoup(html, parser)
        except FeatureNotFound:
            continue
    return BeautifulSoup(html, "html.parser")


def _extract_with_beautifulsoup_parsers(html: str, *, title: str = "") -> str:
    best_markdown = ""
    best_score = 0.0
    for parser in _available_bs4_parsers():
        try:
            soup = BeautifulSoup(html, parser)
        except FeatureNotFound:
            continue
        if _is_challenge_page(soup, title=title):
            raise ScrapeQualityError("Fetched page appears to be a bot-protection or JavaScript challenge page.")
        markdown = _extract_with_soup(soup)
        score = _markdown_score(markdown)
        if score > best_score:
            best_markdown = markdown
            best_score = score
    return best_markdown


def _extract_with_soup(soup: BeautifulSoup) -> str:
    _remove_unusable_nodes(soup)
    candidate = _best_content_node(soup)
    markdown = markdownify(str(candidate), heading_style="ATX")
    return _clean_markdown(markdown)


def _html_without_unusable_nodes(html: str) -> str:
    soup = _parse_with_available_parser(html)
    _remove_unusable_nodes(soup)
    return str(soup)


def _extract_structured_article_text(html: str) -> str:
    candidates: list[str] = []
    for parser in _available_bs4_parsers():
        try:
            soup = BeautifulSoup(html, parser)
        except FeatureNotFound:
            continue
        for script in soup.select("script[type*='ld+json'], script[type='application/json']"):
            raw = script.string or script.get_text(" ", strip=True)
            payload = _load_json_payload(raw)
            if payload is None:
                continue
            for item in _iter_json_objects(payload):
                candidates.extend(
                    text for text in _long_text_values_from_json(item, heading=_heading_from_json(item)) if text and not _is_low_content(text)
                )
    if not candidates:
        return ""
    return max(candidates, key=_markdown_score)


def _load_json_payload(raw: str) -> Any | None:
    cleaned = raw.strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"^<!--|-->$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _iter_json_objects(value: Any) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    if isinstance(value, dict):
        objects.append(value)
        for nested in value.values():
            objects.extend(_iter_json_objects(nested))
    elif isinstance(value, list):
        for item in value:
            objects.extend(_iter_json_objects(item))
    return objects


def _long_text_values_from_json(value: Any, *, heading: str = "") -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for nested in value.values():
            values.extend(_long_text_values_from_json(nested, heading=heading))
    elif isinstance(value, list):
        for item in value:
            values.extend(_long_text_values_from_json(item, heading=heading))
    elif isinstance(value, str):
        text = _json_value_to_text(value)
        if len(text.split()) >= 40:
            if heading and heading.lower() not in text[:200].lower():
                values.append(_clean_markdown(f"# {heading}\n\n{text}"))
            else:
                values.append(_clean_markdown(text))
    return values


def _heading_from_json(value: Any) -> str:
    if isinstance(value, dict):
        for key, nested in value.items():
            if HEADING_KEY_RE.search(str(key)):
                text = _json_value_to_text(nested)
                if 2 <= len(text.split()) <= 20:
                    return text
        for nested in value.values():
            text = _heading_from_json(nested)
            if text:
                return text
    if isinstance(value, list):
        for item in value:
            text = _heading_from_json(item)
            if text:
                return text
    return ""


def _json_value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        if "<" in value and ">" in value:
            value = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, list):
        return " ".join(_json_value_to_text(item) for item in value if item is not None).strip()
    if isinstance(value, dict):
        preferred = value.get("text") or value.get("content") or value.get("html") or value.get("value")
        return _json_value_to_text(preferred)
    return str(value).strip()


def _extract_with_trafilatura_variants(html: str) -> list[tuple[str, str]]:
    try:
        import trafilatura
    except ImportError:
        return []
    variants = [
        ("trafilatura_balanced", {"favor_precision": False, "favor_recall": False}),
        ("trafilatura_precision", {"favor_precision": True, "favor_recall": False}),
        ("trafilatura_recall", {"favor_precision": False, "favor_recall": True}),
    ]
    results: list[tuple[str, str]] = []
    for method, flags in variants:
        try:
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                include_links=False,
                deduplicate=True,
                output_format="markdown",
                **flags,
            )
        except Exception:
            extracted = ""
        results.append((method, _clean_markdown(extracted or "")))
    return results


def _extract_with_readability(html: str) -> str:
    try:
        from readability import Document
    except ImportError:
        return ""
    try:
        document = Document(html)
        summary = document.summary(html_partial=True)
    except Exception:
        return ""
    return _clean_markdown(markdownify(summary, heading_style="ATX"))


def _extract_with_newspaper(html: str) -> str:
    try:
        from newspaper import Article
    except ImportError:
        return ""
    article = Article(url="")
    try:
        article.set_html(html)
        article.parse()
    except Exception:
        return ""
    parts = []
    if article.title:
        parts.append(f"# {article.title}")
    if article.text:
        parts.append(article.text)
    return _clean_markdown("\n\n".join(parts))


def _extract_with_goose(html: str) -> str:
    try:
        from goose3 import Goose
    except ImportError:
        return ""
    goose = Goose({"enable_image_fetching": False})
    try:
        article = goose.extract(raw_html=html)
    except Exception:
        return ""
    finally:
        try:
            goose.close()
        except Exception:
            pass
    parts = []
    if getattr(article, "title", ""):
        parts.append(f"# {article.title}")
    if getattr(article, "cleaned_text", ""):
        parts.append(article.cleaned_text)
    return _clean_markdown("\n\n".join(parts))


def _extract_with_justext(html: str) -> str:
    try:
        import justext
    except ImportError:
        return ""
    try:
        stoplist = justext.get_stoplist("English")
        paragraphs = justext.justext(html, stoplist)
    except Exception:
        return ""
    text = "\n\n".join(paragraph.text for paragraph in paragraphs if not paragraph.is_boilerplate)
    return _clean_markdown(text)


def _extract_with_selectolax(html: str) -> str:
    try:
        from selectolax.lexbor import LexborHTMLParser
    except ImportError:
        try:
            from selectolax.parser import HTMLParser as LexborHTMLParser
        except ImportError:
            return ""
    try:
        tree = LexborHTMLParser(html)
    except Exception:
        return ""
    for node in tree.css("body *"):
        if NON_CONTENT_TAG_RE.match(str(getattr(node, "tag", "") or "")):
            node.remove()
    best_text = ""
    best_score = 0.0
    for node in tree.css("body, body *"):
        if not CONTENT_TAG_RE.match(str(getattr(node, "tag", "") or "")):
            continue
        text = node.text(separator="\n", strip=True)
        if len(text) < 160:
            continue
        link_text_len = sum(len(link.text(separator=" ", strip=True)) for link in node.css("a"))
        if link_text_len / max(len(text), 1) > 0.45:
            continue
        score = _markdown_score(text)
        if score > best_score:
            best_text = text
            best_score = score
    return _clean_markdown(best_text)


def _extract_with_inscriptis(html: str) -> str:
    try:
        from inscriptis import get_text
    except ImportError:
        return ""
    try:
        return _clean_markdown(get_text(html))
    except Exception:
        return ""


def _extract_with_lxml_text(html: str) -> str:
    try:
        from lxml import html as lxml_html
    except ImportError:
        return ""
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return ""
    for element in list(tree.iter()):
        tag = str(getattr(element, "tag", "") or "")
        if not NON_CONTENT_TAG_RE.match(tag):
            continue
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)
    return _clean_markdown(tree.text_content())


def _extract_title(html: str) -> str:
    marker = "<title"
    start = html.lower().find(marker)
    if start == -1:
        return ""
    close = html.find(">", start)
    end = html.lower().find("</title>", close)
    if close == -1 or end == -1:
        return ""
    return html[close + 1 : end].strip()


def _title_from_url(url: str) -> str:
    path = url.split("?", 1)[0].rstrip("/").rsplit("/", 1)[-1]
    return path.replace("-", " ").replace("_", " ").strip() or url


def _is_pdf_response(headers: Any, url: str) -> bool:
    content_type = str(headers.get("content-type", "")).lower() if headers else ""
    return "application/pdf" in content_type or url.lower().split("?", 1)[0].endswith(".pdf")


def _extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ScrapeQualityError("PDF response received, but pypdf is not installed.") from exc
    try:
        reader = PdfReader(BytesIO(content))
        text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
    except Exception as exc:
        raise ScrapeQualityError(f"PDF text extraction failed: {exc}") from exc
    cleaned = _clean_markdown(text)
    if _is_low_content(cleaned):
        raise ScrapeQualityError("PDF response did not contain enough extractable text.")
    return cleaned


def _meta_refresh_url(html: str, base_url: str) -> str:
    soup = _parse_with_available_parser(html)
    meta = soup.find("meta", attrs={"http-equiv": re.compile("^refresh$", re.I)})
    if not isinstance(meta, Tag):
        return ""
    content = str(meta.get("content") or "")
    match = re.search(r"url\s*=\s*['\"]?([^;'\"\s]+)", content, flags=re.I)
    if not match:
        return ""
    return urljoin(base_url, match.group(1).strip())


def _settle_dynamic_page(page: Any) -> None:
    try:
        page.wait_for_selector("article, main, [role='main'], body", timeout=3_000)
    except Exception:
        pass
    # Scroll to trigger lazy-loaded content.  The evaluate() call uses a synchronous
    # scroll pattern (no async/await) to avoid the hanging-Promise failure mode.
    try:
        page.evaluate(
            """
            () => {
              const h = document.body.scrollHeight;
              window.scrollTo(0, Math.floor(h * 0.35));
              window.scrollTo(0, Math.floor(h * 0.70));
              window.scrollTo(0, h);
              window.scrollTo(0, 0);
            }
            """,
            timeout=3_000,
        )
    except Exception:
        pass
    _click_expand_controls(page)


def _click_expand_controls(page: Any) -> None:
    try:
        clicked = page.evaluate(
            """
            () => {
              const textPattern = /\\b(read|show|load|continue|view|expand)\\b.{0,24}\\b(more|full|article|story|content|report)?\\b/i;
              const nodes = Array.from(document.querySelectorAll('button, a, summary, [role="button"]'));
              let clicks = 0;
              for (const node of nodes) {
                const text = (node.innerText || node.textContent || node.getAttribute('aria-label') || '').trim();
                if (!textPattern.test(text)) continue;
                const rect = node.getBoundingClientRect();
                const visible = rect.width > 0 && rect.height > 0 && getComputedStyle(node).visibility !== 'hidden';
                if (!visible) continue;
                node.click();
                clicks += 1;
                if (clicks >= 4) break;
              }
              return clicks;
            }
            """,
            timeout=3_000,
        )
        if clicked:
            page.wait_for_timeout(500)
    except Exception:
        return


def _is_challenge_page(soup: BeautifulSoup, *, title: str = "") -> bool:
    text = " ".join(soup.get_text(" ").lower().split())
    title_text = title.strip().lower()
    content_words = WORD_RE.findall(text)
    paragraph_count = len([p for p in soup.find_all("p") if len(p.get_text(" ", strip=True).split()) >= 12])
    has_access_form = bool(
        soup.find("input", attrs={"type": re.compile(r"password", re.I)})
        or (soup.find("form") and soup.find("iframe") and soup.find("input"))
        or (soup.find("form") and len(content_words) < 80 and paragraph_count == 0)
    )
    challenge_score = 0
    challenge_score += 2 if ACCESS_CONTROL_TEXT_RE.search(f"{title_text} {text}") else 0
    challenge_score += 2 if has_access_form else 0
    challenge_score += 1 if len(content_words) < 80 else 0
    challenge_score += 1 if paragraph_count == 0 else 0
    return challenge_score >= 3


def _remove_unusable_nodes(soup: BeautifulSoup) -> None:
    for node in list(soup.find_all(True)):
        if node.parent is None or getattr(node, "attrs", None) is None:
            continue
        if node.name and NON_CONTENT_TAG_RE.match(node.name):
            node.decompose()
            continue
        text = " ".join(node.get_text(" ").split())
        if text:
            link_text_len = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
            paragraph_count = len(node.find_all("p"))
            if paragraph_count == 0 and link_text_len / max(len(text), 1) > 0.5:
                node.decompose()
                continue
        if _is_structural_noise_node(node):
            node.decompose()


def _best_content_node(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    candidates = _generic_content_candidates(soup)
    if not candidates and soup.body is not None:
        candidates.append(soup.body)

    best: Tag | None = None
    best_score = 0.0
    for candidate in candidates:
        score = _content_score(candidate)
        if score > best_score:
            best = candidate
            best_score = score
    return best or soup


def _generic_content_candidates(soup: BeautifulSoup) -> list[Tag]:
    candidates: list[Tag] = []
    for node in soup.find_all(True):
        if not isinstance(node, Tag) or not node.name or not CONTENT_TAG_RE.match(node.name):
            continue
        text = " ".join(node.get_text(" ").split())
        if len(text) < 160:
            continue
        link_text_len = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
        link_density = link_text_len / max(len(text), 1)
        if link_density > 0.45:
            continue
        candidates.append(node)
    return candidates


def _content_score(node: Tag) -> float:
    text = " ".join(node.get_text(" ").split())
    if not text:
        return 0.0
    paragraph_count = len([p for p in node.find_all("p") if len(p.get_text(" ").strip()) > 40])
    heading_count = len(node.find_all(re.compile("^h[1-3]$")))
    link_text_len = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
    link_density = link_text_len / max(len(text), 1)
    return len(text) + (paragraph_count * 250) + (heading_count * 80) - (link_density * 800)


def _markdown_score(markdown: str) -> float:
    words = WORD_RE.findall(markdown)
    if not words:
        return 0.0
    heading_bonus = len(re.findall(r"(?m)^#{1,3}\s+", markdown)) * 75
    paragraph_bonus = len([part for part in re.split(r"\n\s*\n", markdown) if len(part.split()) >= 20]) * 120
    link_penalty = len(re.findall(r"\[[^\]]+]\([^)]+\)", markdown)) * 15
    return len(words) + heading_bonus + paragraph_bonus - link_penalty


def _clean_markdown(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        stripped = line.rstrip()
        if _is_noise_line(stripped):
            continue
        lines.append(stripped)
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _is_noise_line(line: str) -> bool:
    text = line.strip()
    if not text:
        return False
    if text.startswith("#"):
        return False
    words = text.split()
    if URL_RE.search(text) or MARKDOWN_LINK_RE.search(text):
        return True
    if len(words) <= 5 and not SENTENCE_END_RE.search(text):
        return True
    if len(text) < 60 and _symbol_ratio(text) > 0.24:
        return True
    return False


def _is_structural_noise_node(node: Tag) -> bool:
    if str(node.name or "").lower() in {"p", "li", "blockquote"}:
        return False
    text = " ".join(node.get_text(" ").split())
    if not text:
        return False
    score = _content_score(node)
    links = node.find_all("a")
    buttons = node.find_all(["button", "input", "select", "textarea"])
    paragraphs = [p for p in node.find_all("p") if len(p.get_text(" ", strip=True).split()) >= 12]
    link_text_len = sum(len(a.get_text(" ", strip=True)) for a in links)
    link_density = link_text_len / max(len(text), 1)
    control_density = len(buttons) / max(len(WORD_RE.findall(text)), 1)
    repeated_fraction = _repeated_short_text_fraction(node)
    return (
        score < 500
        and not paragraphs
        and (
            link_density > 0.35
            or control_density > 0.08
            or repeated_fraction > 0.65
            or len(text.split()) <= 12
        )
    )


def _repeated_short_text_fraction(node: Tag) -> float:
    chunks = [
        " ".join(child.get_text(" ").split()).lower()
        for child in node.find_all(True)
        if 0 < len(child.get_text(" ").split()) <= 8
    ]
    if len(chunks) < 4:
        return 0.0
    return (len(chunks) - len(set(chunks))) / len(chunks)


def _symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for char in text if not char.isalnum() and not char.isspace()) / len(text)


def _is_low_content(markdown: str) -> bool:
    words = WORD_RE.findall(markdown)
    return len(words) < 40

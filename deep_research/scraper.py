from __future__ import annotations

from dataclasses import dataclass
import re

import httpx
from bs4 import BeautifulSoup, Tag
from markdownify import markdownify
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


@dataclass(frozen=True)
class ScrapeResult:
    url: str
    title: str
    markdown: str
    extraction_method: str = "playwright"


class ScrapeQualityError(RuntimeError):
    """Raised when a fetched page is not usable research content."""


class PlaywrightScraper:
    def __init__(self, *, timeout_ms: int = 30_000):
        self.timeout_ms = timeout_ms

    def fetch(self, url: str) -> ScrapeResult:
        try:
            return self._fetch_with_playwright(url)
        except ScrapeQualityError:
            raise
        except Exception:
            return self._fetch_with_httpx(url)

    def _fetch_with_playwright(self, url: str) -> ScrapeResult:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                response = page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                if response is not None and response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status} while fetching {url}")
                try:
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except PlaywrightTimeoutError:
                    pass
                title = page.title()
                html = page.content()
                return ScrapeResult(
                    url=page.url,
                    title=title,
                    markdown=html_to_markdown(html, title=title),
                )
            finally:
                browser.close()

    def _fetch_with_httpx(self, url: str) -> ScrapeResult:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = httpx.get(url, headers=headers, timeout=self.timeout_ms / 1000)
        response.raise_for_status()
        title = _extract_title(response.text) or url
        return ScrapeResult(
            url=str(response.url),
            title=title,
            markdown=html_to_markdown(response.text, title=title),
            extraction_method="httpx",
        )


def html_to_markdown(html: str, *, title: str = "") -> str:
    soup = BeautifulSoup(html, "html.parser")
    if _is_challenge_page(soup, title=title):
        raise ScrapeQualityError("Fetched page appears to be a bot-protection or JavaScript challenge page.")

    _remove_unusable_nodes(soup)
    candidate = _best_content_node(soup)
    markdown = markdownify(str(candidate), heading_style="ATX")
    markdown = _clean_markdown(markdown)
    if _is_low_content(markdown):
        raise ScrapeQualityError("Fetched page did not contain enough article text after boilerplate removal.")
    return markdown


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


def _is_challenge_page(soup: BeautifulSoup, *, title: str = "") -> bool:
    text = " ".join(soup.get_text(" ").lower().split())
    title_text = title.strip().lower()
    challenge_markers = (
        "just a moment",
        "403 forbidden",
        "access denied",
        "performing security verification",
        "request blocked",
        "enable javascript and cookies to continue",
        "checking if the site connection is secure",
        "verify you are human",
        "temporarily blocked",
        "cloudflare",
        "cf-browser-verification",
    )
    if title_text in {"just a moment...", "just a moment"}:
        return True
    return any(marker in text for marker in challenge_markers)


def _remove_unusable_nodes(soup: BeautifulSoup) -> None:
    selectors = [
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "iframe",
        "nav",
        "header",
        "footer",
        "aside",
        "form",
        "button",
        "[role='navigation']",
        "[role='banner']",
        "[role='contentinfo']",
        ".navbox",
        ".metadata",
        ".mw-editsection",
        ".reference",
        ".references",
        ".reflist",
        ".infobox",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            node.decompose()

    noisy_tokens = (
        "advert",
        "banner",
        "breadcrumb",
        "cookie",
        "footer",
        "header",
        "login",
        "menu",
        "newsletter",
        "promo",
        "sidebar",
        "signup",
        "subscribe",
    )
    for node in list(soup.find_all(True)):
        if node.parent is None or getattr(node, "attrs", None) is None:
            continue
        haystack = " ".join(
            str(value).lower()
            for value in (
                node.get("id"),
                " ".join(node.get("class", [])),
                node.get("aria-label"),
                node.get("role"),
            )
            if value
        )
        if haystack and any(token in haystack for token in noisy_tokens):
            node.decompose()


def _best_content_node(soup: BeautifulSoup) -> Tag | BeautifulSoup:
    selectors = [
        "article",
        "main article",
        "main",
        "[role='main']",
        ".mw-parser-output",
        "#bodyContent",
        ".article-content",
        ".article-body",
        ".post-content",
        ".entry-content",
        ".markdown-body",
        ".content",
        "#content",
    ]
    candidates: list[Tag] = []
    for selector in selectors:
        candidates.extend(node for node in soup.select(selector) if isinstance(node, Tag))
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


def _content_score(node: Tag) -> float:
    text = " ".join(node.get_text(" ").split())
    if not text:
        return 0.0
    paragraph_count = len([p for p in node.find_all("p") if len(p.get_text(" ").strip()) > 40])
    heading_count = len(node.find_all(re.compile("^h[1-3]$")))
    link_text_len = sum(len(a.get_text(" ", strip=True)) for a in node.find_all("a"))
    link_density = link_text_len / max(len(text), 1)
    return len(text) + (paragraph_count * 250) + (heading_count * 80) - (link_density * 800)


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
    text = line.strip().lower()
    if not text:
        return False
    exact_noise = {
        "log in",
        "login",
        "sign up",
        "sign in",
        "join for free",
        "create account",
        "donate",
        "search",
        "main menu",
        "personal tools",
        "appearance",
        "toggle the table of contents",
    }
    if text in exact_noise:
        return True
    if len(text) < 40 and any(token in text for token in ("cookie", "newsletter", "subscribe", "advertisement")):
        return True
    return False


def _is_low_content(markdown: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z-]+", markdown)
    return len(words) < 40

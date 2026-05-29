from __future__ import annotations

from dataclasses import dataclass

import httpx
from markdownify import markdownify
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


@dataclass(frozen=True)
class ScrapeResult:
    url: str
    title: str
    markdown: str
    extraction_method: str = "playwright"


class PlaywrightScraper:
    def __init__(self, *, timeout_ms: int = 30_000):
        self.timeout_ms = timeout_ms

    def fetch(self, url: str) -> ScrapeResult:
        try:
            return self._fetch_with_playwright(url)
        except Exception:
            return self._fetch_with_httpx(url)

    def _fetch_with_playwright(self, url: str) -> ScrapeResult:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page()
                page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                try:
                    page.wait_for_load_state("networkidle", timeout=5_000)
                except PlaywrightTimeoutError:
                    pass
                title = page.title()
                html = page.content()
                return ScrapeResult(
                    url=page.url,
                    title=title,
                    markdown=markdownify(html),
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
            markdown=markdownify(response.text),
            extraction_method="httpx",
        )


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

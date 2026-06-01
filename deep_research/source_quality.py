from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SourceQuality:
    score: float
    label: str
    source_type: str
    reasons: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ACADEMIC_HOSTS = {
    "arxiv.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "nih.gov",
    "nature.com",
    "science.org",
    "acm.org",
    "ieee.org",
    "springer.com",
    "jmlr.org",
}

STANDARDS_HOSTS = {
    "nist.gov",
    "w3.org",
    "ietf.org",
    "rfc-editor.org",
    "iso.org",
    "ecma-international.org",
}

USER_CONTENT_HOSTS = {
    "medium.com",
    "substack.com",
    "reddit.com",
    "quora.com",
    "linkedin.com",
    "towardsdatascience.com",
    "dev.to",
    "hashnode.dev",
}

REFERENCE_HOSTS = {
    "wikipedia.org",
    "britannica.com",
}

NEWS_HOSTS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "nytimes.com",
    "theguardian.com",
}

SEO_TITLE_PATTERNS = (
    "what is",
    "ultimate guide",
    "complete guide",
    "best ",
    "top ",
    "explained",
)


def score_source(
    *,
    url: str,
    title: str = "",
    snippet: str | None = None,
    markdown: str | None = None,
    search_score: float | None = None,
) -> SourceQuality:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    path = parsed.path.lower()
    title_text = title.lower()
    text = " ".join(part for part in (title, snippet or "", markdown or "") if part).lower()
    score = 0.55
    reasons: list[str] = []
    source_type = "general_web"

    if _host_matches(host, STANDARDS_HOSTS):
        score += 0.28
        source_type = "standards_or_government"
        reasons.append("standards/government source")
    elif host.endswith(".gov"):
        score += 0.25
        source_type = "government"
        reasons.append("government domain")
    elif _host_matches(host, ACADEMIC_HOSTS) or host.endswith(".edu"):
        score += 0.22
        source_type = "academic"
        reasons.append("academic or scholarly source")
    elif _looks_like_official_docs(host, path):
        score += 0.17
        source_type = "official_docs"
        reasons.append("official documentation pattern")
    elif _host_matches(host, NEWS_HOSTS):
        score += 0.04
        source_type = "news"
        reasons.append("established news source")

    if _host_matches(host, USER_CONTENT_HOSTS):
        score -= 0.23
        source_type = "user_content"
        reasons.append("user-generated or blog platform")

    if _host_matches(host, REFERENCE_HOSTS):
        score -= 0.03
        source_type = "reference"
        reasons.append("reference source; verify against primary sources when possible")

    if any(pattern in title_text for pattern in SEO_TITLE_PATTERNS):
        score -= 0.05
        reasons.append("SEO-style title")

    if "/blog" in path or "/resources/" in path or "/articles/" in path:
        score -= 0.04
        reasons.append("article/blog path")

    if any(token in text for token in ("white paper", "technical report", "specification", "documentation")):
        score += 0.04
        reasons.append("technical source language")

    if search_score is not None:
        score += max(0.0, min(float(search_score), 1.0)) * 0.05
        reasons.append("search relevance signal")

    if markdown is not None:
        word_count = len(re.findall(r"[A-Za-z][A-Za-z-]+", markdown))
        if word_count >= 800:
            score += 0.06
            reasons.append("substantial extracted text")
        elif word_count < 120:
            score -= 0.12
            reasons.append("short extracted text")

    score = round(max(0.0, min(score, 1.0)), 4)
    return SourceQuality(
        score=score,
        label=_quality_label(score),
        source_type=source_type,
        reasons=reasons or ["neutral source signals"],
    )


def _host_matches(host: str, candidates: set[str]) -> bool:
    return any(host == candidate or host.endswith(f".{candidate}") for candidate in candidates)


def _looks_like_official_docs(host: str, path: str) -> bool:
    if _host_matches(host, USER_CONTENT_HOSTS):
        return False
    host_tokens = host.split(".")
    return (
        "docs" in host_tokens
        or "developer" in host_tokens
        or "developers" in host_tokens
        or path.startswith("/docs")
        or path.startswith("/developer")
        or path.startswith("/reference")
    )


def _quality_label(score: float) -> str:
    if score >= 0.82:
        return "excellent"
    if score >= 0.70:
        return "strong"
    if score >= 0.55:
        return "usable"
    return "weak"

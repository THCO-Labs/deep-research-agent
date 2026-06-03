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


@dataclass(frozen=True)
class DomainSignals:
    host: str
    domain: str
    suffix: str
    subdomain_tokens: tuple[str, ...]
    host_tokens: tuple[str, ...]
    path_tokens: tuple[str, ...]


TECHNICAL_LANGUAGE_RE = re.compile(
    r"\b(?:documentation|specification|standard|technical\s+report|white\s+paper|manual|reference|guideline|"
    r"api|sdk|protocol|benchmark|dataset)\b",
    flags=re.I,
)
SCHOLARLY_LANGUAGE_RE = re.compile(
    r"\b(?:doi|peer[-\s]?reviewed|journal|abstract|citation|proceedings|preprint|clinical\s+trial|"
    r"systematic\s+review|meta[-\s]?analysis)\b|10\.\d{4,9}/",
    flags=re.I,
)
DOCS_TOKEN_RE = re.compile(r"^(?:docs?|documentation|developer|developers|learn|manual|reference|api|sdk)$", flags=re.I)
REPOSITORY_TOKEN_RE = re.compile(r"^(?:git|github|gitlab|bitbucket|sourceforge|code|repo|repository)$", flags=re.I)
USER_CONTENT_TOKEN_RE = re.compile(
    r"^(?:blog|blogs|forum|forums|community|question|questions|answer|answers|discussion|discuss|"
    r"medium|substack|wordpress|blogspot|reddit|quora|stackoverflow|stackexchange|tumblr|hashnode|dev)$",
    flags=re.I,
)
REFERENCE_TOKEN_RE = re.compile(r"^(?:wiki|wikipedia|encyclopedia|reference|britannica)$", flags=re.I)
PREPRINT_TOKEN_RE = re.compile(r"(?:^|[-_.])(?:arxiv|biorxiv|medrxiv|chemrxiv|psyarxiv|techrxiv|ssrn)(?:$|[-_.])", flags=re.I)
STANDARDS_TOKEN_RE = re.compile(
    r"^(?:rfc|ietf|iana|iso|iec|w3c|whatwg|tc39|ecma|ansi|nist|oasis|unicode|khronos|standards?)$",
    flags=re.I,
)
SEO_TITLE_RE = re.compile(r"\b(?:what\s+is|ultimate\s+guide|complete\s+guide|best|top|explained)\b", flags=re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z-]+")


def score_source(
    *,
    url: str,
    title: str = "",
    snippet: str | None = None,
    markdown: str | None = None,
    search_score: float | None = None,
) -> SourceQuality:
    parsed = urlsplit(url)
    domain = _domain_signals(parsed.hostname or "", parsed.path)
    title_text = title.lower()
    combined_text = " ".join(part for part in (title, snippet or "", markdown or "") if part)
    combined_lower = combined_text.lower()
    path = parsed.path.lower()

    score = 0.55
    reasons: list[str] = []
    source_type = "general_web"

    if _is_standards_source(domain, title_text, combined_lower):
        score += 0.28
        source_type = "standards_or_government"
        reasons.append("standards or specification signal")
    elif _is_government_or_multilateral_source(domain):
        score += 0.25
        source_type = "government"
        reasons.append("government or multilateral domain signal")
    elif _looks_like_official_docs(domain, path, combined_lower):
        score += 0.17
        source_type = "official_docs"
        reasons.append("documentation or developer-reference signal")
    elif _is_scholarly_source(domain, path, combined_lower):
        score += 0.22
        source_type = "academic"
        reasons.append("academic or scholarly publication signal")
    elif _looks_like_software_repository(domain, path):
        score += 0.04
        source_type = "software_repository"
        reasons.append("software repository signal")
    elif _looks_like_established_news(domain, path):
        score += 0.04
        source_type = "news"
        reasons.append("news/publication signal")

    if _looks_like_user_content(domain, path):
        score -= 0.23
        source_type = "user_content"
        reasons.append("user-generated, forum, or personal publishing signal")

    if _looks_like_reference_source(domain):
        score -= 0.03
        source_type = "reference"
        reasons.append("reference source; verify against primary sources when possible")

    if _looks_like_preprint(domain, path, combined_lower):
        score -= 0.04
        reasons.append("preprint source; verify peer-review status")

    if SEO_TITLE_RE.search(title_text):
        score -= 0.05
        reasons.append("SEO-style title")

    if _looks_like_article_path(path):
        score -= 0.04
        reasons.append("article/blog path")

    if TECHNICAL_LANGUAGE_RE.search(combined_text):
        score += 0.04
        reasons.append("technical source language")

    if SCHOLARLY_LANGUAGE_RE.search(combined_text) or _path_has_scholarly_marker(path):
        score += 0.05
        reasons.append("scholarly publication signal")

    if search_score is not None:
        score += max(0.0, min(float(search_score), 1.0)) * 0.05
        reasons.append("search relevance signal")

    if markdown is not None:
        word_count = len(WORD_RE.findall(markdown))
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


def _domain_signals(host: str, path: str) -> DomainSignals:
    normalized_host = host.lower().removeprefix("www.")
    try:
        import tldextract

        extractor = tldextract.TLDExtract(suffix_list_urls=(), fallback_to_snapshot=True)
        extracted = extractor(normalized_host)
        domain = extracted.domain.lower()
        suffix = extracted.suffix.lower()
        subdomain = extracted.subdomain.lower()
    except Exception:
        parts = normalized_host.split(".")
        domain = parts[-2] if len(parts) >= 2 else normalized_host
        suffix = ".".join(parts[-1:]) if len(parts) >= 2 else ""
        subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""

    subdomain_tokens = _tokens_from_text(subdomain)
    host_tokens = _tokens_from_text(normalized_host)
    path_tokens = _tokens_from_text(path)
    return DomainSignals(
        host=normalized_host,
        domain=domain,
        suffix=suffix,
        subdomain_tokens=subdomain_tokens,
        host_tokens=host_tokens,
        path_tokens=path_tokens,
    )


def _tokens_from_text(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.split(r"[^a-z0-9]+", text.lower()) if token)


def _is_standards_source(domain: DomainSignals, title: str, text: str) -> bool:
    tokens = set(domain.host_tokens + domain.path_tokens)
    return bool(
        any(STANDARDS_TOKEN_RE.match(token) for token in tokens)
        or re.search(r"\b(?:rfc\s?\d+|iso\s?\d+|iec\s?\d+|ecma-?\d+|standard|specification)\b", title)
        or re.search(r"\b(?:standard|specification|protocol)\b", text)
        and any(token in tokens for token in {"rfc", "ietf", "iso", "iec", "ecma", "w3c", "whatwg", "standard", "standards"})
    )


def _is_government_or_multilateral_source(domain: DomainSignals) -> bool:
    suffix = domain.suffix
    tokens = set(domain.host_tokens + domain.path_tokens)
    return (
        suffix == "gov"
        or suffix.startswith("gov.")
        or ".gov." in f".{suffix}."
        or suffix == "int"
        or any(token in tokens for token in {"gov", "government", "agency", "ministry", "department", "parliament", "senate"})
    )


def _looks_like_official_docs(domain: DomainSignals, path: str, text: str) -> bool:
    if _looks_like_user_content(domain, path):
        return False
    tokens = set(domain.subdomain_tokens + domain.path_tokens)
    return (
        any(DOCS_TOKEN_RE.match(token) for token in tokens)
        or path.startswith(("/docs", "/developer", "/developers", "/reference", "/manual", "/learn"))
        or re.search(r"\b(?:official\s+documentation|api\s+reference|developer\s+guide|user\s+guide)\b", text)
    )


def _is_scholarly_source(domain: DomainSignals, path: str, text: str) -> bool:
    suffix = domain.suffix
    tokens = set(domain.host_tokens + domain.path_tokens)
    return (
        suffix == "edu"
        or suffix.startswith("edu.")
        or suffix.startswith("ac.")
        or ".ac." in f".{suffix}."
        or _path_has_scholarly_marker(path)
        or SCHOLARLY_LANGUAGE_RE.search(text) is not None
        or any(token in tokens for token in {"journal", "journals", "research", "publication", "publications", "proceedings"})
        or _looks_like_preprint(domain, path, text)
    )


def _looks_like_software_repository(domain: DomainSignals, path: str) -> bool:
    tokens = set(domain.host_tokens + domain.path_tokens)
    return (
        any(REPOSITORY_TOKEN_RE.match(token) for token in tokens)
        or re.search(r"/(?:src|source|tree|blob|commit|pull|issues|releases)(?:/|$)", path)
    )


def _looks_like_established_news(domain: DomainSignals, path: str) -> bool:
    tokens = set(domain.host_tokens + domain.path_tokens)
    return any(token in tokens for token in {"news", "press", "times", "post", "journal", "review"}) or "/news/" in path


def _looks_like_user_content(domain: DomainSignals, path: str) -> bool:
    tokens = set(domain.host_tokens + domain.path_tokens)
    return (
        any(USER_CONTENT_TOKEN_RE.match(token) for token in tokens)
        or re.search(r"/(?:@[^/]+|users?|members?|questions?|answers?|posts?|comments?|forum|forums|discussion|blog)(?:/|$)", path)
        is not None
    )


def _looks_like_reference_source(domain: DomainSignals) -> bool:
    return any(REFERENCE_TOKEN_RE.match(token) for token in domain.host_tokens)


def _looks_like_preprint(domain: DomainSignals, path: str, text: str) -> bool:
    haystack = " ".join((".".join(domain.host_tokens), path, text))
    return bool(PREPRINT_TOKEN_RE.search(haystack) or re.search(r"\bpreprint\b", haystack, flags=re.I))


def _looks_like_article_path(path: str) -> bool:
    return bool(re.search(r"/(?:blog|blogs|resources|articles|posts|news|insights)(?:/|$)", path))


def _path_has_scholarly_marker(path: str) -> bool:
    return bool(
        re.search(
            r"/(?:doi|abs|abstract|article|articles|journal|journals|pmc|pubmed|paper|papers|preprint|preprints|content)(?:/|$)",
            path,
        )
        or re.search(r"/10\.\d{4,9}/", path)
    )


def _quality_label(score: float) -> str:
    if score >= 0.82:
        return "excellent"
    if score >= 0.70:
        return "strong"
    if score >= 0.55:
        return "usable"
    return "weak"

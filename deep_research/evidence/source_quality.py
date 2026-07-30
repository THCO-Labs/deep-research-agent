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


@dataclass(frozen=True)
class ProductDocumentSignals:
    document_hint: bool
    product_location_hint: bool
    product_title_hint: bool
    key_value_count: int
    bullet_count: int
    unit_count: int
    model_count: int
    table_count: int
    structured_score: int

    @property
    def has_technical_evidence(self) -> bool:
        return self.structured_score >= 4 or (
            self.model_count >= 1 and (self.unit_count >= 1 or self.key_value_count >= 2 or self.table_count >= 1)
        )


TECHNICAL_LANGUAGE_RE = re.compile(
    r"\b(?:documentation|specification|specifications|specs?|standard|technical\s+report|technical\s+data|"
    r"white\s+paper|manual|reference|guideline|datasheet|data\s+sheet|brochure|catalog(?:ue)?|"
    r"api|sdk|protocol|benchmark|dataset)\b",
    flags=re.I,
)
SCHOLARLY_LANGUAGE_RE = re.compile(
    r"\b(?:doi|peer[-\s]?reviewed|journal|abstract|citation|proceedings|preprint|clinical\s+trial|"
    r"systematic\s+review|meta[-\s]?analysis)\b|10\.\d{4,9}/",
    flags=re.I,
)
DOCS_TOKEN_RE = re.compile(r"^(?:docs?|documentation|developer|developers|learn|manual|reference|api|sdk)$", flags=re.I)
PRODUCT_PATH_TOKEN_RE = re.compile(
    r"^(?:products?|machines?|equipment|models?|model|catalog(?:ue)?|downloads?|technical|specifications?|"
    r"specs?|datasheets?|brochures?|manuals?|support)$",
    flags=re.I,
)
PRODUCT_TITLE_RE = re.compile(
    r"\b(?:product|machine|equipment|model|series|brochure|catalog(?:ue)?|datasheet|data\s+sheet|"
    r"technical\s+data|specifications?|specs?|manual)\b",
    flags=re.I,
)
TECHNICAL_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:kw|hp|rpm|min-?1|nm|mm|cm|m|in\.?|inch(?:es)?|kg|lb|lbs|v|hz|"
    r"bar|psi|mpa|µm|um|micron|%|°c|deg(?:ree)?s?)\b",
    flags=re.I,
)
MODEL_TOKEN_RE = re.compile(r"\b[A-Z]{1,8}[A-Z0-9-]*\s?\d{2,5}[A-Z0-9-]*\b")
DOCUMENT_EXTENSION_RE = re.compile(r"\.(?:pdf|docx?|xlsx?|csv)(?:$|[?#])", flags=re.I)
SPEC_FIELD_LINE_RE = re.compile(r"(?m)^\s*[A-Za-z][A-Za-z0-9 /().,+#%-]{1,64}\s*(?::|\||\t| {2,})\s*\S+")
BULLET_LINE_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S+")
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

    if _looks_like_video_transcript(domain, combined_text):
        # Search providers (Exa in particular) prepend a metadata block --
        # "Channel:", "Length:", "Views:", "Published:" -- before an embedded video
        # transcript. That block's key-value density otherwise trips the spec_sheet
        # heuristic below, mislabeling a speech/talk/interview transcript as a
        # product datasheet. Checked first so it wins before that heuristic runs.
        score += 0.10
        source_type = "video_transcript"
        reasons.append("video platform URL with an embedded transcript")
    elif _looks_like_product_document(domain, path, title, combined_text, url):
        score += 0.20
        source_type = _product_document_type(domain, path, title, combined_text, url)
        reasons.append("product/specification document signal")
    elif _is_standards_source(domain, title_text, combined_lower):
        score += 0.28
        source_type = "standards_or_government"
        reasons.append("standards or specification signal")
    elif _is_scholarly_source(domain, path, combined_lower):
        score += 0.22
        source_type = "academic"
        reasons.append("academic or scholarly publication signal")
    elif _is_government_or_multilateral_source(domain):
        score += 0.25
        source_type = "government"
        reasons.append("government or multilateral domain signal")
    elif _looks_like_official_docs(domain, path, combined_lower):
        score += 0.17
        source_type = "official_docs"
        reasons.append("documentation or developer-reference signal")
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

    if _technical_spec_signal_count(combined_text) >= 3:
        score += 0.05
        reasons.append("structured technical/specification signal")

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


_VIDEO_PLATFORM_DOMAINS = {"youtube.com", "youtu.be", "m.youtube.com"}
_TRANSCRIPT_HEADING_RE = re.compile(r"##\s*transcript\b", re.I)


def _looks_like_video_transcript(domain: DomainSignals, text: str) -> bool:
    host = f"{domain.domain}.{domain.suffix}".strip(".")
    if host not in _VIDEO_PLATFORM_DOMAINS and domain.host not in _VIDEO_PLATFORM_DOMAINS:
        return False
    return bool(_TRANSCRIPT_HEADING_RE.search(text))


def _looks_like_product_document(
    domain: DomainSignals,
    path: str,
    title: str,
    text: str,
    url: str,
) -> bool:
    if _looks_like_user_content(domain, path):
        return False
    signals = _product_document_signals(domain=domain, path=path, title=title, text=text, url=url)
    has_extractable_spec_evidence = signals.has_technical_evidence or (signals.document_hint and signals.structured_score >= 2)
    return bool(
        has_extractable_spec_evidence
        and (
            signals.product_location_hint
            or signals.product_title_hint
            or signals.document_hint
            or signals.structured_score >= 8
        )
    )


def _product_document_type(
    domain: DomainSignals,
    path: str,
    title: str,
    text: str,
    url: str,
) -> str:
    lower = " ".join([url, path, title, text[:1200]]).lower()
    signals = _product_document_signals(domain=domain, path=path, title=title, text=text, url=url)
    is_pdf = url.lower().split("?", 1)[0].endswith(".pdf")
    if is_pdf and re.search(r"\b(?:manual|user\s+guide|operation|installation)\b", lower):
        return "manual_pdf"
    if is_pdf and re.search(r"\b(?:brochure|catalog(?:ue)?)\b", lower):
        return "brochure_pdf"
    if is_pdf and re.search(r"\b(?:datasheet|data\s+sheet|technical\s+data|specifications?|specs?)\b", lower):
        return "spec_sheet"
    if is_pdf:
        return "brochure_pdf"
    if re.search(r"\b(?:datasheet|data\s+sheet)\b", lower):
        return "datasheet"
    if re.search(r"\b(?:specifications?|specs?|technical\s+data)\b", lower):
        return "spec_sheet"
    if signals.key_value_count >= 3 or signals.table_count >= 2 or signals.unit_count >= 3:
        return "spec_sheet"
    if signals.product_location_hint:
        return "product_page"
    return "vendor_page"


def _product_document_signals(
    *,
    domain: DomainSignals,
    path: str,
    title: str,
    text: str,
    url: str,
) -> ProductDocumentSignals:
    tokens = set(domain.subdomain_tokens + domain.path_tokens)
    lower_url = url.lower()
    key_value_count = len(SPEC_FIELD_LINE_RE.findall(text))
    bullet_count = len(BULLET_LINE_RE.findall(text))
    unit_count = len(TECHNICAL_UNIT_RE.findall(text))
    model_count = len(MODEL_TOKEN_RE.findall(text))
    table_count = _table_like_line_count(text)
    structured_score = (
        min(key_value_count, 8)
        + min(bullet_count, 6)
        + min(unit_count, 12)
        + min(model_count, 8)
        + min(table_count, 8)
    )
    document_hint = bool(DOCUMENT_EXTENSION_RE.search(lower_url) or TECHNICAL_LANGUAGE_RE.search(f"{url} {title}"))
    product_location_hint = bool(
        any(PRODUCT_PATH_TOKEN_RE.match(token) for token in tokens)
        or re.search(r"/(?:products?|machines?|equipment|models?)/", path.lower())
    )
    product_title_hint = bool(PRODUCT_TITLE_RE.search(title))
    return ProductDocumentSignals(
        document_hint=document_hint,
        product_location_hint=product_location_hint,
        product_title_hint=product_title_hint,
        key_value_count=key_value_count,
        bullet_count=bullet_count,
        unit_count=unit_count,
        model_count=model_count,
        table_count=table_count,
        structured_score=structured_score,
    )


def _technical_spec_signal_count(text: str) -> int:
    if not text:
        return 0
    key_value_count = len(SPEC_FIELD_LINE_RE.findall(text))
    bullet_count = len(BULLET_LINE_RE.findall(text))
    unit_count = len(TECHNICAL_UNIT_RE.findall(text))
    model_count = len(MODEL_TOKEN_RE.findall(text))
    table_count = _table_like_line_count(text)
    return key_value_count + min(bullet_count, 8) + min(unit_count, 12) + min(model_count, 8) + min(table_count, 8)


def _table_like_line_count(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line.count("|") >= 2 or line.count("\t") >= 2 or len(re.split(r"\s{2,}", stripped)) >= 3:
            count += 1
    return count


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
        or any(token in tokens for token in {"journal", "journals", "research", "proceedings", "pubmed", "pmc"})
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

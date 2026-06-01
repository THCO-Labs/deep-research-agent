from __future__ import annotations

import re
from dataclasses import asdict, dataclass

TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")

STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "best",
    "can",
    "does",
    "for",
    "from",
    "guide",
    "how",
    "into",
    "its",
    "learn",
    "more",
    "the",
    "this",
    "use",
    "what",
    "when",
    "with",
}


@dataclass(frozen=True)
class SourceRelevance:
    score: float
    matched_terms: list[str]
    missing_terms: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_source_relevance(
    *,
    query: str,
    url: str = "",
    title: str = "",
    snippet: str | None = None,
    markdown: str | None = None,
) -> SourceRelevance:
    query_terms = _tokens(query)
    if not query_terms:
        return SourceRelevance(score=0.0, matched_terms=[], missing_terms=[])

    weighted_terms: dict[str, float] = {}
    for term in _tokens(title):
        weighted_terms[term] = max(weighted_terms.get(term, 0.0), 1.0)
    for term in _tokens(snippet or ""):
        weighted_terms[term] = max(weighted_terms.get(term, 0.0), 0.8)
    for term in _tokens(url.replace("/", " ").replace("-", " ")):
        weighted_terms[term] = max(weighted_terms.get(term, 0.0), 0.6)
    if markdown is not None:
        for term in _tokens(markdown[:4000]):
            weighted_terms[term] = max(weighted_terms.get(term, 0.0), 0.5)

    matched = sorted(term for term in query_terms if term in weighted_terms)
    missing = sorted(query_terms - set(matched))
    raw_score = sum(weighted_terms.get(term, 0.0) for term in query_terms) / len(query_terms)
    return SourceRelevance(
        score=round(max(0.0, min(raw_score, 1.0)), 4),
        matched_terms=matched,
        missing_terms=missing,
    )


def _tokens(text: str) -> set[str]:
    normalized = text.lower().replace("_", " ").replace("-", " ")
    return {
        token
        for token in TOKEN_RE.findall(normalized)
        if token not in STOPWORDS and not token.isdigit()
    }

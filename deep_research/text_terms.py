from __future__ import annotations

import re
from functools import lru_cache

TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")


@lru_cache(maxsize=1)
def english_stopwords() -> frozenset[str]:
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for English stopword filtering.") from exc
    return frozenset(ENGLISH_STOP_WORDS)


def stopword_source() -> str:
    english_stopwords()
    return "sklearn"


def normalize_term_text(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def ordered_terms(text: str, *, extra_stopwords: frozenset[str] | set[str] | None = None) -> list[str]:
    stopwords = english_stopwords()
    if extra_stopwords:
        stopwords = stopwords | frozenset(extra_stopwords)

    terms: list[str] = []
    seen: set[str] = set()
    for token in TOKEN_RE.findall(normalize_term_text(text)):
        if token in stopwords or token.isdigit() or token in seen:
            continue
        terms.append(token)
        seen.add(token)
    return terms


def term_set(text: str, *, extra_stopwords: frozenset[str] | set[str] | None = None) -> set[str]:
    return set(ordered_terms(text, extra_stopwords=extra_stopwords))

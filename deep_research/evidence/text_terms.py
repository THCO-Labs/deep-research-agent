from __future__ import annotations

import re
from functools import lru_cache

try:
    import regex as unicode_regex
except ImportError:  # pragma: no cover - dependency is declared, fallback keeps imports usable.
    unicode_regex = None

LATIN_TOKEN_RE = re.compile(r"[a-z][a-z0-9+]{1,}")
if unicode_regex is not None:
    TOKEN_RE = unicode_regex.compile(r"(?V1)(?:[a-z][a-z0-9+]{1,}|\p{Han})", flags=unicode_regex.I)
    TERM_SPAN_RE = unicode_regex.compile(r"(?V1)(?:\p{Han}+|[a-z][a-z0-9+]{1,})", flags=unicode_regex.I)
    HAN_RE = unicode_regex.compile(r"(?V1)^\p{Han}+$")
else:
    TOKEN_RE = re.compile(r"[a-z][a-z0-9+]{1,}|[\u3400-\u9fff]")
    TERM_SPAN_RE = re.compile(r"[\u3400-\u9fff]+|[a-z][a-z0-9+]{1,}", flags=re.I)
    HAN_RE = re.compile(r"^[\u3400-\u9fff]+$")

CJK_BOUNDARY_WORDS = (
    "请为我",
    "请帮我",
    "收集整理",
    "给我",
    "收集",
    "整理",
    "整合",
    "调研",
    "研究",
    "分析",
    "总结",
    "提供",
    "一份",
    "详尽",
    "详细",
    "报告",
    "请",
    "为我",
    "有关",
    "目前",
    "现在",
    "未来",
    "近几年",
    "近十年",
    "最后",
    "包括",
    "包含",
    "比如",
    "例如",
    "哪些",
    "什么",
    "如何",
    "是否",
    "能否",
    "可以",
    "需要",
    "以及",
    "或者",
    "并且",
    "最终",
    "在此基础上",
    "基础上",
    "方面",
    "情况",
    "资料",
    "相关",
    "这个",
    "它们",
    "我们",
    "你们",
    "他们",
    "问题",
    "和",
    "与",
    "及",
    "对",
    "从",
    "在",
    "上",
    "下",
    "的",
    "了",
    "是",
    "有",
    "等",
)
CJK_BOUNDARY_RE = re.compile("|".join(re.escape(word) for word in sorted(CJK_BOUNDARY_WORDS, key=len, reverse=True)))


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
    normalized = normalize_term_text(text)
    if not extra_stopwords:
        return list(_ordered_terms_cached(normalized))

    stopwords = english_stopwords()
    if extra_stopwords:
        stopwords = stopwords | frozenset(extra_stopwords)
    return _ordered_terms_from_normalized(normalized, stopwords)


@lru_cache(maxsize=8192)
def _ordered_terms_cached(normalized_text: str) -> tuple[str, ...]:
    return tuple(_ordered_terms_from_normalized(normalized_text, english_stopwords()))


def _ordered_terms_from_normalized(normalized_text: str, stopwords: frozenset[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for span in TERM_SPAN_RE.findall(normalized_text):
        for token in _tokens_for_span(span):
            if token in stopwords or token.isdigit() or token in seen:
                continue
            terms.append(token)
            seen.add(token)
    return terms


def term_set(text: str, *, extra_stopwords: frozenset[str] | set[str] | None = None) -> set[str]:
    return set(ordered_terms(text, extra_stopwords=extra_stopwords))


def contains_cjk(text: str) -> bool:
    return bool(TOKEN_RE.search(text) and any(HAN_RE.fullmatch(token) for token in TOKEN_RE.findall(text)))


def cjk_char_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", text))


def latin_letter_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]", text))


def preferred_output_language(text: str) -> str:
    cjk_count = cjk_char_count(text)
    latin_count = latin_letter_count(text)
    if cjk_count >= 4 and cjk_count >= max(4, int(latin_count * 0.25)):
        return "zh"
    return "en"


def _tokens_for_span(span: str) -> list[str]:
    if HAN_RE.fullmatch(span):
        return _han_terms(span)
    if LATIN_TOKEN_RE.fullmatch(span):
        return [span]
    return []


def _han_terms(text: str) -> list[str]:
    if len(text) > 24:
        return _han_ngram_terms(text)
    segmented = _jieba_terms(text)
    if segmented:
        return segmented
    return _han_ngram_terms(text)


def _jieba_terms(text: str) -> list[str]:
    jieba = _jieba_module()
    if jieba is None:
        return []
    return _dedupe(
        token.strip()
        for token in jieba.cut_for_search(text)
        if len(token.strip()) >= 2 and HAN_RE.search(token) and not _is_cjk_noise(token.strip())
    )


@lru_cache(maxsize=1)
def _jieba_module():
    try:
        import jieba
    except ImportError:
        return None
    return jieba


def _han_ngram_terms(text: str) -> list[str]:
    if len(text) <= 1:
        return []
    terms: list[str] = []
    chunks = cjk_content_chunks(text)
    terms.extend(chunks)
    for chunk in chunks:
        if len(chunk) <= 4:
            continue
        for size in (4, 3, 2):
            if len(chunk) < size:
                continue
            for index in range(0, len(chunk) - size + 1):
                terms.append(chunk[index : index + size])
    return _dedupe(terms)


def cjk_content_chunks(text: str) -> list[str]:
    cleaned = re.sub(r"[^\u3400-\u9fff]+", " ", text)
    chunks: list[str] = []
    for span in cleaned.split():
        for piece in CJK_BOUNDARY_RE.split(span):
            piece = piece.strip()
            if len(piece) < 2 or _is_cjk_noise(piece):
                continue
            chunks.extend(_bounded_cjk_chunks(piece))
    return _dedupe(chunks)


def _is_cjk_noise(value: str) -> bool:
    return value in CJK_BOUNDARY_WORDS or len(value) < 2


def _bounded_cjk_chunks(piece: str, *, max_len: int = 14) -> list[str]:
    if len(piece) <= max_len:
        return [piece]
    chunks: list[str] = []
    step = max_len
    for index in range(0, len(piece), step):
        chunk = piece[index : index + max_len]
        if len(chunk) >= 2:
            chunks.append(chunk)
    return chunks


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = str(value).strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result

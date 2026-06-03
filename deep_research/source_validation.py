from __future__ import annotations

import re
from dataclasses import dataclass

from deep_research.schemas import ResearchBranch
from deep_research.text_terms import TOKEN_RE, normalize_term_text, ordered_terms, term_set

URL_RE = re.compile(r"https?://\S+", flags=re.I)
KEY_VALUE_LINE_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 _./-]{1,48}:\s+\S+")
SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?$")


@dataclass(frozen=True)
class SourceValidation:
    usable: bool
    relevance_score: float
    word_count: int
    relevant_chunk_count: int
    reasons: list[str]


def validate_source_content(
    *,
    title: str,
    content: str,
    branch: ResearchBranch,
    min_words: int,
    min_relevant_chunks: int,
    question: str = "",
) -> SourceValidation:
    normalized = _normalize(content)
    words = TOKEN_RE.findall(normalize_term_text(normalized))
    word_count = len(words)
    reasons: list[str] = []
    if word_count < min_words:
        reasons.append(f"short extracted text: {word_count} words < {min_words}")

    if _looks_like_boilerplate(normalized):
        reasons.append("extracted text appears to be mostly boilerplate or related-link content")

    terms = _branch_terms(branch)
    if not terms:
        terms = _tokens(branch.title + " " + branch.objective)
    content_terms = _tokens(title + " " + normalized)
    question_terms = _tokens(question)
    if question_terms:
        question_matches = _matched_terms(question_terms, content_terms)
        if len(question_matches) < min(2, len(question_terms)):
            reasons.append("source relevance to the original question below threshold")
    matched_terms = _matched_terms(terms, content_terms)
    relevance_score = round(len(matched_terms) / max(len(terms), 1), 4)
    if relevance_score < 0.30:
        reasons.append("branch relevance below threshold")

    relevant_chunks = [
        chunk for chunk in _chunks(normalized) if len(_matched_terms(terms, _tokens(chunk))) >= max(1, min(3, len(terms)))
    ]
    if len(relevant_chunks) < min_relevant_chunks:
        reasons.append(
            f"only {len(relevant_chunks)} relevant chunk(s); need {min_relevant_chunks}"
        )

    return SourceValidation(
        usable=not reasons,
        relevance_score=relevance_score,
        word_count=word_count,
        relevant_chunk_count=len(relevant_chunks),
        reasons=reasons,
    )


def branch_terms(branch: ResearchBranch) -> set[str]:
    return _branch_terms(branch)


def content_terms(text: str) -> set[str]:
    return _tokens(text)


def _branch_terms(branch: ResearchBranch) -> set[str]:
    seed = " ".join(
        [
            branch.title,
            branch.objective,
            " ".join(branch.queries),
            " ".join(branch.required_terms),
        ]
    )
    expanded = _expand_query_terms(seed)
    return _tokens(seed + " " + expanded)


def _expand_query_terms(text: str) -> str:
    tokens = _ordered_tokens(text)
    joined = " ".join(tokens)
    bigrams = " ".join(f"{left} {right}" for left, right in zip(tokens, tokens[1:]))
    return f"{joined} {bigrams}"


def _tokens(text: str) -> set[str]:
    return term_set(text)


def _matched_terms(expected_terms: set[str], observed_terms: set[str]) -> set[str]:
    return {
        term
        for term in expected_terms
        if term in observed_terms or any(variant in observed_terms for variant in _term_variants(term))
    }


def _term_variants(term: str) -> set[str]:
    variants = {term}
    if len(term) > 4 and term.endswith("ies"):
        variants.add(term[:-3] + "y")
    if len(term) > 4 and term.endswith("es"):
        variants.add(term[:-2])
    if len(term) > 3 and term.endswith("s"):
        variants.add(term[:-1])
    if len(term) > 3:
        variants.add(term + "s")
    return variants


def _ordered_tokens(text: str) -> list[str]:
    return ordered_terms(text)


def _chunks(text: str) -> list[str]:
    chunks = []
    for block in re.split(r"\n\s*\n|(?<=[.!?])\s+", text):
        chunk = block.strip()
        if len(chunk) >= 80:
            chunks.append(chunk)
    return chunks


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _looks_like_boilerplate(text: str) -> bool:
    if not text:
        return True
    lines = [line.strip() for line in re.split(r"\n+|(?<=\S)\s{2,}(?=\S)", text) if line.strip()]
    words = TOKEN_RE.findall(normalize_term_text(text))
    if not words:
        return True
    unique_terms = len(_tokens(text))
    url_chars = sum(len(match.group(0)) for match in URL_RE.finditer(text))
    symbol_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
    metadata_lines = sum(1 for line in lines if KEY_VALUE_LINE_RE.search(line) and (URL_RE.search(line) or len(TOKEN_RE.findall(normalize_term_text(line))) <= 8))
    sentence_lines = sum(1 for line in lines if SENTENCE_END_RE.search(line))
    repeated_line_fraction = _repeated_line_fraction(lines)
    url_ratio = url_chars / max(len(text), 1)
    symbol_ratio = symbol_chars / max(len(text), 1)
    metadata_ratio = metadata_lines / max(len(lines), 1)
    sentence_ratio = sentence_lines / max(len(lines), 1)
    return (
        (url_ratio > 0.20 and unique_terms < 120)
        or (symbol_ratio > 0.32 and unique_terms < 120)
        or (metadata_ratio > 0.45 and sentence_ratio < 0.25)
        or repeated_line_fraction > 0.70
    )


def _repeated_line_fraction(lines: list[str]) -> float:
    if len(lines) < 4:
        return 0.0
    normalized = [re.sub(r"\s+", " ", line.lower()).strip() for line in lines]
    repeated = len(normalized) - len(set(normalized))
    return repeated / len(normalized)

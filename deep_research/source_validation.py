from __future__ import annotations

import re
from dataclasses import dataclass

from deep_research.schemas import ResearchBranch
from deep_research.text_terms import TOKEN_RE, cjk_char_count, contains_cjk, latin_letter_count, normalize_term_text, ordered_terms, term_set

URL_RE = re.compile(r"https?://\S+", flags=re.I)
KEY_VALUE_LINE_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 _./-]{1,48}:\s+\S+")
SENTENCE_END_RE = re.compile(r"[.!?。！？][\"')\]）】」』”’]?$")
SENTENCE_SPLIT_RE = re.compile(r"\n\s*\n|(?<=[.!?。！？])\s*|[;；]\s*")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


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
    branch_anchor_groups = anchor_groups_for_branch(branch)
    question_anchor_groups = anchor_groups_for_question(question)
    if uses_translated_branch_context(question=question, branch=branch, title=title, content=normalized):
        question_terms = set()
        question_anchor_groups = []
    branch_anchor_matches = _matched_anchor_groups(branch_anchor_groups, content_terms)
    question_anchor_matches = _matched_anchor_groups(question_anchor_groups, content_terms)
    partial_question_anchor_collisions = _partial_anchor_collisions(question_anchor_groups, content_terms, question_anchor_matches)
    matched_terms = _matched_terms(terms, content_terms)
    branch_semantic_match = _has_strong_branch_semantic_match(
        matched_terms=matched_terms,
        branch_terms=terms,
        question_terms=question_terms,
        content_terms=content_terms,
    )
    if question_terms:
        question_matches = _matched_terms(question_terms, content_terms)
        if len(question_matches) < min(2, len(question_terms)):
            reasons.append("source relevance to the original question below threshold")
        elif question_anchor_groups and not question_anchor_matches and len(question_matches) / max(len(question_terms), 1) < 0.45:
            reasons.append("source lacks a question-specific anchor phrase")
        elif partial_question_anchor_collisions and len(question_anchor_matches) < 2:
            reasons.append("source partially matches a question anchor without the complete phrase")
    if branch_anchor_groups and not branch_anchor_matches and not branch_semantic_match:
        reasons.append("source lacks a branch-specific anchor phrase")
    reasons.extend(_concept_dominance_rejections(title=title, content=normalized, branch=branch, question=question))
    term_score = len(matched_terms) / max(len(terms), 1)
    anchor_score = len(branch_anchor_matches) / max(len(branch_anchor_groups), 1) if branch_anchor_groups else term_score
    relevance_score = round((term_score * 0.70) + (anchor_score * 0.30), 4)
    if branch_semantic_match:
        relevance_score = max(relevance_score, 0.30)
    if relevance_score < 0.30:
        reasons.append("branch relevance below threshold")

    relevant_chunks = [
        chunk
        for chunk in _chunks(normalized)
        if len(_matched_terms(terms, _tokens(chunk))) >= max(1, min(3, len(terms)))
        and (not branch_anchor_groups or _matched_anchor_groups(branch_anchor_groups, _tokens(chunk)))
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


def anchor_groups_for_branch(branch: ResearchBranch) -> list[frozenset[str]]:
    groups: list[frozenset[str]] = []
    for text in [branch.title, *branch.required_terms]:
        groups.extend(_anchor_groups_from_text(text))
    if not groups:
        groups.extend(_anchor_groups_from_text(branch.objective))
    return _dedupe_anchor_groups(groups)


def anchor_groups_for_question(question: str) -> list[frozenset[str]]:
    return _dedupe_anchor_groups(_anchor_groups_from_text(question))


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


def uses_translated_branch_context(*, question: str, branch: ResearchBranch, title: str, content: str) -> bool:
    if not question.strip():
        return False
    branch_text = " ".join(
        [
            branch.title,
            branch.objective,
            " ".join(branch.queries),
            " ".join(branch.required_terms),
        ]
    )
    question_language = _dominant_validation_language(question)
    branch_language = _dominant_validation_language(branch_text)
    source_language = _dominant_validation_language(f"{title} {content[:6000]}")
    return question_language != source_language and branch_language == source_language


def _uses_translated_branch_context(*, question: str, branch: ResearchBranch, title: str, content: str) -> bool:
    return uses_translated_branch_context(question=question, branch=branch, title=title, content=content)


def _dominant_validation_language(text: str) -> str:
    cjk_count = cjk_char_count(text)
    latin_count = latin_letter_count(text)
    if cjk_count >= 4 and cjk_count > latin_count:
        return "zh"
    return "en"


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


def _matched_anchor_groups(
    anchor_groups: list[frozenset[str]],
    observed_terms: set[str],
) -> list[frozenset[str]]:
    return [
        group
        for group in anchor_groups
        if group and len(_matched_terms(set(group), observed_terms)) == len(group)
    ]


def _partial_anchor_collisions(
    anchor_groups: list[frozenset[str]],
    observed_terms: set[str],
    matched_groups: list[frozenset[str]],
) -> list[frozenset[str]]:
    collisions: list[frozenset[str]] = []
    fully_covered_terms = set().union(*(set(group) for group in matched_groups)) if matched_groups else set()
    for group in anchor_groups:
        if len(group) < 2:
            continue
        matched = _matched_terms(set(group), observed_terms)
        if matched and len(matched) < len(group):
            if matched <= fully_covered_terms:
                continue
            collisions.append(group)
    return collisions


def _has_strong_branch_semantic_match(
    *,
    matched_terms: set[str],
    branch_terms: set[str],
    question_terms: set[str],
    content_terms: set[str],
) -> bool:
    if not matched_terms:
        return False
    if question_terms and len(_matched_terms(question_terms, content_terms)) < min(2, len(question_terms)):
        return False
    absolute_hits = len(matched_terms)
    relative_hits = absolute_hits / max(len(branch_terms), 1)
    return absolute_hits >= 6 or (absolute_hits >= 4 and relative_hits >= 0.28)


def _anchor_groups_from_text(text: str) -> list[frozenset[str]]:
    terms = ordered_terms(text)
    groups: list[frozenset[str]] = []
    for size in (3, 2):
        if len(terms) < size:
            continue
        for index in range(0, len(terms) - size + 1):
            group = frozenset(terms[index : index + size])
            if len(group) == size:
                groups.append(group)
    return groups


def _dedupe_anchor_groups(groups: list[frozenset[str]]) -> list[frozenset[str]]:
    seen: set[frozenset[str]] = set()
    result: list[frozenset[str]] = []
    for group in groups:
        if not group or group in seen:
            continue
        seen.add(group)
        result.append(group)
    return result[:24]


def _concept_dominance_rejections(
    *,
    title: str,
    content: str,
    branch: ResearchBranch,
    question: str,
) -> list[str]:
    if contains_cjk(f"{title} {content} {branch.title} {question}"):
        return []
    protected_phrases = _protected_concept_phrases(branch, question)
    if not protected_phrases:
        return []
    source_text = f"{title}\n{content}"
    title_core_terms = _title_core_terms(title)
    protected_phrase_sets = {frozenset(phrase) for phrase in protected_phrases}
    reasons: list[str] = []
    for phrase_terms in protected_phrases:
        target_count = _phrase_count(phrase_terms, source_text)
        competing_count = _competing_title_phrase_count(
            title_core_terms,
            phrase_terms,
            source_text,
            protected_phrase_sets=protected_phrase_sets,
        )
        if _title_competes_with_phrase(title_core_terms, phrase_terms) and (
            target_count <= 1 or competing_count >= max(2, target_count * 1.5)
        ):
            reasons.append(
                "source main topic appears to be a neighboring concept rather than the requested concept"
            )
            break
    acronym_conflict = _acronym_expansion_conflict(source_text=source_text, branch=branch, question=question)
    if acronym_conflict:
        reasons.append("source appears to expand an acronym differently from the requested concept")
    return reasons


def _protected_concept_phrases(branch: ResearchBranch, question: str) -> list[tuple[str, ...]]:
    phrases: list[tuple[str, ...]] = []
    anchor_terms = set(ordered_terms(f"{question} {branch.title}"))
    for text in [question]:
        phrases.extend(_concept_phrases_from_text(text))
    for text in branch.required_terms:
        terms = ordered_terms(text)
        if 2 <= len(terms) <= 5 and set(terms) <= anchor_terms and len(set(terms)) == len(terms):
            phrases.append(tuple(terms))
    return _dedupe_phrases(phrases)[:24]


def _concept_phrases_from_text(text: str) -> list[tuple[str, ...]]:
    phrases: list[tuple[str, ...]] = []
    for segment in _concept_segments(text):
        terms = ordered_terms(segment)
        if len(terms) == 2 and len(set(terms)) == 2:
            phrases.append(tuple(terms))
            continue
        for index in range(0, len(terms) - 1):
            window = tuple(terms[index : index + 2])
            if len(set(window)) == 2:
                phrases.append(window)
    return phrases


def _concept_segments(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    parts = re.split(
        r"\s*(?:[:;|/(){}\[\]]| - |\u2013|\u2014|,|\b(?:and|or|of|on|to|between|with)\b)\s*",
        normalized,
        flags=re.I,
    )
    return [part.strip(" .:-") for part in parts if len(ordered_terms(part)) >= 2]


def _dedupe_phrases(phrases: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for phrase in phrases:
        if phrase in seen:
            continue
        seen.add(phrase)
        result.append(phrase)
    return result


def _title_competes_with_phrase(title_terms: list[str], phrase_terms: tuple[str, ...]) -> bool:
    if len(phrase_terms) < 2 or len(title_terms) < len(phrase_terms):
        return False
    title_set = set(title_terms)
    phrase_set = set(phrase_terms)
    if phrase_set <= title_set:
        return False
    overlap = len(title_set & phrase_set)
    if overlap <= 0:
        return False
    title_extra = title_set - phrase_set
    phrase_missing = phrase_set - title_set
    if not title_extra or not phrase_missing:
        return False
    if len(phrase_terms) == 2:
        return overlap == 1
    return overlap >= len(phrase_terms) - 1


def _competing_title_phrase_count(
    title_terms: list[str],
    phrase_terms: tuple[str, ...],
    source_text: str,
    *,
    protected_phrase_sets: set[frozenset[str]] | None = None,
) -> int:
    if len(phrase_terms) < 2:
        return 0
    phrase_set = set(phrase_terms)
    protected_phrase_sets = protected_phrase_sets or set()
    best = 0
    for index in range(0, max(0, len(title_terms) - len(phrase_terms) + 1)):
        window = tuple(title_terms[index : index + len(phrase_terms)])
        window_set = set(window)
        if window_set == phrase_set:
            continue
        if frozenset(window_set) in protected_phrase_sets:
            continue
        if len(window_set & phrase_set) <= 0:
            continue
        if not (window_set - phrase_set):
            continue
        best = max(best, _phrase_count(window, source_text))
    return best


def _title_core_terms(title: str) -> list[str]:
    core = re.split(r"\s[-|:]\s| - |\|", title, maxsplit=1)[0]
    return ordered_terms(core)


def _acronym_expansion_conflict(*, source_text: str, branch: ResearchBranch, question: str) -> bool:
    planning_text = " ".join(
        [
            question,
            branch.title,
            branch.objective,
            " ".join(branch.queries),
            " ".join(branch.required_terms),
        ]
    )
    acronyms = sorted(set(ACRONYM_RE.findall(planning_text)))
    if not acronyms:
        return False
    source_intro = source_text[:5000]
    for acronym in acronyms:
        target_expansions = _expansions_for_acronym(acronym, planning_text)
        if not target_expansions:
            continue
        source_expansions = _expansions_for_acronym(acronym, source_intro)
        if not source_expansions:
            continue
        target_sets = {tuple(ordered_terms(expansion)) for expansion in target_expansions}
        for expansion in source_expansions:
            expansion_terms = tuple(ordered_terms(expansion))
            if not expansion_terms or expansion_terms in target_sets:
                continue
            if any(set(target_terms) <= set(expansion_terms) for target_terms in target_sets):
                continue
            target_count = max(_phrase_count(target_terms, source_text) for target_terms in target_sets)
            source_count = _phrase_count(expansion_terms, source_text)
            if target_count <= 1 and source_count >= 2:
                return True
    return False


def _expansions_for_acronym(acronym: str, text: str) -> list[str]:
    words = [word for word in WORD_RE.findall(text) if word]
    expansions: list[str] = []
    target = acronym.lower()
    max_len = min(6, len(target) + 2)
    for start in range(0, len(words)):
        for size in range(2, max_len + 1):
            window = words[start : start + size]
            if len(window) != size:
                continue
            initials = "".join(word[0].lower() for word in window)
            if initials == target:
                expansions.append(" ".join(window))
    return _dedupe_text(expansions)[:12]


def _phrase_count(phrase_terms: tuple[str, ...], text: str) -> int:
    if not phrase_terms:
        return 0
    if len(phrase_terms) == 1:
        return len(re.findall(rf"\b{re.escape(phrase_terms[0])}\b", normalize_term_text(text)))
    gap = r"(?:\W+[a-z0-9-]+){0,3}\W+"
    pattern = r"\b" + gap.join(re.escape(term) for term in phrase_terms) + r"\b"
    return len(re.findall(pattern, normalize_term_text(text), flags=re.I))


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", value.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


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
    for block in SENTENCE_SPLIT_RE.split(text):
        chunk = block.strip()
        if len(chunk) >= 80 or len(TOKEN_RE.findall(normalize_term_text(chunk))) >= 30:
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

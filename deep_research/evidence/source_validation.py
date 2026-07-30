from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from deep_research.core.schemas import ResearchBranch
from deep_research.evidence.text_terms import TOKEN_RE, cjk_char_count, contains_cjk, latin_letter_count, normalize_term_text, ordered_terms, term_set

URL_RE = re.compile(r"https?://\S+", flags=re.I)
KEY_VALUE_LINE_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 _./-]{1,48}:\s+\S+")
SENTENCE_END_RE = re.compile(r"[.!?。！？][\"')\]）】」』”’]?$")
SENTENCE_SPLIT_RE = re.compile(r"\n\s*\n|(?<=[.!?。！？])\s*|[;；]\s*")
ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,5}\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
TECHNICAL_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:kw|hp|rpm|min-?1|nm|mm|cm|m|in\.?|inch(?:es)?|kg|lb|lbs|v|hz|"
    r"bar|psi|mpa|µm|um|micron|%|°c|deg(?:ree)?s?)\b",
    flags=re.I,
)
MODEL_TOKEN_RE = re.compile(r"\b[A-Z]{1,8}[A-Z0-9-]*\s?\d{2,5}[A-Z0-9-]*\b")
PRODUCT_SPEC_INTENT_RE = re.compile(
    r"\b(?:compare|comparison|choose|selection|select|between|versus|vs\.?|specs?|specifications?|"
    r"features?|compatibility|integration|costs?|pricing|vendor|manufacturer|model|models|hardware|"
    r"software|machine|equipment|device|datasheet|data\s+sheet|manual|brochure|technical\s+data)\b",
    flags=re.I,
)
TECHNICAL_FIELD_RE = re.compile(
    r"\b(?:spindle|torque|power|speed|rpm|axis|axes|stroke|capacity|diameter|length|width|height|"
    r"weight|tolerance|accuracy|thermal|control|software|integration|protocol|api|connectivity|"
    r"voltage|frequency|consumption|tool|tooling|material|coolant|standard|certification|"
    r"specification|datasheet|manual|brochure|model)\b",
    flags=re.I,
)
PRODUCT_SPEC_SOURCE_TYPES = frozenset(
    {
        "product_page",
        "vendor_page",
        "spec_sheet",
        "brochure_pdf",
        "manual_pdf",
        "datasheet",
        "official_docs",
        "standards_or_government",
    }
)
PRODUCT_SPEC_MIN_WORDS = 40
ValidationPolicy = Literal["default", "product_spec"]


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
    source_type: str = "",
    url: str = "",
    extraction_method: str = "",
) -> SourceValidation:
    normalized = _normalize(content)
    words = TOKEN_RE.findall(normalize_term_text(normalized))
    word_count = len(words)
    reasons: list[str] = []
    policy = validation_policy_for_source(
        question=question,
        branch=branch,
        title=title,
        content=content,
        source_type=source_type,
        url=url,
        extraction_method=extraction_method,
    )
    product_spec_candidate = policy == "product_spec"
    has_product_spec_evidence = _has_product_spec_evidence(
        question=question,
        branch=branch,
        title=title,
        content=content,
    )
    effective_min_words = PRODUCT_SPEC_MIN_WORDS if product_spec_candidate and has_product_spec_evidence else min_words
    if word_count < effective_min_words:
        reasons.append(f"short extracted text: {word_count} words < {effective_min_words}")

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
        elif (
            not product_spec_candidate
            and question_anchor_groups
            and not question_anchor_matches
            and len(question_matches) / max(len(question_terms), 1) < 0.45
        ):
            reasons.append("source lacks a question-specific anchor phrase")
        elif not product_spec_candidate and partial_question_anchor_collisions and len(question_anchor_matches) < 2:
            reasons.append("source partially matches a question anchor without the complete phrase")
    if branch_anchor_groups and not branch_anchor_matches and not branch_semantic_match and not product_spec_candidate:
        reasons.append("source lacks a branch-specific anchor phrase")
    reasons.extend(_concept_dominance_rejections(
        title=title, content=normalized, branch=branch, question=question,
        branch_semantic_match=branch_semantic_match,
    ))
    term_score = len(matched_terms) / max(len(terms), 1)
    anchor_score = len(branch_anchor_matches) / max(len(branch_anchor_groups), 1) if branch_anchor_groups else term_score
    relevance_score = round((term_score * 0.70) + (anchor_score * 0.30), 4)
    if branch_semantic_match:
        relevance_score = max(relevance_score, 0.30)
    if product_spec_candidate and has_product_spec_evidence:
        relevance_score = max(relevance_score, 0.34)
    if relevance_score < 0.30:
        reasons.append("branch relevance below threshold")

    relevant_chunks = []
    for chunk in _chunks(normalized):
        chunk_terms = _tokens(chunk)
        default_match = len(_matched_terms(terms, chunk_terms)) >= max(1, min(3, len(terms))) and (
            not branch_anchor_groups or _matched_anchor_groups(branch_anchor_groups, chunk_terms)
        )
        product_spec_match = product_spec_candidate and _chunk_has_product_spec_evidence(
            question=question,
            branch=branch,
            title=title,
            chunk=chunk,
        )
        if default_match or product_spec_match:
            relevant_chunks.append(chunk)
    if len(relevant_chunks) < min_relevant_chunks:
        reasons.append(
            f"only {len(relevant_chunks)} relevant chunk(s); need {min_relevant_chunks}"
        )

    if product_spec_candidate and not has_product_spec_evidence:
        reasons.append("product/spec source lacks matching entity and technical field evidence")

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


def validation_policy_for_source(
    *,
    question: str,
    branch: ResearchBranch,
    title: str,
    content: str,
    source_type: str = "",
    url: str = "",
    extraction_method: str = "",
) -> ValidationPolicy:
    task_text = " ".join(
        [
            question,
            branch.title,
            branch.objective,
            " ".join(branch.queries),
            " ".join(branch.required_terms),
        ]
    )
    if not PRODUCT_SPEC_INTENT_RE.search(task_text):
        return "default"
    source_marker = " ".join([source_type, url, extraction_method, title]).lower()
    typed_source = source_type in PRODUCT_SPEC_SOURCE_TYPES or bool(
        re.search(r"\b(?:pdf|product|products|machine|equipment|spec|specification|datasheet|manual|brochure)\b", source_marker)
    )
    if not typed_source:
        return "default"
    return "product_spec"


def _has_product_spec_evidence(
    *,
    question: str,
    branch: ResearchBranch,
    title: str,
    content: str,
) -> bool:
    source_text = f"{title}\n{content}"
    source_terms = _tokens(source_text)
    task_text = " ".join(
        [
            question,
            branch.title,
            branch.objective,
            " ".join(branch.queries),
            " ".join(branch.required_terms),
        ]
    )
    task_terms = _tokens(task_text)
    entity_hits = _entity_overlap(task_text, source_text)
    branch_hits = len(_matched_terms(_branch_terms(branch), source_terms))
    question_hits = len(_matched_terms(task_terms, source_terms))
    technical_score = _technical_evidence_score(source_text)
    return technical_score >= 3 and (
        entity_hits >= 1
        or branch_hits >= 4
        or question_hits >= max(4, min(10, len(task_terms) // 6))
    )


def _chunk_has_product_spec_evidence(
    *,
    question: str,
    branch: ResearchBranch,
    title: str,
    chunk: str,
) -> bool:
    source_text = f"{title}\n{chunk}"
    task_text = " ".join(
        [
            question,
            branch.title,
            branch.objective,
            " ".join(branch.queries),
            " ".join(branch.required_terms),
        ]
    )
    source_terms = _tokens(source_text)
    branch_hits = len(_matched_terms(_branch_terms(branch), source_terms))
    return _technical_evidence_score(source_text) >= 2 and (
        _entity_overlap(task_text, source_text) >= 1 or branch_hits >= 3
    )


def _technical_evidence_score(text: str) -> int:
    key_value_count = len(re.findall(r"(?m)^\s*[A-Za-z][A-Za-z0-9 /().+-]{1,48}\s*[:|]\s*\S+", text))
    bullet_count = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S+", text))
    unit_count = len(TECHNICAL_UNIT_RE.findall(text))
    model_count = len(MODEL_TOKEN_RE.findall(text))
    field_count = len(TECHNICAL_FIELD_RE.findall(text))
    table_count = sum(1 for line in text.splitlines() if line.count("|") >= 2 or len(re.split(r"\s{2,}", line.strip())) >= 3)
    return (
        min(key_value_count, 4)
        + min(bullet_count, 4)
        + min(unit_count, 6)
        + min(model_count, 4)
        + min(field_count, 6)
        + min(table_count, 4)
    )


def _entity_overlap(task_text: str, source_text: str) -> int:
    task_entities = _entity_markers(task_text)
    source_entities = _entity_markers(source_text)
    if not task_entities or not source_entities:
        return 0
    normalized_sources = {_normalize_entity(entity) for entity in source_entities}
    hits = 0
    for entity in task_entities:
        normalized = _normalize_entity(entity)
        if normalized in normalized_sources or any(
            normalized and (normalized in source_entity or source_entity in normalized)
            for source_entity in normalized_sources
        ):
            hits += 1
    return hits


def _entity_markers(text: str) -> set[str]:
    markers = set(MODEL_TOKEN_RE.findall(text))
    # Capture mixed alphanumeric product/entity phrases without binding to any
    # specific vendor or domain.
    phrase_re = re.compile(r"\b(?:[A-Z][A-Za-z0-9+-]{1,12}\s+){0,3}[A-Z0-9][A-Za-z0-9+-]*\d[A-Za-z0-9+-]*\b")
    markers.update(match.group(0).strip() for match in phrase_re.finditer(text))
    return {marker for marker in markers if len(_normalize_entity(marker)) >= 3}


def _normalize_entity(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


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
    branch_semantic_match: bool = False,
) -> list[str]:
    if contains_cjk(f"{title} {content} {branch.title} {question}"):
        return []
    # If the source already proved itself by content (strong branch term coverage),
    # skip the title-based dominance check — it would only produce false positives.
    if branch_semantic_match:
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
        # Reject only when the target concept is completely absent (not merely sparse)
        # and the competing concept dominates strongly — 2.5× ratio with a minimum of 4.
        if _title_competes_with_phrase(title_core_terms, phrase_terms) and (
            target_count == 0 or competing_count >= max(4, target_count * 2.5)
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
    # Derive protected phrases from branch.required_terms ONLY — not from the
    # question text.  The question is long and unstructured; sliding a bigram window
    # over it produces sentence-structure artifacts like ('transportation', 'based')
    # or ('report', 'elderly') that are not meaningful concept discriminators and
    # cause false rejections on any source whose title shares a single generic word
    # with those noise phrases.  branch.required_terms are the planner's explicit,
    # curated list of concepts a source must address — they are the right source.
    phrases: list[tuple[str, ...]] = []
    for text in branch.required_terms:
        terms = ordered_terms(text)
        if 2 <= len(terms) <= 5 and len(set(terms)) == len(terms):
            phrases.append(tuple(terms))
        elif len(terms) == 1:
            # single-word required term: pair it with the branch title's first
            # distinctive word so we still get a 2-term protected phrase
            title_terms = [t for t in ordered_terms(branch.title) if t not in terms and len(t) > 3]
            if title_terms:
                phrases.append((terms[0], title_terms[0]))
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

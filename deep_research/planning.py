from __future__ import annotations

import re
from dataclasses import replace

from deep_research.schemas import ResearchBranch, ResearchIntent, ResearchPlan, SourceRequirement
from deep_research.source_limits import MINIMUM_SOURCE_TARGET
from deep_research.text_terms import cjk_content_chunks, contains_cjk, ordered_terms

MAX_DYNAMIC_BRANCHES = 14
MAX_BRANCH_QUERIES = 4
SEGMENT_TRIGGER_RE = re.compile(
    r"\b(?:address|analyze|assess|compare|cover|describe|evaluate|explain|explore|find|include|investigate|research|review|summarize)\b",
    flags=re.I,
)

def build_research_plan(question: str, *, audience: str = "technical generalist") -> ResearchPlan:
    intent: ResearchIntent = "general"
    branches = _branches_for_question(question)
    outline = [branch.title for branch in branches]
    return ResearchPlan(
        question=question.strip(),
        intent=intent,
        audience=audience,
        report_outline=outline,
        branches=branches,
        source_requirements=_source_requirements(),
        acceptance_criteria=[
            "Every branch has evidence cards from usable sources.",
            "Every factual paragraph has inline citations.",
            "The report answers the question directly before deep detail.",
            "Verification passes citation, evidence, coverage, and structure gates.",
        ],
    )


def _branches_for_question(question: str) -> list[ResearchBranch]:
    topic = _topic_label(question)
    seeds = _branch_seeds(question, topic)
    branches: list[ResearchBranch] = []
    for index, seed in enumerate(seeds, start=1):
        if isinstance(seed, dict):
            title = str(seed["title"])
            objective = str(seed["objective"])
            queries = list(seed["queries"])
            required_terms = list(seed["required_terms"])
        else:
            title = _title_from_seed(seed)
            objective = _objective_for_seed(seed, question)
            queries = _queries_for_seed(seed, topic, question, is_primary=index == 1)
            required_terms = _required_terms_for_seed(seed, topic)
        branches.append(
            _branch(
                f"branch_{index}",
                title,
                objective,
                queries,
                required_terms,
            )
        )
    return _raise_branch_source_floor(branches, MINIMUM_SOURCE_TARGET)


def _branch(
    branch_id: str,
    title: str,
    objective: str,
    queries: list[str],
    required_terms: list[str],
    *,
    min_sources: int = 1,
) -> ResearchBranch:
    return ResearchBranch(
        id=branch_id,
        title=title,
        objective=objective,
        queries=queries,
        source_types=["academic", "official_docs", "standards_or_government", "general_web"],
        min_sources=min_sources,
        required_terms=required_terms,
        completion_criteria=[
            "At least the minimum source count is usable.",
            "Evidence cards cover the branch-specific required terms.",
        ],
    )


def _source_requirements() -> list[SourceRequirement]:
    return [
        SourceRequirement("academic", min_count=1, rationale="Anchor technical claims in research where available."),
        SourceRequirement("official_docs", min_count=1, rationale="Prefer primary documentation for implementation details."),
        SourceRequirement("recent_or_contextual", min_count=1, rationale="Capture current context when recency matters."),
    ]


def _raise_branch_source_floor(branches: list[ResearchBranch], target: int) -> list[ResearchBranch]:
    total = sum(branch.min_sources for branch in branches)
    if total >= target or not branches:
        return branches

    raised = list(branches)
    index = 0
    while total < target:
        branch = raised[index % len(raised)]
        raised[index % len(raised)] = replace(branch, min_sources=branch.min_sources + 1)
        total += 1
        index += 1
    return raised


def _core_terms(question: str) -> list[str]:
    return ordered_terms(question)[:8] or ["research"]


def _branch_seeds(question: str, topic: str) -> list[str | dict[str, object]]:
    explicit_segments = _explicit_request_segments(question)
    if contains_cjk(question):
        candidates = _dedupe(explicit_segments + [topic, _trim_query(question, limit=220)])
        candidates = [candidate for candidate in candidates if candidate.strip()]
        target = _target_branch_count(question, explicit_segments)
        selected = _select_distinct_seeds(candidates, target)
        return selected or [topic]

    terms = ordered_terms(question)
    keyphrases = _keyphrases(question)
    term_windows = _term_windows(terms, size=3) + _term_windows(terms, size=2)

    candidates = _dedupe(explicit_segments + keyphrases + term_windows + [topic, _trim_query(question, limit=160)])
    candidates = [candidate for candidate in candidates if len(ordered_terms(candidate)) >= 1]

    target = _target_branch_count(question, explicit_segments)
    selected = _select_distinct_seeds(candidates, target)
    return selected or [topic]


def _target_branch_count(question: str, explicit_segments: list[str]) -> int:
    if contains_cjk(question):
        if explicit_segments:
            return min(MAX_DYNAMIC_BRANCHES, max(1, len(explicit_segments)))
        length = len(re.sub(r"\s+", "", question))
        if length <= 80:
            return 1
        if length <= 180:
            return 2
        return min(MAX_DYNAMIC_BRANCHES, 4)

    term_count = len(ordered_terms(question))
    if explicit_segments:
        return min(MAX_DYNAMIC_BRANCHES, max(2, len(explicit_segments) + min(4, term_count // 12)))
    if term_count <= 5:
        return 2
    if term_count <= 10:
        return 3
    if term_count <= 18:
        return 4
    return min(MAX_DYNAMIC_BRANCHES, 4 + min(8, term_count // 12))


def _explicit_request_segments(question: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", question.strip())
    sentence_parts = re.split(r"(?<=[.!?。！？])\s*|[;；\n]+", normalized)
    segments: list[str] = []
    for sentence in sentence_parts:
        sentence = sentence.strip(" ,.:，。：")
        if not sentence:
            continue
        if contains_cjk(sentence):
            segments.extend(_cjk_request_segments(sentence))
            continue
        candidate = sentence
        trigger = SEGMENT_TRIGGER_RE.search(sentence)
        if trigger:
            candidate = sentence[trigger.end() :].strip(" :,")
        if "," in candidate or re.search(r"\band\b", candidate, flags=re.I):
            for part in re.split(r",|\band\b", candidate, flags=re.I):
                cleaned = part.strip(" ,.:")
                if len(ordered_terms(cleaned)) >= 2:
                    segments.append(cleaned)
            continue
        if len(ordered_terms(candidate)) >= 3:
            segments.append(candidate)
    return _dedupe(segments)


def _cjk_request_segments(sentence: str) -> list[str]:
    parts = [part.strip(" ,.:，。：、“”\"'()（）") for part in re.split(r"[，,、:：]+", sentence)]
    result: list[str] = []
    for part in parts:
        if not part or _looks_like_cjk_meta_fragment(part):
            continue
        if not cjk_content_chunks(part):
            continue
        compact = re.sub(r"\s+", "", part)
        if len(compact) < 4:
            continue
        if len(ordered_terms(part)) < 1:
            continue
        result.append(part)
    if result:
        return _dedupe(result)
    compact_sentence = re.sub(r"\s+", "", sentence)
    if len(compact_sentence) >= 4 and not _looks_like_cjk_meta_fragment(compact_sentence):
        return [sentence]
    return []


def _looks_like_cjk_meta_fragment(text: str) -> bool:
    normalized = re.sub(r"\s+", "", text)
    return bool(re.search(r"(为什么)?这个问题有价值|问题有价值|未来发展预测", normalized))


def _keyphrases(question: str) -> list[str]:
    try:
        import yake
    except ImportError:
        return _fallback_keyphrases(question)

    try:
        extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.85, top=16)
        extracted = [keyword for keyword, _score in extractor.extract_keywords(question)]
    except Exception:
        return _fallback_keyphrases(question)
    return _dedupe([phrase for phrase in extracted if len(ordered_terms(phrase)) >= 1])


def _fallback_keyphrases(question: str) -> list[str]:
    terms = ordered_terms(question)
    return _dedupe(_term_windows(terms, size=3) + _term_windows(terms, size=2) + terms[:8])


def _term_windows(terms: list[str], *, size: int) -> list[str]:
    if len(terms) < size:
        return []
    return [" ".join(terms[index : index + size]) for index in range(0, len(terms) - size + 1)]


def _select_distinct_seeds(candidates: list[str], target: int) -> list[str]:
    selected: list[str] = []
    deferred: list[tuple[str, set[str]]] = []
    for candidate in candidates:
        terms = set(ordered_terms(candidate))
        if not terms:
            continue
        if any(_too_similar(terms, set(ordered_terms(existing))) for existing in selected):
            deferred.append((candidate, terms))
            continue
        selected.append(candidate)
        if len(selected) >= target:
            break
    covered_terms = set().union(*(set(ordered_terms(seed)) for seed in selected)) if selected else set()
    for candidate, terms in deferred:
        if len(selected) >= target:
            break
        if terms <= covered_terms and selected:
            continue
        selected.append(candidate)
        covered_terms |= terms
    return selected


def _too_similar(left: set[str], right: set[str]) -> bool:
    if not left or not right:
        return False
    if left <= right or right <= left:
        return True
    similarity = len(left & right) / max(len(left | right), 1)
    return similarity >= 0.78


def _title_from_seed(seed: str) -> str:
    title = re.sub(r"\s+", " ", seed.strip(" .,:;!?")).strip()
    if not title:
        return "Research Angle"
    title = _trim_query(title, limit=78)
    return title[0].upper() + title[1:]


def _objective_for_seed(seed: str, question: str) -> str:
    return (
        f"Research this aspect of the request: {seed}. "
        f"Keep findings grounded in the full user question: {_trim_query(question, limit=260)}"
    )


def _queries_for_seed(seed: str, topic: str, question: str, *, is_primary: bool) -> list[str]:
    queries = [seed]
    if topic.lower() not in seed.lower():
        queries.append(f"{topic} {seed}")
    if is_primary:
        queries.append(_trim_query(question))
    return _queries(*queries[:MAX_BRANCH_QUERIES])


def _required_terms_for_seed(seed: str, topic: str) -> list[str]:
    return _dedupe(ordered_terms(seed)[:6] + ordered_terms(topic)[:4])[:8]


def _topic_label(question: str) -> str:
    normalized = re.sub(r"\s+", " ", question.strip(" ?.!")).strip()
    if contains_cjk(normalized):
        return _trim_query(normalized, limit=180)
    terms = _core_terms(normalized)
    if terms:
        return " ".join(terms[:12])
    first_sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    if 20 <= len(first_sentence) <= 140:
        return first_sentence
    return _trim_query(question, limit=140)


def _trim_query(question: str, *, limit: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", question.strip()).strip()
    if len(normalized) <= limit:
        return normalized
    boundary = normalized.rfind(" ", 0, limit)
    return normalized[: boundary if boundary > 80 else limit].strip()


def _queries(*queries: str) -> list[str]:
    return _dedupe([query.strip() for query in queries if query and query.strip()])


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result

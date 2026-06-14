from __future__ import annotations

from collections import defaultdict
import json
import re
from typing import Any

from langchain_core.language_models import BaseChatModel

from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchPlan, ResearchBranch, SourceRecordV2
from deep_research.settings import Settings
from deep_research.source_validation import content_terms
from deep_research.synthesis_formatting import _coverage_repair_labels, _criteria_rich_plan, _report_subject
from deep_research.synthesis_runtime import _invoke_with_synthesis_budget
from deep_research.synthesis_selection import (
    _cards_by_branch,
    _cards_for_synthesis,
    _rank_cards,
    _source_diverse_cards,
)

INDIVIDUAL_CITATION_REPAIR_THRESHOLD = 0.30
SPECIFIC_FACT_REPAIR_THRESHOLD = 0.40
_REWRITE_MAX_SENTENCES = 30
_REWRITE_MIN_OVERLAP = 0.42


def _normalize_report_markdown(report: str, sources: list[SourceRecordV2]) -> str:
    cleaned = report.strip()
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = _normalize_markdown_headings(cleaned)
    cleaned = _separate_heading_blocks(cleaned)
    cleaned = _clean_malformed_citation_punctuation(cleaned)
    cleaned = _strip_report_chrome_lines(cleaned)
    cleaned = _remove_existing_source_listing(cleaned)
    cleaned = _strip_source_artifact_lines(cleaned)
    cleaned = _remove_unknown_numeric_citations(cleaned, sources)
    cleaned = _repair_uncited_body_paragraphs(cleaned, sources)
    if not re.search(r"(?im)^#\s+", cleaned):
        cleaned = "# Research Report\n\n" + cleaned
    cleaned = _remove_existing_source_listing(cleaned)
    cleaned = _clean_malformed_citation_punctuation(cleaned)
    cleaned = _strip_report_chrome_lines(cleaned)
    cleaned = _strip_source_artifact_lines(cleaned)
    source_section = _sources_section(sources, cleaned)
    if re.search(r"(?ims)^##\s+Sources\s*$", cleaned):
        cleaned = re.sub(r"(?ims)^##\s+Sources\s*$.*\Z", source_section, cleaned).strip()
    else:
        cleaned = cleaned.rstrip() + "\n\n" + source_section
    return cleaned.rstrip() + "\n"


def _clean_malformed_citation_punctuation(report: str) -> str:
    cleaned = report
    cleaned = re.sub(r"\((\s*\[[0-9][0-9,;\s]*\])\s*;\s*\)", r"(\1)", cleaned)
    cleaned = re.sub(r"\[\s*([0-9][0-9,;\s]*)\s*;\s*]", r"[\1]", cleaned)
    cleaned = re.sub(r"\[\s*([0-9][0-9,;\s]*)\s*,\s*]", r"[\1]", cleaned)
    cleaned = re.sub(r"\[\s*([0-9][0-9,;\s]*)\s+]", r"[\1]", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def _is_degenerate_model_report(report: str, evidence_cards: list[EvidenceCard]) -> bool:
    body, _separator, _source_tail = _split_sources(report)
    body_without_headings = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("#")
    ).strip()
    normalized = body_without_headings.lower().strip(" .`'\"")
    if normalized in {"none", "null", "n/a", "na"}:
        return True
    if evidence_cards and len(content_terms(body_without_headings)) < 20:
        return True
    if evidence_cards and not _numeric_citation_ids(body_without_headings) and len(body_without_headings) < 1200:
        return True
    return False


def _normalize_markdown_headings(report: str) -> str:
    normalized_lines: list[str] = []
    for line in report.splitlines():
        stripped = line.strip()
        bold_match = re.fullmatch(r"\*\*([^*]{2,120})\*\*", stripped)
        if bold_match:
            heading = bold_match.group(1).strip()
            normalized_lines.append(f"## {heading}")
            continue
        heading_match = re.fullmatch(r"(#{1,6})[ \t]+(.+?)\s*", stripped)
        if heading_match:
            level = heading_match.group(1)
            heading = re.sub(r"\*\*", "", heading_match.group(2)).strip()
            normalized_lines.append(f"{level} {heading}")
            continue
        normalized_lines.append(line)
    return "\n".join(normalized_lines)


def _separate_heading_blocks(report: str) -> str:
    return re.sub(r"(?m)^(#{1,6}\s+.+?)\s*\n(?!\s*\n)", r"\1\n\n", report)


def _strip_report_chrome_lines(report: str) -> str:
    cleaned_lines: list[str] = []
    for line in report.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", stripped):
            continue
        if re.fullmatch(r"\*?\s*(?:end of report|end)\s*\*?", stripped, flags=re.I):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _remove_existing_source_listing(report: str) -> str:
    if re.search(r"(?ims)^##\s+Sources\s*$", report):
        return re.sub(r"(?ims)^##\s+Sources\s*$.*\Z", "", report).strip()
    bibliography_heading = _malformed_bibliography_heading_match(report)
    if bibliography_heading is not None:
        return report[: bibliography_heading.start()].strip()

    lines = report.splitlines()
    index = len(lines) - 1
    while index >= 0 and not lines[index].strip():
        index -= 1
    if index < 0:
        return report

    source_line_count = 0
    block_start = index + 1
    while index >= 0:
        stripped = lines[index].strip()
        if not stripped:
            block_start = index
            index -= 1
            continue
        if _looks_like_source_entry(stripped):
            source_line_count += 1
            block_start = index
            index -= 1
            continue
        break

    if source_line_count == 0:
        return report
    if index >= 0 and _looks_like_bibliography_heading(lines[index].strip()):
        block_start = index
    return "\n".join(lines[:block_start]).rstrip()


def _looks_like_source_entry(line: str) -> bool:
    stripped = line.strip()
    if not re.search(r"https?://\S+", stripped, flags=re.I):
        return False
    return bool(
        re.search(r"^\s*\[[0-9]+]\s+.+https?://\S+", stripped, flags=re.I)
        or re.search(r"^\s*(?:[-*]\s*)?[^:]{2,240}:\s+https?://\S+", stripped, flags=re.I)
        or re.search(r"^\s*(?:[-*]\s*)?https?://\S+", stripped, flags=re.I)
    )


def _malformed_bibliography_heading_match(report: str) -> re.Match[str] | None:
    for match in re.finditer(r"(?im)^#{1,6}\s+(.+?)\s*$", report):
        if not _looks_like_bibliography_heading(match.group(0)):
            continue
        tail = report[match.end() :]
        preview_lines = [line.strip() for line in tail.splitlines()[:12] if line.strip()]
        if _looks_like_source_entry(match.group(0)) or any(_looks_like_source_entry(line) for line in preview_lines):
            return match
    return None


def _looks_like_bibliography_heading(line: str) -> bool:
    stripped = re.sub(r"^\s*#{1,6}\s*", "", line.strip())
    stripped = re.sub(r"^\*\*|\*\*$", "", stripped).strip()
    return bool(
        re.match(
            r"^(?:sources|references|bibliography|works cited)\s*(?:$|[:\-]|\[[0-9]+])",
            stripped,
            flags=re.I,
        )
    )


def _strip_source_artifact_lines(report: str) -> str:
    body, separator, source_tail = _split_sources(report)
    cleaned_lines = [
        line
        for line in body.splitlines()
        if not _looks_like_source_artifact_line(line)
    ]
    cleaned_body = "\n".join(cleaned_lines).strip()
    return cleaned_body + (separator + source_tail if separator else "")


def _looks_like_source_artifact_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if _looks_like_bibliography_heading(stripped) and (
        _looks_like_source_entry(stripped) or re.search(r"\[[0-9]+]\s+.+https?://", stripped, flags=re.I)
    ):
        return True
    if _looks_like_source_entry(stripped):
        return True
    if re.search(r"https?://\S+", stripped, flags=re.I):
        word_count = len(re.findall(r"\b[A-Za-z][A-Za-z0-9-]*\b", stripped))
        url_chars = sum(len(match.group(0)) for match in re.finditer(r"https?://\S+", stripped, flags=re.I))
        if word_count <= 24 or url_chars / max(len(stripped), 1) > 0.10:
            return True
    return False


def _remove_unknown_numeric_citations(report: str, sources: list[SourceRecordV2]) -> str:
    allowed = {source.id for source in sources}

    def replace(match: re.Match[str]) -> str:
        kept = [
            int(value)
            for value in re.findall(r"\d+", match.group(1))
            if int(value) in allowed
        ]
        if not kept:
            return ""
        return "[" + ", ".join(str(value) for value in sorted(set(kept))) + "]"

    return re.sub(r"\[([0-9][0-9,;\s]*)\]", replace, report)


def _repair_uncited_body_paragraphs(report: str, sources: list[SourceRecordV2]) -> str:
    body, separator, source_tail = _split_sources(report)
    fallback = _fallback_citations(body, sources)
    if not fallback:
        return report
    repaired: list[str] = []
    for paragraph in re.split(r"(\n\s*\n)", body):
        if not paragraph.strip() or re.fullmatch(r"\n\s*\n", paragraph):
            repaired.append(paragraph)
            continue
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if _paragraph_needs_repair(text):
            paragraph = paragraph.rstrip() + f" {fallback}"
        repaired.append(paragraph)
    return "".join(repaired) + (separator + source_tail if separator else "")


_SPECIFIC_FACT_RE = re.compile(
    r"("
    r"\d{1,3}(?:[,.]\d{3})+"                                  # 1,000 / 1.000.000
    r"|\d+\.\d+"                                              # 3.14
    r"|\d+\s?%"                                               # 25% or 25 %
    r"|\$\s?\d+"                                              # $1000 / $ 1000
    r"|[€£¥]\s?\d+"                                           # €1000
    r"|\b\d{4}\b"                                             # 1999 / 2024 (4-digit years)
    r"|\b\d+\s?(?:billion|million|trillion|bn|mn|tn)\b"       # 3 billion
    r")",
    re.IGNORECASE,
)


def _has_specific_facts(text: str) -> bool:
    """True if the sentence contains a number, money, year, or percentage —
    the kinds of claims that need precise citation backing, not just topical."""
    return bool(_SPECIFIC_FACT_RE.search(text))


def _repair_weak_citation_support(
    report: str,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    *,
    threshold: float = 0.35,
) -> str:
    source_ids = {source.id for source in sources}
    terms_by_source = _evidence_terms_by_source(evidence_cards, source_ids)
    if not terms_by_source:
        return report
    body, separator, source_tail = _split_sources(report)
    repaired: list[str] = []
    for paragraph in re.split(r"(\n\s*\n)", body):
        if not paragraph.strip() or re.fullmatch(r"\n\s*\n", paragraph):
            repaired.append(paragraph)
            continue
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not _paragraph_can_receive_support_repair(text):
            repaired.append(paragraph)
            continue
        claim_terms = content_terms(_strip_numeric_citations(text))
        # Lowered from 6 → 4 so shorter sentences with specific facts also get checked.
        if len(claim_terms) < 4:
            repaired.append(paragraph)
            continue
        cited_ids = {source_id for source_id in _numeric_citation_ids(text) if source_id in source_ids}
        if not cited_ids:
            repaired.append(paragraph)
            continue
        # Apply stricter threshold for paragraphs with specific facts (numbers,
        # money, percentages, years) — these need precise citation grounding.
        has_facts = _has_specific_facts(_strip_numeric_citations(text))
        individual_threshold = SPECIFIC_FACT_REPAIR_THRESHOLD if has_facts else INDIVIDUAL_CITATION_REPAIR_THRESHOLD
        paragraph_threshold = max(threshold, 0.45) if has_facts else threshold
        individual_scores = _individual_source_support_scores(claim_terms, terms_by_source, cited_ids)
        weak_ids = {
            source_id
            for source_id, score in individual_scores.items()
            if score < individual_threshold
        }
        strong_ids = cited_ids - weak_ids
        support_terms = set().union(*(terms_by_source.get(source_id, set()) for source_id in cited_ids))
        support_score = len(claim_terms & support_terms) / max(len(claim_terms), 1)
        if support_score >= paragraph_threshold and not weak_ids:
            repaired.append(paragraph)
            continue
        additions = _best_supporting_source_ids(claim_terms, terms_by_source, cited_ids, paragraph_threshold)
        if weak_ids and (strong_ids or additions):
            paragraph = _remove_numeric_citation_ids(paragraph, weak_ids)
        if additions:
            paragraph = paragraph.rstrip() + " " + " ".join(f"[{source_id}]" for source_id in additions)
        repaired.append(paragraph)
    return "".join(repaired) + (separator + source_tail if separator else "")


# Pattern matches specific facts: numbers, percentages, money, years, units.
# Captures the whole token so we can search for it in cited cards.
_SPECIFIC_FACT_TOKEN_RE = re.compile(
    r"("
    r"\$\s?\d+(?:[.,]\d+)*\s?(?:billion|million|trillion|bn|mn|tn|k|m|b)?"
    r"|[€£¥]\s?\d+(?:[.,]\d+)*"
    r"|\d{1,3}(?:[,.]\d{3})+(?:\.\d+)?"
    r"|\d+\.\d+\s?%?"
    r"|\d+\s?%"
    r"|\b\d{4}\b"
    r"|\b\d+\s?(?:billion|million|trillion|bn|mn|tn)\b"
    r")",
    re.IGNORECASE,
)


def _normalize_fact_token(token: str) -> str:
    """Reduce a specific-fact token to a comparable form so '3.5 billion'
    matches '3.5 bn' and '$ 1,000' matches '$1000'. Lowercase, no spaces,
    money symbols dropped, commas dropped, common unit abbreviations expanded."""
    t = token.strip().lower()
    t = t.replace("$", "").replace("€", "").replace("£", "").replace("¥", "")
    t = t.replace(",", "").replace(" ", "")
    # Unit abbreviation harmonisation.
    replacements = (
        ("billion", "bn"),
        ("million", "mn"),
        ("trillion", "tn"),
    )
    for full, short in replacements:
        t = t.replace(full, short)
    return t


def _strip_hallucinated_specific_citations(
    report: str,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
) -> str:
    """Post-synthesis fact-check.

    Strip a [N] citation ONLY when both conditions hold:
      (a) the sentence's specific facts (numbers, $, %, years) do not appear
          in source N's evidence cards, AND
      (b) the cited card from source N also does NOT meaningfully support the
          general claim of the sentence (term overlap < SUPPORT_FLOOR).

    Earlier we stripped on (a) alone, which was too aggressive: a sentence
    like "Square offers a tablet POS in 2020 [3]" was losing [3] because the
    card didn't say "2020", even though the card fully supported "Square
    offers a tablet POS". The conjunction with (b) makes the heuristic safe:
    we only strip when the source supports NEITHER the general claim NOR
    the specific fact.
    """
    SUPPORT_FLOOR = 0.20  # term overlap below this counts as 'not supporting general claim'

    source_ids = {source.id for source in sources}
    if not source_ids:
        return report

    # Per-source pool of normalised fact tokens from card claims/excerpts.
    tokens_by_source: dict[int, set[str]] = defaultdict(set)
    # Per-source pool of content terms for general-claim overlap check.
    terms_by_source = _evidence_terms_by_source(evidence_cards, source_ids)
    for card in evidence_cards:
        if card.source_id not in source_ids:
            continue
        blob = f"{card.claim} {card.supporting_excerpt}"
        for match in _SPECIFIC_FACT_TOKEN_RE.finditer(blob):
            tokens_by_source[card.source_id].add(_normalize_fact_token(match.group(1)))

    body, separator, source_tail = _split_sources(report)

    def _scrub_sentence(sentence: str) -> str:
        cited_ids = [int(m) for m in re.findall(r"\[(\d+)\]", sentence) if int(m) in source_ids]
        if not cited_ids:
            return sentence
        text_no_cites = re.sub(r"\s*\[\d+\]", "", sentence)
        fact_tokens = [
            _normalize_fact_token(m.group(1))
            for m in _SPECIFIC_FACT_TOKEN_RE.finditer(text_no_cites)
        ]
        fact_tokens = [t for t in fact_tokens if len(t) >= 2 and t not in {"100", "10", "0", "1"}]
        if not fact_tokens:
            return sentence
        claim_terms = content_terms(text_no_cites)
        if len(claim_terms) < 4:
            return sentence
        for source_id in set(cited_ids):
            supporting_tokens = tokens_by_source.get(source_id, set())
            specific_matched = any(token in supporting_tokens for token in fact_tokens)
            if specific_matched:
                continue  # cited source contains the specific fact — keep citation
            # No specific match. Check if the source supports the general claim.
            source_terms = terms_by_source.get(source_id, set())
            general_overlap = (
                len(claim_terms & source_terms) / max(len(claim_terms), 1)
                if source_terms
                else 0.0
            )
            if general_overlap < SUPPORT_FLOOR:
                # Neither specific facts nor general claim are supported —
                # safe to strip this misleading citation.
                sentence = re.sub(rf"\s*\[{source_id}\]", "", sentence)
        return sentence

    repaired_parts: list[str] = []
    for paragraph in re.split(r"(\n\s*\n)", body):
        if not paragraph.strip() or re.fullmatch(r"\n\s*\n", paragraph):
            repaired_parts.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])(\s+)", paragraph)
        rebuilt: list[str] = []
        for i, chunk in enumerate(sentences):
            if i % 2 == 0:
                rebuilt.append(_scrub_sentence(chunk))
            else:
                rebuilt.append(chunk)
        repaired_parts.append("".join(rebuilt))
    return "".join(repaired_parts) + (separator + source_tail if separator else "")


_REWRITE_OVERLAP_THRESHOLD = 0.35  # below this, sentence needs rewrite or strip
_REWRITE_MAX_SENTENCES = 30        # cap to keep prompt size reasonable
_REWRITE_MIN_CARDS_PER_SOURCE = 1  # need at least one card with excerpt


def _rewrite_low_overlap_cited_sentences(
    report: str,
    evidence_cards: list[EvidenceCard],
    sources: list[SourceRecordV2],
    *,
    model: BaseChatModel,
    settings: Settings,
    model_spec: str,
) -> str:
    """For every cited sentence whose content terms overlap weakly with the
    cited card's verbatim excerpt, ask the model to rewrite that sentence
    using the excerpt's exact phrasing.

    This directly attacks the FACT-pipeline "claim not in URL" rejection by
    aligning the report's wording to what the URL literally contains. One
    batched LLM call per repair pass, not one per sentence — keeps overhead
    bounded.
    """
    source_ids = {source.id for source in sources}
    if not source_ids:
        return report
    cards_by_source: dict[int, list[EvidenceCard]] = defaultdict(list)
    for card in evidence_cards:
        if card.source_id in source_ids:
            cards_by_source[card.source_id].append(card)
    if not cards_by_source:
        return report

    body, separator, source_tail = _split_sources(report)

    # Walk sentences, find weak ones, queue them up for batched rewrite.
    paragraphs = re.split(r"(\n\s*\n)", body)
    rewrite_targets: list[dict[str, Any]] = []
    sentence_index: list[tuple[int, int, str]] = []  # (para_idx, sent_idx, original_text)
    para_sentences: dict[int, list[str]] = {}

    for p_idx, paragraph in enumerate(paragraphs):
        if not paragraph.strip() or re.fullmatch(r"\n\s*\n", paragraph):
            para_sentences[p_idx] = [paragraph]
            continue
        # Split into sentence + trailing whitespace pairs to preserve formatting.
        chunks = re.split(r"(?<=[.!?])(\s+)", paragraph)
        para_sentences[p_idx] = chunks
        for s_idx, chunk in enumerate(chunks):
            if s_idx % 2 != 0:
                continue  # whitespace token
            sentence_text = chunk
            cited_ids = [
                int(m) for m in re.findall(r"\[(\d+)\]", sentence_text)
                if int(m) in source_ids
            ]
            if not cited_ids:
                continue
            # Only attempt rewrite when the sentence makes a substantive claim
            # — same gating as _repair_weak_citation_support.
            if not _paragraph_can_receive_support_repair(sentence_text):
                continue
            claim_text = _strip_numeric_citations(sentence_text)
            claim_terms = content_terms(claim_text)
            if len(claim_terms) < 6:
                continue
            # Pick the FIRST cited source as the rewrite target (others can
            # stay as supplementary citations).
            target_source_id = cited_ids[0]
            target_cards = cards_by_source.get(target_source_id, [])
            if len(target_cards) < _REWRITE_MIN_CARDS_PER_SOURCE:
                continue
            # Pick the card whose excerpt has the most overlap with this claim.
            best_card = None
            best_overlap = 0.0
            for card in target_cards:
                excerpt_terms = content_terms(card.supporting_excerpt or card.claim or "")
                if not excerpt_terms:
                    continue
                overlap = (
                    len(claim_terms & excerpt_terms) / max(len(claim_terms), 1)
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_card = card
            if best_card is None:
                continue
            if best_overlap >= _REWRITE_OVERLAP_THRESHOLD:
                continue  # claim is well-anchored — leave it alone
            excerpt = (best_card.supporting_excerpt or "").strip()
            if len(excerpt) < 40:
                continue
            rewrite_targets.append(
                {
                    "para_idx": p_idx,
                    "sent_idx": s_idx,
                    "original": sentence_text.strip(),
                    "source_id": target_source_id,
                    "excerpt": excerpt[:520],
                    "overlap": round(best_overlap, 3),
                }
            )
            if len(rewrite_targets) >= _REWRITE_MAX_SENTENCES:
                break
        if len(rewrite_targets) >= _REWRITE_MAX_SENTENCES:
            break

    if not rewrite_targets:
        return report

    rewrites = _batch_rewrite_sentences(
        targets=rewrite_targets,
        model=model,
        settings=settings,
        model_spec=model_spec,
    )
    if not rewrites:
        return report

    # Splice rewrites back into the paragraph chunks.
    rewritten_by_position: dict[tuple[int, int], str] = {}
    for target, rewritten in zip(rewrite_targets, rewrites):
        if not rewritten:
            continue
        rewritten_by_position[(target["para_idx"], target["sent_idx"])] = rewritten

    out_paragraphs: list[str] = []
    for p_idx, chunks in para_sentences.items():
        if len(chunks) == 1:
            out_paragraphs.append(chunks[0])
            continue
        new_chunks: list[str] = []
        for s_idx, chunk in enumerate(chunks):
            replacement = rewritten_by_position.get((p_idx, s_idx))
            if replacement is not None:
                # Preserve any leading whitespace at the start of the original
                # sentence (typical when it begins a new line in markdown).
                leading_ws = re.match(r"^\s*", chunk).group(0)
                new_chunks.append(leading_ws + replacement.strip())
            else:
                new_chunks.append(chunk)
        out_paragraphs.append("".join(new_chunks))
    return "".join(out_paragraphs) + (separator + source_tail if separator else "")


def _batch_rewrite_sentences(
    *,
    targets: list[dict[str, Any]],
    model: BaseChatModel,
    settings: Settings,
    model_spec: str,
) -> list[str]:
    """One LLM call that rewrites up to N sentences to match their cited
    excerpts. Returns a list of rewritten sentences indexed to `targets`.
    On any failure, returns an empty list so the caller leaves the report
    untouched."""
    if not targets:
        return []
    entries_text = []
    for i, target in enumerate(targets, start=1):
        entries_text.append(
            f"ENTRY {i}:\n"
            f"SOURCE_ID: {target['source_id']}\n"
            f"ORIGINAL_SENTENCE: {target['original']}\n"
            f"VERBATIM_EXCERPT_FROM_SOURCE: {target['excerpt']}\n"
        )
    prompt = (
        "You are a citation alignment editor. For each ENTRY below, rewrite "
        "the ORIGINAL_SENTENCE so its phrasing reuses the key tokens (names, "
        "numbers, percentages, dates, technical terms) from the "
        "VERBATIM_EXCERPT_FROM_SOURCE. Keep the citation marker (e.g. "
        f"[{targets[0]['source_id']}]) at the end of the rewritten sentence. "
        "Do NOT invent facts that aren't in the excerpt — if the excerpt is "
        "vague, write a softer sentence rather than fabricating specifics. "
        "Keep the sentence single-line, professional prose, similar length.\n\n"
        "Return STRICT JSON: {\"rewrites\": [{\"entry\": 1, \"sentence\": \"...\"}, ...]} "
        "with one entry per input. No commentary, no fences.\n\n"
        + "\n".join(entries_text)
    )
    try:
        response = _invoke_with_synthesis_budget(
            model, prompt=prompt, settings=settings, model_spec=model_spec
        )
        text = str(getattr(response, "content", response)).strip()
    except Exception:
        return []
    if not text:
        return []
    # Strip code fences if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = parsed.get("rewrites") if isinstance(parsed, dict) else None
    if not isinstance(items, list):
        return []
    by_index: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            entry_n = int(item.get("entry"))
        except (TypeError, ValueError):
            continue
        sentence = item.get("sentence")
        if not isinstance(sentence, str) or not sentence.strip():
            continue
        by_index[entry_n - 1] = sentence
    result: list[str] = []
    for i in range(len(targets)):
        result.append(by_index.get(i, ""))
    return result


def _append_evidence_coverage_if_needed(
    report: str,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    *,
    max_added_cards: int = 24,
) -> str:
    if not evidence_cards:
        return report
    if _criteria_rich_plan(plan):
        return report
    body, separator, source_tail = _split_sources(report)
    cited_source_ids = set(_numeric_citation_ids(body))
    report_terms = content_terms(body)
    cards_by_branch = _cards_by_branch(evidence_cards)
    additions: list[tuple[ResearchBranch | None, EvidenceCard]] = []
    selected_card_ids: set[int] = set()

    for branch in plan.branches:
        branch_cards = cards_by_branch.get(branch.id, [])
        if not branch_cards:
            continue
        branch_terms = content_terms(branch.title + " " + branch.objective + " " + " ".join(branch.required_terms))
        branch_coverage = len(branch_terms & report_terms) / max(len(branch_terms), 1) if branch_terms else 1.0
        branch_source_ids = {card.source_id for card in branch_cards}
        branch_is_cited = bool(branch_source_ids & cited_source_ids)
        if branch_coverage >= 0.35 and branch_is_cited:
            continue
        for card in _source_diverse_cards(_rank_cards(branch_cards, question=plan.question), limit=2):
            if card.id in selected_card_ids:
                continue
            additions.append((branch, card))
            selected_card_ids.add(card.id)
            cited_source_ids.add(card.source_id)
            if len(additions) >= max_added_cards:
                break
        if len(additions) >= max_added_cards:
            break

    target_sources = min(17, len({card.source_id for card in evidence_cards}))
    if len(cited_source_ids) < target_sources and len(additions) < max_added_cards:
        needed = target_sources - len(cited_source_ids)
        breadth_cards = [
            card
            for card in _cards_for_synthesis(plan, evidence_cards)
            if card.source_id not in cited_source_ids and card.id not in selected_card_ids
        ]
        for card in _source_diverse_cards(breadth_cards, limit=min(needed, max_added_cards - len(additions))):
            additions.append((None, card))
            selected_card_ids.add(card.id)
            cited_source_ids.add(card.source_id)

    if not additions:
        return report

    labels = _coverage_repair_labels(plan.question)
    lines = [
        "",
        f"## {labels['heading']} {_report_subject(plan.question)}",
        "",
    ]
    for branch, card in additions:
        if branch is not None:
            prefix = f"{labels['branch_prefix']} {branch.title}, {labels['evidence_adds']}"
        else:
            prefix = labels["breadth_prefix"]
        lines.append(f"{prefix} {card.claim.rstrip('. ')}. [{card.source_id}]")
        lines.append("")
    repaired_body = body.rstrip() + "\n" + "\n".join(lines).rstrip() + "\n"
    return repaired_body + (separator + source_tail if separator else "")


def _paragraph_can_receive_support_repair(text: str) -> bool:
    if len(text) < 80:
        return False
    if text.startswith("#") or text.startswith("|"):
        return False
    if not re.search(r"\[[0-9][0-9,;\s]*\]", text):
        return False
    return True


def _evidence_terms_by_source(
    evidence_cards: list[EvidenceCard],
    source_ids: set[int],
) -> dict[int, set[str]]:
    terms: dict[int, set[str]] = {}
    for card in evidence_cards:
        if card.source_id not in source_ids:
            continue
        terms.setdefault(card.source_id, set()).update(content_terms(card.claim + " " + card.supporting_excerpt))
    return terms


def _individual_source_support_scores(
    claim_terms: set[str],
    terms_by_source: dict[int, set[str]],
    cited_ids: set[int],
) -> dict[int, float]:
    return {
        source_id: len(claim_terms & terms_by_source.get(source_id, set())) / max(len(claim_terms), 1)
        for source_id in cited_ids
    }


def _remove_numeric_citation_ids(paragraph: str, remove_ids: set[int]) -> str:
    if not remove_ids:
        return paragraph

    def replace(match: re.Match[str]) -> str:
        kept = [
            int(value)
            for value in re.findall(r"\d+", match.group(1))
            if int(value) not in remove_ids
        ]
        if not kept:
            return ""
        return "[" + ", ".join(str(value) for value in sorted(set(kept))) + "]"

    cleaned = re.sub(r"\[([0-9][0-9,;\s]*)\]", replace, paragraph)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def _best_supporting_source_ids(
    claim_terms: set[str],
    terms_by_source: dict[int, set[str]],
    cited_ids: set[int],
    threshold: float,
) -> list[int]:
    ranked = sorted(
        (
            (
                len(claim_terms & terms) / max(len(claim_terms), 1),
                len(claim_terms & terms),
                source_id,
            )
            for source_id, terms in terms_by_source.items()
            if source_id not in cited_ids
        ),
        reverse=True,
    )
    additions: list[int] = []
    support_terms = set().union(*(terms_by_source.get(source_id, set()) for source_id in cited_ids))
    for score, _hits, source_id in ranked:
        if score <= 0:
            continue
        additions.append(source_id)
        support_terms |= terms_by_source.get(source_id, set())
        support_score = len(claim_terms & support_terms) / max(len(claim_terms), 1)
        if support_score >= threshold or len(additions) >= 3:
            break
    return sorted(additions)


def _split_sources(report: str) -> tuple[str, str, str]:
    match = re.search(r"(?ims)^##\s+Sources\s*$", report)
    if not match:
        return report, "", ""
    return report[: match.start()], report[match.start() : match.end()], report[match.end() :]


def _strip_numeric_citations(text: str) -> str:
    return re.sub(r"\[([0-9][0-9,;\s]*)\]", "", text)


def _numeric_citation_ids(text: str) -> list[int]:
    return [
        int(value)
        for block in re.findall(r"\[([0-9][0-9,;\s]*)\]", text)
        for value in re.findall(r"\d+", block)
    ]


def _fallback_citations(body: str, sources: list[SourceRecordV2]) -> str:
    cited = [
        source_id
        for source_id in _numeric_citation_ids(body)
        if any(source.id == source_id for source in sources)
    ]
    if not cited:
        cited = [source.id for source in sorted(sources, key=lambda item: (-item.quality_score, item.id))[:2]]
    return " ".join(f"[{source_id}]" for source_id in sorted(set(cited))[:3])


def _paragraph_needs_repair(text: str) -> bool:
    if len(text) < 80:
        return False
    if text.startswith("#") or text.startswith("|"):
        return False
    if re.search(r"\[[0-9][0-9,;\s]*\]", text):
        return False
    return True


def _sources_section(sources: list[SourceRecordV2], report: str) -> str:
    cited_ids = sorted(
        {
            int(value)
            for block in re.findall(r"\[([0-9][0-9,;\s]*)\]", report)
            for value in re.findall(r"\d+", block)
        }
    )
    source_by_id = {source.id: source for source in sources}
    listed = [source_by_id[source_id] for source_id in cited_ids if source_id in source_by_id]
    if not listed:
        listed = sorted(sources, key=lambda item: item.id)
    lines = ["## Sources", ""]
    for source in listed:
        lines.append(f"[{source.id}] {source.title}: {source.url}")
    return "\n".join(lines)



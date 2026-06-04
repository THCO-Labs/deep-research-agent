from __future__ import annotations

import re
from dataclasses import dataclass
from statistics import mean

from deep_research.schemas import EvidenceCard
from deep_research.source_validation import content_terms
from deep_research.text_terms import TOKEN_RE, normalize_term_text


URL_RE = re.compile(r"https?://\S+", flags=re.I)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]{0,160}]\((https?://[^)]+)\)", flags=re.I)
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)", flags=re.I)
KEY_VALUE_LINE_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9 _./-]{1,48}:\s+\S+")
FENCE_RE = re.compile(r"```")
SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?$")
INTERNAL_ARTIFACT_RE = re.compile(
    r"\b(?:evidence\s+cards?|verified\s+evidence\s+cards?|provided\s+evidence|"
    r"available\s+evidence\s+cards?|internal\s+coverage\s+score|verification\s+failures?)\b"
    r"|\bbranch_[0-9]+\b",
    flags=re.I,
)


@dataclass(frozen=True)
class EvidenceHygieneResult:
    kept: list[EvidenceCard]
    rejected: list[dict[str, object]]


@dataclass(frozen=True)
class TextQualityMetrics:
    char_count: int
    word_count: int
    sentence_like_lines: int
    url_to_text_ratio: float
    markdown_link_ratio: float
    markdown_image_count: int
    metadata_line_ratio: float
    symbol_to_text_ratio: float
    alphabetic_word_ratio: float
    punctuation_failure_ratio: float
    repeated_trigram_ratio: float
    mean_word_length: float
    unique_term_count: int

    @property
    def structural_artifact_score(self) -> float:
        weighted = (
            min(1.0, self.url_to_text_ratio / 0.20) * 0.24
            + min(1.0, self.markdown_link_ratio / 0.20) * 0.18
            + min(1.0, self.metadata_line_ratio / 0.35) * 0.20
            + min(1.0, self.symbol_to_text_ratio / 0.28) * 0.16
            + min(1.0, self.punctuation_failure_ratio / 0.85) * 0.12
            + min(1.0, self.repeated_trigram_ratio / 0.20) * 0.10
        )
        if self.markdown_image_count:
            weighted += 0.30
        return round(min(1.0, weighted), 4)


def apply_evidence_hygiene(cards: list[EvidenceCard]) -> EvidenceHygieneResult:
    kept: list[EvidenceCard] = []
    rejected: list[dict[str, object]] = []
    seen_claims: set[str] = set()
    for card in cards:
        reasons = evidence_card_rejection_reasons(card)
        normalized = _dedupe_key(card.claim)
        if normalized in seen_claims:
            reasons.append("duplicate evidence claim")
        if reasons:
            rejected.append({"card": card.to_dict(), "reasons": reasons})
            continue
        seen_claims.add(normalized)
        kept.append(card)
    return EvidenceHygieneResult(kept=kept, rejected=rejected)


def evidence_card_rejection_reasons(card: EvidenceCard) -> list[str]:
    reasons: list[str] = []
    claim = _normalize(card.claim)
    excerpt = _normalize(card.supporting_excerpt)
    claim_metrics = text_quality_metrics(claim)
    combined_metrics = text_quality_metrics(f"{claim}\n{excerpt}")

    if claim_metrics.char_count < 50:
        reasons.append("claim is too short to be useful evidence")
    if claim_metrics.char_count > 800:
        reasons.append("claim is too long and likely unprocessed source text")
    if claim_metrics.unique_term_count < 7 or claim_metrics.alphabetic_word_ratio < 0.72:
        reasons.append("claim lacks enough natural-language signal")
    if claim_metrics.structural_artifact_score >= 0.55 or combined_metrics.structural_artifact_score >= 0.65:
        reasons.append("claim has high structural artifact score")
    if claim_metrics.url_to_text_ratio > 0.05:
        reasons.append("claim contains URL-heavy text")
    if claim_metrics.markdown_image_count > 0 or claim_metrics.markdown_link_ratio > 0.08:
        reasons.append("claim contains markdown media/link artifacts")
    if claim_metrics.metadata_line_ratio > 0.0 and claim_metrics.sentence_like_lines == 0:
        reasons.append("claim is shaped like extracted metadata rather than evidence")
    if claim_metrics.repeated_trigram_ratio > 0.35:
        reasons.append("claim has excessive repeated phrase content")
    if card.confidence < 0.35:
        reasons.append("evidence confidence is below threshold")
    return reasons


def report_quality_issues(markdown: str) -> list[str]:
    body = _without_sources(markdown)
    issues: list[str] = []
    for line_number, line in enumerate(body.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        metrics = text_quality_metrics(stripped)
        if metrics.markdown_image_count:
            issues.append(f"Report line {line_number} contains markdown media artifact.")
            continue
        if metrics.url_to_text_ratio > 0.05 or metrics.markdown_link_ratio > 0.05:
            issues.append(f"Report line {line_number} contains URL-heavy or markdown-link artifact outside Sources.")
            continue
        if metrics.metadata_line_ratio > 0.0 and metrics.sentence_like_lines == 0 and metrics.word_count <= 18:
            issues.append(f"Report line {line_number} is shaped like extracted metadata rather than report prose.")
            continue
        if FENCE_RE.search(stripped) and metrics.sentence_like_lines == 0:
            issues.append(f"Report line {line_number} contains a code-fence artifact.")
            continue
        if INTERNAL_ARTIFACT_RE.search(stripped):
            issues.append(f"Report line {line_number} leaks internal research artifact language.")
    return issues


def text_quality_metrics(text: str) -> TextQualityMetrics:
    normalized = _normalize(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    words = TOKEN_RE.findall(normalize_term_text(normalized))
    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    url_chars = sum(len(match.group(0)) for match in URL_RE.finditer(normalized))
    link_chars = sum(len(match.group(0)) for match in MARKDOWN_LINK_RE.finditer(normalized))
    metadata_lines = [line for line in lines if _is_metadata_shaped_line(line)]
    sentence_like = [line for line in lines if SENTENCE_END_RE.search(line)]
    symbol_chars = sum(1 for char in normalized if not char.isalnum() and not char.isspace())
    punctuation_failures = max(0, len(lines) - len(sentence_like))
    return TextQualityMetrics(
        char_count=len(normalized),
        word_count=len(words),
        sentence_like_lines=len(sentence_like),
        url_to_text_ratio=round(url_chars / max(len(normalized), 1), 4),
        markdown_link_ratio=round(link_chars / max(len(normalized), 1), 4),
        markdown_image_count=len(MARKDOWN_IMAGE_RE.findall(normalized)),
        metadata_line_ratio=round(len(metadata_lines) / max(len(lines), 1), 4),
        symbol_to_text_ratio=round(symbol_chars / max(len(normalized), 1), 4),
        alphabetic_word_ratio=round(len(alpha_words) / max(len(words), 1), 4),
        punctuation_failure_ratio=round(punctuation_failures / max(len(lines), 1), 4),
        repeated_trigram_ratio=_repeated_ngram_ratio(words, 3),
        mean_word_length=round(mean(len(word) for word in words), 4) if words else 0.0,
        unique_term_count=len(content_terms(normalized)),
    )


def _is_metadata_shaped_line(line: str) -> bool:
    if not KEY_VALUE_LINE_RE.search(line):
        return False
    words = TOKEN_RE.findall(normalize_term_text(line))
    if URL_RE.search(line):
        return True
    return len(words) <= 8 and not SENTENCE_END_RE.search(line)


def _repeated_ngram_ratio(words: list[str], n: int) -> float:
    if len(words) < n * 2:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(0, len(words) - n + 1)]
    unique = len(set(grams))
    repeated = len(grams) - unique
    return round(repeated / max(len(grams), 1), 4)


def _dedupe_key(text: str) -> str:
    terms = list(content_terms(text))
    return " ".join(sorted(terms)[:40])


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _without_sources(markdown: str) -> str:
    match = re.search(r"(?ims)^#{2,3}\s+sources\s*$", markdown)
    return markdown if not match else markdown[: match.start()]

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from deep_research.lexical_expansion import expand_terms


@dataclass(frozen=True)
class ContradictionQuery:
    claim_id: str
    branch_id: str
    claim: str
    query: str
    purpose: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONTRADICTION_INTENTS: dict[str, tuple[str, ...]] = {
    "limitation": ("limitation", "constraint", "qualification", "exception"),
    "criticism": ("criticism", "critique", "challenge", "dispute"),
    "freshness": ("outdated", "newer", "updated", "superseded"),
    "opposing_evidence": ("contrary", "opposing", "conflicting", "negative"),
    "controversy": ("controversy", "debate", "disagreement", "uncertainty"),
    "falsification": ("false", "incorrect", "not true", "overstated"),
}


def generate_contradiction_queries(
    claims: list[dict[str, Any]],
    *,
    question: str,
    limit: int = 12,
    per_claim: int = 2,
) -> list[ContradictionQuery]:
    selected = _rank_claims(claims)
    rows: list[ContradictionQuery] = []
    seen: set[str] = set()
    for claim in selected:
        claim_text = _compact_claim(str(claim.get("claim") or ""))
        if not claim_text:
            continue
        for purpose, term in _intent_terms()[: max(per_claim, 1)]:
            query = _trim_query(f"{question} {claim_text} {term}")
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                ContradictionQuery(
                    claim_id=str(claim.get("id") or ""),
                    branch_id=str(claim.get("branch_id") or ""),
                    claim=claim_text,
                    query=query,
                    purpose=purpose,
                )
            )
            if len(rows) >= limit:
                return rows
    return rows


def _intent_terms() -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for purpose, seeds in CONTRADICTION_INTENTS.items():
        expanded = sorted(expand_terms(seeds, max_per_term=4), key=lambda value: (value not in seeds, len(value), value))
        for term in expanded:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append((purpose, term))
            break
    return rows


def _rank_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [claim for claim in claims if isinstance(claim, dict)],
        key=lambda claim: (
            not bool(claim.get("high_impact")),
            bool(claim.get("weak")),
            -float(claim.get("support_count", 0) or 0),
            -float(claim.get("average_confidence", 0.0) or 0.0),
        ),
    )


def _compact_claim(claim: str) -> str:
    cleaned = re.sub(r"\[[0-9,\s]+\]", "", claim)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .;:")
    if len(cleaned) <= 140:
        return cleaned
    boundary = cleaned.rfind(" ", 0, 140)
    return cleaned[: boundary if boundary > 80 else 140].strip()


def _trim_query(query: str, *, max_chars: int = 240) -> str:
    cleaned = re.sub(r"\s+", " ", query).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    boundary = cleaned.rfind(" ", 0, max_chars)
    return cleaned[: boundary if boundary > 120 else max_chars].strip()

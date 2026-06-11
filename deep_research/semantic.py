from __future__ import annotations

import json
import re
import hashlib
import ast
import threading
from dataclasses import dataclass, replace
from typing import Any, Callable

from langchain_core.messages import HumanMessage

from deep_research.errors import classify_exception
from deep_research.model_router import model_for_role
from deep_research.schemas import EvidenceCard, ResearchPlan, VerificationResultV2
from deep_research.settings import Settings

EVIDENCE_BATCH_SIZE = 20
SEMANTIC_CARD_THRESHOLD = 0.55
SEMANTIC_REPORT_THRESHOLD = 0.75
RECOVERABLE_SEMANTIC_PROVIDER_FAILURES = {
    "quota_or_rate_limit",
    "provider_timeout",
    "token_budget_exceeded",
    "tool_call_parse_error",
}


@dataclass(frozen=True)
class SemanticEvidenceResult:
    cards: list[EvidenceCard]
    judgments: list[dict[str, Any]]
    rejected: list[dict[str, Any]]
    failures: list[str]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class SemanticReportResult:
    judgment: dict[str, Any]
    failures: list[str]
    score: float


def enrich_evidence_cards_with_semantics(
    *,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    settings: Settings,
    model: Any | None = None,
    prior_judgments: list[dict[str, Any]] | None = None,
    batch_checkpoint_callback: Callable[[list[dict[str, Any]]], None] | None = None,
) -> SemanticEvidenceResult:
    if not settings.semantic_verification:
        return SemanticEvidenceResult(
            cards=evidence_cards,
            judgments=[],
            rejected=[],
            failures=[],
            metrics={"semantic_evidence_enabled": False, "semantic_evidence_batches": 0},
        )
    if not evidence_cards:
        return SemanticEvidenceResult(
            cards=[],
            judgments=[],
            rejected=[],
            failures=[],
            metrics={"semantic_evidence_enabled": True, "semantic_evidence_batches": 0},
        )

    judge = model if model is not None else _judge_model(settings)
    cached_judgments, cards_to_judge = _reuse_prior_card_judgments(evidence_cards, prior_judgments or [])
    max_llm_cards = int(getattr(settings, "semantic_evidence_max_llm_cards", 120))
    if max_llm_cards > 0 and len(cards_to_judge) > max_llm_cards:
        judgments = _fallback_card_judgments(
            cards_to_judge,
            reason=f"evidence deck has {len(cards_to_judge)} uncached cards; configured semantic LLM card limit is {max_llm_cards}",
        )
        judgments = cached_judgments + judgments
        cards, rejected, application_failures = _apply_card_judgments(evidence_cards, judgments)
        return SemanticEvidenceResult(
            cards=cards,
            judgments=judgments,
            rejected=rejected,
            failures=application_failures,
            metrics={
                "semantic_evidence_enabled": True,
                "semantic_evidence_batches": 0,
                "semantic_evidence_judgment_count": len(judgments),
                "semantic_evidence_cached_judgment_count": len(cached_judgments),
                "semantic_evidence_rejected_count": len(rejected),
                "semantic_evidence_large_deck_fallback": True,
                "semantic_evidence_max_llm_cards": max_llm_cards,
                "semantic_evidence_parse_failure_count": 0,
                "semantic_evidence_failure_count": len(application_failures),
            },
        )
    judgments: list[dict[str, Any]] = list(cached_judgments)
    failures: list[str] = []
    parse_failures = 0
    provider_failures = 0
    batches = 0
    for batch in _chunks(cards_to_judge, EVIDENCE_BATCH_SIZE):
        batches += 1
        try:
            payload = _invoke_json(judge, _evidence_prompt(plan, batch))
            judgments.extend(_validate_card_judgments(payload, batch))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_failures += 1
            judgments.extend(_fallback_card_judgments(batch, reason=str(exc)))
        except Exception as exc:
            failure = classify_exception(exc)
            if failure.category not in RECOVERABLE_SEMANTIC_PROVIDER_FAILURES:
                raise
            provider_failures += 1
            failures.append(
                f"Semantic evidence judge unavailable for batch; used deterministic fallback ({failure.category})."
            )
            judgments.extend(_fallback_card_judgments(batch, reason=f"{failure.category}: {failure.message}"))

        # Mid-node checkpoint: persist accumulated judgments after each batch so a
        # crash+resume can reuse them via prior_judgments and skip re-judging.
        if batch_checkpoint_callback is not None:
            try:
                batch_checkpoint_callback(judgments)
            except Exception:
                pass

    cards, rejected, application_failures = _apply_card_judgments(evidence_cards, judgments)
    failures.extend(application_failures)
    return SemanticEvidenceResult(
        cards=cards,
        judgments=judgments,
        rejected=rejected,
        failures=failures,
        metrics={
            "semantic_evidence_enabled": True,
            "semantic_evidence_batches": batches,
            "semantic_evidence_judgment_count": len(judgments),
            "semantic_evidence_cached_judgment_count": len(cached_judgments),
            "semantic_evidence_rejected_count": len(rejected),
            "semantic_evidence_parse_failure_count": parse_failures,
            "semantic_evidence_provider_failure_count": provider_failures,
            "semantic_evidence_failure_count": len(failures),
        },
    )


def verify_report_with_semantics(
    *,
    report_markdown: str,
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    settings: Settings,
    model: Any | None = None,
) -> SemanticReportResult:
    if not settings.semantic_verification:
        return SemanticReportResult(judgment={"enabled": False}, failures=[], score=1.0)
    judge = model if model is not None else _judge_model(settings)
    try:
        payload = _invoke_json(judge, _report_prompt(plan, report_markdown, evidence_cards))
        judgment = _validate_report_judgment(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return SemanticReportResult(
            judgment={"enabled": True, "valid": False, "error": str(exc)},
            failures=[f"Semantic report judge returned invalid structured output: {exc}"],
            score=0.0,
        )
    except Exception as exc:
        failure = classify_exception(exc)
        if failure.category not in RECOVERABLE_SEMANTIC_PROVIDER_FAILURES:
            raise
        return SemanticReportResult(
            judgment={"enabled": True, "valid": False, "error": failure.message, "failure_category": failure.category},
            failures=[f"Semantic report judge unavailable: {failure.category}"],
            score=0.0,
        )

    score = float(judgment["overall_score"])
    failures = list(judgment.get("failures", []))
    if score < SEMANTIC_REPORT_THRESHOLD:
        failures.append(f"Semantic report verification below threshold: {score}")
    for claim in judgment.get("unsupported_claims", []):
        failures.append(f"Semantic judge found unsupported claim: {str(claim)[:160]}")
    for contradiction in judgment.get("contradictions", []):
        failures.append(f"Semantic judge found potential contradiction: {str(contradiction)[:160]}")
    return SemanticReportResult(judgment=judgment, failures=failures, score=score)


def apply_semantic_report_result(
    result: VerificationResultV2,
    semantic: SemanticReportResult,
) -> VerificationResultV2:
    if not semantic.failures and semantic.score >= SEMANTIC_REPORT_THRESHOLD:
        return replace(
            result,
            semantic_verification_score=semantic.score,
            semantic_verification=semantic.judgment,
        )
    failures = list(result.failures)
    failures.extend(semantic.failures)
    return replace(
        result,
        valid=False,
        failures=failures,
        semantic_verification_score=semantic.score,
        semantic_verification=semantic.judgment,
    )


def _judge_model(settings: Settings) -> Any:
    model = model_for_role(settings, "judge", settings.judge_model)
    if not hasattr(model, "invoke"):
        raise RuntimeError(f"Semantic judge role did not resolve to an invokable model: {model!r}")
    return model


class JudgeTimeoutError(RuntimeError):
    """Raised when a semantic judge call exceeds the wall-clock budget."""


def _judge_wall_clock_timeout(model: Any) -> float:
    """Wall-clock cap for any one judge call. Pulled from the model's own
    timeout attribute if it exposes one; otherwise 300s, with a hard ceiling
    of 600s. Matches the same logic the synthesis watchdog uses."""
    inner = getattr(model, "primary", model)  # unwrap FallbackChatModel
    candidate = getattr(inner, "timeout", None) or getattr(inner, "request_timeout", None)
    base = float(candidate) if candidate else 300.0
    return min(600.0, max(180.0, base * 1.5))


def _invoke_json(model: Any, prompt: str) -> dict[str, Any]:
    timeout_s = _judge_wall_clock_timeout(model)
    result_holder: list[Any] = []
    error_holder: list[BaseException] = []

    def _runner() -> None:
        try:
            result_holder.append(model.invoke([HumanMessage(content=prompt)]))
        except BaseException as exc:  # noqa: BLE001
            error_holder.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        raise JudgeTimeoutError(
            f"semantic judge did not return within {timeout_s:.0f}s wall-clock budget; "
            f"abandoning thread and propagating as a soft failure."
        )
    if error_holder:
        raise error_holder[0]
    if not result_holder:
        raise ValueError("empty response")

    response = result_holder[0]
    text = str(getattr(response, "content", response)).strip()
    if not text:
        raise ValueError("empty response")
    return _load_json_object(text)


def _load_json_object(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = _raw_decode_first_object_or_literal(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("response JSON must be an object")
    return parsed


def _strip_json_fence(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _raw_decode_first_object(text: str) -> Any:
    start = text.find("{")
    if start == -1:
        raise ValueError("response did not contain a JSON object")
    decoder = json.JSONDecoder()
    parsed, _end = decoder.raw_decode(text[start:])
    return parsed


def _raw_decode_first_object_or_literal(text: str) -> Any:
    try:
        return _raw_decode_first_object(text)
    except json.JSONDecodeError:
        extracted = _extract_json_object(text)
        try:
            return ast.literal_eval(extracted)
        except (SyntaxError, ValueError) as exc:
            raise json.JSONDecodeError(str(exc), text, 0) from exc


def _extract_json_object(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response did not contain a JSON object")
    return cleaned[start : end + 1]


def _fallback_card_judgments(batch: list[EvidenceCard], *, reason: str) -> list[dict[str, Any]]:
    judgments: list[dict[str, Any]] = []
    for card in batch:
        relevance = max(0.0, min(1.0, float(card.relevance_score)))
        confidence = max(0.0, min(1.0, float(card.confidence)))
        quality = max(0.0, min(1.0, float(card.quality_score)))
        judgments.append(
            {
                "id": card.id,
                "keep": True,
                "branch_alignment_score": max(SEMANTIC_CARD_THRESHOLD, relevance),
                "entailment_score": max(SEMANTIC_CARD_THRESHOLD, confidence),
                "evidence_relevance_score": max(SEMANTIC_CARD_THRESHOLD, (relevance + quality) / 2),
                "normalized_claim": "",
                "key_points": [],
                "limitations": [],
                "failure_reasons": [],
                "fallback": "deterministic_after_invalid_judge_output",
                "fallback_reason": reason[:240],
                "card_fingerprint": _card_fingerprint(card),
            }
        )
    return judgments


def _validate_card_judgments(payload: dict[str, Any], batch: list[EvidenceCard]) -> list[dict[str, Any]]:
    rows = payload.get("cards")
    if not isinstance(rows, list):
        raise ValueError("expected top-level cards list")
    expected_ids = {card.id for card in batch}
    seen_ids: set[int] = set()
    judgments: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("card judgment must be an object")
        card_id = _int_field(row, "id")
        if card_id not in expected_ids:
            raise ValueError(f"judgment references unknown card id {card_id}")
        if card_id in seen_ids:
            raise ValueError(f"duplicate judgment for card id {card_id}")
        seen_ids.add(card_id)
        judgments.append(
            {
                "id": card_id,
                "keep": _bool_field(row, "keep", default=True),
                "branch_alignment_score": _score_field(row, "branch_alignment_score"),
                "entailment_score": _score_field(row, "entailment_score"),
                "evidence_relevance_score": _score_field(row, "evidence_relevance_score"),
                "normalized_claim": _string_field(row, "normalized_claim", default=""),
                "key_points": _string_list_field(row, "key_points"),
                "limitations": _string_list_field(row, "limitations"),
                "failure_reasons": _string_list_field(row, "failure_reasons"),
                "card_fingerprint": _card_fingerprint(next(card for card in batch if card.id == card_id)),
            }
        )
    returned_ids = {row["id"] for row in judgments}
    missing = expected_ids - returned_ids
    if missing:
        raise ValueError(f"missing judgments for card ids {sorted(missing)}")
    return judgments


def _apply_card_judgments(
    cards: list[EvidenceCard],
    judgments: list[dict[str, Any]],
) -> tuple[list[EvidenceCard], list[dict[str, Any]], list[str]]:
    by_id = {int(row["id"]): row for row in judgments}
    kept: list[EvidenceCard] = []
    rejected: list[dict[str, Any]] = []
    failures: list[str] = []
    for card in cards:
        judgment = by_id.get(card.id)
        if judgment is None:
            failures.append(f"Semantic evidence judge did not return a judgment for card {card.id}.")
            kept.append(card)
            continue
        semantic_score = round(
            (
                float(judgment["branch_alignment_score"])
                + float(judgment["entailment_score"])
                + float(judgment["evidence_relevance_score"])
            )
            / 3,
            4,
        )
        reasons = list(judgment.get("failure_reasons", []))
        if semantic_score < SEMANTIC_CARD_THRESHOLD:
            reasons.append(f"semantic score below threshold: {semantic_score}")
        if not judgment.get("keep", True):
            reasons.append("semantic judge marked card as not keepable")
        if reasons:
            rejected.append({"card": card.to_dict(), "judgment": judgment, "reasons": _dedupe(reasons)})
            continue
        normalized_claim = str(judgment.get("normalized_claim") or "").strip()
        claim = normalized_claim if len(normalized_claim) >= 40 else card.claim
        semantic_notes = _dedupe(list(judgment.get("key_points", [])) + list(judgment.get("limitations", [])))
        kept.append(
            replace(
                card,
                claim=claim,
                confidence=round(min(card.confidence, semantic_score), 4),
                limitations=_dedupe(card.limitations + list(judgment.get("limitations", []))),
                semantic_score=semantic_score,
                semantic_notes=semantic_notes,
            )
        )
    return kept, rejected, failures


def _validate_report_judgment(payload: dict[str, Any]) -> dict[str, Any]:
    judgment = {
        "enabled": True,
        "answer_completeness_score": _score_field(payload, "answer_completeness_score"),
        "citation_entailment_score": _score_field(payload, "citation_entailment_score"),
        "evidence_use_score": _score_field(payload, "evidence_use_score"),
        "contradiction_safety_score": _score_field(payload, "contradiction_safety_score"),
        # Prose quality scores — optional for backwards compatibility with cached judgments.
        "opening_directness_score": float(payload.get("opening_directness_score") or 1.0),
        "argumentative_coherence_score": float(payload.get("argumentative_coherence_score") or 1.0),
        "closing_synthesis_score": float(payload.get("closing_synthesis_score") or 1.0),
        "overall_score": _score_field(payload, "overall_score"),
        "failures": _string_list_field(payload, "failures"),
        "missing_context": _string_list_field(payload, "missing_context"),
        "unsupported_claims": _string_list_field(payload, "unsupported_claims"),
        "contradictions": _string_list_field(payload, "contradictions"),
        "search_focus": _string_list_field(payload, "search_focus"),
    }
    return judgment


def _evidence_prompt(plan: ResearchPlan, batch: list[EvidenceCard]) -> str:
    branches = {
        branch.id: {
            "title": branch.title,
            "objective": branch.objective,
            "required_terms": branch.required_terms,
        }
        for branch in plan.branches
    }
    cards = [
        {
            "id": card.id,
            "branch_id": card.branch_id,
            "claim": card.claim,
            "supporting_excerpt": card.supporting_excerpt[:1200],
            "source_title": card.source_title,
            "quality_score": card.quality_score,
            "relevance_score": card.relevance_score,
        }
        for card in batch
    ]
    return f"""You are a strict semantic evidence judge for a deep research system.

Judge each evidence card against the user's question and its assigned branch. Use only the card claim and supporting excerpt. Do not add outside facts.

User question:
{plan.question}

Branch definitions:
{json.dumps(branches, ensure_ascii=True)}

Evidence cards:
{json.dumps(cards, ensure_ascii=True)}

Return exactly one JSON object:
{{
  "cards": [
    {{
      "id": 1,
      "keep": true,
      "branch_alignment_score": 0.0,
      "entailment_score": 0.0,
      "evidence_relevance_score": 0.0,
      "normalized_claim": "one precise claim entailed by the excerpt",
      "key_points": ["short semantic coverage point"],
      "limitations": ["short limitation if any"],
      "failure_reasons": []
    }}
  ]
}}

Scoring rubric:
- branch_alignment_score: whether the card belongs to the assigned branch.
- entailment_score: whether the excerpt directly supports the claim.
- evidence_relevance_score: whether this evidence helps answer the user's question.
- keep must be false when the card is mostly irrelevant, not entailed, too vague, or only page chrome/metadata.
"""


def _report_prompt(plan: ResearchPlan, report: str, evidence_cards: list[EvidenceCard]) -> str:
    branch_payload = [
        {
            "id": branch.id,
            "title": branch.title,
            "objective": branch.objective,
            "required_terms": branch.required_terms,
        }
        for branch in plan.branches
    ]
    evidence_payload = [
        {
            "id": card.id,
            "source_id": card.source_id,
            "branch_id": card.branch_id,
            "claim": card.claim,
            "excerpt": card.supporting_excerpt[:500],
        }
        for card in sorted(evidence_cards, key=lambda item: (-item.confidence, item.id))[:120]
    ]
    return f"""You are a strict semantic verifier for a cited research report.

Use only the report text and the evidence-card deck. Do not add outside facts. Judge whether the report answers the question, uses evidence correctly, and avoids contradictions.

User question:
{plan.question}

Branches:
{json.dumps(branch_payload, ensure_ascii=True)}

Evidence cards:
{json.dumps(evidence_payload, ensure_ascii=True)}

Report:
{report[:20000]}

Return exactly one JSON object:
{{
  "answer_completeness_score": 0.0,
  "citation_entailment_score": 0.0,
  "evidence_use_score": 0.0,
  "contradiction_safety_score": 0.0,
  "opening_directness_score": 0.0,
  "argumentative_coherence_score": 0.0,
  "closing_synthesis_score": 0.0,
  "overall_score": 0.0,
  "failures": [],
  "missing_context": [],
  "unsupported_claims": [],
  "contradictions": [],
  "search_focus": []
}}

Rubric:
- answer_completeness_score: all important branches of the user's request are answered.
- citation_entailment_score: cited claims are supported by the evidence-card deck.
- evidence_use_score: the report synthesizes evidence instead of copying fragments or adding unsupported material.
- contradiction_safety_score: the report does not contain contradictions or unresolved tensions.
- opening_directness_score: the first substantive paragraph directly answers the user's question with a clear claim and at least one citation, rather than restating the question or saying "this report examines..."
- argumentative_coherence_score: sections develop a logical argument — each section advances the reader's understanding rather than just listing facts; transitions explain the logical connection to the next topic.
- closing_synthesis_score: the report ends with a conclusion that synthesizes the key findings into a unified takeaway, not just a bullet summary of each section.
- overall_score should be the minimum score you would defend from ALL of those criteria — factual accuracy and prose quality both count.
- When missing_context is non-empty, search_focus must include concrete search phrases that would help repair those gaps.
- search_focus should name topics to search, not internal branch IDs, card IDs, or generic complaints.
"""


def _score_field(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    return round(max(0.0, min(1.0, float(value))), 4)


def _int_field(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _bool_field(payload: dict[str, Any], key: str, *, default: bool) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _string_field(payload: dict[str, Any], key: str, *, default: str) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip()


def _string_list_field(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    result = []
    for item in value:
        if isinstance(item, str):
            cleaned = item.strip()
        elif isinstance(item, int | float | bool):
            cleaned = str(item)
        elif isinstance(item, dict | list):
            cleaned = json.dumps(item, ensure_ascii=True, sort_keys=True)[:500]
        else:
            raise ValueError(f"{key} entries must be strings or JSON-compatible values")
        if cleaned:
            result.append(cleaned)
    return result


def _chunks(values: list[EvidenceCard], size: int) -> list[list[EvidenceCard]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _reuse_prior_card_judgments(
    cards: list[EvidenceCard],
    prior_judgments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[EvidenceCard]]:
    cache: dict[str, dict[str, Any]] = {}
    for row in prior_judgments:
        if not isinstance(row, dict):
            continue
        fingerprint = str(row.get("card_fingerprint") or "").strip()
        if not fingerprint:
            continue
        cache[fingerprint] = row

    reused: list[dict[str, Any]] = []
    remaining: list[EvidenceCard] = []
    for card in cards:
        fingerprint = _card_fingerprint(card)
        cached = cache.get(fingerprint)
        if cached is None:
            remaining.append(card)
            continue
        row = dict(cached)
        row["id"] = card.id
        row["card_fingerprint"] = fingerprint
        reused.append(row)
    return reused, remaining


def _card_fingerprint(card: EvidenceCard) -> str:
    payload = json.dumps(
        {
            "source_id": card.source_id,
            "branch_id": card.branch_id,
            "claim": card.claim,
            "supporting_excerpt": card.supporting_excerpt,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value.strip())
    return result

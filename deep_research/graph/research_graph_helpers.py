from __future__ import annotations

import re
import time
from typing import Any

from deep_research.runtime.artifacts_v2 import ResearchArtifactsV2
from deep_research.reasoning.coverage import build_coverage_matrix
from deep_research.core.schemas import (
    BranchCoverage,
    CoverageMatrix,
    EvidenceCard,
    ResearchBranch,
    ResearchPlan,
    ResearchState,
    SourceCandidate,
    SourceRecordV2,
)

MAX_IMPROVING_VERIFICATION_ROUNDS = 8
QUALITY_SCORE_REGRESSION_TOLERANCE = 0.015
GROUNDING_SCORE_REGRESSION_TOLERANCE = 0.01
MIN_PARTIAL_SYNTHESIS_COVERAGE_SCORE = 0.75
MIN_PARTIAL_SYNTHESIS_EVIDENCE_CARDS = 12
MIN_EXHAUSTED_SYNTHESIS_COVERAGE_SCORE = 0.70
MIN_EXHAUSTED_SYNTHESIS_EVIDENCE_CARDS = 12
MIN_SEMANTIC_GATE_COVERAGE_SCORE = 0.75
MIN_SEMANTIC_GATE_EVIDENCE_CARDS = 12
ResearchGraphRuntime = Any


def _next_draft_index(artifacts: ResearchArtifactsV2) -> int:
    """Return 1, 2, 3... — the next available draft_report_N.md number."""
    return _next_indexed_file(artifacts, "draft_report_*.md")


def _draft_history_entry(
    *,
    current_draft: dict[str, Any],
    draft: str,
    verification_index: int,
    verification: dict[str, Any],
) -> dict[str, Any] | None:
    draft_index = current_draft.get("index")
    draft_path = str(current_draft.get("path") or "").strip()
    if not isinstance(draft_index, int) or not draft_path:
        return None
    failures = list(verification.get("failures", []))
    scores = _draft_quality_scores(verification)
    return {
        "draft_index": draft_index,
        "draft_path": draft_path,
        "draft_chars": int(current_draft.get("chars") or len(draft)),
        "verification_index": verification_index,
        "verification_path": f"verification_{verification_index}.json",
        "valid": bool(verification.get("valid")),
        "failure_count": len(failures),
        "quality_score": _draft_quality_score(scores, len(failures), bool(verification.get("valid"))),
        "quality_scores": scores,
    }


def _publish_best_draft(artifacts: ResearchArtifactsV2, metrics: dict[str, Any]) -> None:
    best = _select_best_draft(metrics.get("draft_history", []))
    if best is None:
        return
    best_path = str(best.get("draft_path") or "")
    source_path = artifacts.resolve_path(best_path)
    if not source_path.exists():
        return
    draft = source_path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n"
    artifacts.write_text("best_draft.md", draft)
    verification_path = str(best.get("verification_path") or "")
    source_verification = artifacts.resolve_path(verification_path)
    if source_verification.exists():
        artifacts.write_text(
            "best_verification.json",
            source_verification.read_text(encoding="utf-8", errors="replace").rstrip() + "\n",
        )
    metrics["best_draft_index"] = best.get("draft_index")
    metrics["best_draft_path"] = best_path
    metrics["best_draft_valid"] = bool(best.get("valid"))
    metrics["best_draft_failure_count"] = int(best.get("failure_count", 0) or 0)
    metrics["best_verification_path"] = verification_path
    metrics["best_draft_quality_score"] = best.get("quality_score")
    metrics["best_draft_quality_scores"] = best.get("quality_scores")


def _select_best_draft(history: Any) -> dict[str, Any] | None:
    if not isinstance(history, list):
        return None
    candidates = [entry for entry in history if isinstance(entry, dict) and entry.get("draft_path")]
    if not candidates:
        return None

    def rank(entry: dict[str, Any]) -> tuple[int, float, int, int]:
        valid_rank = 0 if entry.get("valid") else 1
        failure_count = int(entry.get("failure_count", 10_000) or 0)
        quality_score = float(entry.get("quality_score", 0.0) or 0.0)
        draft_index = int(entry.get("draft_index", 0) or 0)
        return (valid_rank, -quality_score, failure_count, -draft_index)

    return min(candidates, key=rank)


def _draft_quality_scores(verification: dict[str, Any]) -> dict[str, float]:
    keys = (
        "source_support_score",
        "evidence_linkage_score",
        "citation_validity_score",
        "request_alignment_score",
        "criteria_coverage_score",
        "answer_coverage_score",
        "branch_coverage_score",
        "report_depth_score",
        "semantic_verification_score",
    )
    scores: dict[str, float] = {}
    for key in keys:
        try:
            scores[key] = float(verification.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            scores[key] = 0.0
    return scores


def _draft_quality_score(scores: dict[str, float], failure_count: int, valid: bool) -> float:
    # semantic_verification_score was previously weighted at 0.04 against
    # source_support_score's 0.60 -- a large coherence regression (e.g. 0.7 -> 0.5) could
    # be outweighed by a small citation-support gain, letting a less-coherent draft win
    # best-draft selection even when the repair loop's own regression check (tolerance
    # 0.015) had already flagged that same drop as a fall. Rebalanced so semantic
    # coherence has real influence on which draft gets published, while source_support
    # remains the single largest factor (citation grounding still matters most for FACT).
    weighted = (
        0.45 * scores.get("source_support_score", 0.0)
        + 0.20 * scores.get("semantic_verification_score", 0.0)
        + 0.10 * scores.get("evidence_linkage_score", 0.0)
        + 0.08 * scores.get("request_alignment_score", 0.0)
        + 0.08 * scores.get("criteria_coverage_score", 0.0)
        + 0.06 * scores.get("answer_coverage_score", 0.0)
        + 0.03 * scores.get("report_depth_score", 0.0)
    )
    failure_penalty = min(max(failure_count, 0), 50) * 0.008
    valid_bonus = 0.25 if valid else 0.0
    return round(weighted + valid_bonus - failure_penalty, 6)


def _draft_quality_improved(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_scores = dict(previous.get("quality_scores", {}) or {})
    current_scores = dict(current.get("quality_scores", {}) or {})
    if not previous_scores or not current_scores:
        return int(current.get("failure_count", 10_000) or 10_000) < int(previous.get("failure_count", 10_000) or 10_000)

    grounding_keys = ("source_support_score", "evidence_linkage_score", "citation_validity_score")
    for key in grounding_keys:
        if float(current_scores.get(key, 0.0) or 0.0) + GROUNDING_SCORE_REGRESSION_TOLERANCE < float(previous_scores.get(key, 0.0) or 0.0):
            return False

    quality_keys = (
        "request_alignment_score",
        "criteria_coverage_score",
        "answer_coverage_score",
        "branch_coverage_score",
        "report_depth_score",
        "semantic_verification_score",
    )
    for key in quality_keys:
        if float(current_scores.get(key, 0.0) or 0.0) + QUALITY_SCORE_REGRESSION_TOLERANCE < float(previous_scores.get(key, 0.0) or 0.0):
            return False

    current_quality = float(current.get("quality_score", 0.0) or 0.0)
    previous_quality = float(previous.get("quality_score", 0.0) or 0.0)
    current_failures = int(current.get("failure_count", 10_000) or 10_000)
    previous_failures = int(previous.get("failure_count", 10_000) or 10_000)
    return current_quality >= previous_quality or current_failures < previous_failures


def _repair_fell(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_failures = int(previous.get("failure_count", 10_000) or 10_000)
    current_failures = int(current.get("failure_count", 10_000) or 10_000)
    return current_failures > previous_failures or not _draft_quality_improved(previous, current)


def _repair_fall_count(
    failure_history: list[Any],
    draft_history: list[dict[str, Any]],
) -> int:
    if len(failure_history) < 2:
        return 0
    falls = 0
    for index in range(1, len(failure_history)):
        try:
            previous_failures = int(failure_history[index - 1])
            current_failures = int(failure_history[index])
        except (TypeError, ValueError):
            continue
        issue_count_fell = current_failures > previous_failures
        quality_fell = False
        if len(draft_history) > index:
            quality_fell = _repair_fell(draft_history[index - 1], draft_history[index])
        if issue_count_fell or quality_fell:
            falls += 1
    return falls


def _selected_failed_draft(artifacts: ResearchArtifactsV2, fallback_draft: str) -> str:
    for file_name in ("best_draft.md", "failed_report.md"):
        path = artifacts.resolve_path(file_name)
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return text.rstrip() + "\n"
    return fallback_draft.rstrip() + "\n"


def _write_run_health(
    artifacts: ResearchArtifactsV2,
    metrics: dict[str, Any],
    verification: dict[str, Any],
) -> None:
    draft_history = [entry for entry in metrics.get("draft_history", []) if isinstance(entry, dict)]
    payload = {
        "schema_version": 1,
        "status": "passed" if verification.get("valid") else "failed_verification",
        "verification_valid": bool(verification.get("valid")),
        "verification_failures": len(list(verification.get("failures", []))),
        "verification_rounds": int(metrics.get("verification_rounds", 0) or 0),
        "failure_history": list(metrics.get("verification_failure_history", [])),
        "draft_count": len(draft_history),
        "latest_draft_path": "draft_report.md" if artifacts.resolve_path("draft_report.md").exists() else None,
        "best_draft_path": metrics.get("best_draft_path"),
        "best_draft_index": metrics.get("best_draft_index"),
        "best_draft_failure_count": metrics.get("best_draft_failure_count"),
        "best_draft_valid": metrics.get("best_draft_valid"),
        "best_draft_quality_score": metrics.get("best_draft_quality_score"),
        "best_draft_quality_scores": metrics.get("best_draft_quality_scores"),
        "section_audit_all_locked": metrics.get("section_audit_all_locked"),
        "section_audit_locked_count": metrics.get("section_audit_locked_count"),
        "section_audit_section_count": metrics.get("section_audit_section_count"),
        "section_audit_llm_review_applied": metrics.get("section_audit_llm_review_applied"),
        "knowledge_context_refinement_applied": metrics.get("knowledge_context_refinement_applied"),
        "knowledge_context_refinement_reason": metrics.get("knowledge_context_refinement_reason"),
        "knowledge_context_refinement_history": metrics.get("knowledge_context_refinement_history"),
        "knowledge_context_section_packets": metrics.get("knowledge_context_section_packets"),
        "reasoning_decision": metrics.get("reasoning_decision"),
        "reasoning_iteration_count": metrics.get("reasoning_iteration_count"),
        "reasoning_weak_claim_count": metrics.get("reasoning_weak_claim_count"),
        "reasoning_unknown_count": metrics.get("reasoning_unknown_count"),
        "reasoning_contradiction_count": metrics.get("reasoning_contradiction_count"),
        "reasoning_model_refinement_applied": metrics.get("reasoning_model_refinement_applied"),
        "reasoning_model_refinement_reason": metrics.get("reasoning_model_refinement_reason"),
        "search_intent_count": metrics.get("search_intent_count"),
        "search_intent_model_applied": metrics.get("search_intent_model_applied"),
        "search_intent_generation_reason": metrics.get("search_intent_generation_reason"),
        "search_intent_result_count": metrics.get("search_intent_result_count"),
        "search_intent_satisfied_count": metrics.get("search_intent_satisfied_count"),
        "search_intent_result_model_applied": metrics.get("search_intent_result_model_applied"),
        "replan_iteration_count": metrics.get("replan_iteration_count"),
        "plan_revision_count": metrics.get("plan_revision_count"),
        "source_policy_label": metrics.get("source_policy_label"),
        "best_section_count": metrics.get("best_section_count"),
        "locked_best_section_count": metrics.get("locked_best_section_count"),
        "assembled_best_report_usable": metrics.get("assembled_best_report_usable"),
        "assembled_best_report_reason": metrics.get("assembled_best_report_reason"),
        "assembled_best_report_section_count": metrics.get("assembled_best_report_section_count"),
        "assembled_best_report_locked_count": metrics.get("assembled_best_report_locked_count"),
        "draft_history": draft_history,
        "source_count": metrics.get("source_count"),
        "candidate_count": metrics.get("candidate_count_total", metrics.get("candidate_count")),
        "evidence_card_count": metrics.get("evidence_card_count"),
        "elapsed_seconds": metrics.get("elapsed_seconds"),
        "token_usage": metrics.get("token_usage"),
    }
    artifacts.write_json("run_health.json", payload)


def _next_verification_index(artifacts: ResearchArtifactsV2) -> int:
    """Return 1, 2, 3... — the next available verification_N.json number."""
    return _next_indexed_file(artifacts, "verification_*.json")


def _next_indexed_file(artifacts: ResearchArtifactsV2, glob_pattern: str) -> int:
    run_dir = artifacts.resolve_path(".")
    existing = list(run_dir.glob(glob_pattern))
    indices = []
    for path in existing:
        suffix = path.stem.rsplit("_", 1)[-1]
        try:
            indices.append(int(suffix))
        except ValueError:
            continue
    return max(indices, default=0) + 1


def _coverage_route(state: ResearchState) -> str:
    coverage = state.get("coverage_matrix", {})
    metrics = state.get("metrics", {})
    reasoning = state.get("reasoning_decision", {})
    rounds = int(metrics.get("coverage_rounds", 0))
    max_rounds = int(metrics.get("max_rounds", 4) or 4)
    partial_ready = _partial_coverage_ready_for_synthesis(state)
    exhausted_ready = _exhausted_coverage_ready_for_synthesis(state)
    # Absolute ceiling, checked before anything else below (including the
    # reasoning-driven "search_more" branch). A stale reasoning_decision that
    # never gets re-evaluated -- e.g. because acquire_sources keeps taking the
    # "reuse_evidence" shortcut with no new search intents queued -- must never
    # be able to bypass this. Mirrors the same unconditional-first-check
    # pattern _verification_route already uses for its own loop.
    hard_round_cap = max(max_rounds, MAX_IMPROVING_VERIFICATION_ROUNDS)
    if rounds > hard_round_cap:
        return "synthesize" if partial_ready or exhausted_ready else "finish"
    if (
        reasoning.get("action") == "search_more"
        and _reasoning_budget_available(metrics)
        and _reasoning_search_more_available(metrics)
    ):
        return "more_sources"
    if reasoning.get("action") == "synthesize" and state.get("evidence_cards"):
        return "synthesize"
    if coverage.get("complete"):
        return "synthesize"
    if not state.get("evidence_cards") and _no_evidence_acquisition_stalled(metrics):
        return "finish"
    if not state.get("evidence_cards") and _source_acquisition_plateaued(metrics):
        return "finish"
    search_count = int(metrics.get("search_count", 0))
    # Pipeline wall-clock budget: if we've already spent more than 25 minutes
    # acquiring/processing, stop looping for more sources. Only synthesize a
    # partial report when evidence coverage is still strong enough to score.
    started = metrics.get("started_at_monotonic")
    if started is not None and (time.perf_counter() - float(started)) > 1500:
        return "synthesize" if partial_ready or exhausted_ready else "finish"
    if _source_acquisition_plateaued(metrics):
        return "synthesize" if partial_ready or exhausted_ready else "finish"
    if rounds <= max_rounds and search_count < int(metrics.get("max_search_queries", 10_000) or 10_000):
        return "more_sources"
    if rounds <= max_rounds and _has_unsearched_branch_queries(state):
        return "more_sources"
    if search_count >= int(metrics.get("max_search_queries", 10_000) or 10_000):
        return "synthesize" if partial_ready or exhausted_ready else "finish"
    return "synthesize" if partial_ready or exhausted_ready else "finish"


def _reasoning_route(state: ResearchState) -> str:
    decision = state.get("reasoning_decision", {})
    metrics = state.get("metrics", {})
    if (
        decision.get("action") == "search_more"
        and _reasoning_budget_available(metrics)
        and _reasoning_search_more_available(metrics)
    ):
        return "search_intents"
    if (
        decision.get("action") == "contradiction_search"
        and _reasoning_budget_available(metrics)
        and int(metrics.get("contradiction_search_iterations", 0) or 0) < 1
    ):
        return "contradiction_search"
    return "continue"


def _reasoning_budget_available(metrics: dict[str, Any]) -> bool:
    iteration_count = int(metrics.get("reasoning_iteration_count", 0) or 0)
    max_iterations = int(metrics.get("max_reasoning_iterations", 3) or 3)
    return iteration_count <= max_iterations


def _reasoning_search_more_available(metrics: dict[str, Any]) -> bool:
    if metrics.get("acquisition_time_budget_exhausted"):
        return False
    search_count = int(metrics.get("search_count", 0) or 0)
    max_search_queries = int(metrics.get("max_search_queries", 10_000) or 10_000)
    candidate_count = int(metrics.get("candidate_count_total", metrics.get("candidate_count", 0)) or 0)
    max_candidates = int(metrics.get("max_candidates", 0) or 0)
    if max_candidates > 0 and candidate_count >= max_candidates:
        return False
    return search_count < max_search_queries


def _partial_coverage_ready_for_synthesis(state: ResearchState) -> bool:
    coverage = state.get("coverage_matrix", {})
    try:
        coverage_score = float(coverage.get("coverage_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        coverage_score = 0.0
    evidence_count = len(state.get("evidence_cards", []) or [])
    return (
        coverage_score >= MIN_PARTIAL_SYNTHESIS_COVERAGE_SCORE
        and evidence_count >= MIN_PARTIAL_SYNTHESIS_EVIDENCE_CARDS
    )


def _exhausted_coverage_ready_for_synthesis(state: ResearchState) -> bool:
    metrics = state.get("metrics", {})
    if not _source_acquisition_exhausted(metrics):
        return False
    coverage = state.get("coverage_matrix", {})
    try:
        coverage_score = float(coverage.get("coverage_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        coverage_score = 0.0
    evidence_count = len(state.get("evidence_cards", []) or [])
    return (
        coverage_score >= MIN_EXHAUSTED_SYNTHESIS_COVERAGE_SCORE
        and evidence_count >= MIN_EXHAUSTED_SYNTHESIS_EVIDENCE_CARDS
    )


def _semantic_gate_collapsed_coverage(
    plan: ResearchPlan,
    sources: list[SourceRecordV2],
    *,
    before_cards: list[EvidenceCard],
    after_cards: list[EvidenceCard],
) -> bool:
    if len(after_cards) >= MIN_SEMANTIC_GATE_EVIDENCE_CARDS:
        return False
    if len(before_cards) < MIN_SEMANTIC_GATE_EVIDENCE_CARDS:
        return False

    cards_by_branch_before = {}
    for card in before_cards:
        if card.branch_id:
            cards_by_branch_before[card.branch_id] = cards_by_branch_before.get(card.branch_id, 0) + 1

    cards_by_branch_after = {}
    for card in after_cards:
        if card.branch_id:
            cards_by_branch_after[card.branch_id] = cards_by_branch_after.get(card.branch_id, 0) + 1

    for branch in plan.branches:
        before_cnt = cards_by_branch_before.get(branch.id, 0)
        after_cnt = cards_by_branch_after.get(branch.id, 0)
        if before_cnt >= 1 and after_cnt == 0:
            return True
        if before_cnt >= 2 and after_cnt < 2:
            return True
        if before_cnt >= 4 and after_cnt < 3:
            return True
        if before_cnt >= 6 and after_cnt < 4:
            return True

    before = build_coverage_matrix(branches=plan.branches, evidence_cards=before_cards, sources=sources)
    after = build_coverage_matrix(branches=plan.branches, evidence_cards=after_cards, sources=sources)
    return (
        before.coverage_score >= MIN_SEMANTIC_GATE_COVERAGE_SCORE
        and after.coverage_score < MIN_SEMANTIC_GATE_COVERAGE_SCORE
    )


def _resume_entry_point(state: ResearchState) -> str:
    phase = str(state.get("checkpoint_phase") or "").strip()
    if phase in {"classify_request", "plan"} and state.get("evidence_cards"):
        return "semantic_enrichment"
    if phase in {"classify_request", "plan"} and state.get("source_records"):
        return "read_sources"
    return {
        "classify_request": "plan",
        "plan": "acquire_sources",
        # Partial checkpoints — re-enter the node with saved state so work isn't repeated.
        "acquire_sources_partial": "acquire_sources",
        "acquire_sources": "read_sources",
        "read_sources": "build_evidence",
        "build_evidence_partial": "build_evidence",
        "build_evidence": "evidence_hygiene",
        "evidence_hygiene": "semantic_enrichment",
        "semantic_enrichment_partial": "semantic_enrichment",
        "semantic_enrichment": "build_evidence_graph",
        "build_evidence_graph": "evaluate_search_intents",
        "evaluate_search_intents": "update_reasoning_state",
        "update_reasoning_state": "replan_from_reasoning",
        "replan_from_reasoning": "decide_next_action",
        "decide_next_action": "check_coverage",
        "generate_search_intents": "acquire_sources",
        "prepare_contradiction_search": "acquire_sources",
        "check_coverage": "check_coverage",
        "synthesize": "verify",
        "verify": "verify",
        "repair_or_finish": "repair_or_finish",
        "finish": "repair_or_finish",
    }.get(phase, "classify_request")


def _acquire_route(state: ResearchState) -> str:
    metrics = state.get("metrics", {})
    if int(metrics.get("last_acquire_added_sources", 0) or 0) > 0:
        return "read_sources"
    if _source_acquisition_plateaued(metrics) and state.get("evidence_cards"):
        return "reuse_evidence"
    return "read_sources"


def _source_acquisition_plateaued(metrics: dict[str, Any]) -> bool:
    if _source_acquisition_exhausted(metrics):
        return True
    if (
        int(metrics.get("last_acquire_added_sources", 1)) <= 0
        and int(metrics.get("last_acquire_added_candidates", 1)) <= 0
        and int(metrics.get("last_acquire_searches", 1)) <= 0
    ):
        return True
    candidate_total = int(metrics.get("candidate_count_total", metrics.get("candidate_count", 0)) or 0)
    candidate_budget = int(metrics.get("max_candidates", 0) or 0)
    search_count = int(metrics.get("search_count", 0) or 0)
    search_budget = int(metrics.get("max_search_queries", 0) or 0)
    if candidate_budget > 0 and candidate_total < candidate_budget and (search_budget <= 0 or search_count < search_budget):
        return False
    return (
        int(metrics.get("last_acquire_added_sources", 1)) <= 0
        and int(metrics.get("last_acquire_added_candidates", 1)) <= 0
    )


def _source_acquisition_exhausted(metrics: dict[str, Any]) -> bool:
    source_count = int(metrics.get("source_count", 0) or 0)
    source_budget = int(metrics.get("max_sources", 0) or 0)
    if source_budget > 0 and source_count >= source_budget:
        return True
    candidate_total = int(metrics.get("candidate_count_total", metrics.get("candidate_count", 0)) or 0)
    candidate_budget = int(metrics.get("max_candidates", 0) or 0)
    if candidate_budget > 0 and candidate_total >= candidate_budget:
        return True
    search_count = int(metrics.get("search_count", 0) or 0)
    search_budget = int(metrics.get("max_search_queries", 0) or 0)
    return search_budget > 0 and search_count >= search_budget


def _no_evidence_acquisition_stalled(metrics: dict[str, Any]) -> bool:
    return (
        int(metrics.get("last_acquire_added_sources", 1)) <= 0
        and int(metrics.get("last_acquire_added_candidates", 1)) <= 0
    )


def _verification_route(state: ResearchState) -> str:
    verification = state.get("verification", {})
    if verification.get("valid"):
        return "finish"
    metrics = state.get("metrics", {})
    rounds = int(metrics.get("verification_rounds", 0))
    max_rounds = int(metrics.get("max_rounds", 4) or 4)
    # Anti-thrashing: allow one bad repair so the writer can recover, then stop
    # after the second fall and publish the best draft selected so far.
    failure_history = list(metrics.get("verification_failure_history", []))
    draft_history = [entry for entry in metrics.get("draft_history", []) if isinstance(entry, dict)]
    if _repair_fall_count(failure_history, draft_history) >= 2:
        return "finish"
    hard_round_cap = max(max_rounds, MAX_IMPROVING_VERIFICATION_ROUNDS)
    if rounds >= hard_round_cap:
        return "finish"
    if len(failure_history) >= 3:
        recent = failure_history[-3:]
        # No strict improvement across three rounds.
        if min(recent) >= recent[0] * 0.9:
            return "finish"
    failures = [str(failure).lower() for failure in verification.get("failures", [])]
    # If every failure is purely a judge availability problem, finish with what we
    # have — re-synthesising cannot fix a quota error, and discarding the draft
    # would be wrong.
    judge_unavailable_only = failures and all(
        "quota_or_rate_limit" in f or "judge unavailable" in f for f in failures
    )
    if judge_unavailable_only:
        return "finish"
    rewrite_only = (
        any("semantic judge found unsupported claim" in failure for failure in failures)
        or any("some claims are not directly supported" in failure for failure in failures)
        or any("cited evidence-backed source count" in failure for failure in failures)
        or any("weakly supported cited paragraph" in failure for failure in failures)
        or any("acceptance criteria coverage below threshold" in failure for failure in failures)
        or any("under-covered acceptance criterion" in failure for failure in failures)
        or any("report depth below threshold" in failure for failure in failures)
        or any("semantic report judge returned invalid structured output" in failure for failure in failures)
    )
    if rewrite_only:
        return "rewrite"
    coverage_complete = bool(state.get("coverage_matrix", {}).get("complete"))
    if coverage_complete and any("semantic report verification below threshold" in failure for failure in failures):
        return "rewrite"
    needs_more_sources = (
        any("coverage below threshold" in failure for failure in failures)
        or any("coverage incomplete" in failure for failure in failures)
        or any("source quality" in failure for failure in failures)
        or any("semantic report verification below threshold" in failure for failure in failures)
        or any("missing context" in failure for failure in failures)
    )
    if needs_more_sources:
        return "rewrite" if _source_acquisition_plateaued(metrics) else "more_sources"
    return "rewrite"


def _focus_terms_from_state(state: ResearchState) -> dict[str, list[str]]:
    coverage = state.get("coverage_matrix", {})
    focus: dict[str, list[str]] = {}
    reasoning_focus = state.get("reasoning_focus_terms", {})
    if isinstance(reasoning_focus, dict):
        for branch_id, terms in reasoning_focus.items():
            cleaned = _clean_focus_terms(list(terms) if isinstance(terms, list) else [terms])
            if cleaned:
                focus[str(branch_id)] = cleaned
    missing_branch_ids = {str(branch_id) for branch_id in coverage.get("missing_branches", [])}
    for row in coverage.get("branches", []):
        if row.get("complete"):
            continue
        branch_id = str(row.get("branch_id", ""))
        raw_terms = list(row.get("missing_points", [])) or list(row.get("required_points", []))
        terms = _clean_focus_terms(raw_terms)
        if not terms and raw_terms and all(str(term).startswith(("usable sources", "branch evidence")) for term in raw_terms):
            for branch in state.get("plan", {}).get("branches", []):
                if str(branch.get("id", "")) == branch_id:
                    terms = _clean_focus_terms(list(branch.get("required_terms", [])) or terms)
                    break
        if branch_id and terms:
            focus.setdefault(branch_id, [])
            focus[branch_id].extend(terms)
    verification = state.get("verification", {})
    failures = " ".join(str(failure).lower() for failure in verification.get("failures", []))
    if "answer coverage" in failures or "source quality" in failures or "weakly supported" in failures:
        plan = state.get("plan", {})
        for branch in plan.get("branches", []):
            branch_id = str(branch.get("id", ""))
            terms = _clean_focus_terms(list(branch.get("required_terms", [])))
            if branch_id and terms:
                focus.setdefault(branch_id, [])
                focus[branch_id].extend(terms)
    semantic = verification.get("semantic_verification", {})
    semantic_focus = list(semantic.get("search_focus", [])) + list(semantic.get("missing_context", []))
    if semantic_focus:
        plan = state.get("plan", {})
        for branch in plan.get("branches", []):
            branch_id = str(branch.get("id", ""))
            if branch_id and (not missing_branch_ids or branch_id in missing_branch_ids):
                focus.setdefault(branch_id, [])
                focus[branch_id].extend(_clean_focus_terms(semantic_focus))
    if missing_branch_ids:
        for branch in state.get("plan", {}).get("branches", []):
            branch_id = str(branch.get("id", ""))
            if branch_id in missing_branch_ids and branch_id not in focus:
                terms = _clean_focus_terms(
                    list(branch.get("required_terms", []))
                    + [branch.get("title", ""), branch.get("objective", "")]
                )
                if terms:
                    focus.setdefault(branch_id, [])
                    focus[branch_id].extend(terms)
    return {branch_id: _dedupe_focus_terms(terms) for branch_id, terms in focus.items() if terms}


def _synthesis_repair_guidance_from_state(state: ResearchState) -> list[str]:
    verification = state.get("verification", {})
    guidance = [str(failure) for failure in state.get("failures", []) if str(failure).strip()]
    for weak in verification.get("weakly_supported_claims", []) if isinstance(verification, dict) else []:
        if not isinstance(weak, dict):
            continue
        source_ids = [int(value) for value in weak.get("cited_source_ids", []) if str(value).isdigit()]
        if not source_ids:
            continue
        snippet = re.sub(r"\s+", " ", str(weak.get("paragraph", "")).strip())[:180]
        score = weak.get("support_score")
        kind = str(weak.get("support_kind") or "citation")
        guidance.append(
            "Repair weak citation support: "
            f"{kind} using source(s) {', '.join(f'[{source_id}]' for source_id in source_ids)} "
            f"scored {score}; rewrite or remove the claim unless an evidence card directly supports it. "
            f"Claim context: {snippet}"
        )
    for criterion in verification.get("undercovered_criteria", []) if isinstance(verification, dict) else []:
        if not isinstance(criterion, dict):
            continue
        text = re.sub(r"\s+", " ", str(criterion.get("criterion", "")).strip())[:220]
        if text:
            guidance.append(f"Expand under-covered report criterion with cited evidence: {text}")
    return _dedupe_focus_terms(guidance)


def _active_branch_ids_from_state(state: ResearchState) -> set[str] | None:
    reasoning = state.get("reasoning_decision", {})
    if reasoning.get("action") == "search_more":
        intent_branch_ids = {
            str(row.get("branch_id") or "")
            for row in state.get("search_intents", []) or []
            if isinstance(row, dict) and str(row.get("branch_id") or "").strip()
        }
        if intent_branch_ids:
            return intent_branch_ids
    if reasoning.get("action") in {"search_more", "contradiction_search"}:
        branch_ids = {str(branch_id) for branch_id in reasoning.get("branch_ids", []) if str(branch_id).strip()}
        if branch_ids:
            return branch_ids
    coverage = state.get("coverage_matrix", {})
    if not coverage or coverage.get("complete"):
        return None
    missing = {str(branch_id) for branch_id in coverage.get("missing_branches", []) if str(branch_id).strip()}
    return missing or None


def _clean_focus_terms(terms: list[Any]) -> list[str]:
    cleaned_terms: list[str] = []
    for term in terms:
        cleaned = str(term).strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower.startswith(("usable sources", "branch evidence")):
            continue
        if "required term coverage" in lower:
            continue
        cleaned = re.sub(r"(?i)^required\s+term\s*:\s*", "", cleaned).strip()
        cleaned = re.sub(r"\s*\(\s*actual\s+\d+%?\s*\)\s*$", "", cleaned, flags=re.I).strip()
        if not cleaned or ">=" in cleaned:
            continue
        cleaned_terms.append(cleaned)
    return _dedupe_focus_terms(cleaned_terms)


def _dedupe_focus_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        key = " ".join(term.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(term)
    return result


def _has_unsearched_branch_queries(state: ResearchState) -> bool:
    plan = state.get("plan", {})
    candidates = state.get("source_candidates", [])
    searched = {str(candidate.get("query", "")) for candidate in candidates}
    for branch in plan.get("branches", []):
        if branch.get("id") not in set(state.get("coverage_matrix", {}).get("missing_branches", [])):
            continue
        for query in branch.get("queries", []):
            if query not in searched:
                return True
    return False


def _with_checkpoint(
    runtime: ResearchGraphRuntime,
    phase: str,
    state: ResearchState,
    updates: ResearchState,
) -> ResearchState:
    next_state = dict(state)
    next_state.update(updates)
    runtime.checkpoint(phase, next_state)
    return updates


def _plan_from_state(state: ResearchState) -> ResearchPlan:
    payload = dict(state.get("plan") or {})
    branches = [_branch_from_dict(row) for row in payload.get("branches", [])]
    return ResearchPlan(
        question=str(payload.get("question") or state.get("request", {}).get("question") or ""),
        intent=payload.get("intent", "general"),
        audience=str(payload.get("audience", "technical generalist")),
        report_outline=list(payload.get("report_outline", [])),
        branches=branches,
        source_requirements=[],  # SourceRequirement is only needed for persisted plan metadata.
        acceptance_criteria=list(payload.get("acceptance_criteria", [])),
    )


def _branch_from_dict(payload: dict[str, Any]) -> ResearchBranch:
    return ResearchBranch(
        id=str(payload.get("id", "")),
        title=str(payload.get("title", "")),
        objective=str(payload.get("objective", "")),
        queries=list(payload.get("queries", [])),
        source_types=list(payload.get("source_types", [])),
        min_sources=int(payload.get("min_sources", 1)),
        required_terms=list(payload.get("required_terms", [])),
        completion_criteria=list(payload.get("completion_criteria", [])),
    )


def _candidate_from_dict(payload: dict[str, Any]) -> SourceCandidate:
    return SourceCandidate(
        id=int(payload["id"]),
        branch_id=str(payload["branch_id"]),
        title=str(payload["title"]),
        url=str(payload["url"]),
        query=str(payload.get("query", "")),
        snippet=str(payload.get("snippet", "")),
        search_score=payload.get("search_score"),
        raw_content=payload.get("raw_content"),
        provenance=payload.get("provenance", "web"),
        search_intent_id=str(payload.get("search_intent_id", "")),
        search_intent_goal=str(payload.get("search_intent_goal", "")),
    )


def _source_from_dict(payload: dict[str, Any]) -> SourceRecordV2:
    return SourceRecordV2(
        id=int(payload["id"]),
        branch_id=str(payload["branch_id"]),
        title=str(payload["title"]),
        url=str(payload["url"]),
        canonical_url=str(payload.get("canonical_url") or payload["url"]),
        provenance=payload.get("provenance", "web"),
        content_path=str(payload.get("content_path", "")),
        content_hash=str(payload.get("content_hash", "")),
        extraction_method=str(payload.get("extraction_method", "")),
        word_count=int(payload.get("word_count", 0)),
        quality_score=float(payload.get("quality_score", 0.0)),
        quality_label=str(payload.get("quality_label", "")),
        quality_type=str(payload.get("quality_type", "")),
        relevance_score=float(payload.get("relevance_score", 0.0)),
        usable=bool(payload.get("usable", True)),
        rejection_reasons=list(payload.get("rejection_reasons", [])),
        metadata=dict(payload.get("metadata", {})),
    )


def _card_from_dict(payload: dict[str, Any]) -> EvidenceCard:
    return EvidenceCard(
        id=int(payload["id"]),
        source_id=int(payload["source_id"]),
        branch_id=str(payload["branch_id"]),
        claim=str(payload["claim"]),
        supporting_excerpt=str(payload["supporting_excerpt"]),
        source_url=str(payload["source_url"]),
        source_title=str(payload["source_title"]),
        quality_score=float(payload.get("quality_score", 0.0)),
        relevance_score=float(payload.get("relevance_score", 0.0)),
        confidence=float(payload.get("confidence", 0.0)),
        limitations=list(payload.get("limitations", [])),
        semantic_score=payload.get("semantic_score"),
        semantic_notes=list(payload.get("semantic_notes", [])),
    )


def _coverage_from_dict(payload: dict[str, Any]) -> CoverageMatrix:
    rows = [
        BranchCoverage(
            branch_id=str(row.get("branch_id", "")),
            required_points=list(row.get("required_points", [])),
            covered_points=list(row.get("covered_points", [])),
            missing_points=list(row.get("missing_points", [])),
            source_count=int(row.get("source_count", 0)),
            complete=bool(row.get("complete", False)),
        )
        for row in payload.get("branches", [])
    ]
    return CoverageMatrix(
        branches=rows,
        complete=bool(payload.get("complete", False)),
        coverage_score=float(payload.get("coverage_score", 0.0)),
        missing_branches=list(payload.get("missing_branches", [])),
    )


def _load_source_texts(artifacts: ResearchArtifactsV2, sources: list[SourceRecordV2]) -> dict[int, str]:
    texts: dict[int, str] = {}
    for source in sources:
        if not source.content_path:
            continue
        path = artifacts.resolve_path(source.content_path)
        if path.exists():
            texts[source.id] = _strip_source_file_header(path.read_text(encoding="utf-8"))
    return texts


_SOURCE_METADATA_FIELD_RE = re.compile(
    r"^(URL:\s|Canonical URL:\s|Branch:\s|Extraction method:\s|Word count:\s)"
)


def _strip_source_file_header(raw: str) -> str:
    """Remove the metadata header written by _write_source() before passing
    content to evidence extraction and synthesis.  The header is kept in the
    file on disk for debugging but must not be treated as source content.

    Header format:
        # <title>

        URL: ...
        Canonical URL: ...
        Branch: ...
        Extraction method: ...
        Word count: ...

        <actual page content starts here>
    """
    lines = raw.splitlines(keepends=True)
    saw_metadata_field = False
    for i, line in enumerate(lines):
        if _SOURCE_METADATA_FIELD_RE.match(line):
            saw_metadata_field = True
        elif saw_metadata_field and line.strip() == "":
            # First blank line after the last metadata field — content follows
            return "".join(lines[i + 1:])
    return raw


def _merge_metrics(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in updates.items():
        if isinstance(value, int) and not isinstance(value, bool) and isinstance(merged.get(key), int) and not isinstance(merged.get(key), bool):
            merged[key] = int(merged[key]) + value
        elif key == "failures":
            merged[key] = list(merged.get(key, [])) + list(value)
        else:
            merged[key] = value
    return merged


def _public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"started_at_monotonic", "finished_at_monotonic"}
    }



from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from deep_research.acquisition.acquisition import acquire_sources
from deep_research.runtime.artifacts_v2 import ResearchArtifactsV2
from deep_research.reasoning.coverage import build_coverage_matrix
from deep_research.evidence.evidence import build_evidence_cards
from deep_research.evidence.evidence_hygiene import apply_evidence_hygiene
from deep_research.graph.context_builder import build_knowledge_base, write_knowledge_base
from deep_research.graph.context_builder_runtime import refine_knowledge_base_with_model
from deep_research.reasoning.contradiction_search import generate_contradiction_queries
from deep_research.evidence.evidence_graph import build_evidence_graph as build_evidence_graph_artifact
from deep_research.acquisition.ingestion import IngestedDocument
from deep_research.runtime.progress import ActivityLog
from deep_research.reasoning.reasoning_state import build_research_state_artifact, decide_next_action
from deep_research.reasoning.reasoning_runtime import refine_reasoning_state_with_model
from deep_research.reasoning.reasoning_summary import reasoning_brief_for_prompt, render_reasoning_summary
from deep_research.planning.search_intents import (
    SearchIntent,
    apply_plan_revisions_with_model,
    evaluate_search_intent_results_with_model,
    generate_search_intents_with_model,
)
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
from deep_research.core.settings import Settings
from deep_research.verification.semantic import (
    apply_semantic_report_result,
    enrich_evidence_cards_with_semantics,
    verify_report_with_semantics,
)
from deep_research.synthesis.section_audit_runtime import audit_report_sections_with_model, section_audit_failures
from deep_research.models.model_router import get_and_reset_token_usage
from deep_research.planning.semantic_planning import build_or_enrich_research_plan
from deep_research.synthesis.section_history import assemble_best_section_report, publish_section_versions
from deep_research.synthesis.section_writing import build_adaptive_section_plan
from deep_research.evidence.source_policy import infer_source_policy, score_sources_against_policy
from deep_research.synthesis.synthesis import (
    build_claim_ledger,
    build_report_blueprint,
    build_sentence_plan,
    synthesize_report,
    synthesize_report_with_model,
)
from deep_research.verification.verifier_v2 import verify_report_v2
from deep_research.verification.section_batch_judge import batched_section_failures, judge_report_sections_batched
from deep_research.verification.race_judge import race_self_judge
from deep_research.synthesis.targeted_repair import apply_targeted_citation_repair, classify_repair_failures
from deep_research.synthesis.citation_agent import apply_citations
from deep_research.graph.research_graph_helpers import (
    _acquire_route,
    _active_branch_ids_from_state,
    _branch_from_dict,
    _candidate_from_dict,
    _card_from_dict,
    _coverage_from_dict,
    _coverage_route,
    _draft_history_entry,
    _focus_terms_from_state,
    _load_source_texts,
    _merge_metrics,
    _next_draft_index,
    _next_verification_index,
    _no_evidence_acquisition_stalled,
    _partial_coverage_ready_for_synthesis,
    _plan_from_state,
    _public_metrics,
    _publish_best_draft,
    _reasoning_route,
    _resume_entry_point,
    _select_best_draft,
    _selected_failed_draft,
    _semantic_gate_collapsed_coverage,
    _source_from_dict,
    _synthesis_repair_guidance_from_state,
    _verification_route,
    _with_checkpoint,
    _write_run_health,
)

MAX_IMPROVING_VERIFICATION_ROUNDS = 8
QUALITY_SCORE_REGRESSION_TOLERANCE = 0.015
GROUNDING_SCORE_REGRESSION_TOLERANCE = 0.01
MIN_PARTIAL_SYNTHESIS_COVERAGE_SCORE = 0.75
MIN_PARTIAL_SYNTHESIS_EVIDENCE_CARDS = 12
MIN_SEMANTIC_GATE_COVERAGE_SCORE = 0.75
MIN_SEMANTIC_GATE_EVIDENCE_CARDS = 12


@dataclass
class ResearchGraphRuntime:
    settings: Settings
    artifacts: ResearchArtifactsV2
    activity: ActivityLog | None = None
    search_client: Any | None = None
    scraper: Any | None = None
    local_documents: list[IngestedDocument] = field(default_factory=list)
    mcp_documents: list[IngestedDocument] = field(default_factory=list)
    source_texts: dict[int, str] = field(default_factory=dict)
    searched_queries: set[str] = field(default_factory=set)

    def emit_status(self, phase: str, message: str, **data: Any) -> None:
        if self.activity is None:
            return
        self.activity.emit(
            "research_status",
            message,
            kind="research_status",
            data={"phase": phase, **data},
        )

    def checkpoint(self, phase: str, state: ResearchState) -> None:
        serializable = dict(state)
        serializable["checkpoint_phase"] = phase
        self.artifacts.write_json("checkpoints/latest.json", serializable)
        self.artifacts.write_json(f"checkpoints/{phase}.json", serializable)


def run_local_research_graph(
    *,
    question: str,
    settings: Settings,
    artifacts: ResearchArtifactsV2,
    activity: ActivityLog | None = None,
    search_client: Any | None = None,
    scraper: Any | None = None,
    local_documents: list[IngestedDocument] | None = None,
    mcp_documents: list[IngestedDocument] | None = None,
    initial_state: ResearchState | None = None,
    writing_guidance: str = "",
) -> ResearchState:
    runtime = ResearchGraphRuntime(
        settings=settings,
        artifacts=artifacts,
        activity=activity,
        search_client=search_client,
        scraper=scraper,
        local_documents=list(local_documents or []),
        mcp_documents=list(mcp_documents or []),
    )
    if initial_state:
        state = dict(initial_state)
        request = dict(state.get("request", {}))
        request["question"] = str(request.get("question") or question).strip()
        request["engine"] = "local_langgraph"
        request["mode"] = settings.mode
        if writing_guidance.strip():
            request["writing_guidance"] = writing_guidance.strip()
        state["request"] = request
        metrics = dict(state.get("metrics", {}))
        metrics["engine"] = "local_langgraph"
        metrics["max_search_queries"] = settings.max_search_queries
        metrics["max_candidates"] = settings.max_candidates
        metrics["max_sources"] = settings.max_sources
        metrics["max_rounds"] = settings.max_rounds
        metrics.setdefault("max_reasoning_iterations", 5)
        metrics["started_at_monotonic"] = time.perf_counter()
        state["metrics"] = metrics
        runtime.searched_queries.update(
            str(candidate.get("query", ""))
            for candidate in state.get("source_candidates", [])
            if str(candidate.get("query", "")).strip()
        )
        entry_point = _resume_entry_point(state)
    else:
        state = {
            "request": {
                "question": question.strip(),
                "engine": "local_langgraph",
                "mode": settings.mode,
                "writing_guidance": writing_guidance.strip(),
            },
            "source_candidates": [],
            "source_records": [],
            "evidence_cards": [],
            "metrics": {
                "engine": "local_langgraph",
                "coverage_rounds": 0,
                "max_search_queries": settings.max_search_queries,
                "max_candidates": settings.max_candidates,
                "max_sources": settings.max_sources,
                "max_rounds": settings.max_rounds,
                "max_reasoning_iterations": 5,
                "started_at_monotonic": time.perf_counter(),
            },
            "failures": [],
        }
        entry_point = "classify_request"
    graph = build_research_graph(runtime, entry_point=entry_point)
    final_state = graph.invoke(state, config={"configurable": {"thread_id": artifacts.run_dir.name}})
    return final_state


def build_research_graph(runtime: ResearchGraphRuntime, *, entry_point: str = "classify_request"):
    graph = StateGraph(ResearchState)

    def classify_request(state: ResearchState) -> ResearchState:
        request = dict(state.get("request", {}))
        question = str(request.get("question", "")).strip()
        if not question:
            raise ValueError("Research question cannot be empty.")
        request["engine"] = "local_langgraph"
        request["mode"] = runtime.settings.mode
        runtime.artifacts.write_json("request.json", request)
        runtime.emit_status("classify_request", "classified request", question=question)
        return _with_checkpoint(runtime, "classify_request", state, {"request": request})

    def plan(state: ResearchState) -> ResearchState:
        request = state["request"]
        enrichment = build_or_enrich_research_plan(
            str(request["question"]),
            settings=runtime.settings,
            planning_guidance=str(request.get("writing_guidance") or ""),
        )
        plan_obj = enrichment.plan
        plan_dict = plan_obj.to_dict()
        runtime.artifacts.write_json("plan.json", plan_dict)
        runtime.artifacts.write_json("plan_enrichment.json", enrichment.to_dict())
        metrics = dict(state.get("metrics", {}))
        metrics["llm_planning_enabled"] = runtime.settings.llm_planning
        metrics["llm_planning_accepted"] = enrichment.accepted
        metrics["llm_planning_failure_count"] = len(enrichment.failures)
        runtime.emit_status(
            "plan",
            f"planned {len(plan_obj.branches)} research branches"
            + (" with LLM semantic enrichment" if enrichment.accepted else " with deterministic fallback"),
            branches=[branch.id for branch in plan_obj.branches],
            llm_planning_accepted=enrichment.accepted,
            llm_planning_failures=enrichment.failures[:3],
        )
        return _with_checkpoint(runtime, "plan", state, {"plan": plan_dict, "metrics": metrics})

    def check_plan_quality(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        quality = _assess_plan_quality(plan_obj)
        runtime.artifacts.write_json("plan_quality.json", quality)
        status = quality["status"]
        issues = quality.get("issues", [])
        runtime.emit_status(
            "check_plan_quality",
            f"plan quality {status}: score={quality['overall_score']:.2f}, branches={quality['branch_count']}",
            issues=issues[:5],
        )
        return _with_checkpoint(runtime, "check_plan_quality", state, {})

    def acquire(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        existing_candidates = [_candidate_from_dict(row) for row in state.get("source_candidates", [])]
        existing_sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, existing_sources))

        def _mid_checkpoint(
            candidates: list,
            sources: list,
        ) -> None:
            """Save partial acquisition state so a crash+resume can continue mid-node."""
            mid_state: ResearchState = {
                **state,
                "source_candidates": [c.to_dict() for c in candidates],
                "source_records": [s.to_dict() for s in sources],
                "checkpoint_phase": "acquire_sources_partial",
            }
            runtime.artifacts.write_json("checkpoints/acquire_sources_partial.json", mid_state)
            runtime.artifacts.write_json("checkpoints/latest.json", mid_state)

        result = acquire_sources(
            question=plan_obj.question,
            branches=plan_obj.branches,
            artifacts=runtime.artifacts,
            settings=runtime.settings,
            search_client=runtime.search_client,
            scraper=runtime.scraper,
            local_documents=runtime.local_documents,
            mcp_documents=runtime.mcp_documents,
            existing_candidates=existing_candidates,
            existing_sources=existing_sources,
            existing_source_texts=runtime.source_texts,
            searched_queries=runtime.searched_queries,
            focus_terms_by_branch=_focus_terms_from_state(state),
            active_branch_ids=_active_branch_ids_from_state(state),
            search_intents=(
                _search_intents_from_state(state)
                if dict(state.get("reasoning_decision", {}) or {}).get("action") == "search_more"
                else None
            ),
            progress_callback=lambda message, data: runtime.emit_status("acquire_sources", message, **data),
            mid_checkpoint_callback=_mid_checkpoint,
        )
        runtime.source_texts.update(result.source_texts)
        runtime.searched_queries.update(candidate.query for candidate in result.candidates)
        runtime.artifacts.write_jsonl("sources.jsonl", [source.to_dict() for source in result.sources])
        metrics = _merge_metrics(state.get("metrics", {}), result.metrics.to_dict())
        metrics["last_acquire_added_sources"] = len(result.sources) - len(existing_sources)
        metrics["last_acquire_added_candidates"] = len(result.candidates) - len(existing_candidates)
        metrics["last_acquire_searches"] = result.metrics.search_count
        metrics["source_count"] = len(result.sources)
        metrics["candidate_count_total"] = len(result.candidates)
        runtime.emit_status(
            "acquire_sources",
            f"collected {len(result.sources)} usable sources from {len(result.candidates)} candidates",
            sources=len(result.sources),
            candidates=len(result.candidates),
            searches=result.metrics.search_count,
        )
        updates: ResearchState = {
            "source_candidates": [candidate.to_dict() for candidate in result.candidates],
            "source_records": [source.to_dict() for source in result.sources],
            "metrics": metrics,
        }
        return _with_checkpoint(runtime, "acquire_sources", state, updates)

    def read_sources(state: ResearchState) -> ResearchState:
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, sources))
        runtime.emit_status("read_sources", f"read {len(runtime.source_texts)} source documents")
        return _with_checkpoint(runtime, "read_sources", state, {})

    def build_evidence(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, sources))
        # Restore any cards already built in a partial run so processed sources are skipped.
        existing_cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]

        def _evidence_mid_checkpoint(cards: list) -> None:
            partial: ResearchState = {
                **state,
                "evidence_cards": [c.to_dict() for c in cards],
                "checkpoint_phase": "build_evidence_partial",
            }
            runtime.artifacts.write_json("checkpoints/build_evidence_partial.json", partial)
            runtime.artifacts.write_json("checkpoints/latest.json", partial)

        cards = build_evidence_cards(
            branches=plan_obj.branches,
            sources=sources,
            source_texts=runtime.source_texts,
            question=plan_obj.question,
            max_cards_per_source=3,
            existing_cards=existing_cards,
            source_checkpoint_callback=_evidence_mid_checkpoint,
        )
        runtime.artifacts.write_jsonl("evidence_cards.jsonl", [card.to_dict() for card in cards])
        metrics = dict(state.get("metrics", {}))
        metrics["raw_evidence_card_count"] = len(cards)
        runtime.emit_status("build_evidence", f"built {len(cards)} raw evidence cards", cards=len(cards))
        return _with_checkpoint(
            runtime,
            "build_evidence",
            state,
            {"evidence_cards": [card.to_dict() for card in cards], "metrics": metrics},
        )

    def evidence_hygiene(state: ResearchState) -> ResearchState:
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        result = apply_evidence_hygiene(cards)
        runtime.artifacts.write_jsonl("evidence_cards.jsonl", [card.to_dict() for card in result.kept])
        runtime.artifacts.write_jsonl("evidence_rejections.jsonl", result.rejected)
        metrics = dict(state.get("metrics", {}))
        metrics["evidence_card_count"] = len(result.kept)
        metrics["rejected_evidence_card_count"] = len(result.rejected)
        runtime.emit_status(
            "evidence_hygiene",
            f"kept {len(result.kept)} evidence cards; rejected {len(result.rejected)}",
            kept=len(result.kept),
            rejected=len(result.rejected),
        )
        return _with_checkpoint(
            runtime,
            "evidence_hygiene",
            state,
            {
                "evidence_cards": [card.to_dict() for card in result.kept],
                "evidence_rejections": result.rejected,
                "metrics": metrics,
            },
        )

    def semantic_enrichment(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        prior_judgments = list(dict(state.get("semantic_judgments", {})).get("judgments", []))

        def _semantic_mid_checkpoint(judgments: list) -> None:
            partial: ResearchState = {
                **state,
                "semantic_judgments": {"judgments": judgments},
                "checkpoint_phase": "semantic_enrichment_partial",
            }
            runtime.artifacts.write_json("checkpoints/semantic_enrichment_partial.json", partial)
            runtime.artifacts.write_json("checkpoints/latest.json", partial)

        result = enrich_evidence_cards_with_semantics(
            plan=plan_obj,
            evidence_cards=cards,
            settings=runtime.settings,
            prior_judgments=prior_judgments,
            batch_checkpoint_callback=_semantic_mid_checkpoint,
        )
        semantic_payload = {
            "enabled": runtime.settings.semantic_verification,
            "judgments": result.judgments,
            "rejected": result.rejected,
            "failures": result.failures,
            "metrics": result.metrics,
        }
        runtime.artifacts.write_json("semantic_judgments.json", semantic_payload)
        runtime.artifacts.write_jsonl("semantic_evidence_rejections.jsonl", result.rejected)
        selected_cards = result.cards
        metrics = _merge_metrics(state.get("metrics", {}), result.metrics)
        if _semantic_gate_collapsed_coverage(plan_obj, sources, before_cards=cards, after_cards=result.cards):
            selected_cards = cards
            metrics["semantic_evidence_gate_fallback_coverage_collapse"] = True
            metrics["semantic_evidence_gate_original_kept_count"] = len(result.cards)
            metrics["semantic_evidence_gate_fallback_card_count"] = len(selected_cards)
            runtime.emit_status(
                "semantic_enrichment",
                "semantic evidence gate fallback: restored pre-gate cards after coverage collapse",
                enabled=runtime.settings.semantic_verification,
                kept=len(result.cards),
                restored=len(selected_cards),
                rejected=len(result.rejected),
            )
        runtime.artifacts.write_jsonl("evidence_cards.jsonl", [card.to_dict() for card in selected_cards])
        metrics["evidence_card_count"] = len(selected_cards)
        failures = list(state.get("failures", [])) + result.failures
        runtime.emit_status(
            "semantic_enrichment",
            f"semantic evidence gate kept {len(selected_cards)} cards; rejected {len(result.rejected)}",
            enabled=runtime.settings.semantic_verification,
            kept=len(selected_cards),
            rejected=len(result.rejected),
            failures=result.failures[:5],
        )
        return _with_checkpoint(
            runtime,
            "semantic_enrichment",
            state,
            {
                "evidence_cards": [card.to_dict() for card in selected_cards],
                "semantic_judgments": semantic_payload,
                "metrics": metrics,
                "failures": failures,
            },
        )

    def build_evidence_graph(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, sources))
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        graph_payload = build_evidence_graph_artifact(
            plan=plan_obj,
            sources=sources,
            evidence_cards=cards,
            source_texts=runtime.source_texts,
            semantic_judgments=dict(state.get("semantic_judgments", {}) or {}),
        ).to_dict()
        runtime.artifacts.write_json("evidence_graph.json", graph_payload)
        runtime.artifacts.write_json("contradictions.json", graph_payload.get("contradiction_edges", []))
        metrics = dict(state.get("metrics", {}))
        graph_metrics = dict(graph_payload.get("metrics", {}) or {})
        metrics["evidence_graph_claim_count"] = graph_metrics.get("claim_count", 0)
        metrics["evidence_graph_weak_claim_count"] = graph_metrics.get("weak_claim_count", 0)
        metrics["evidence_graph_contradiction_count"] = graph_metrics.get("contradiction_count", 0)
        runtime.emit_status(
            "build_evidence_graph",
            (
                f"built evidence graph with {graph_metrics.get('claim_count', 0)} claim(s), "
                f"{graph_metrics.get('weak_claim_count', 0)} weak"
            ),
            claims=graph_metrics.get("claim_count", 0),
            weak_claims=graph_metrics.get("weak_claim_count", 0),
            contradictions=graph_metrics.get("contradiction_count", 0),
        )
        return _with_checkpoint(
            runtime,
            "build_evidence_graph",
            state,
            {
                "evidence_graph": graph_payload,
                "contradictions": graph_payload.get("contradiction_edges", []),
                "metrics": metrics,
            },
        )

    def evaluate_search_intents(state: ResearchState) -> ResearchState:
        intents = _search_intents_from_state(state)
        if not intents:
            runtime.artifacts.write_json("search_intent_results.json", [])
            return _with_checkpoint(runtime, "evaluate_search_intents", state, {"search_intent_results": []})
        plan_obj = _plan_from_state(state)
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        results, metadata = evaluate_search_intent_results_with_model(
            intents=intents,
            sources=sources,
            evidence_cards=cards,
            settings=runtime.settings,
            plan=plan_obj,
            writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
        )
        result_rows = [row.to_dict() for row in results]
        runtime.artifacts.write_json("search_intent_results.json", result_rows)
        runtime.artifacts.write_json("search_intent_result_evaluation.json", metadata)
        metrics = dict(state.get("metrics", {}))
        metrics["search_intent_result_count"] = len(result_rows)
        metrics["search_intent_satisfied_count"] = sum(1 for row in result_rows if row.get("status") == "satisfied")
        metrics["search_intent_result_model_applied"] = bool(metadata.get("applied"))
        runtime.emit_status(
            "evaluate_search_intents",
            f"evaluated {len(result_rows)} search intent result(s)",
            satisfied=metrics["search_intent_satisfied_count"],
            model_applied=metadata.get("applied", False),
        )
        return _with_checkpoint(
            runtime,
            "evaluate_search_intents",
            state,
            {"search_intent_results": result_rows, "metrics": metrics},
        )

    def update_reasoning_state(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        coverage = build_coverage_matrix(branches=plan_obj.branches, evidence_cards=cards, sources=sources)
        policy = infer_source_policy(plan_obj)
        policy_payload = score_sources_against_policy(policy, sources)
        policy_payload["policy"] = policy.to_dict()
        reasoning = build_research_state_artifact(
            plan=plan_obj,
            evidence_graph=dict(state.get("evidence_graph", {}) or {}),
            coverage=coverage,
            source_policy=policy_payload,
            metrics=dict(state.get("metrics", {}) or {}),
        ).to_dict()
        metrics = dict(state.get("metrics", {}))
        reflection_log = list(metrics.get("reasoning_reflection_log", []) or [])
        reasoning = refine_reasoning_state_with_model(
            reasoning_state=reasoning,
            evidence_graph=dict(state.get("evidence_graph", {}) or {}),
            plan=plan_obj,
            evidence_cards=cards,
            sources=sources,
            settings=runtime.settings,
            writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
            reflection_log=reflection_log,
        )
        recommended = reasoning.get("model_recommended_action")
        if isinstance(recommended, dict) and recommended.get("rationale"):
            reflection_log.append(f"decided '{recommended.get('action')}' -- {recommended['rationale']}")
            metrics["reasoning_reflection_log"] = reflection_log
        runtime.artifacts.write_json("source_policy.json", policy_payload)
        runtime.artifacts.write_json("reasoning_state.json", reasoning)
        runtime.artifacts.write_json("reasoning_refinement.json", reasoning.get("model_refinement", {}))
        metrics["reasoning_weak_claim_count"] = int(reasoning.get("summary", {}).get("weak_claim_count", 0) or 0)
        metrics["reasoning_unknown_count"] = int(reasoning.get("summary", {}).get("unknown_count", 0) or 0)
        metrics["reasoning_contradiction_count"] = int(reasoning.get("summary", {}).get("contradiction_count", 0) or 0)
        metrics["reasoning_model_refinement_applied"] = bool(reasoning.get("model_refinement", {}).get("applied"))
        metrics["reasoning_model_refinement_reason"] = reasoning.get("model_refinement", {}).get("reason")
        metrics["source_policy_label"] = policy.label
        metrics["source_policy_score"] = policy_payload.get("score")
        runtime.emit_status(
            "update_reasoning_state",
            (
                f"reasoning state: {reasoning.get('readiness_status')}; "
                f"weak={metrics['reasoning_weak_claim_count']}, unknown={metrics['reasoning_unknown_count']}"
            ),
            readiness=reasoning.get("readiness_status"),
            weak_claims=metrics["reasoning_weak_claim_count"],
            unknowns=metrics["reasoning_unknown_count"],
            source_policy=policy.label,
        )
        return _with_checkpoint(
            runtime,
            "update_reasoning_state",
            state,
            {
                "coverage_matrix": coverage.to_dict(),
                "source_policy": policy_payload,
                "reasoning_state": reasoning,
                "metrics": metrics,
            },
        )

    def decide_reasoning_next_action(state: ResearchState) -> ResearchState:
        metrics = dict(state.get("metrics", {}))
        metrics["reasoning_iteration_count"] = int(metrics.get("reasoning_iteration_count", 0) or 0) + 1
        decision = decide_next_action(
            reasoning_state=dict(state.get("reasoning_state", {}) or {}),
            evidence_graph=dict(state.get("evidence_graph", {}) or {}),
            coverage=dict(state.get("coverage_matrix", {}) or {}),
            metrics=metrics,
        )
        metrics["reasoning_decision"] = decision.get("action")
        focus_terms = _focus_terms_by_branch(decision, dict(state.get("reasoning_state", {}) or {}))
        runtime.artifacts.write_json("reasoning_decision.json", decision)
        runtime.artifacts.write_text(
            "reasoning_summary.md",
            render_reasoning_summary(
                query=str(state.get("request", {}).get("question") or ""),
                source_policy=dict(state.get("source_policy", {}) or {}),
                reasoning_state=dict(state.get("reasoning_state", {}) or {}),
                reasoning_decision=decision,
                contradiction_queries=list(state.get("contradiction_queries", []) or []),
                search_intents=list(state.get("search_intents", []) or []),
                search_intent_results=list(state.get("search_intent_results", []) or []),
                plan_revisions=dict(state.get("plan_revisions", {}) or {}),
            ),
        )
        runtime.emit_status(
            "decide_next_action",
            f"reasoning decision: {decision.get('action')}",
            rationale=decision.get("rationale"),
            branch_ids=decision.get("branch_ids", []),
            deferred=decision.get("deferred", False),
        )
        return _with_checkpoint(
            runtime,
            "decide_next_action",
            state,
            {
                "reasoning_decision": decision,
                "reasoning_focus_terms": focus_terms,
                "metrics": metrics,
            },
        )

    def replan_from_reasoning(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        metrics = dict(state.get("metrics", {}))
        revised_plan, metadata = apply_plan_revisions_with_model(
            plan=plan_obj,
            reasoning_state=dict(state.get("reasoning_state", {}) or {}),
            search_intent_results=list(state.get("search_intent_results", []) or []),
            settings=runtime.settings,
            writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
            iteration_count=int(metrics.get("replan_iteration_count", 0) or 0),
        )
        runtime.artifacts.write_json("plan_revisions.json", metadata)
        updates: ResearchState = {"plan_revisions": metadata}
        if metadata.get("reason") not in {"no_unsatisfied_intents_or_unknowns", "replan_iteration_budget_exhausted"}:
            metrics["replan_iteration_count"] = int(metrics.get("replan_iteration_count", 0) or 0) + 1
        if metadata.get("applied"):
            metrics["plan_revision_count"] = len(metadata.get("applied_revisions", []) or [])
            plan_dict = revised_plan.to_dict()
            runtime.artifacts.write_json("plan.json", plan_dict)
            updates["plan"] = plan_dict
            runtime.emit_status(
                "replan_from_reasoning",
                f"applied {metrics['plan_revision_count']} additive plan revision(s)",
                revisions=metadata.get("applied_revisions", []),
            )
        else:
            runtime.emit_status(
                "replan_from_reasoning",
                "no additive plan revision applied",
                reason=metadata.get("reason"),
            )
        updates["metrics"] = metrics
        return _with_checkpoint(runtime, "replan_from_reasoning", state, updates)

    def generate_search_intents(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        coverage = _coverage_from_dict(dict(state.get("coverage_matrix", {}) or {}))
        intents, metadata = generate_search_intents_with_model(
            plan=plan_obj,
            coverage=coverage,
            reasoning_state=dict(state.get("reasoning_state", {}) or {}),
            evidence_graph=dict(state.get("evidence_graph", {}) or {}),
            source_policy=dict(state.get("source_policy", {}) or {}),
            settings=runtime.settings,
            writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
        )
        intent_rows = [intent.to_dict() for intent in intents]
        by_branch: dict[str, list[str]] = {}
        for intent in intents:
            by_branch.setdefault(intent.branch_id, []).append(f"search_query: {intent.query}")
        metrics = dict(state.get("metrics", {}))
        metrics["search_intent_count"] = len(intent_rows)
        metrics["search_intent_model_applied"] = bool(metadata.get("applied"))
        metrics["search_intent_generation_reason"] = metadata.get("reason")
        runtime.artifacts.write_json("search_intents.json", intent_rows)
        runtime.artifacts.write_json("search_intent_generation.json", metadata)
        runtime.artifacts.write_text(
            "reasoning_summary.md",
            render_reasoning_summary(
                query=str(state.get("request", {}).get("question") or ""),
                source_policy=dict(state.get("source_policy", {}) or {}),
                reasoning_state=dict(state.get("reasoning_state", {}) or {}),
                reasoning_decision=dict(state.get("reasoning_decision", {}) or {}),
                contradiction_queries=list(state.get("contradiction_queries", []) or []),
                search_intents=intent_rows,
                search_intent_results=list(state.get("search_intent_results", []) or []),
                plan_revisions=dict(state.get("plan_revisions", {}) or {}),
            ),
        )
        runtime.emit_status(
            "generate_search_intents",
            f"generated {len(intent_rows)} targeted search intent(s)",
            model_applied=metadata.get("applied", False),
            reason=metadata.get("reason"),
            branches=sorted(by_branch),
        )
        return _with_checkpoint(
            runtime,
            "generate_search_intents",
            state,
            {
                "search_intents": intent_rows,
                "reasoning_focus_terms": by_branch,
                "metrics": metrics,
            },
        )

    def prepare_contradiction_search(state: ResearchState) -> ResearchState:
        graph_payload = dict(state.get("evidence_graph", {}) or {})
        queries = generate_contradiction_queries(
            [row for row in graph_payload.get("claims", []) if isinstance(row, dict)],
            question=str(state.get("request", {}).get("question") or ""),
            limit=12,
            per_claim=2,
        )
        query_rows = [query.to_dict() for query in queries]
        by_branch: dict[str, list[str]] = {}
        for query in queries:
            if query.branch_id:
                by_branch.setdefault(query.branch_id, []).append(f"search_query: {query.query}")
        metrics = dict(state.get("metrics", {}))
        metrics["contradiction_search_iterations"] = int(metrics.get("contradiction_search_iterations", 0) or 0) + 1
        metrics["contradiction_query_count"] = len(query_rows)
        decision = dict(state.get("reasoning_decision", {}) or {})
        if by_branch:
            decision["branch_ids"] = sorted(by_branch)
        runtime.artifacts.write_json("contradiction_queries.json", query_rows)
        runtime.artifacts.write_json("contradictions.json", graph_payload.get("contradiction_edges", []))
        runtime.artifacts.write_text(
            "reasoning_summary.md",
            render_reasoning_summary(
                query=str(state.get("request", {}).get("question") or ""),
                source_policy=dict(state.get("source_policy", {}) or {}),
                reasoning_state=dict(state.get("reasoning_state", {}) or {}),
                reasoning_decision=decision,
                contradiction_queries=query_rows,
                search_intents=list(state.get("search_intents", []) or []),
                search_intent_results=list(state.get("search_intent_results", []) or []),
                plan_revisions=dict(state.get("plan_revisions", {}) or {}),
            ),
        )
        runtime.emit_status(
            "prepare_contradiction_search",
            f"prepared {len(query_rows)} contradiction follow-up querie(s)",
            queries=len(query_rows),
            branches=sorted(by_branch),
        )
        return _with_checkpoint(
            runtime,
            "prepare_contradiction_search",
            state,
            {
                "contradiction_queries": query_rows,
                "reasoning_focus_terms": by_branch,
                "reasoning_decision": decision,
                "metrics": metrics,
            },
        )

    def check_coverage(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, sources))
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        coverage = build_coverage_matrix(branches=plan_obj.branches, evidence_cards=cards, sources=sources)
        runtime.artifacts.write_json("coverage.json", coverage.to_dict())
        metrics = dict(state.get("metrics", {}))
        metrics["coverage_score"] = coverage.coverage_score
        metrics["coverage_rounds"] = int(metrics.get("coverage_rounds", 0)) + 1
        no_evidence_failures: list[str] = []
        if not cards and _no_evidence_acquisition_stalled(metrics):
            no_evidence_failures = [
                "No evidence cards were retrieved; synthesis was skipped because source acquisition made no progress.",
            ]
            no_evidence_failures.extend(str(failure) for failure in list(metrics.get("failures", []))[:5])
            runtime.artifacts.write_json(
                "verification.json",
                {
                    "schema_version": 2,
                    "valid": False,
                    "failures": no_evidence_failures,
                    "semantic_verification": {
                        "enabled": runtime.settings.semantic_verification,
                        "overall_score": 0.0,
                        "failures": ["No evidence cards were retrieved to support any claim"],
                    },
                },
            )
        runtime.emit_status(
            "check_coverage",
            f"coverage {coverage.coverage_score:.2f}; missing: {', '.join(coverage.missing_branches) or 'none'}",
            coverage=coverage.coverage_score,
            missing_branches=coverage.missing_branches,
        )
        return _with_checkpoint(
            runtime,
            "check_coverage",
            state,
            {
                "coverage_matrix": coverage.to_dict(),
                "metrics": metrics,
                **(
                    {
                        "verification": {"schema_version": 2, "valid": False, "failures": no_evidence_failures},
                        "failures": no_evidence_failures,
                    }
                    if no_evidence_failures
                    else {}
                ),
            },
        )

    def synthesize(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, sources))
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        coverage = _coverage_from_dict(state.get("coverage_matrix", {}))
        blueprint = build_report_blueprint(plan=plan_obj, evidence_cards=cards, coverage=coverage, sources=sources)
        claim_ledger = build_claim_ledger(plan=plan_obj, evidence_cards=cards, sources=sources)
        sentence_plan = build_sentence_plan(
            plan=plan_obj,
            evidence_cards=cards,
            sources=sources,
            coverage=coverage,
            claim_ledger=claim_ledger,
        )
        section_plan = build_adaptive_section_plan(
            plan=plan_obj,
            evidence_cards=cards,
            coverage=coverage,
            sources=sources,
        )
        knowledge_base = build_knowledge_base(
            plan=plan_obj,
            evidence_cards=cards,
            sources=sources,
            coverage=coverage,
            section_plan=section_plan.to_dict(),
        )
        knowledge_base["reasoning_brief"] = reasoning_brief_for_prompt(
            dict(state.get("reasoning_state", {}) or {}),
            dict(state.get("reasoning_decision", {}) or {}),
        )
        runtime.artifacts.write_json("knowledge_base/base_manifest.json", knowledge_base)
        runtime.emit_status(
            "synthesize",
            "refining knowledge-base context",
            section_packets=len(knowledge_base.get("section_packets", [])),
        )
        knowledge_base = refine_knowledge_base_with_model(
            knowledge_base=knowledge_base,
            plan=plan_obj,
            evidence_cards=cards,
            sources=sources,
            settings=runtime.settings,
            writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
        )
        write_knowledge_base(artifacts=runtime.artifacts, knowledge_base=knowledge_base)
        runtime.artifacts.write_json("knowledge_base/refinement.json", knowledge_base.get("model_refinement", {}))
        runtime.artifacts.write_json("report_blueprint.json", blueprint)
        runtime.artifacts.write_json("claim_ledger.json", claim_ledger)
        runtime.artifacts.write_json("sentence_plan.json", sentence_plan)
        runtime.artifacts.write_json("section_plan.json", section_plan.to_dict())
        context_refinement = dict(knowledge_base.get("model_refinement", {}) or {})
        runtime.emit_status(
            "synthesize",
            "knowledge-base context refinement applied"
            if context_refinement.get("applied")
            else "knowledge-base context refinement skipped",
            reason=context_refinement.get("reason"),
        )
        previous_draft = str(state.get("draft_report") or "")
        used_deterministic_fallback = False
        report: str | None = None
        used_targeted_repair = False
        # Tracks the citation-free draft in sync with `report`, if any -- only
        # ever set when this round's `report` came from a two-pass path (first
        # synthesis or two-pass repair below). Any other path (targeted repair,
        # old single-pass repair, deterministic fallback) changes `report`
        # without updating a matching citation-free version, so it's left as ""
        # to signal "no synced base" -- reusing a stale one would let the writer
        # silently discard whatever that other path just fixed.
        writer_pass_report: str = ""
        if previous_draft.strip() and runtime.settings.llm_synthesis:
            # Repair round: try a targeted paragraph patch before paying for a full
            # re-synthesis. Only attempt it when every failure this round names a
            # specific span of text (deterministic citation/support checks) rather
            # than a whole-report property like coverage or alignment — those still
            # need the full rewrite path below.
            verification_state = dict(state.get("verification", {}) or {})
            # Advisory findings (section-batched judge, RACE self-judge) are tagged
            # "[advisory]" and never gate `valid` -- exclude them here so a round
            # that's otherwise fully local-patchable isn't forced into a full
            # re-synthesis just because an advisory judge left a non-blocking note.
            gating_failures = [f for f in state.get("failures", []) if not str(f).startswith("[advisory]")]
            all_local, _structural = classify_repair_failures(gating_failures)
            weak_claims = list(verification_state.get("weakly_supported_claims", []) or [])
            unsupported = list(verification_state.get("unsupported_claims", []) or [])
            if all_local and (weak_claims or unsupported):
                patched_report, patched_count = apply_targeted_citation_repair(
                    report=previous_draft,
                    weakly_supported_claims=weak_claims,
                    unsupported_claims=unsupported,
                    evidence_cards=cards,
                    sources=sources,
                    plan=plan_obj,
                    settings=runtime.settings,
                )
                if patched_count > 0:
                    runtime.emit_status(
                        "synthesize",
                        f"targeted citation repair patched {patched_count} paragraph(s); skipping full re-synthesis",
                    )
                    report = patched_report
                    used_targeted_repair = True
        writing_guidance_text = str(state.get("request", {}).get("writing_guidance") or "")
        used_two_pass_synthesis = False
        if report is not None:
            pass
        elif (
            runtime.settings.llm_synthesis
            and runtime.settings.two_pass_synthesis_enabled
            and not previous_draft.strip()
        ):
            # First synthesis of the run only (repair rounds keep using the
            # single-pass repair path below -- it already assumes bracket
            # citations are present in previous_draft, which only the two-pass
            # first draft or a prior single-pass draft can guarantee).
            # Same synthesis pipeline as the repair path (same context: blueprint,
            # sentence plan, knowledge base, argumentative outline) but with
            # citations_enabled=False, so its effort goes entirely into
            # structure/tables/diagrams/insight instead of splitting attention
            # with per-sentence citation bookkeeping. citation_agent then inserts
            # [N] markers section-by-section, in parallel, over the finished prose.
            try:
                runtime.emit_status(
                    "synthesize",
                    "starting two-pass synthesis (writer pass, then citation agent)",
                    evidence_cards=len(cards),
                    sources=len(sources),
                )
                written = synthesize_report_with_model(
                    plan=plan_obj,
                    evidence_cards=cards,
                    coverage=coverage,
                    sources=sources,
                    settings=runtime.settings,
                    blueprint=blueprint,
                    sentence_plan=sentence_plan,
                    section_plan=section_plan,
                    knowledge_base=knowledge_base,
                    writing_guidance=writing_guidance_text,
                    citations_enabled=False,
                )
                runtime.artifacts.write_text("writer_pass_draft.md", written)
                report, citation_diagnostics = apply_citations(
                    report_markdown=written,
                    section_plan=section_plan,
                    evidence_cards=cards,
                    sources=sources,
                    plan=plan_obj,
                    settings=runtime.settings,
                    writing_guidance=writing_guidance_text,
                )
                runtime.artifacts.write_json("citation_agent_diagnostics.json", citation_diagnostics)
                used_two_pass_synthesis = True
                writer_pass_report = written
                runtime.emit_status(
                    "synthesize",
                    "two-pass synthesis complete",
                    sections_cited=sum(
                        1 for v in citation_diagnostics.get("sections", {}).values() if v.get("status") == "cited"
                    ),
                    sections_total=len(citation_diagnostics.get("sections", {})),
                )
            except Exception as exc:
                runtime.emit_status(
                    "synthesize",
                    f"two-pass synthesis failed ({type(exc).__name__}: {exc}); falling back to single-pass synthesis",
                )
                report = None
        if report is not None:
            pass
        elif (
            runtime.settings.llm_synthesis
            and runtime.settings.two_pass_synthesis_enabled
            and previous_draft.strip()
            and str(state.get("writer_pass_report") or "").strip()
        ):
            # Repair round, and we have a citation-free base in sync with the
            # draft that just failed verification: give comprehension/coverage
            # failures to the writer (still citations_enabled=False -- same
            # clean split as the first draft, so repair doesn't reintroduce the
            # prose-vs-citation split-attention problem two-pass exists to
            # avoid), then run the citation agent fresh over the result. It
            # gets this round's citation-specific failures as "watch out for
            # this again" guidance -- otherwise it has no memory and could
            # repeat the exact same mistake (e.g. over-applying one source
            # across claims outside its actual topic) on a second pass over
            # text it's already seen.
            try:
                gating_failures = [f for f in state.get("failures", []) if not str(f).startswith("[advisory]")]
                _all_local, structural_failures = classify_repair_failures(gating_failures)
                citation_type_failures = [f for f in gating_failures if f not in structural_failures]
                runtime.emit_status(
                    "synthesize",
                    "starting two-pass repair (writer revise, then citation agent)",
                    structural_failures=len(structural_failures),
                    citation_failures=len(citation_type_failures),
                )
                written = synthesize_report_with_model(
                    plan=plan_obj,
                    evidence_cards=cards,
                    coverage=coverage,
                    sources=sources,
                    settings=runtime.settings,
                    previous_report=str(state.get("writer_pass_report") or ""),
                    verification_failures=structural_failures or gating_failures,
                    blueprint=blueprint,
                    sentence_plan=sentence_plan,
                    section_plan=section_plan,
                    knowledge_base=knowledge_base,
                    writing_guidance=writing_guidance_text,
                    citations_enabled=False,
                )
                runtime.artifacts.write_text("writer_pass_draft.md", written)
                report, citation_diagnostics = apply_citations(
                    report_markdown=written,
                    section_plan=section_plan,
                    evidence_cards=cards,
                    sources=sources,
                    plan=plan_obj,
                    settings=runtime.settings,
                    writing_guidance=writing_guidance_text,
                    citation_failures=citation_type_failures,
                )
                runtime.artifacts.write_json("citation_agent_diagnostics.json", citation_diagnostics)
                used_two_pass_synthesis = True
                writer_pass_report = written
                runtime.emit_status(
                    "synthesize",
                    "two-pass repair complete",
                    sections_cited=sum(
                        1 for v in citation_diagnostics.get("sections", {}).values() if v.get("status") == "cited"
                    ),
                    sections_total=len(citation_diagnostics.get("sections", {})),
                )
            except Exception as exc:
                runtime.emit_status(
                    "synthesize",
                    f"two-pass repair failed ({type(exc).__name__}: {exc}); falling back to single-pass repair synthesis",
                )
                report = None
        if report is not None:
            pass
        elif runtime.settings.llm_synthesis:
            try:
                runtime.emit_status(
                    "synthesize",
                    "starting LLM repair synthesis" if previous_draft.strip() else "starting LLM synthesis",
                    evidence_cards=len(cards),
                    sources=len(sources),
                    previous_draft_chars=len(previous_draft),
                )
                report = synthesize_report_with_model(
                    plan=plan_obj,
                    evidence_cards=cards,
                    coverage=coverage,
                    sources=sources,
                    settings=runtime.settings,
                    previous_report=previous_draft,
                    verification_failures=_synthesis_repair_guidance_from_state(state),
                    blueprint=blueprint,
                    sentence_plan=sentence_plan,
                    section_plan=section_plan,
                    knowledge_base=knowledge_base,
                    writing_guidance=writing_guidance_text,
                )
            except Exception as exc:
                # LLM synthesis failed — typically a network/quota/timeout after all fallbacks
                # were exhausted. Log it and fall back to deterministic synthesis so the
                # 41 sources / 61 evidence cards aren't wasted.
                runtime.emit_status(
                    "synthesize",
                    f"LLM synthesis failed ({type(exc).__name__}: {exc}); falling back to deterministic synthesis",
                )
                runtime.artifacts.write_text(
                    "synthesis_error.txt",
                    f"{type(exc).__name__}: {exc}\n",
                )
                report = synthesize_report(plan=plan_obj, evidence_cards=cards, coverage=coverage, sources=sources)
                used_deterministic_fallback = True
        else:
            report = synthesize_report(plan=plan_obj, evidence_cards=cards, coverage=coverage, sources=sources)
        # If we fell back to deterministic and the previous draft was meaningfully
        # longer (i.e. an LLM had already produced a real report), keep the previous
        # draft instead of regressing to a worse template-based one. This prevents
        # the loop we saw where an OpenRouter 429 silently replaces a good draft.
        if used_deterministic_fallback and previous_draft and len(previous_draft) > len(report) * 1.5:
            runtime.emit_status(
                "synthesize",
                f"deterministic fallback ({len(report)} chars) shorter than previous LLM draft ({len(previous_draft)} chars); keeping previous draft",
            )
            report = previous_draft
        # Keep a numbered history of every draft so we can compare across
        # repair cycles. draft_report.md always points at the latest.
        draft_index = _next_draft_index(runtime.artifacts)
        runtime.artifacts.write_text(f"draft_report_{draft_index}.md", report)
        runtime.artifacts.write_text("draft_report.md", report)
        runtime.artifacts.write_text(
            "report.md",
            "# Research Draft Pending Verification\n\n"
            "This run has produced a draft, but it has not passed verification yet. "
            "Inspect draft_report.md for the current unaccepted draft.\n",
        )
        runtime.emit_status("synthesize", f"wrote draft_report_{draft_index}.md (and draft_report.md)")
        current_draft = {
            "index": draft_index,
            "path": f"draft_report_{draft_index}.md",
            "chars": len(report),
        }
        metrics = dict(state.get("metrics", {}))
        refinement_history = [
            row for row in metrics.get("knowledge_context_refinement_history", []) if isinstance(row, dict)
        ]
        refinement_history.append(
            {
                "applied": bool(context_refinement.get("applied")),
                "reason": context_refinement.get("reason"),
                "json_repair_applied": bool(context_refinement.get("json_repair_applied")),
                "section_packets": len(knowledge_base.get("section_packets", [])),
            }
        )
        metrics["knowledge_context_refinement_history"] = refinement_history
        metrics["knowledge_context_refinement_applied"] = bool(context_refinement.get("applied"))
        metrics["knowledge_context_refinement_reason"] = context_refinement.get("reason")
        metrics["knowledge_context_section_packets"] = len(knowledge_base.get("section_packets", []))
        return _with_checkpoint(
            runtime,
            "synthesize",
            state,
            {
                "draft_report": report,
                "writer_pass_report": writer_pass_report,
                "current_draft": current_draft,
                "metrics": metrics,
                "section_plan": section_plan.to_dict(),
                "knowledge_base": {
                    "schema_version": knowledge_base.get("schema_version"),
                    "branch_count": len(knowledge_base.get("branches", [])),
                    "section_packet_count": len(knowledge_base.get("section_packets", [])),
                    "model_refinement": knowledge_base.get("model_refinement", {}),
                    "index_path": "knowledge_base/index.md",
                },
            },
        )

    def verify(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, sources))
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        coverage = _coverage_from_dict(state.get("coverage_matrix", {}))
        report = str(state.get("draft_report") or runtime.artifacts.read_text("report.md"))
        runtime.emit_status("verify", "running deterministic verification", evidence_cards=len(cards), sources=len(sources))
        result = verify_report_v2(
            report_markdown=report,
            plan=plan_obj,
            sources=sources,
            evidence_cards=cards,
            coverage=coverage,
            source_texts=runtime.source_texts,
        )
        if runtime.settings.semantic_verification:
            runtime.emit_status("verify", "running semantic verification", evidence_cards=len(cards))
            semantic = verify_report_with_semantics(
                report_markdown=report,
                plan=plan_obj,
                evidence_cards=cards,
                settings=runtime.settings,
            )
            runtime.artifacts.write_json(
                "semantic_verification.json",
                {
                    "score": semantic.score,
                    "failures": semantic.failures,
                    "judgment": semantic.judgment,
                },
            )
            result = apply_semantic_report_result(result, semantic)
        verification_payload = result.to_dict()
        section_plan_payload = dict(state.get("section_plan", {}) or {})
        if section_plan_payload:
            runtime.emit_status("verify", "running section-level audit", evidence_cards=len(cards))
            section_audit = audit_report_sections_with_model(
                section_plan=section_plan_payload,
                report_markdown=report,
                plan=plan_obj,
                evidence_cards=cards,
                sources=sources,
                source_texts=runtime.source_texts,
                settings=runtime.settings,
                writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
            )
            runtime.artifacts.write_json("section_audit.json", section_audit)
            current_draft = dict(state.get("current_draft", {}) or {})
            draft_index = int(current_draft.get("index", 0) or 0)
            if draft_index:
                metrics_for_sections = dict(state.get("metrics", {}))
                publish_section_versions(
                    artifacts=runtime.artifacts,
                    metrics=metrics_for_sections,
                    section_audit=section_audit,
                    draft_index=draft_index,
                )
                state["metrics"] = metrics_for_sections
            section_failures = section_audit_failures(section_audit)
            if section_failures:
                verification_payload["valid"] = False
                verification_payload["failures"] = list(verification_payload.get("failures", [])) + section_failures
            verification_payload["section_audit"] = {
                "all_locked": section_audit.get("all_locked"),
                "locked_count": section_audit.get("locked_count"),
                "section_count": section_audit.get("section_count"),
                "llm_review_applied": section_audit.get("llm_review_applied"),
            }
            # Advisory pass: judges each section against its OWN full text and OWN
            # evidence cards in parallel, instead of the whole-report compaction to
            # 6,000 chars above (which silently drops the middle of long reports).
            # Findings are surfaced to the repair loop but do not gate `valid` --
            # this judge is new and unproven at scale, so it earns gating power only
            # after we've watched it run for a while (same treatment source_breadth
            # got before it was trusted).
            if runtime.settings.section_batch_judge_enabled:
                runtime.emit_status("verify", "running section-batched grounding judge (advisory)", evidence_cards=len(cards))
                try:
                    batch_audit = judge_report_sections_batched(
                        report_markdown=report,
                        section_plan=section_plan_payload,
                        evidence_cards=cards,
                        plan=plan_obj,
                        settings=runtime.settings,
                        writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
                    )
                    runtime.artifacts.write_json("section_batch_audit.json", batch_audit)
                    batch_failures = batched_section_failures(batch_audit)
                    if batch_failures:
                        verification_payload["failures"] = list(verification_payload.get("failures", [])) + [
                            f"[advisory] {failure}" for failure in batch_failures
                        ]
                    verification_payload["section_batch_audit"] = {
                        "all_locked": batch_audit.get("all_locked"),
                        "locked_count": batch_audit.get("locked_count"),
                        "section_count": batch_audit.get("section_count"),
                    }
                except Exception as exc:
                    runtime.emit_status("verify", f"section-batched judge failed ({type(exc).__name__}); skipping this round")
        # Advisory pass: scores the full report against the ORIGINAL QUESTION on the
        # axes the external RACE benchmark grades (comprehensiveness, insight,
        # instruction-following, readability) rather than against evidence cards.
        # This is the first internal signal correlated with that external score
        # instead of with citation grounding. Does not gate `valid` for the same
        # reason as the section-batched judge above.
        if runtime.settings.race_self_judge_enabled:
            runtime.emit_status("verify", "running RACE-shaped self-judge (advisory)")
            try:
                race_result = race_self_judge(
                    report_markdown=report,
                    plan=plan_obj,
                    settings=runtime.settings,
                    writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
                )
                runtime.artifacts.write_json("race_judgment.json", race_result.to_dict())
                verification_payload["race_judgment"] = {
                    "comprehensiveness_score": race_result.comprehensiveness_score,
                    "insight_score": race_result.insight_score,
                    "instruction_following_score": race_result.instruction_following_score,
                    "readability_score": race_result.readability_score,
                    "overall_score": race_result.overall_score,
                }
                if race_result.weaknesses:
                    verification_payload["failures"] = list(verification_payload.get("failures", [])) + [
                        f"[advisory] RACE self-judge weakness: {weakness}" for weakness in race_result.weaknesses
                    ]
            except Exception as exc:
                runtime.emit_status("verify", f"RACE self-judge failed ({type(exc).__name__}); skipping this round")
        # Keep a numbered history of every verification cycle alongside the
        # latest snapshot. verification.json always points at the most recent.
        verify_index = _next_verification_index(runtime.artifacts)
        runtime.artifacts.write_json(f"verification_{verify_index}.json", verification_payload)
        runtime.artifacts.write_json("verification.json", verification_payload)
        metrics = dict(state.get("metrics", {}))
        metrics["verification_rounds"] = int(metrics.get("verification_rounds", 0)) + 1
        metrics["verification_valid"] = bool(verification_payload.get("valid"))
        metrics["verification_failures"] = len(verification_payload.get("failures", []))
        section_summary = dict(verification_payload.get("section_audit", {}) or {})
        if section_summary:
            metrics["section_audit_all_locked"] = bool(section_summary.get("all_locked"))
            metrics["section_audit_locked_count"] = section_summary.get("locked_count")
            metrics["section_audit_section_count"] = section_summary.get("section_count")
            metrics["section_audit_llm_review_applied"] = bool(section_summary.get("llm_review_applied"))
        # Track failure count over time so _verification_route can detect when
        # repair cycles are no longer making net progress.
        history = list(metrics.get("verification_failure_history", []))
        history.append(len(verification_payload.get("failures", [])))
        metrics["verification_failure_history"] = history
        current_draft = dict(state.get("current_draft", {}) or {})
        draft_entry = _draft_history_entry(
            current_draft=current_draft,
            draft=report,
            verification_index=verify_index,
            verification=verification_payload,
        )
        if draft_entry is not None:
            draft_history = list(metrics.get("draft_history", []))
            draft_history.append(draft_entry)
            metrics["draft_history"] = draft_history
            _publish_best_draft(runtime.artifacts, metrics)
        runtime.emit_status(
            "verify",
            "verification passed"
            if verification_payload.get("valid")
            else f"verification failed with {len(verification_payload.get('failures', []))} issue(s)",
            valid=bool(verification_payload.get("valid")),
            failures=list(verification_payload.get("failures", []))[:10],
        )
        return _with_checkpoint(
            runtime,
            "verify",
            state,
            {
                "verification": verification_payload,
                "metrics": metrics,
                "failures": list(verification_payload.get("failures", [])),
            },
        )

    def repair_or_finish(state: ResearchState) -> ResearchState:
        metrics = dict(state.get("metrics", {}))
        metrics["finished_at_monotonic"] = time.perf_counter()
        metrics["elapsed_seconds"] = round(metrics["finished_at_monotonic"] - metrics.get("started_at_monotonic", metrics["finished_at_monotonic"]), 3)
        token_usage = get_and_reset_token_usage()
        if token_usage.get("llm_calls", 0) > 0:
            metrics["token_usage"] = token_usage
        verification = dict(state.get("verification", {}))
        if not verification and state.get("coverage_matrix", {}).get("complete") is False:
            coverage = state.get("coverage_matrix", {})
            evidence_count = len(state.get("evidence_cards", []) or [])
            missing = list(coverage.get("missing_branches", []) or [])
            verification = {
                "schema_version": 2,
                "valid": False,
                "failures": [
                    (
                        "Coverage incomplete before synthesis: "
                        f"coverage={float(coverage.get('coverage_score', 0.0) or 0.0):.2f}, "
                        f"evidence_cards={evidence_count}, missing_branches={missing}"
                    )
                ],
            }
            runtime.artifacts.write_json("verification.json", verification)
        draft = str(state.get("draft_report") or "")
        assembled_report = ""
        section_plan_payload = dict(state.get("section_plan", {}) or {})
        if draft and section_plan_payload:
            assembly = assemble_best_section_report(
                artifacts=runtime.artifacts,
                latest_report=draft,
                section_plan=section_plan_payload,
            )
            metrics["assembled_best_report_usable"] = bool(assembly.get("usable_for_final"))
            metrics["assembled_best_report_reason"] = assembly.get("reason")
            metrics["assembled_best_report_section_count"] = assembly.get("assembled_section_count")
            metrics["assembled_best_report_locked_count"] = assembly.get("locked_section_count")
            assembled_report = str(assembly.get("report") or "")
            if assembled_report.strip():
                runtime.artifacts.write_text("assembled_best_report.md", assembled_report)
        failures = [str(f).lower() for f in verification.get("failures", [])]
        judge_unavailable = failures and all(
            "quota_or_rate_limit" in f or "judge unavailable" in f for f in failures
        )
        if verification.get("valid") or runtime.settings.allow_failed_verification or judge_unavailable:
            runtime.artifacts.write_text(
                "report.md",
                assembled_report if metrics.get("assembled_best_report_usable") else draft,
            )
        elif draft:
            selected_draft = assembled_report if metrics.get("assembled_best_report_usable") else _selected_failed_draft(runtime.artifacts, draft)
            runtime.artifacts.write_text("failed_report.md", selected_draft)
            runtime.artifacts.write_text("report.md", _failed_report_notice(verification, metrics))
        else:
            runtime.artifacts.write_text("report.md", _failed_report_notice(verification, metrics))
        _write_run_health(runtime.artifacts, metrics, verification)
        runtime.artifacts.write_json("metrics.json", _public_metrics(metrics))
        runtime.emit_status("finish", "local LangGraph run complete", valid=state.get("verification", {}).get("valid"))
        return _with_checkpoint(runtime, "finish", state, {"metrics": metrics})

    graph.add_node("classify_request", classify_request)
    graph.add_node("plan", plan)
    graph.add_node("check_plan_quality", check_plan_quality)
    graph.add_node("acquire_sources", acquire)
    graph.add_node("read_sources", read_sources)
    graph.add_node("build_evidence", build_evidence)
    graph.add_node("evidence_hygiene", evidence_hygiene)
    graph.add_node("semantic_enrichment", semantic_enrichment)
    graph.add_node("build_evidence_graph", build_evidence_graph)
    graph.add_node("evaluate_search_intents", evaluate_search_intents)
    graph.add_node("update_reasoning_state", update_reasoning_state)
    graph.add_node("replan_from_reasoning", replan_from_reasoning)
    graph.add_node("decide_next_action", decide_reasoning_next_action)
    graph.add_node("generate_search_intents", generate_search_intents)
    graph.add_node("prepare_contradiction_search", prepare_contradiction_search)
    graph.add_node("check_coverage", check_coverage)
    graph.add_node("synthesize", synthesize)
    graph.add_node("verify", verify)
    graph.add_node("repair_or_finish", repair_or_finish)

    graph.set_entry_point(entry_point)
    graph.add_edge("classify_request", "plan")
    graph.add_edge("plan", "check_plan_quality")
    graph.add_edge("check_plan_quality", "acquire_sources")
    graph.add_conditional_edges(
        "acquire_sources",
        _acquire_route,
        {
            "read_sources": "read_sources",
            "reuse_evidence": "check_coverage",
        },
    )
    graph.add_edge("read_sources", "build_evidence")
    graph.add_edge("build_evidence", "evidence_hygiene")
    graph.add_edge("evidence_hygiene", "semantic_enrichment")
    graph.add_edge("semantic_enrichment", "build_evidence_graph")
    graph.add_edge("build_evidence_graph", "evaluate_search_intents")
    graph.add_edge("evaluate_search_intents", "update_reasoning_state")
    graph.add_edge("update_reasoning_state", "replan_from_reasoning")
    graph.add_edge("replan_from_reasoning", "decide_next_action")
    graph.add_conditional_edges(
        "decide_next_action",
        _reasoning_route,
        {
            "search_intents": "generate_search_intents",
            "contradiction_search": "prepare_contradiction_search",
            "continue": "check_coverage",
        },
    )
    graph.add_edge("generate_search_intents", "acquire_sources")
    graph.add_edge("prepare_contradiction_search", "acquire_sources")
    graph.add_conditional_edges(
        "check_coverage",
        _coverage_route,
        {
            # Route through generate_search_intents rather than straight to
            # acquire_sources: acquire_sources only re-searches the intents
            # already sitting in state, and after the first pass those are the
            # exact queries already recorded in runtime.searched_queries, so a
            # direct loop back finds nothing new and bounces straight back to
            # check_coverage forever (see _coverage_route's hard_round_cap
            # comment). generate_search_intents produces fresh, reasoning-
            # informed queries each call, which is what "search_more" actually
            # requires to make progress.
            "more_sources": "generate_search_intents",
            "synthesize": "synthesize",
            "finish": "repair_or_finish",
        },
    )
    graph.add_edge("synthesize", "verify")
    graph.add_conditional_edges(
        "verify",
        _verification_route,
        {
            "more_sources": "acquire_sources",
            "rewrite": "synthesize",
            "finish": "repair_or_finish",
        },
    )
    graph.add_edge("repair_or_finish", END)
    return graph.compile()


def _search_intents_from_state(state: ResearchState) -> list[SearchIntent]:
    intents: list[SearchIntent] = []
    for row in state.get("search_intents", []) or []:
        if not isinstance(row, dict):
            continue
        try:
            intents.append(
                SearchIntent(
                    id=str(row.get("id") or ""),
                    branch_id=str(row.get("branch_id") or ""),
                    gap=str(row.get("gap") or ""),
                    query=str(row.get("query") or ""),
                    expected_evidence=str(row.get("expected_evidence") or ""),
                    success_criteria=str(row.get("success_criteria") or ""),
                    source_preference=str(row.get("source_preference") or ""),
                    priority=str(row.get("priority") or "medium"),
                    origin=str(row.get("origin") or "state"),
                    rationale=str(row.get("rationale") or ""),
                    claim_ids=[str(value) for value in (row.get("claim_ids") or []) if str(value).strip()],
                    source_ids=[int(value) for value in (row.get("source_ids") or []) if str(value).isdigit()],
                )
            )
        except (TypeError, ValueError):
            continue
    return intents


def _focus_terms_by_branch(decision: dict[str, Any], reasoning_state: dict[str, Any] | None = None) -> dict[str, list[str]]:
    reasoning_state = reasoning_state or {}
    branch_ids = [str(branch_id) for branch_id in decision.get("branch_ids", []) if str(branch_id).strip()]
    terms = [str(term) for term in decision.get("focus_terms", []) if str(term).strip()]
    model_action = reasoning_state.get("model_recommended_action", {})
    if isinstance(model_action, dict):
        if not branch_ids:
            branch_ids = [str(branch_id) for branch_id in model_action.get("branch_ids", []) if str(branch_id).strip()]
        if not terms:
            terms = [str(term) for term in model_action.get("focus_terms", []) if str(term).strip()]
    unknowns = [row for row in reasoning_state.get("unknowns", []) if isinstance(row, dict)]
    if not branch_ids:
        branch_ids = [str(row.get("branch_id") or "") for row in unknowns if str(row.get("branch_id") or "").strip()]
    if not branch_ids or not terms:
        return {}
    by_branch = {branch_id: list(terms) for branch_id in branch_ids}
    for row in unknowns:
        branch_id = str(row.get("branch_id") or "").strip()
        if branch_id not in by_branch:
            continue
        by_branch[branch_id].extend(str(term) for term in row.get("focus_terms", []) if str(term).strip())
    return {branch_id: _dedupe_reasoning_focus_terms(values)[:12] for branch_id, values in by_branch.items()}


def _dedupe_reasoning_focus_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        cleaned = re.sub(r"\s+", " ", str(term)).strip(" .;:")
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _assess_plan_quality(plan: ResearchPlan) -> dict[str, Any]:
    """Deterministic plan quality check using TF-IDF branch similarity and term coverage.

    Writes issues to plan_quality.json before source acquisition so problems are
    visible without waiting for a full run to fail.
    """
    from deep_research.evidence.text_terms import ordered_terms

    issues: list[str] = []

    # 1. Branch query diversity: flag pairs with >0.85 cosine similarity
    branch_texts = [
        " ".join([b.title, b.objective] + b.queries + b.required_terms)
        for b in plan.branches
    ]
    branch_sim: dict[str, float] = {}
    if len(branch_texts) >= 2:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(branch_texts)
            sim_matrix = cosine_similarity(vec)
            for i in range(len(plan.branches)):
                for j in range(i + 1, len(plan.branches)):
                    score = float(sim_matrix[i, j])
                    pair_key = f"{plan.branches[i].id}_{plan.branches[j].id}"
                    branch_sim[pair_key] = round(score, 3)
                    if score > 0.85:
                        issues.append(
                            f"branches {plan.branches[i].id} and {plan.branches[j].id} "
                            f"overlap significantly (similarity={score:.2f}); consider merging"
                        )
        except Exception:
            pass

    # 2. Question term coverage: how much of the question is addressed across all branches
    question_terms = set(ordered_terms(plan.question))
    combined_branch_terms = set(ordered_terms(" ".join(branch_texts)))
    q_coverage = (
        len(question_terms & combined_branch_terms) / max(len(question_terms), 1)
        if question_terms else 1.0
    )
    if q_coverage < 0.45:
        issues.append(
            f"plan covers only {q_coverage:.0%} of question terms; "
            "some parts of the request may be missed"
        )

    # 3. Generic acceptance criteria: fewer than 3 unique content terms
    generic_criteria = [
        c for c in plan.acceptance_criteria if len(ordered_terms(c)) < 3
    ]
    if len(generic_criteria) > max(1, len(plan.acceptance_criteria) // 2):
        issues.append(
            f"{len(generic_criteria)}/{len(plan.acceptance_criteria)} acceptance criteria "
            "are too generic to gate evidence quality"
        )

    overall_score = max(0.0, round(1.0 - len(issues) * 0.2, 3))
    return {
        "overall_score": overall_score,
        "status": "ok" if not issues else "warnings",
        "issues": issues,
        "branch_count": len(plan.branches),
        "question_term_coverage": round(q_coverage, 3),
        "branch_similarity": branch_sim,
        "branch_details": [
            {
                "id": b.id,
                "title": b.title,
                "query_count": len(b.queries),
                "min_sources": b.min_sources,
                "required_terms": b.required_terms,
            }
            for b in plan.branches
        ],
    }


def _failed_report_notice(verification: dict[str, Any], metrics: dict[str, Any] | None = None) -> str:
    metrics = metrics or {}
    failures = [str(failure) for failure in verification.get("failures", [])][:12]
    lines = [
        "# Research Run Failed Verification",
        "",
        "This run did not produce an accepted final report. The best rejected draft is stored in `failed_report.md` and `best_draft.md`; the latest draft remains in `draft_report.md`.",
        "",
        "## Run Summary",
        "",
        f"- Verification rounds: {int(metrics.get('verification_rounds', 0) or 0)}",
        f"- Failure history: {list(metrics.get('verification_failure_history', []))}",
        f"- Best draft: {metrics.get('best_draft_path') or 'not available'}",
        f"- Best draft issues: {metrics.get('best_draft_failure_count', 'unknown')}",
        f"- Source count: {metrics.get('source_count', 'unknown')}",
        f"- Evidence cards: {metrics.get('evidence_card_count', 'unknown')}",
        "",
        "## Verification Failures",
        "",
    ]
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- Verification did not mark the draft as valid.")
    return "\n".join(lines).rstrip() + "\n"


def load_latest_checkpoint(run_dir: Path) -> ResearchState:
    path = run_dir / "checkpoints" / "latest.json"
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            state = _load_newest_valid_checkpoint(run_dir)
    else:
        state = _load_newest_valid_checkpoint(run_dir)
    if isinstance(state, dict) and not state.get("checkpoint_phase"):
        state["checkpoint_phase"] = _infer_checkpoint_phase(run_dir)
    return state


def _load_newest_valid_checkpoint(run_dir: Path) -> ResearchState:
    checkpoint_dir = run_dir / "checkpoints"
    candidates = [
        path
        for path in checkpoint_dir.glob("*.json")
        if path.name != "latest.json" and path.is_file()
    ]
    for candidate in sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            state = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(state, dict):
            state.setdefault("checkpoint_phase", candidate.stem)
            return state
    raise FileNotFoundError(f"No readable checkpoint found under {checkpoint_dir}")


def _infer_checkpoint_phase(run_dir: Path) -> str:
    checkpoint_dir = run_dir / "checkpoints"
    candidates = [
        path
        for path in checkpoint_dir.glob("*.json")
        if path.name != "latest.json" and path.is_file()
    ]
    if not candidates:
        return ""
    return max(candidates, key=lambda path: path.stat().st_mtime).stem

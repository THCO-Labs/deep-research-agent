from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph

from deep_research.acquisition import acquire_sources
from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.coverage import build_coverage_matrix
from deep_research.evidence import build_evidence_cards
from deep_research.evidence_hygiene import apply_evidence_hygiene
from deep_research.ingestion import IngestedDocument
from deep_research.progress import ActivityLog
from deep_research.schemas import (
    BranchCoverage,
    CoverageMatrix,
    EvidenceCard,
    ResearchBranch,
    ResearchPlan,
    ResearchState,
    SourceCandidate,
    SourceRecordV2,
)
from deep_research.settings import Settings
from deep_research.semantic import (
    apply_semantic_report_result,
    enrich_evidence_cards_with_semantics,
    verify_report_with_semantics,
)
from deep_research.model_router import get_and_reset_token_usage
from deep_research.semantic_planning import build_or_enrich_research_plan
from deep_research.synthesis import (
    build_claim_ledger,
    build_report_blueprint,
    build_sentence_plan,
    synthesize_report,
    synthesize_report_with_model,
)
from deep_research.verifier_v2 import verify_report_v2
from deep_research.research_graph_helpers import (
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
        runtime.artifacts.write_json("report_blueprint.json", blueprint)
        runtime.artifacts.write_json("claim_ledger.json", claim_ledger)
        runtime.artifacts.write_json("sentence_plan.json", sentence_plan)
        previous_draft = str(state.get("draft_report") or "")
        used_deterministic_fallback = False
        if runtime.settings.llm_synthesis:
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
                    writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
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
        return _with_checkpoint(runtime, "synthesize", state, {"draft_report": report, "current_draft": current_draft})

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
        # Keep a numbered history of every verification cycle alongside the
        # latest snapshot. verification.json always points at the most recent.
        verify_index = _next_verification_index(runtime.artifacts)
        runtime.artifacts.write_json(f"verification_{verify_index}.json", result.to_dict())
        runtime.artifacts.write_json("verification.json", result.to_dict())
        metrics = dict(state.get("metrics", {}))
        metrics["verification_rounds"] = int(metrics.get("verification_rounds", 0)) + 1
        metrics["verification_valid"] = result.valid
        metrics["verification_failures"] = len(result.failures)
        # Track failure count over time so _verification_route can detect when
        # repair cycles are no longer making net progress.
        history = list(metrics.get("verification_failure_history", []))
        history.append(len(result.failures))
        metrics["verification_failure_history"] = history
        current_draft = dict(state.get("current_draft", {}) or {})
        draft_entry = _draft_history_entry(
            current_draft=current_draft,
            draft=report,
            verification_index=verify_index,
            verification=result.to_dict(),
        )
        if draft_entry is not None:
            draft_history = list(metrics.get("draft_history", []))
            draft_history.append(draft_entry)
            metrics["draft_history"] = draft_history
            _publish_best_draft(runtime.artifacts, metrics)
        runtime.emit_status(
            "verify",
            "verification passed" if result.valid else f"verification failed with {len(result.failures)} issue(s)",
            valid=result.valid,
            failures=result.failures[:10],
        )
        return _with_checkpoint(
            runtime,
            "verify",
            state,
            {"verification": result.to_dict(), "metrics": metrics, "failures": result.failures},
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
        failures = [str(f).lower() for f in verification.get("failures", [])]
        judge_unavailable = failures and all(
            "quota_or_rate_limit" in f or "judge unavailable" in f for f in failures
        )
        if verification.get("valid") or runtime.settings.allow_failed_verification or judge_unavailable:
            runtime.artifacts.write_text("report.md", draft)
        elif draft:
            selected_draft = _selected_failed_draft(runtime.artifacts, draft)
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
    graph.add_edge("semantic_enrichment", "check_coverage")
    graph.add_conditional_edges(
        "check_coverage",
        _coverage_route,
        {
            "more_sources": "acquire_sources",
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


def _assess_plan_quality(plan: ResearchPlan) -> dict[str, Any]:
    """Deterministic plan quality check using TF-IDF branch similarity and term coverage.

    Writes issues to plan_quality.json before source acquisition so problems are
    visible without waiting for a full run to fail.
    """
    from deep_research.text_terms import ordered_terms

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

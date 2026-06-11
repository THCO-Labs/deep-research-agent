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
from deep_research.synthesis import build_report_blueprint, synthesize_report, synthesize_report_with_model
from deep_research.verifier_v2 import verify_report_v2


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
        runtime.artifacts.write_jsonl("evidence_cards.jsonl", [card.to_dict() for card in result.cards])
        metrics = _merge_metrics(state.get("metrics", {}), result.metrics)
        metrics["evidence_card_count"] = len(result.cards)
        failures = list(state.get("failures", [])) + result.failures
        runtime.emit_status(
            "semantic_enrichment",
            f"semantic evidence gate kept {len(result.cards)} cards; rejected {len(result.rejected)}",
            enabled=runtime.settings.semantic_verification,
            kept=len(result.cards),
            rejected=len(result.rejected),
            failures=result.failures[:5],
        )
        return _with_checkpoint(
            runtime,
            "semantic_enrichment",
            state,
            {
                "evidence_cards": [card.to_dict() for card in result.cards],
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
        if not cards and _source_acquisition_plateaued(metrics):
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
        runtime.artifacts.write_json("report_blueprint.json", blueprint)
        previous_draft = str(state.get("draft_report") or "")
        used_deterministic_fallback = False
        if runtime.settings.llm_synthesis:
            try:
                report = synthesize_report_with_model(
                    plan=plan_obj,
                    evidence_cards=cards,
                    coverage=coverage,
                    sources=sources,
                    settings=runtime.settings,
                    previous_report=previous_draft,
                    verification_failures=_synthesis_repair_guidance_from_state(state),
                    blueprint=blueprint,
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
        return _with_checkpoint(runtime, "synthesize", state, {"draft_report": report})

    def verify(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, sources))
        cards = [_card_from_dict(row) for row in state.get("evidence_cards", [])]
        coverage = _coverage_from_dict(state.get("coverage_matrix", {}))
        report = str(state.get("draft_report") or runtime.artifacts.read_text("report.md"))
        result = verify_report_v2(
            report_markdown=report,
            plan=plan_obj,
            sources=sources,
            evidence_cards=cards,
            coverage=coverage,
            source_texts=runtime.source_texts,
        )
        if runtime.settings.semantic_verification:
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
        draft = str(state.get("draft_report") or "")
        failures = [str(f).lower() for f in verification.get("failures", [])]
        judge_unavailable = failures and all(
            "quota_or_rate_limit" in f or "judge unavailable" in f for f in failures
        )
        if verification.get("valid") or runtime.settings.allow_failed_verification or judge_unavailable:
            runtime.artifacts.write_text("report.md", draft)
        elif draft:
            runtime.artifacts.write_text("failed_report.md", draft)
            runtime.artifacts.write_text("report.md", _failed_report_notice(verification))
        else:
            runtime.artifacts.write_text("report.md", _failed_report_notice(verification))
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


def _next_draft_index(artifacts: ResearchArtifactsV2) -> int:
    """Return 1, 2, 3... — the next available draft_report_N.md number."""
    return _next_indexed_file(artifacts, "draft_report_*.md")


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
    if coverage.get("complete"):
        return "synthesize"
    if not state.get("evidence_cards") and _source_acquisition_plateaued(metrics):
        return "finish"
    rounds = int(metrics.get("coverage_rounds", 0))
    search_count = int(metrics.get("search_count", 0))
    max_rounds = int(metrics.get("max_rounds", 4) or 4)
    # Pipeline wall-clock budget: if we've already spent more than 25 minutes
    # acquiring/processing, stop looping for more sources and synthesize with what we have.
    started = metrics.get("started_at_monotonic")
    if started is not None and (time.perf_counter() - float(started)) > 1500:
        return "synthesize"
    if _source_acquisition_plateaued(metrics):
        return "synthesize"
    if rounds <= max_rounds and search_count < int(metrics.get("max_search_queries", 10_000) or 10_000):
        return "more_sources"
    if rounds <= max_rounds and _has_unsearched_branch_queries(state):
        return "more_sources"
    return "synthesize"


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
        "semantic_enrichment": "check_coverage",
        "check_coverage": "check_coverage",
        "synthesize": "verify",
        "verify": "verify",
        "repair_or_finish": "repair_or_finish",
        "finish": "repair_or_finish",
    }.get(phase, "classify_request")


def _acquire_route(state: ResearchState) -> str:
    metrics = state.get("metrics", {})
    if _source_acquisition_plateaued(metrics) and state.get("evidence_cards"):
        return "reuse_evidence"
    return "read_sources"


def _source_acquisition_plateaued(metrics: dict[str, Any]) -> bool:
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


def _verification_route(state: ResearchState) -> str:
    verification = state.get("verification", {})
    if verification.get("valid"):
        return "finish"
    metrics = state.get("metrics", {})
    rounds = int(metrics.get("verification_rounds", 0))
    max_rounds = int(metrics.get("max_rounds", 4) or 4)
    if rounds >= max_rounds:
        return "finish"
    # Anti-thrashing: if the last 3 verification cycles produced no net
    # improvement in failure count, stop repairing. We're stuck in a loop
    # where the LLM trades one set of issues for another, like we saw with
    # the Mistral/OpenRouter cycle: 14→27→16→27 issues.
    failure_history = list(metrics.get("verification_failure_history", []))
    failure_history.append(len(verification.get("failures", [])))
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
            focus[branch_id] = terms
    verification = state.get("verification", {})
    failures = " ".join(str(failure).lower() for failure in verification.get("failures", []))
    if "answer coverage" in failures or "source quality" in failures or "weakly supported" in failures:
        plan = state.get("plan", {})
        for branch in plan.get("branches", []):
            branch_id = str(branch.get("id", ""))
            terms = _clean_focus_terms(list(branch.get("required_terms", [])))
            if branch_id and terms:
                focus.setdefault(branch_id, terms)
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
                    focus[branch_id] = terms
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
    requirements = [
        requirement
        for requirement in payload.get("source_requirements", [])
        if isinstance(requirement, dict)
    ]
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


def _failed_report_notice(verification: dict[str, Any]) -> str:
    failures = [str(failure) for failure in verification.get("failures", [])][:12]
    lines = [
        "# Research Run Failed Verification",
        "",
        "This run did not produce an accepted final report. The unaccepted draft is stored in `failed_report.md` and `draft_report.md` for debugging.",
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

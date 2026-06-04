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

    def acquire(state: ResearchState) -> ResearchState:
        plan_obj = _plan_from_state(state)
        existing_candidates = [_candidate_from_dict(row) for row in state.get("source_candidates", [])]
        existing_sources = [_source_from_dict(row) for row in state.get("source_records", [])]
        runtime.source_texts.update(_load_source_texts(runtime.artifacts, existing_sources))
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
            progress_callback=lambda message, data: runtime.emit_status("acquire_sources", message, **data),
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
        cards = build_evidence_cards(
            branches=plan_obj.branches,
            sources=sources,
            source_texts=runtime.source_texts,
            question=plan_obj.question,
            max_cards_per_source=3,
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
        result = enrich_evidence_cards_with_semantics(
            plan=plan_obj,
            evidence_cards=cards,
            settings=runtime.settings,
            prior_judgments=list(dict(state.get("semantic_judgments", {})).get("judgments", [])),
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
        if runtime.settings.llm_synthesis:
            report = synthesize_report_with_model(
                plan=plan_obj,
                evidence_cards=cards,
                coverage=coverage,
                sources=sources,
                settings=runtime.settings,
                previous_report=str(state.get("draft_report") or ""),
                verification_failures=list(state.get("failures", [])),
                blueprint=blueprint,
                writing_guidance=str(state.get("request", {}).get("writing_guidance") or ""),
            )
        else:
            report = synthesize_report(plan=plan_obj, evidence_cards=cards, coverage=coverage, sources=sources)
        runtime.artifacts.write_text("draft_report.md", report)
        runtime.artifacts.write_text(
            "report.md",
            "# Research Draft Pending Verification\n\n"
            "This run has produced a draft, but it has not passed verification yet. "
            "Inspect draft_report.md for the current unaccepted draft.\n",
        )
        runtime.emit_status("synthesize", "wrote draft_report.md")
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
        runtime.artifacts.write_json("verification.json", result.to_dict())
        metrics = dict(state.get("metrics", {}))
        metrics["verification_rounds"] = int(metrics.get("verification_rounds", 0)) + 1
        metrics["verification_valid"] = result.valid
        metrics["verification_failures"] = len(result.failures)
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
        verification = dict(state.get("verification", {}))
        draft = str(state.get("draft_report") or "")
        if verification.get("valid") or runtime.settings.allow_failed_verification:
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
    graph.add_edge("plan", "acquire_sources")
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
        "acquire_sources": "read_sources",
        "read_sources": "build_evidence",
        "build_evidence": "evidence_hygiene",
        "evidence_hygiene": "semantic_enrichment",
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
    failures = [str(failure).lower() for failure in verification.get("failures", [])]
    rewrite_only = (
        any("semantic judge found unsupported claim" in failure for failure in failures)
        or any("some claims are not directly supported" in failure for failure in failures)
        or any("cited evidence-backed source count" in failure for failure in failures)
    )
    if rewrite_only:
        return "rewrite"
    needs_more_sources = (
        any("coverage below threshold" in failure for failure in failures)
        or any("coverage incomplete" in failure for failure in failures)
        or any("source quality" in failure for failure in failures)
        or any("weakly supported" in failure for failure in failures)
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
    return {branch_id: _dedupe_focus_terms(terms) for branch_id, terms in focus.items() if terms}


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
            texts[source.id] = path.read_text(encoding="utf-8")
    return texts


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

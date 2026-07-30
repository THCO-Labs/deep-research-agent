from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import HumanMessage

from deep_research.planning.guidance import criteria_acceptance_lines
from deep_research.models.model_router import model_for_role
from deep_research.planning.planning import build_research_plan
from deep_research.core.schemas import ResearchBranch, ResearchPlan, SourceRequirement
from deep_research.core.settings import Settings
from deep_research.acquisition.source_limits import MINIMUM_SOURCE_TARGET
from deep_research.evidence.text_terms import cjk_content_chunks, contains_cjk, ordered_terms

MAX_LLM_BRANCHES = 14
MAX_QUERIES_PER_BRANCH = 5
MAX_REQUIRED_TERMS = 12
PLACEHOLDER_RE = re.compile(
    r"(?:"
    r"\b(?:company|organization|entity|institution|brand|product|person|author|country|city|region|topic|subject|keyword|item|case)\s+[XYZ]\b"
    r"|\b(?:placeholder|sample\s+query|example\s+query|replace\s+with|tbd|n/a)\b"
    r"|\[[^\]]{1,80}\]"
    r"|<[A-Za-z][^>]{1,80}>"
    r"|\{[^{}]{1,80}\}"
    r")",
    flags=re.I,
)


@dataclass(frozen=True)
class PlanEnrichmentResult:
    plan: ResearchPlan
    accepted: bool
    used_model: bool
    payload: dict[str, Any]
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "accepted": self.accepted,
            "used_model": self.used_model,
            "failures": self.failures,
            "payload": self.payload,
            "plan": self.plan.to_dict(),
        }


def build_or_enrich_research_plan(
    question: str,
    *,
    settings: Settings,
    model: Any | None = None,
    planning_guidance: str = "",
) -> PlanEnrichmentResult:
    base_plan = _plan_with_guidance_criteria(build_research_plan(question), planning_guidance)
    if not settings.llm_planning:
        return PlanEnrichmentResult(
            plan=base_plan,
            accepted=False,
            used_model=False,
            payload={"enabled": False},
            failures=["LLM planning enrichment disabled."],
        )

    planner = model if model is not None else model_for_role(settings, "planner", settings.planner_model)
    if not hasattr(planner, "invoke"):
        return PlanEnrichmentResult(
            plan=base_plan,
            accepted=False,
            used_model=False,
            payload={"model": repr(planner)},
            failures=["Planner role did not resolve to an invokable chat model."],
        )

    try:
        payload = _invoke_json(planner, _planning_prompt(base_plan, planning_guidance=planning_guidance))
        enriched = _plan_from_payload(base_plan, payload, planning_guidance=planning_guidance)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return PlanEnrichmentResult(
            plan=base_plan,
            accepted=False,
            used_model=True,
            payload={"error": str(exc)},
            failures=[f"LLM planning enrichment rejected: {exc}"],
        )

    # Self-reflection: planner reviews its own output for gaps and redundancy.
    # Only runs for non-trivial questions to avoid burning quota on simple lookups.
    if len(enriched.branches) >= 2 and len(ordered_terms(enriched.question)) >= 5:
        enriched = _critique_and_refine_plan(enriched, planner, planning_guidance=planning_guidance)

    # Infer a dynamic writer persona from the question and plan domain.
    enriched = _infer_writer_persona(enriched, planner)

    return PlanEnrichmentResult(
        plan=enriched,
        accepted=True,
        used_model=True,
        payload=payload,
        failures=[],
    )


def _critique_and_refine_plan(
    plan: ResearchPlan,
    planner: Any,
    *,
    planning_guidance: str = "",
) -> ResearchPlan:
    """Second LLM pass: planner reviews its own output for gaps and redundancy."""
    branches_summary = "\n".join(
        f"- {b.id}: {b.title} | queries: {', '.join(b.queries[:2])} | required_terms: {', '.join(b.required_terms[:3])}"
        for b in plan.branches
    )
    prompt = f"""You are a structural QA gatekeeper evaluating a newly generated research plan.

<INPUT_DATA>
User request:
{plan.question}

Plan branches:
{branches_summary}

Additional guidance:
{planning_guidance.strip()[:2000] if planning_guidance.strip() else "None"}
</INPUT_DATA>

Evaluate these three items only:
1. Missing angles: Are core constraints, entities, decision criteria, or evidence requirements from the user request entirely omitted by the branches?
2. Overlap redundancy: Do any two branches share a >80% semantic footprint or use near-identical search strings?
3. Weak queries: Do any search queries contain fewer than 3 descriptive topic words, lack search specificity, or lack concrete terms from the request?

Approval standard:
- Approve if the plan covers the user request well enough for acquisition to start.
- Do not suggest cosmetic rewrites, style changes, or merely nicer wording.
- Only reject for issues that would materially harm source acquisition or final answer coverage.

Return exactly one JSON object. Do not include markdown code blocks or extra text.

If the plan is sound:
{{"approved": true}}

If there are real issues:
{{
  "approved": false,
  "suggested_fixes": [
    {{"type": "add_queries", "branch_id": "branch_N", "queries": ["more specific query"]}},
    {{"type": "remove_branch", "branch_id": "branch_N", "reason": "fully covered by branch_M"}},
    {{"type": "add_branch", "title": "Missing angle title", "objective": "what to research", "queries": ["query 1", "query 2"], "required_terms": ["term1", "term2"]}}
  ]
}}

Be highly conservative. JSON only."""
    try:
        response = planner.invoke([HumanMessage(content=prompt)])
        text = str(getattr(response, "content", response)).strip()
        if not text:
            return plan
        payload = _loads_json_object(_extract_json_object(text))
        if payload.get("approved"):
            return plan
        fixes = payload.get("suggested_fixes")
        if not isinstance(fixes, list) or not fixes:
            return plan
        return _apply_critique_fixes(plan, fixes)
    except Exception:
        return plan


def _infer_writer_persona(plan: ResearchPlan, planner: Any) -> ResearchPlan:
    """Ask the planner to infer a one-sentence writer persona from the question and domain.

    The persona is used at synthesis time to give the writer a domain-appropriate
    voice and expertise framing without any hardcoded domain lists.
    """
    branch_titles = ", ".join(b.title for b in plan.branches[:6])
    prompt = f"""Analyze the following scope to determine the ideal drafting profile.

<SCOPE>
User question:
{plan.question[:600]}

Research branches:
{branch_titles}
</SCOPE>

Task: Write a single sentence briefing that establishes the expert identity for drafting the final report.

The sentence must:
1. Name the domain or field.
2. Define the audience framing.
3. Read like an internal role briefing, not a job posting.

Return exactly one raw sentence. Do not wrap in quotes. Do not add JSON or introductory phrases."""
    try:
        response = planner.invoke([HumanMessage(content=prompt)])
        raw = str(getattr(response, "content", response)).strip()
        persona = raw.strip('"').strip("'").replace("\n", " ").replace("  ", " ").strip()
        if not persona:
            return plan
        if len(persona) < 40:
            persona = persona.strip() + " " + raw.strip('"').strip("'").replace("\n", " ")
        persona = persona[:400].strip().rstrip(".")
        if len(persona) >= 40:
            return replace(plan, writer_persona=persona)
    except Exception:
        pass
    return plan


def _apply_critique_fixes(plan: ResearchPlan, fixes: list[dict[str, Any]]) -> ResearchPlan:
    branches = list(plan.branches)
    branch_index = {b.id: i for i, b in enumerate(branches)}
    for fix in fixes[:6]:
        if not isinstance(fix, dict):
            continue
        fix_type = str(fix.get("type", ""))

        if fix_type == "add_queries":
            bid = str(fix.get("branch_id", ""))
            new_queries = [str(q) for q in (fix.get("queries") or []) if isinstance(q, str) and q.strip()]
            if bid in branch_index and new_queries:
                b = branches[branch_index[bid]]
                merged = _dedupe(list(b.queries) + new_queries)[:MAX_QUERIES_PER_BRANCH]
                branches[branch_index[bid]] = ResearchBranch(
                    id=b.id, title=b.title, objective=b.objective,
                    queries=merged, source_types=b.source_types,
                    min_sources=b.min_sources, required_terms=b.required_terms,
                    completion_criteria=b.completion_criteria,
                )

        elif fix_type == "remove_branch":
            bid = str(fix.get("branch_id", ""))
            if bid in branch_index and len(branches) > 1:
                branches.pop(branch_index[bid])
                branch_index = {b.id: i for i, b in enumerate(branches)}

        elif fix_type == "add_branch" and len(branches) < MAX_LLM_BRANCHES:
            title = str(fix.get("title", "")).strip()
            objective = str(fix.get("objective", "")).strip()
            queries = [str(q) for q in (fix.get("queries") or []) if isinstance(q, str) and q.strip()]
            required_terms = [str(t) for t in (fix.get("required_terms") or []) if isinstance(t, str) and t.strip()]
            if len(ordered_terms(title)) >= 1 and len(ordered_terms(objective)) >= 3 and queries:
                new_id = f"branch_{len(branches) + 1}"
                branches.append(ResearchBranch(
                    id=new_id, title=title, objective=objective,
                    queries=queries[:MAX_QUERIES_PER_BRANCH],
                    source_types=["academic", "official_docs", "general_web"],
                    min_sources=1, required_terms=required_terms[:MAX_REQUIRED_TERMS],
                    completion_criteria=["Evidence addresses the branch objective."],
                ))

    if not branches:
        return plan
    branches = _renumber_branches(branches)
    branches = _raise_branch_source_floor(branches, MINIMUM_SOURCE_TARGET)
    return replace(plan, branches=branches, report_outline=[b.title for b in branches])


def _planning_prompt(base_plan: ResearchPlan, *, planning_guidance: str = "") -> str:
    baseline = {
        "question": base_plan.question,
        "baseline_branch_count": len(base_plan.branches),
        "minimum_total_sources": MINIMUM_SOURCE_TARGET,
        "baseline_terms": ordered_terms(base_plan.question)[:20],
    }
    return f"""You are the planning node in a deep research graph.

Analyze the provided user request and execution context to synthesize a custom, domain-specific research plan. Do not use a fixed template. Do not expose reasoning.

<CONTEXT>
User request:
{base_plan.question}

Additional task-specific guidance, if any:
{planning_guidance.strip()[:8000] if planning_guidance.strip() else "None"}

Deterministic baseline constraints:
{json.dumps(baseline, ensure_ascii=True)}
</CONTEXT>

Return exactly one valid JSON object matching this schema:
{{
  "audience": "specific intended reader",
  "report_outline": ["section heading"],
  "source_requirements": [
    {{"source_type": "academic|official_docs|standards_or_government|government|news|general_web|local_file|mcp", "min_count": 1, "rationale": "why this source type matters"}}
  ],
  "acceptance_criteria": ["specific pass criterion"],
  "branches": [
    {{
      "title": "specific branch title",
      "objective": "what this branch must learn",
      "queries": ["search query tailored to this branch to help the researcher find relevant sources"],
      "source_types": ["academic", "official_docs", "general_web"],
      "min_sources": "scale to this branch's actual depth, not a fixed number",
      "required_terms": ["semantic coverage point"],
      "completion_criteria": ["what evidence must be present"]
    }}
  ]
}}

Critical directions:
- Create only as many branches as the request and task-specific guidance semantically require.
- A narrow single-question prompt may be one branch; multi-part or criteria-rich prompts should become multiple branches.
- Never pad the plan with duplicate, generic, or loosely related branches.
- Do not exceed {MAX_LLM_BRANCHES} branches.
- Branch titles must be specific to the user request, not generic labels like "background" alone.
- The sum of min_sources across branches must be at least {MINIMUM_SOURCE_TARGET}. Reach this by sizing each
  branch's min_sources to the depth it actually needs, not by adding more branches to hit the total -- a narrow
  question with one genuinely broad branch can and should set that branch's min_sources well above the schema
  example (e.g. 20-40 on a single branch), rather than splitting into several thinner ones just to pad the count.
- Each branch needs 2-{MAX_QUERIES_PER_BRANCH} diverse search queries.
- Queries should include concrete topic terms from the user request plus the branch's semantic angle.
- Never use unresolved placeholders such as "Company X", "Topic Y", bracketed variables, sample queries, or TODO text.
- Required terms should describe meaning to cover, not exact words to force.
- Keep the plan proportional to the prompt. Include only branches that are directly necessary to answer the user's wording.
- Do not add medical, legal, financial, or software assumptions unless the request implies them.
- For comparison or ranking requests, include the comparison dimensions and evidence requirements needed to compare options fairly.
- For contradiction-heavy requests, include at least one branch or query angle that can find limitations, exceptions, or opposing evidence.
- JSON only.
"""


def _invoke_json(model: Any, prompt: str) -> dict[str, Any]:
    response = model.invoke([HumanMessage(content=prompt)])
    text = str(getattr(response, "content", response)).strip()
    if not text:
        raise ValueError("empty planner response")
    return _loads_json_object(_extract_json_object(text))


def _loads_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import dirtyjson
        except ImportError:
            raise
        payload = dirtyjson.loads(text)
    plain = _plain_json(payload)
    if not isinstance(plain, dict):
        raise ValueError("planner response JSON root must be an object")
    return plain


def _plain_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_json(item) for item in value]
    return value


def _extract_json_object(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("planner response did not contain a JSON object")
    return cleaned[start : end + 1]


def _plan_from_payload(base_plan: ResearchPlan, payload: dict[str, Any], *, planning_guidance: str = "") -> ResearchPlan:
    rows = payload.get("branches")
    if not isinstance(rows, list):
        raise ValueError("expected branches list")
    if len(rows) < 1:
        raise ValueError("planner returned too few branches")

    branches: list[ResearchBranch] = []
    seen_titles: set[str] = set()
    for row in rows[:MAX_LLM_BRANCHES]:
        if not isinstance(row, dict):
            raise ValueError("branch must be an object")
        branch = _branch_from_payload(base_plan, len(branches) + 1, row)
        title_key = branch.title.lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        branches.append(branch)

    if not branches:
        for fallback in base_plan.branches:
            if fallback.title.lower() in seen_titles:
                continue
            branches.append(
                ResearchBranch(
                    id=f"branch_{len(branches) + 1}",
                    title=fallback.title,
                    objective=fallback.objective,
                    queries=fallback.queries,
                    source_types=fallback.source_types,
                    min_sources=fallback.min_sources,
                    required_terms=fallback.required_terms,
                    completion_criteria=fallback.completion_criteria,
                )
            )
            seen_titles.add(fallback.title.lower())
            break

    failures = _plan_acceptance_failures(branches)
    if failures:
        raise ValueError("; ".join(failures[:5]))

    branches = _raise_branch_source_floor(branches, MINIMUM_SOURCE_TARGET)
    return ResearchPlan(
        question=base_plan.question,
        intent="general",
        audience=_string_field(payload, "audience", default=base_plan.audience) or base_plan.audience,
        report_outline=_string_list_field(payload, "report_outline") or base_plan.report_outline,
        branches=branches,
        source_requirements=_source_requirements(payload) or base_plan.source_requirements,
        acceptance_criteria=_acceptance_criteria(base_plan, payload, planning_guidance),
    )


def _branch_from_payload(base_plan: ResearchPlan, index: int, row: dict[str, Any]) -> ResearchBranch:
    title = _string_field(row, "title", default=f"Research Branch {index}")
    objective = _string_field(row, "objective", default=f"Research branch {index} for {base_plan.question}")
    if len(ordered_terms(title)) < 1:
        raise ValueError(f"branch {index} title lacks semantic terms")
    if len(ordered_terms(objective)) < 3:
        raise ValueError(f"branch {index} objective is too vague")
    queries = _queries(row, base_plan.question, title, objective)
    required_terms = _required_terms(row, title, objective, base_plan.question)
    source_types = _string_list_field(row, "source_types") or ["academic", "official_docs", "government", "general_web"]
    completion = _string_list_field(row, "completion_criteria") or [
        "At least the minimum source count is usable.",
        "Evidence cards answer the branch objective.",
    ]
    return ResearchBranch(
        id=f"branch_{index}",
        title=title,
        objective=objective,
        queries=queries,
        source_types=_dedupe(source_types)[:8],
        min_sources=_min_sources(row),
        required_terms=required_terms,
        completion_criteria=completion[:8],
    )


def _plan_acceptance_failures(branches: list[ResearchBranch]) -> list[str]:
    failures: list[str] = []
    for branch in branches:
        texts = [
            ("title", branch.title),
            ("objective", branch.objective),
            *[(f"query {index}", query) for index, query in enumerate(branch.queries, start=1)],
            *[(f"required term {index}", term) for index, term in enumerate(branch.required_terms, start=1)],
            *[
                (f"completion criterion {index}", criterion)
                for index, criterion in enumerate(branch.completion_criteria, start=1)
            ],
        ]
        for label, text in texts:
            match = PLACEHOLDER_RE.search(text)
            if match:
                failures.append(
                    f"LLM planning enrichment rejected: branch {branch.id} {label} contains unresolved placeholder {match.group(0)!r}"
                )
                break
    return failures


def _criteria_from_guidance(planning_guidance: str) -> list[str]:
    structured = criteria_acceptance_lines(planning_guidance)
    if structured:
        return structured
    criteria: list[str] = []
    for line in planning_guidance.splitlines():
        match = re.match(r"\s*-\s+(.+?)(?:\s+\(weight:\s*[^)]*\))?\s*$", line)
        if not match:
            continue
        criterion = re.sub(r"\s+", " ", match.group(1)).strip(" .:")
        if len(ordered_terms(criterion)) < 3:
            continue
        criteria.append(criterion)
    return _dedupe(criteria)


def _plan_with_guidance_criteria(plan: ResearchPlan, planning_guidance: str) -> ResearchPlan:
    guidance_criteria = _criteria_from_guidance(planning_guidance)
    if not guidance_criteria:
        return plan
    branches = _branches_with_guidance_criteria(plan, list(plan.branches), guidance_criteria)
    merged = _dedupe(
        list(plan.acceptance_criteria)
        + [f"Cover this task-specific criterion in synthesis: {criterion}" for criterion in guidance_criteria]
    )
    return replace(plan, branches=branches, report_outline=[branch.title for branch in branches], acceptance_criteria=merged)


def _branches_with_guidance_criteria(
    plan: ResearchPlan,
    branches: list[ResearchBranch],
    guidance_criteria: list[str],
) -> list[ResearchBranch]:
    if not branches:
        return branches

    source_criteria = [
        (criterion, _criterion_seed_text(criterion))
        for criterion in guidance_criteria
        if _criterion_should_affect_source_plan(criterion)
    ]
    source_criteria = [(criterion, seed) for criterion, seed in source_criteria if seed]
    expanded = _reserve_branch_slots_for_guidance(list(branches), len(source_criteria))
    for criterion, seed in source_criteria:
        if _criterion_is_covered_by_branches(expanded, seed):
            continue
        target_index, target_score = _best_branch_for_criterion(expanded, seed)
        if target_index is not None and (target_score >= 0.2 or len(expanded) >= MAX_LLM_BRANCHES):
            expanded[target_index] = _branch_enriched_by_criterion(expanded[target_index], plan, criterion, seed)
            continue
        if len(expanded) < MAX_LLM_BRANCHES and _criterion_should_drive_research_branch(plan, expanded, seed):
            expanded.append(_branch_for_guidance_criterion(plan, len(expanded) + 1, criterion, seed=seed))
    return _raise_branch_source_floor(_renumber_branches(expanded), MINIMUM_SOURCE_TARGET)


def _reserve_branch_slots_for_guidance(branches: list[ResearchBranch], source_criteria_count: int) -> list[ResearchBranch]:
    if len(branches) < MAX_LLM_BRANCHES or source_criteria_count < 4:
        return branches
    reserve = min(8, max(4, source_criteria_count // 3))
    base_budget = max(4, MAX_LLM_BRANCHES - reserve)
    return branches[:base_budget]


def _criterion_should_drive_research_branch(
    plan: ResearchPlan,
    branches: list[ResearchBranch],
    seed: str,
) -> bool:
    criterion_terms = set(ordered_terms(seed))
    if len(criterion_terms) < 3:
        return False
    branch_text_terms = set(
        ordered_terms(
            " ".join(
                branch.title
                + " "
                + branch.objective
                + " "
                + " ".join(branch.queries)
                + " "
                + " ".join(branch.required_terms)
                for branch in branches
            )
        )
    )
    if _term_coverage(criterion_terms, branch_text_terms) >= 0.65:
        return False
    question_terms = set(ordered_terms(plan.question))
    if question_terms and len(criterion_terms & question_terms) >= min(2, len(question_terms)):
        return True
    new_terms = criterion_terms - branch_text_terms
    return len(new_terms) >= 4 and len(criterion_terms) >= 5


def _criterion_is_covered_by_branches(branches: list[ResearchBranch], seed: str) -> bool:
    terms = set(ordered_terms(seed))
    if not terms:
        return True
    for branch in branches:
        branch_terms = set(
            ordered_terms(
                branch.title
                + " "
                + branch.objective
                + " "
                + " ".join(branch.queries)
                + " "
                + " ".join(branch.required_terms)
                + " "
                + " ".join(branch.completion_criteria)
            )
        )
        if _term_coverage(terms, branch_terms) >= 0.65:
            return True
    return False


def _best_branch_for_criterion(branches: list[ResearchBranch], seed: str) -> tuple[int | None, float]:
    terms = set(ordered_terms(seed))
    if not terms:
        return None, 0.0
    best_index: int | None = None
    best_score = 0.0
    for index, branch in enumerate(branches):
        branch_terms = set(ordered_terms(branch.title + " " + branch.objective + " " + " ".join(branch.required_terms)))
        score = _term_coverage(terms, branch_terms)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index, best_score


def _branch_enriched_by_criterion(
    branch: ResearchBranch,
    plan: ResearchPlan,
    criterion: str,
    seed: str,
) -> ResearchBranch:
    seed_terms = ordered_terms(seed)
    question_terms = ordered_terms(plan.question)
    query = " ".join(_dedupe(question_terms[:8] + seed_terms[:8]))
    queries = _dedupe(list(branch.queries) + [query])[:MAX_QUERIES_PER_BRANCH]
    required_terms = _dedupe(list(branch.required_terms) + seed_terms[:MAX_REQUIRED_TERMS])[:MAX_REQUIRED_TERMS]
    completion = _dedupe(
        list(branch.completion_criteria)
        + [f"Evidence addresses task-specific requirement: {_trim_criterion(criterion)}"]
    )[:10]
    return ResearchBranch(
        id=branch.id,
        title=branch.title,
        objective=branch.objective,
        queries=queries,
        source_types=branch.source_types,
        min_sources=branch.min_sources,
        required_terms=required_terms,
        completion_criteria=completion,
    )


def _branch_for_guidance_criterion(plan: ResearchPlan, index: int, criterion: str, *, seed: str) -> ResearchBranch:
    terms = ordered_terms(seed)
    question_terms = ordered_terms(plan.question)
    title_terms = _dedupe((terms[:8] or question_terms[:8]) + question_terms[:4])
    title = _sentence_case(" ".join(title_terms[:10]) or f"Task-specific coverage {index}")
    criterion_summary = _trim_criterion(criterion, limit=280)
    question_summary = _trim_criterion(plan.question, limit=220)
    query_seed = " ".join(_dedupe(question_terms[:8] + terms[:8]))
    queries = _dedupe(
        [
            query_seed,
            f"{question_summary} {criterion_summary}",
            f"{query_seed} evidence",
        ]
    )
    required_terms = _dedupe(terms[:MAX_REQUIRED_TERMS] + question_terms[:4])
    return ResearchBranch(
        id=f"branch_{index}",
        title=title,
        objective=(
            f"Research evidence needed to satisfy this task-specific requirement: {criterion_summary}. "
            f"Keep the evidence anchored to the user request: {question_summary}"
        ),
        queries=[query for query in queries if len(ordered_terms(query)) >= 2][:MAX_QUERIES_PER_BRANCH],
        source_types=["academic", "official_docs", "standards_or_government", "general_web"],
        min_sources=1,
        required_terms=required_terms[:MAX_REQUIRED_TERMS],
        completion_criteria=[
            f"Evidence addresses task-specific requirement: {criterion_summary}",
            "Evidence remains anchored to the user's original request.",
        ],
    )


def _renumber_branches(branches: list[ResearchBranch]) -> list[ResearchBranch]:
    renumbered: list[ResearchBranch] = []
    for index, branch in enumerate(branches, start=1):
        renumbered.append(
            ResearchBranch(
                id=f"branch_{index}",
                title=branch.title,
                objective=branch.objective,
                queries=branch.queries,
                source_types=branch.source_types,
                min_sources=branch.min_sources,
                required_terms=branch.required_terms,
                completion_criteria=branch.completion_criteria,
            )
        )
    return renumbered


def _term_coverage(required_terms: set[str], text_terms: set[str]) -> float:
    if not required_terms:
        return 1.0
    return len(required_terms & text_terms) / max(len(required_terms), 1)


def _trim_criterion(text: str, *, limit: int = 220) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip()).strip(" .:")
    if len(cleaned) <= limit:
        return cleaned
    boundary = cleaned.rfind(" ", 0, limit)
    return cleaned[: boundary if boundary > 80 else limit].strip()


def _criterion_seed_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip()).strip(" .:")
    cleaned = re.sub(r"^task-specific\s+[^:]{1,80}\s+criterion:\s*", "", cleaned, flags=re.I)
    if " - " in cleaned:
        cleaned = cleaned.split(" - ", 1)[0].strip()
    return cleaned


def _criterion_should_affect_source_plan(text: str) -> bool:
    match = re.match(r"^task-specific\s+([^:]{1,80})\s+criterion:", text.strip(), flags=re.I)
    if not match:
        return True
    dimension = match.group(1).strip().lower()
    return dimension != "readability"


def _sentence_case(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip()).strip(" .:")
    if not cleaned:
        return "Task-specific coverage"
    return cleaned[0].upper() + cleaned[1:]


def _acceptance_criteria(
    base_plan: ResearchPlan,
    payload: dict[str, Any],
    planning_guidance: str,
) -> list[str]:
    criteria = _string_list_field(payload, "acceptance_criteria") or list(base_plan.acceptance_criteria)
    guidance_criteria = _criteria_from_guidance(planning_guidance)
    if guidance_criteria:
        criteria.extend(f"Cover this task-specific criterion in synthesis: {criterion}" for criterion in guidance_criteria)
    return _dedupe(criteria)


def _queries(row: dict[str, Any], question: str, title: str, objective: str) -> list[str]:
    raw = _string_list_field(row, "queries")
    if contains_cjk(f"{question} {title} {objective}"):
        topic_chunks = cjk_content_chunks(question)
        title_chunks = cjk_content_chunks(title)
        objective_chunks = cjk_content_chunks(objective)
        candidates = raw + [
            title,
            " ".join(_dedupe(topic_chunks[:3] + title_chunks[:4])),
            " ".join(_dedupe(topic_chunks[:3] + objective_chunks[:4])),
        ]
        queries = [query for query in _dedupe(candidates) if len(ordered_terms(query)) >= 1]
        if len(queries) < 2:
            queries.append(question)
        return queries[:MAX_QUERIES_PER_BRANCH]

    topic_terms = " ".join(ordered_terms(question)[:10])
    candidates = raw + [
        f"{question} {title}",
        f"{topic_terms} {title}",
        f"{topic_terms} {objective}",
    ]
    queries = [query for query in _dedupe(candidates) if len(ordered_terms(query)) >= 2]
    if len(queries) < 2:
        queries.append(question)
    return queries[:MAX_QUERIES_PER_BRANCH]


def _required_terms(row: dict[str, Any], title: str, objective: str, question: str) -> list[str]:
    raw = _string_list_field(row, "required_terms")
    if raw:
        cleaned = _clean_required_terms(raw)
        if cleaned:
            return cleaned[:MAX_REQUIRED_TERMS]
    fallback = _fallback_required_terms(title=title, objective=objective, question=question)
    return fallback[:MAX_REQUIRED_TERMS]


def _clean_required_terms(raw_terms: list[str]) -> list[str]:
    cleaned: list[str] = []
    for term in raw_terms:
        normalized = re.sub(r"\s+", " ", term).strip(" .:;,-")
        if not normalized:
            continue
        if contains_cjk(normalized):
            cleaned.extend(cjk_content_chunks(normalized) or [normalized])
            continue
        terms = ordered_terms(normalized)
        if len(terms) >= 2 or _is_high_information_single_term(normalized, terms):
            cleaned.append(normalized)
    return _dedupe(cleaned)


def _is_high_information_single_term(raw: str, terms: list[str]) -> bool:
    if len(terms) != 1:
        return False
    token = terms[0]
    if re.search(r"[A-Z]{2,}|[A-Z][a-z]+[A-Z]", raw):
        return True
    return len(token) >= 12


def _fallback_required_terms(*, title: str, objective: str, question: str) -> list[str]:
    if contains_cjk(f"{title} {objective} {question}"):
        chunks = cjk_content_chunks(f"{title} {objective}") or cjk_content_chunks(question)
        return _dedupe(chunks)[:MAX_REQUIRED_TERMS]

    seed_terms = ordered_terms(f"{title} {objective}") or ordered_terms(question)
    phrases: list[str] = []
    for size in (3, 2):
        if len(seed_terms) < size:
            continue
        for index in range(0, len(seed_terms) - size + 1):
            phrases.append(" ".join(seed_terms[index : index + size]))
    if not phrases and seed_terms:
        phrases = [term for term in seed_terms if len(term) >= 10]
    return _dedupe(phrases) or _dedupe(ordered_terms(question)[:MAX_REQUIRED_TERMS])


def _source_requirements(payload: dict[str, Any]) -> list[SourceRequirement]:
    rows = payload.get("source_requirements", [])
    if not isinstance(rows, list):
        return []
    requirements: list[SourceRequirement] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source_type = _string_field(row, "source_type", default="")
        if not source_type:
            continue
        requirements.append(
            SourceRequirement(
                source_type=source_type,
                min_count=max(1, _safe_int(row.get("min_count"), default=1)),
                rationale=_string_field(row, "rationale", default=""),
            )
        )
    return requirements


def _min_sources(row: dict[str, Any]) -> int:
    return max(1, _safe_int(row.get("min_sources"), default=1))


def _raise_branch_source_floor(branches: list[ResearchBranch], target: int) -> list[ResearchBranch]:
    if not branches:
        return branches
    total = sum(branch.min_sources for branch in branches)
    raised = list(branches)
    index = 0
    while total < target:
        branch = raised[index % len(raised)]
        raised[index % len(raised)] = ResearchBranch(
            id=branch.id,
            title=branch.title,
            objective=branch.objective,
            queries=branch.queries,
            source_types=branch.source_types,
            min_sources=branch.min_sources + 1,
            required_terms=branch.required_terms,
            completion_criteria=branch.completion_criteria,
        )
        total += 1
        index += 1
    return raised


def _string_field(payload: dict[str, Any], key: str, *, default: str) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return re.sub(r"\s+", " ", value).strip()


def _string_list_field(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{key} entries must be strings")
        cleaned = re.sub(r"\s+", " ", item).strip()
        if cleaned:
            result.append(cleaned)
    return result


def _safe_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", value).strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result

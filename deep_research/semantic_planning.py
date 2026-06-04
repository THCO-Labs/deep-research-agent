from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import HumanMessage

from deep_research.guidance import criteria_acceptance_lines
from deep_research.model_router import model_for_role
from deep_research.planning import build_research_plan
from deep_research.schemas import ResearchBranch, ResearchPlan, SourceRequirement
from deep_research.settings import Settings
from deep_research.source_limits import MINIMUM_SOURCE_TARGET
from deep_research.text_terms import ordered_terms

MAX_LLM_BRANCHES = 14
MAX_QUERIES_PER_BRANCH = 5
MAX_REQUIRED_TERMS = 12


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

    return PlanEnrichmentResult(
        plan=enriched,
        accepted=True,
        used_model=True,
        payload=payload,
        failures=[],
    )


def _planning_prompt(base_plan: ResearchPlan, *, planning_guidance: str = "") -> str:
    baseline = {
        "question": base_plan.question,
        "baseline_branch_count": len(base_plan.branches),
        "minimum_total_sources": MINIMUM_SOURCE_TARGET,
        "baseline_terms": ordered_terms(base_plan.question)[:20],
    }
    return f"""You are the planning node in a deep research graph.

Infer a domain-specific research plan from the user's exact request. Do not use a fixed template. Do not expose reasoning.

User request:
{base_plan.question}

Additional task-specific guidance, if any:
{planning_guidance.strip()[:8000] if planning_guidance.strip() else "None"}

Baseline constraints:
{json.dumps(baseline, ensure_ascii=True)}

Return exactly one JSON object:
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
      "queries": ["search query tailored to this branch"],
      "source_types": ["academic", "official_docs", "general_web"],
      "min_sources": 3,
      "required_terms": ["semantic coverage point"],
      "completion_criteria": ["what evidence must be present"]
    }}
  ]
}}

Hard requirements:
- Create only as many branches as the request and task-specific guidance semantically require.
- A narrow single-question prompt may be one branch; multi-part or criteria-rich prompts should become multiple branches.
- Never pad the plan with duplicate, generic, or loosely related branches.
- Do not exceed {MAX_LLM_BRANCHES} branches.
- Branch titles must be specific to the user request, not generic labels like "background" alone.
- The sum of min_sources across branches must be at least {MINIMUM_SOURCE_TARGET}.
- Each branch needs 2-{MAX_QUERIES_PER_BRANCH} diverse search queries.
- Queries should include concrete topic terms from the user request plus the branch's semantic angle.
- Required terms should describe meaning to cover, not exact words to force.
- Keep the plan proportional to the prompt. Include only branches that are directly necessary to answer the user's wording.
- Do not add medical, legal, financial, or software assumptions unless the request implies them.
- JSON only.
"""


def _invoke_json(model: Any, prompt: str) -> dict[str, Any]:
    response = model.invoke([HumanMessage(content=prompt)])
    text = str(getattr(response, "content", response)).strip()
    if not text:
        raise ValueError("empty planner response")
    return json.loads(_extract_json_object(text))


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
    terms = raw + ordered_terms(title + " " + objective + " " + question)[:MAX_REQUIRED_TERMS]
    cleaned = [term for term in _dedupe(terms) if len(term.strip()) >= 3]
    return cleaned[:MAX_REQUIRED_TERMS] or ordered_terms(question)[:MAX_REQUIRED_TERMS]


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

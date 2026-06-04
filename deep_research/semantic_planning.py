from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

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
    base_plan = build_research_plan(question)
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

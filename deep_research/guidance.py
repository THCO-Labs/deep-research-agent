from __future__ import annotations

import json
import re
from typing import Any

from deep_research.text_terms import ordered_terms

CRITERIA_BLOCK_START = "<deep_research_task_criteria_json>"
CRITERIA_BLOCK_END = "</deep_research_task_criteria_json>"


def format_criteria_guidance_block(criteria: dict[str, Any]) -> str:
    payload = structured_criteria_payload(criteria)
    if not payload["criteria"]:
        return ""
    return "\n".join(
        [
            CRITERIA_BLOCK_START,
            json.dumps(payload, ensure_ascii=True, sort_keys=True),
            CRITERIA_BLOCK_END,
        ]
    )


def structured_criteria_payload(criteria: dict[str, Any]) -> dict[str, Any]:
    weights = criteria.get("dimension_weight", {})
    criterions = criteria.get("criterions", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(criterions, dict):
        return {"schema_version": 1, "criteria": rows}
    for dimension, dimension_rows in criterions.items():
        if not isinstance(dimension_rows, list):
            continue
        dimension_name = str(dimension).strip()
        for row in dimension_rows:
            if not isinstance(row, dict):
                continue
            criterion = str(row.get("criterion") or "").strip()
            if not criterion:
                continue
            rows.append(
                {
                    "dimension": dimension_name,
                    "dimension_weight": weights.get(dimension_name) if isinstance(weights, dict) else None,
                    "criterion": criterion,
                    "explanation": str(row.get("explanation") or "").strip(),
                    "weight": row.get("weight"),
                }
            )
    return {"schema_version": 1, "criteria": rows}


def extract_structured_criteria(guidance: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in _criteria_blocks(guidance):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        criteria = payload.get("criteria") if isinstance(payload, dict) else None
        if not isinstance(criteria, list):
            continue
        for row in criteria:
            if not isinstance(row, dict):
                continue
            criterion = str(row.get("criterion") or "").strip()
            if len(ordered_terms(criterion)) < 2:
                continue
            rows.append(
                {
                    "dimension": str(row.get("dimension") or "").strip(),
                    "dimension_weight": row.get("dimension_weight"),
                    "criterion": criterion,
                    "explanation": str(row.get("explanation") or "").strip(),
                    "weight": row.get("weight"),
                }
            )
    return _dedupe_criteria(rows)


def criteria_acceptance_lines(guidance: str) -> list[str]:
    rows = extract_structured_criteria(guidance)
    if not rows:
        return []
    lines: list[str] = []
    for row in rows:
        dimension = str(row.get("dimension") or "task").strip()
        criterion = str(row.get("criterion") or "").strip()
        explanation = str(row.get("explanation") or "").strip()
        if not criterion:
            continue
        line = f"Task-specific {dimension} criterion: {criterion}"
        if explanation:
            line += f" - {explanation}"
        lines.append(line)
    return _dedupe_text(lines)


def _criteria_blocks(guidance: str) -> list[str]:
    pattern = re.compile(
        re.escape(CRITERIA_BLOCK_START) + r"\s*(.*?)\s*" + re.escape(CRITERIA_BLOCK_END),
        flags=re.S,
    )
    return [match.group(1).strip() for match in pattern.finditer(guidance)]


def _dedupe_criteria(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("dimension") or "").strip().lower(),
            re.sub(r"\s+", " ", str(row.get("criterion") or "").strip().lower()),
        )
        if not key[1] or key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = re.sub(r"\s+", " ", value.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result

from __future__ import annotations

import re
from typing import Any

from deep_research.artifacts_v2 import ResearchArtifactsV2


def publish_section_versions(
    *,
    artifacts: ResearchArtifactsV2,
    metrics: dict[str, Any],
    section_audit: dict[str, Any],
    draft_index: int,
) -> None:
    history = [row for row in metrics.get("section_history", []) if isinstance(row, dict)]
    for audit in section_audit.get("audits", []):
        if not isinstance(audit, dict):
            continue
        section_id = _safe_section_id(str(audit.get("section_id") or "section"))
        markdown = str(audit.get("matched_report_section_markdown") or "").strip()
        if not markdown:
            continue
        section_path = f"section_drafts/draft_{draft_index}_{section_id}.md"
        artifacts.write_text(section_path, markdown.rstrip() + "\n")
        history.append(
            {
                "section_id": section_id,
                "draft_index": draft_index,
                "section_path": section_path,
                "heading": str(audit.get("heading") or ""),
                "locked": bool(audit.get("locked")),
                "failure_count": len(list(audit.get("failures", []))),
                "citation_support_score": _float_score(audit.get("citation_support_score")),
                "evidence_linkage_score": _float_score(audit.get("evidence_linkage_score")),
                "cited_source_ids": list(audit.get("cited_source_ids", [])),
            }
        )
    metrics["section_history"] = history
    best = select_best_section_versions(history)
    metrics["best_section_count"] = len(best)
    metrics["locked_best_section_count"] = sum(1 for row in best.values() if row.get("locked"))
    artifacts.write_json("best_sections.json", {"schema_version": 1, "sections": best})
    for section_id, row in best.items():
        path = artifacts.resolve_path(str(row.get("section_path") or ""))
        if not path.exists():
            continue
        artifacts.write_text(
            f"best_sections/{section_id}.md",
            path.read_text(encoding="utf-8", errors="replace").rstrip() + "\n",
        )


def select_best_section_versions(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in history:
        section_id = str(row.get("section_id") or "")
        if not section_id:
            continue
        current = best.get(section_id)
        if current is None or _section_rank(row) > _section_rank(current):
            best[section_id] = dict(row)
    return best


def assemble_best_section_report(
    *,
    artifacts: ResearchArtifactsV2,
    latest_report: str,
    section_plan: dict[str, Any],
) -> dict[str, Any]:
    best_payload = artifacts.read_json("best_sections.json")
    sections = best_payload.get("sections", {}) if isinstance(best_payload, dict) else {}
    if not isinstance(sections, dict) or not sections:
        return {
            "schema_version": 1,
            "usable_for_final": False,
            "reason": "no best section versions available",
            "report": "",
        }
    ordered_ids = _section_order(section_plan, sections)
    parts: list[str] = []
    locked_count = 0
    used_section_ids: list[str] = []
    for section_id in ordered_ids:
        row = sections.get(section_id)
        if not isinstance(row, dict):
            continue
        path = artifacts.resolve_path(str(row.get("section_path") or ""))
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        parts.append(text)
        used_section_ids.append(section_id)
        if row.get("locked"):
            locked_count += 1
    if not parts:
        return {
            "schema_version": 1,
            "usable_for_final": False,
            "reason": "best section files were empty or missing",
            "report": "",
        }

    planned_count = len(_planned_section_ids(section_plan))
    section_count = len(used_section_ids)
    all_planned_present = planned_count > 0 and section_count >= planned_count
    all_used_locked = locked_count == section_count
    report = _assemble_markdown(latest_report=latest_report, section_parts=parts)
    usable = bool(report.strip()) and all_planned_present and all_used_locked
    return {
        "schema_version": 1,
        "usable_for_final": usable,
        "reason": "complete locked section assembly" if usable else "assembly is partial or contains unlocked sections",
        "planned_section_count": planned_count,
        "assembled_section_count": section_count,
        "locked_section_count": locked_count,
        "used_section_ids": used_section_ids,
        "report": report,
    }


def _section_rank(row: dict[str, Any]) -> tuple[int, float, float, int, int]:
    return (
        1 if row.get("locked") else 0,
        _float_score(row.get("citation_support_score")),
        _float_score(row.get("evidence_linkage_score")),
        -int(row.get("failure_count", 10_000) or 0),
        int(row.get("draft_index", 0) or 0),
    )


def _float_score(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_section_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip().lower()).strip("_")
    return slug or "section"


def _section_order(section_plan: dict[str, Any], sections: dict[str, Any]) -> list[str]:
    planned = _planned_section_ids(section_plan)
    remaining = [section_id for section_id in sections if section_id not in planned]
    return planned + sorted(remaining)


def _planned_section_ids(section_plan: dict[str, Any]) -> list[str]:
    if not isinstance(section_plan, dict):
        return []
    result: list[str] = []
    for row in section_plan.get("sections", []):
        if not isinstance(row, dict):
            continue
        section_id = _safe_section_id(str(row.get("id") or ""))
        if section_id and section_id not in result:
            result.append(section_id)
    return result


def _assemble_markdown(*, latest_report: str, section_parts: list[str]) -> str:
    title = _report_title(latest_report)
    body = "\n\n".join(_strip_report_title(part) for part in section_parts if part.strip()).strip()
    source_entries = _source_entries_by_id(latest_report)
    cited_ids = _inline_citation_ids(body)
    source_lines = [
        source_entries[source_id]
        for source_id in sorted(cited_ids)
        if source_id in source_entries
    ]
    lines = [title, "", body]
    if source_lines:
        lines.extend(["", "## Sources", "", *source_lines])
    return "\n".join(lines).rstrip() + "\n"


def _report_title(markdown: str) -> str:
    match = re.search(r"(?m)^#\s+(.+?)\s*$", markdown)
    if not match:
        return "# Research Report"
    return f"# {match.group(1).strip()}"


def _strip_report_title(markdown: str) -> str:
    return re.sub(r"(?m)^#\s+.+?\s*\n+", "", markdown, count=1).strip()


def _source_entries_by_id(markdown: str) -> dict[int, str]:
    entries: dict[int, str] = {}
    for match in re.finditer(r"(?m)^\[(\d+)]\s+.+?:\s+https?://\S+\s*$", markdown):
        entries[int(match.group(1))] = match.group(0).strip()
    return entries


def _inline_citation_ids(markdown: str) -> set[int]:
    return {int(value) for value in re.findall(r"\[(\d+)]", markdown)}

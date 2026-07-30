from __future__ import annotations

from deep_research.core.models import SourceRecord, VerificationResult


def render_verification_repair_markdown(
    result: VerificationResult,
    source_records: list[SourceRecord],
    *,
    report_exists: bool,
) -> str:
    """Render deterministic next steps for a failed report verification."""
    lines = [
        "# Verification Repair Checklist",
        "",
        "This file is generated deterministically from `verification.json`.",
        "Fix these items, then rerun `verify_report_file(\"report.md\")`.",
        "",
        "## Status",
        "",
        f"- Valid: `{str(result.valid).lower()}`",
        f"- Citation validity score: `{result.citation_validity_score}`",
        f"- Source support score: `{result.source_support_score}`",
        f"- Verification round: `{result.verification_rounds}`",
        "",
    ]

    if result.valid:
        lines.extend(["## Required Repairs", "", "- None. Verification passed.", ""])
        return "\n".join(lines)

    lines.extend(["## Required Repairs", ""])
    repairs = _required_repairs(result, report_exists=report_exists)
    lines.extend(f"- {repair}" for repair in repairs)
    lines.append("")

    if result.weakly_supported_claims:
        lines.extend(["## Weakly Supported Claims", ""])
        for index, claim in enumerate(result.weakly_supported_claims, start=1):
            source_ids = ", ".join(f"[{source_id}]" for source_id in claim.get("cited_source_ids", []))
            missing_terms = ", ".join(str(term) for term in claim.get("missing_terms", [])[:12])
            lines.append(f"{index}. Cited sources: {source_ids or 'none'}")
            lines.append(f"   Claim: {claim.get('paragraph', '')}")
            lines.append(f"   Missing terms: {missing_terms or 'none recorded'}")
        lines.append("")

    if result.unsupported_claims:
        lines.extend(["## Uncited Paragraphs", ""])
        for index, paragraph in enumerate(result.unsupported_claims, start=1):
            lines.append(f"{index}. {paragraph}")
        lines.append("")

    if result.source_list_errors:
        lines.extend(["## Source List Errors", ""])
        lines.extend(f"- {error}" for error in result.source_list_errors)
        lines.append("")

    if result.missing_sources:
        lines.extend(["## Missing Sources", ""])
        lines.extend(f"- {missing}" for missing in result.missing_sources)
        lines.append("")

    if result.unscraped_sources:
        lines.extend(["## Cited But Unscraped", ""])
        source_lookup = {record.id: record for record in source_records}
        for source_id in result.unscraped_sources:
            record = source_lookup.get(source_id)
            url = record.url if record else "unknown URL"
            lines.append(f"- [{source_id}] must be scraped or replaced: {url}")
        lines.append("")

    if result.unused_sources:
        lines.extend(["## Optional Cleanup", ""])
        lines.append(
            "- Unused source IDs are available but not cited: "
            + ", ".join(f"[{source_id}]" for source_id in result.unused_sources)
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _required_repairs(result: VerificationResult, *, report_exists: bool) -> list[str]:
    repairs: list[str] = []
    if not report_exists:
        repairs.append("Create `report.md` with a source-backed answer and a parseable `## Sources` section.")
    if result.source_list_errors:
        repairs.append("Fix the `## Sources` section so every entry matches `[N] Title: https://url`.")
    if result.missing_sources:
        repairs.append("Remove invented citations or add the cited scraped sources to the registry.")
    if result.unscraped_sources:
        repairs.append("Scrape every cited search candidate or replace it with a scraped usable source.")
    if result.unsupported_claims:
        repairs.append("Add inline `[N]` citations to every factual paragraph, or remove unsupported prose.")
    if result.weakly_supported_claims:
        repairs.append("Rewrite weakly supported claims to match source text or gather stronger sources.")
    if not result.cited_source_ids:
        repairs.append("Cite at least one scraped source in the body of `report.md`.")
    return repairs or ["Inspect `verification.json` for the failing invariant and rerun verification."]

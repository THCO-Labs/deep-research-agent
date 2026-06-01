---
name: comprehensive-report
description: Orchestrates a benchmark-grade public-web research lifecycle with source-backed reports and verification. Use when the user asks for deep research, a comprehensive report, a comparison, or benchmark-quality research.
---

# Comprehensive Deep Research Skill

## Workflow

1. Plan the question and save `research_plan.md`.
2. Dispatch 1-3 `researcher` tasks based on the plan; use more only for clearly independent branches.
3. Require researchers to prefer `collect_sources` for normal research branches so blocked, bot-protected, or low-content pages are skipped before citation.
4. Use `web_search` and `deep_scrape` manually only for targeted follow-up on a specific URL, and never rely on snippets alone.
5. Dispatch `analyst` only for numeric, tabular, or code-based analysis.
6. Synthesize all files under `findings/` into `report.md`.
7. Cite every factual paragraph with `[source_id]` numbers returned by usable scraped sources.
8. End with `## Sources` using `[source_id] Title: URL` lines.
9. Dispatch `verifier` and run `verify_report_file("report.md")`.
10. If verification fails, repair the report and rerun verification. In balanced mode, perform at most 2 repair rounds.
11. If residual issues remain after the repair budget, include a brief `## Verification Notes` section and leave details in `verification.json` plus `findings/verification_repair.md`.

## Report Structure

Use the structure that best fits the question:

- Comparisons: executive summary, entity sections, comparison table, implications, conclusion.
- Market or technical deep dives: executive summary, background, current state, evidence, risks/limits, outlook.
- Rankings/lists: ranked entries with criteria, evidence, caveats, and sources.

## Quality Gates

- `request.md`, `research_plan.md`, `sources.jsonl`, `report.md`, `verification.json`, and `metrics.json` must exist.
- Every relied-on source must be scraped, not just discovered.
- Prefer sources with `source_quality_label` of `excellent` or `strong`; weak sources need a clear reason.
- `collect_sources` entries under `unusable_sources` or `deep_scrape` results with `source_usable: false` must not be cited.
- Every factual paragraph must have at least one inline citation.
- Every cited paragraph must be supported by the scraped source text strongly enough to avoid `weakly_supported_claims`.
- Source IDs in citations must match `sources.jsonl`.
- Failed verification checks must be visible in `verification.json` and summarized in `findings/verification_repair.md`.

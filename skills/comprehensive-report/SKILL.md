---
name: comprehensive-report
description: Orchestrates a benchmark-grade public-web research lifecycle with source-backed reports and verification. Use when the user asks for deep research, a comprehensive report, a comparison, or benchmark-quality research.
---

# Comprehensive Deep Research Skill

## Workflow

1. Plan the question and save `research_plan.md`.
2. Dispatch 1-3 `researcher` tasks based on the plan; use more only for clearly independent branches.
3. Require researchers to call `web_search` and then `deep_scrape` before relying on a source.
4. Dispatch `analyst` only for numeric, tabular, or code-based analysis.
5. Synthesize all files under `findings/` into `report.md`.
6. Cite every factual paragraph with `[source_id]` numbers returned by the tools.
7. End with `## Sources` using `[source_id] Title: URL` lines.
8. Dispatch `verifier` and run `verify_report_file("report.md")`.
9. If verification fails, repair the report and rerun verification. In balanced mode, perform at most 2 repair rounds.
10. If residual issues remain after the repair budget, include a brief `## Verification Notes` section and leave details in `verification.json`.

## Report Structure

Use the structure that best fits the question:

- Comparisons: executive summary, entity sections, comparison table, implications, conclusion.
- Market or technical deep dives: executive summary, background, current state, evidence, risks/limits, outlook.
- Rankings/lists: ranked entries with criteria, evidence, caveats, and sources.

## Quality Gates

- `request.md`, `research_plan.md`, `sources.jsonl`, `report.md`, `verification.json`, and `metrics.json` must exist.
- Every relied-on source must be scraped, not just discovered.
- Every factual paragraph must have at least one inline citation.
- Source IDs in citations must match `sources.jsonl`.
- Failed verification checks must be visible in `verification.json`.

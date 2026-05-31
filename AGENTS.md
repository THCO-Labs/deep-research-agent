# Deep Research Agent

## Architecture

The agent is a multi-tier orchestrated research pipeline. A single `run_research()` call
produces a sandboxed run directory under `runs/` containing the report, sources, verification
result, and metrics.

```
CLI (python -m deep_research "question")
  └─ run_research() [deep_research/agent.py]
       ├─ RunArtifacts       – sandboxed per-run filesystem (path-safe writes/reads)
       ├─ SourceRegistry     – URL deduplication and source provenance tracking
       ├─ ToolContext        – shared mutable state (metrics, scraper, search client)
       ├─ build_tools()      – 6 LangChain tools exposed to the LLM
       └─ create_deep_agent()
            ├─ Orchestrator (main model, e.g. gemini-2.5-flash)
            └─ Subagents (defined in subagents.yaml)
                 ├─ planner   – decomposes question, writes research_plan.md
                 ├─ researcher – web_search + deep_scrape + write findings
                 ├─ analyst   – python_repl for quantitative analysis
                 └─ verifier  – runs verify_report_file, writes repair checklist
```

## Subagent Tools

| Tool                | Available to            | Purpose                                  |
|---------------------|-------------------------|------------------------------------------|
| `web_search`        | orchestrator, researcher | Tavily search; registers source candidates |
| `deep_scrape`       | orchestrator, researcher | Playwright fetch → markdown; records scrape |
| `write_file`        | all subagents           | Write UTF-8 file inside run directory    |
| `read_file`         | researcher, analyst, verifier | Read file inside run directory    |
| `verify_report_file`| orchestrator, verifier  | Deterministic citation + source checks   |
| `python_repl`       | analyst                 | Execute Python for numeric analysis      |

## Research Workflow

1. Orchestrator writes `research_plan.md` via the `planner` subagent.
2. `researcher` subagent(s) call `web_search` then `deep_scrape` on every source they cite.
3. Findings are saved under `findings/`.
4. Orchestrator writes final `report.md` with inline citations `[source_id]` and a `## Sources` section.
5. `verifier` subagent calls `verify_report_file("report.md")`.
6. If verification fails, the orchestrator repairs unsupported/uncited content and reruns verification.
7. After `max_rounds` repair attempts, residual issues are noted in a "Verification Notes" section.

## Citation Rules

- Every factual paragraph must have at least one inline citation `[N]`.
- Citation numbers must match the `source_id` values returned by `web_search` or `deep_scrape`.
- The `## Sources` section must list entries as: `[N] Title: https://url`.
- Entries must be sequential without gaps.
- Never cite a source that has not been scraped with `deep_scrape`.

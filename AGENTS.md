# Deep Research Agent

## Architecture

The agent is a multi-tier orchestrated research pipeline. A single `run_research()` call
produces a sandboxed run directory under `runs/` containing the report, sources, verification
result, activity log, model route manifest, failure classification, and metrics.
Each run also writes `run_manifest.json`, a redacted reproducibility manifest
with settings, runtime metadata, package versions, and model routes.

```text
CLI (python -m deep_research "question")
  -> run_research() [deep_research/agent.py]
       -> RunArtifacts       - sandboxed per-run filesystem (path-safe writes/reads)
       -> SourceRegistry     - URL deduplication and source provenance tracking
       -> ToolContext        - shared mutable state (metrics, scraper, search client, activity log)
       -> build_tools()      - 7 LangChain tools exposed to the LLM
       -> create_deep_agent()
            -> Orchestrator (main model)
            -> Subagents (defined in subagents.yaml)
                 -> planner    - decomposes question, writes research_plan.md
                 -> researcher - collect_sources + targeted search/scrape + write findings
                 -> analyst    - python_repl for quantitative analysis
                 -> verifier   - runs verify_report_file, writes repair checklist
```

## Subagent Tools

| Tool | Available to | Purpose |
| --- | --- | --- |
| `collect_sources` | orchestrator, researcher | Tavily search plus deterministic scrape/recovery loop; returns usable scraped sources |
| `web_search` | orchestrator, researcher | Tavily search; registers source candidates |
| `deep_scrape` | orchestrator, researcher | Playwright/HTTP fetch to markdown; records usable scrapes or returns unusable-source payloads |
| `write_file` | all subagents | Write UTF-8 file inside run directory |
| `read_file` | researcher, analyst, verifier | Read bounded file preview inside run directory |
| `verify_report_file` | orchestrator, verifier | Deterministic citation and source checks |
| `python_repl` | analyst | Execute Python for numeric analysis |

## Research Workflow

1. The runner writes a deterministic baseline `research_plan.md`; the orchestrator or planner may refine it.
2. `researcher` subagent(s) prefer `collect_sources` for normal branches so blocked, bot-protected, and low-content pages are skipped.
3. Researchers use manual `web_search` plus `deep_scrape` only for targeted follow-up on specific URLs.
4. Findings are saved under `findings/`.
5. Orchestrator writes final `report.md` with inline citations `[source_id]` and a `## Sources` section.
6. `verifier` subagent calls `verify_report_file("report.md")`.
7. If verification fails, the orchestrator repairs unsupported/uncited content and reruns verification.
8. After `max_rounds` repair attempts, residual issues are noted in a "Verification Notes" section.

## Citation Rules

- Every factual paragraph must have at least one inline citation `[N]`.
- Citation numbers must match `source_id` values from usable scraped sources.
- The `## Sources` section must list entries as: `[N] Title: https://url`.
- Source IDs may be sparse because search-only candidates also receive IDs; cite only usable scraped source IDs and list those exact IDs in `## Sources`.
- Never cite search-only candidates, `collect_sources.unusable_sources`, or `deep_scrape` results with `source_usable: false`.
- Use `model_routes.json` to verify role/model/key-slot and fallback routing without exposing API key values.
- Use `run_manifest.json` to compare redacted settings, runtime metadata, and package versions across runs.
- Use `activity.jsonl` / `activity.md` `model_fallback` events to see when a role switched to an alternate key or provider.
- Use `model_retry` events to see bounded provider retry-window waits.
- Open `activity.html` or run `python -m deep_research.activity --follow` to watch the latest run's visible progress.
- Use `verification.json` to inspect `weakly_supported_claims` when cited paragraphs do not match scraped source text.
- Use `findings/verification_repair.md` for the deterministic human repair checklist when final verification fails.
- Use `failure.json` to inspect quota, token-budget, tool-call, and permission failures after a failed run.
- The deployment branch is 'current-setup'. All commits to this branch trigger a CI/CD pipeline to Azure Container Apps.
- The persistent run directory on the container app is `/mnt/runs`, mounted from Azure File Share `deepresearch-runs`.
- The GitHub Actions deploy workflow (`.github/workflows/deploy.yml`) must include a post-deploy step to ensure the Azure File volume mount is present.
- All API keys (e.g., TAVILY_API_KEY, BRAVE_SEARCH_API_KEY, EXA_API_KEY) must be configured as GitHub Actions secrets; the deploy pipeline wires them into the container environment.
- On AzureFile mounts, Path.replace may fail with OSError; implement a direct write fallback (e.g., catch OSError and write directly).
- When constructing Settings, ensure the project_root is included in the kwargs to avoid relative path breakage.

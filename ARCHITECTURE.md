# Deep Research Agent Architecture

## 1. Purpose

Deep Research Agent is a public-web research system designed to produce source-backed research reports with reproducible artifacts, visible progress, deterministic citation verification, and benchmark evaluation support.

The system is built around four principles:

- **Evidence first**: search results are only candidates; relied-on sources must be scraped before they can pass verification.
- **Recoverable acquisition**: normal research uses `collect_sources` to over-fetch candidates, scrape them, skip blocked or low-quality pages, and return only usable sources for citation.
- **Source quality ranking**: source candidates receive deterministic quality metadata so scrape budget is spent on stronger public-web evidence first.
- **Reproducibility**: every run gets its own artifact directory with the request, sources, report, transcript, metrics, and verification output.
- **Provider flexibility**: model execution can use Google Gemini or Groq through LangChain model strings.
- **Measurability**: reports are checked by deterministic citation validation and can be scored through a benchmark harness.

The current scope is public-web research. It does not yet support private browsing sessions, authenticated sources, local PDFs, or uploaded corpora.

## 2. High-Level System

```mermaid
flowchart TD
    User["CLI User"] --> CLI["deep_research.cli"]
    CLI --> Settings["Settings / .env"]
    CLI --> Runner["run_research"]
    Runner --> Artifacts["RunArtifacts"]
    Runner --> Registry["SourceRegistry"]
    Runner --> Tools["Research Tools"]
    Runner --> Agent["DeepAgents Graph"]
    Agent --> RootTools["Root Tools"]
    Agent --> Subagents["Planner / Researcher / Analyst / Verifier"]
    RootTools --> Collector["collect_sources Recovery Loop"]
    Collector --> Tavily["Tavily Search"]
    Collector --> Scraper["Playwright + HTTP Scraper"]
    RootTools --> Tavily["Tavily Search"]
    RootTools --> Scraper["Playwright + HTTP Scraper"]
    RootTools --> Verifier["Citation Verifier"]
    Subagents --> Tools
    Tavily --> Registry
    Scraper --> Registry
    Registry --> Artifacts
    Verifier --> Artifacts
    Agent --> Report["report.md"]
    Runner --> Activity["activity.md / activity.jsonl"]
    Runner --> Manifest["run_manifest.json"]
    Runner --> Routes["model_routes.json"]
    Runner --> Metrics["metrics.json"]
```

The user starts the system with `python -m deep_research "question"`. The CLI loads configuration, creates a run directory, builds tools and subagents, streams DeepAgents updates, finalizes artifacts, and prints the generated paths.

## 3. Runtime Entry Points

### Primary CLI

`deep_research/cli.py` is the main user entry point.

Primary command:

```powershell
python -m deep_research "research question"
```

Important options:

- `--provider auto|google|groq|hybrid`: selects model provider. `auto` uses `hybrid` when both Groq and Google keys exist, otherwise it uses the available provider.
- `--model`: overrides the main model. Short names are prefixed by the selected provider.
- `--fast-model`: overrides the subagent and judge model.
- `--planner-model`, `--researcher-model`, `--analyst-model`, `--verifier-model`, `--judge-model`: override individual role models.
- `--mode fast|balanced|max_quality`: controls default source and repair budgets.
- `--max-sources`: caps results per search call.
- `--max-rounds`: caps verifier repair rounds.
- `--scrape-char-limit`: controls how much scraped text is saved per source.
- `--progress live|raw|quiet`: controls console progress display.

### Compatibility Wrappers

`agent.py` and `deep_research_agent.py` are thin compatibility wrappers around the canonical CLI. They exist so older commands still route into the package entry point.

### Evaluation Commands

`deep_research/eval.py` runs benchmark datasets:

```powershell
python -m deep_research.eval --dataset benchmarks/seed.jsonl --out eval_runs --limit 1
```

`deep_research/eval_report.py` summarizes a benchmark result file:

```powershell
python -m deep_research.eval_report eval_runs/<run>/results.jsonl
```

## 4. Configuration

Configuration lives in `deep_research/settings.py`.

`Settings.from_env()` loads `.env` from the project root and validates required keys.

Required keys:

- `TAVILY_API_KEY`: required for public-web search.
- `GROQ_API_KEY`: required when provider resolves to `groq` or a role uses a `groq:...` model.
- `GOOGLE_API_KEY`: required when provider resolves to `google` or a role uses a `google_genai:...` model.

Optional keys:

- `DEEP_RESEARCH_PROVIDER`
- `DEEP_RESEARCH_MODEL`
- `DEEP_RESEARCH_FAST_MODEL`
- `DEEP_RESEARCH_PLANNER_MODEL`
- `DEEP_RESEARCH_RESEARCHER_MODEL`
- `DEEP_RESEARCH_ANALYST_MODEL`
- `DEEP_RESEARCH_VERIFIER_MODEL`
- `DEEP_RESEARCH_JUDGE_MODEL`
- `DEEP_RESEARCH_SCRAPE_CHAR_LIMIT`
- `DEEP_RESEARCH_TOOL_EXCERPT_CHAR_LIMIT`
- `DEEP_RESEARCH_MODEL_FALLBACKS`
- `DEEP_RESEARCH_PROVIDER_RETRY_ATTEMPTS`
- `DEEP_RESEARCH_PROVIDER_RETRY_MAX_WAIT_SECONDS`

API key pools:

- `GROQ_API_KEY`, `GROQ_API_KEY1`, `GROQ_API_KEY2`, ...
- `GOOGLE_API_KEY`, `GOOGLE_API_KEY1`, `GOOGLE_API_KEY2`, ...

The unnumbered key and numbered keys are deduplicated and sorted with the
unnumbered key first. `deep_research/model_router.py` assigns role models across
the provider's available key pool. In hybrid mode with two Groq keys and two
Google keys, the active default graph uses all four credentials:

| Role | Default provider | Key slot |
| --- | --- | --- |
| Orchestrator | Groq | `GROQ_API_KEY` |
| Researcher | Groq | `GROQ_API_KEY1` |
| Planner | Google | `GOOGLE_API_KEY` |
| Verifier | Google | `GOOGLE_API_KEY1` |
| Analyst | Groq | wraps across the Groq key pool |
| Eval judge | Google | wraps across the Google key pool |

Key values are never written to progress output or metrics.

Each run writes `model_routes.json` before the graph starts. The manifest
contains role, provider, model, key pool size, key slot, safe key label, and
fallback route list for each role. It is the authoritative artifact for
confirming that a hybrid run is actually distributing work across
`GROQ_API_KEY`, `GROQ_API_KEY1`, `GOOGLE_API_KEY`, and `GOOGLE_API_KEY1` without
leaking secret values.

Each run also writes `run_manifest.json`. This higher-level reproducibility
manifest includes the redacted `Settings` payload, progress mode, runtime
metadata, selected package versions, and the full `model_routes.json` payload.
It records key-pool counts and whether a Tavily key was present, but never
stores API key values.

Model fallback is enabled by default through `DEEP_RESEARCH_MODEL_FALLBACKS`.
When a model call fails with a classified quota/rate-limit, token-budget, or
tool-call parse failure, the wrapper tries same-provider alternate keys first.
In hybrid mode it then tries a cross-provider fallback route, such as Groq to
Google or Google to Groq. Fallback attempts are emitted to `activity.md` and
`activity.jsonl` under `model_fallback`.

If all available fallback candidates return retryable provider windows, the
wrapper waits only when the shortest retry delay is within
`DEEP_RESEARCH_PROVIDER_RETRY_MAX_WAIT_SECONDS`, then retries the candidate
chain up to `DEEP_RESEARCH_PROVIDER_RETRY_ATTEMPTS`. Waits are visible as
`model_retry` events.

Provider defaults:

| Provider | Main model | Fast model |
| --- | --- | --- |
| `groq` | `groq:openai/gpt-oss-20b` | `groq:openai/gpt-oss-20b` |
| `google` | `google_genai:gemini-2.5-flash` | `google_genai:gemini-2.5-flash` |
| `hybrid` | `groq:openai/gpt-oss-20b` | `groq:openai/gpt-oss-20b` |

Hybrid is preferred automatically when both Groq and Google keys are present because it spreads active subagent work across both providers. If only Groq keys are present, Groq is preferred to avoid Gemini free-tier quota exhaustion. The default Groq model is intentionally the 20B tool-call-capable model because larger Groq models can exceed on-demand token-per-minute limits in multi-step agent flows.

Mode defaults:

| Mode | Default max sources | Default repair rounds |
| --- | ---: | ---: |
| `fast` on Google | 6 | 1 |
| `balanced` on Google | 12 | 2 |
| `max_quality` on Google | 24 | 3 |
| `fast` on Groq | 2 | 1 |
| `balanced` on Groq | 3 | 1 |
| `max_quality` on Groq | 5 | 2 |
| `fast` on Hybrid | 2 | 1 |
| `balanced` on Hybrid | 3 | 1 |
| `max_quality` on Hybrid | 5 | 2 |

Provider-specific scraping defaults:

| Provider | Saved scrape chars | Tool-return excerpt chars |
| --- | ---: | ---: |
| `groq` | 6,000 | 900 |
| `hybrid` | 6,000 | 900 |
| `google` | 15,000 | 2,500 |

The scraper and file reader save more text to disk than they return to the model. This keeps the run reproducible without overloading low-TPM providers with huge tool responses. Before truncation, scraper output is parsed as HTML, obvious site chrome is removed, and the best article/main-content node is converted to markdown. Bot checks, Cloudflare challenges, JavaScript-only pages, very low-content extracts, and fetch failures such as 403 responses are surfaced as unusable source results instead of being saved as citable source documents or aborting the run.

Role-specific model routing:

| Role | Settings field | Env var |
| --- | --- | --- |
| Root orchestrator | `model` | `DEEP_RESEARCH_MODEL` |
| Default subagent fallback | `fast_model` | `DEEP_RESEARCH_FAST_MODEL` |
| Planner | `planner_model` | `DEEP_RESEARCH_PLANNER_MODEL` |
| Researcher | `researcher_model` | `DEEP_RESEARCH_RESEARCHER_MODEL` |
| Analyst | `analyst_model` | `DEEP_RESEARCH_ANALYST_MODEL` |
| Verifier | `verifier_model` | `DEEP_RESEARCH_VERIFIER_MODEL` |
| Eval judge | `judge_model` | `DEEP_RESEARCH_JUDGE_MODEL` |

Role model values can use any LangChain-supported model string, such as `groq:...`, `google_genai:...`, or another provider prefix when the required package and API key are installed. Tool-using roles require models that support chat tool calls. This is why Groq's tool-capable `openai/gpt-oss-20b` remains the default for Groq runs.

## 5. Package Layout

```text
deep_research/
  __main__.py             Module entry point for `python -m deep_research`
  cli.py                  CLI parsing, settings construction, final status output
  settings.py             Env loading, provider/model resolution, budget defaults
  manifest.py             Redacted run manifest and runtime metadata
  model_router.py         Provider model construction and API key-pool routing
  errors.py               Provider/tool failure classification
  agent.py                DeepAgents graph creation and run lifecycle
  deepagents_profiles.py  Provider-specific DeepAgents runtime profile patches
  prompts.py              Root orchestrator prompt
  subagents.py            YAML subagent loading and tool binding
  tools.py                Search, scrape, file, verifier, and analyst tools
  scraper.py              Playwright scraper with HTTP fallback
  source_registry.py      Source dedupe, source IDs, source persistence
  artifacts.py            Run directory creation and path safety
  verifier.py             Deterministic citation/source verifier
  models.py               SourceRecord, Metrics, VerificationResult dataclasses
  urls.py                 URL canonicalization
  progress.py             Concise live progress rendering
  activity.py             Terminal/HTML activity viewer for run progress
  repair.py               Deterministic verification repair checklist renderer
  eval.py                 Benchmark execution harness
  eval_report.py          Benchmark result summarizer
```

Repository-level support files:

- `subagents.yaml`: planner, researcher, analyst, and verifier subagent definitions.
- `skills/comprehensive-report/SKILL.md`: workflow instructions for comprehensive reports.
- `benchmarks/seed.jsonl`: seed benchmark dataset.
- `tests/`: unit and mocked integration tests.
- `.env.example`: expected environment variables.
- `pyproject.toml`: package metadata and dependencies.

## 6. Core Run Lifecycle

The main lifecycle is implemented in `run_research()` in `deep_research/agent.py`.

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLI
    participant S as Settings
    participant R as Runner
    participant A as Artifacts
    participant G as DeepAgents Graph
    participant T as Tools
    participant V as Verifier

    U->>CLI: python -m deep_research "question"
    CLI->>S: Settings.from_env()
    CLI->>R: run_research(question, settings)
    R->>A: create run directory
    R->>A: write request.md and initial research_plan.md
    R->>G: create_deep_agent(...)
    G->>T: search, scrape, read/write, verify
    T->>A: persist sources and files
    G->>A: write or return report content
    R->>A: write transcript.log
    R->>V: verify report.md
    V->>A: write verification.json
    R->>A: write metrics.json
    CLI->>U: print artifact paths
```

Detailed steps:

1. The CLI configures UTF-8 console output to avoid Windows encoding crashes.
2. `Settings.from_env()` loads `.env`, resolves provider/model defaults, and validates keys.
3. `RunArtifacts.create()` creates `runs/<timestamp-slug>/`.
4. `request.md`, `activity.md`, `activity.jsonl`, `activity.html`, `run_manifest.json`, `model_routes.json`, an initial `research_plan.md`, empty `sources.jsonl`, `findings/`, and `source_docs/` are created.
5. A `SourceRegistry` is attached to the run.
6. A `ToolContext` is created with settings, artifacts, registry, Tavily client, scraper, Python REPL, metrics, and progress callback.
7. `create_deep_agent()` builds the root DeepAgents graph with root tools and subagents.
8. The graph streams updates. In `live` progress mode these are summarized into concise progress lines. All observable progress events are also persisted to `activity.md` and `activity.jsonl`; these logs expose actions, sources, and status, not hidden chain-of-thought.
9. Tool calls mutate only the run artifact directory and the source registry.
10. If the graph fails, `failure.json`, `error.txt`, `verification.json`, and `metrics.json` are still written.
11. If the graph returns final report text but does not write `report.md`, the runner reconstructs `report.md` and appends a real `## Sources` section from the registry.
12. Deterministic verification runs against the final report and source registry.
13. If verification fails, `findings/verification_repair.md` is generated from the deterministic verifier output.
14. Metrics are written and artifact paths are printed.

## 7. Agents and Subagents

The root agent is built in `deep_research/agent.py` with:

- `settings.model` as the primary model.
- Root tools: `web_search`, `deep_scrape`, `collect_sources`, `write_file`, `read_file`, `verify_report_file`.
- System prompt from `deep_research/prompts.py`.
- Subagents loaded from `subagents.yaml`.

Subagents are loaded by `deep_research/subagents.py`. Their configured model is overridden at runtime with role-specific settings. If a role-specific model is not provided, it falls back to `settings.fast_model`.

Current subagents:

| Subagent | Purpose | Tools |
| --- | --- | --- |
| `planner` | Decompose the request and save `research_plan.md`. | `write_file` |
| `researcher` | Collect usable sources and save source-backed findings. | `collect_sources`, `web_search`, `deep_scrape`, `write_file`, `read_file` |
| `analyst` | Run Python for numeric/data analysis. | `python_repl`, `write_file`, `read_file` |
| `verifier` | Run deterministic report verification and save repair notes. | `read_file`, `verify_report_file`, `write_file` |

The root graph also has direct access to search and scrape as a recovery path. This prevents failures when a model tries to research directly instead of delegating.

For Groq-backed runs, `deep_research/deepagents_profiles.py` registers a
DeepAgents harness profile for `groq` and any explicit `groq:...` model strings.
That profile excludes the built-in `write_todos` tool while preserving subagent
dispatch, file middleware, citation verification, and the app's own live
progress feed. This avoids provider-side `tool_use_failed` errors when a Groq
model emits malformed JSON for DeepAgents' internal todo tool.

## 8. Tooling Architecture

Tools are defined in `deep_research/tools.py` by `build_tools(context)`.

### ToolContext

`ToolContext` carries all mutable run state:

- `settings`
- `artifacts`
- `registry`
- `search_client`
- `scraper`
- `activity`
- `on_progress`
- `repl`
- `metrics`

Each tool updates metrics and can emit progress events.

### collect_sources

`collect_sources(query, target_count, max_results)` is the preferred source acquisition tool for normal research branches.

Important behavior:

- Searches Tavily for more candidates than the requested usable-source target, capped at a small recovery budget.
- Registers every candidate in `SourceRegistry` so IDs remain deterministic.
- Ranks candidates by `source_quality_score` before scraping, preserving stable source IDs.
- Scrapes ranked candidates until the target number of usable sources is collected.
- Skips blocked pages, bot checks, low-content pages, and fetch failures by returning them under `unusable_sources`.
- Returns citable entries only under `usable_sources`; each usable entry has `source_usable: true` and a saved `content_path`.
- Sets `needs_more_sources: true` when the query did not produce enough usable sources, telling the researcher to run a better follow-up query.

This tool prevents one bad page, such as a 403 or a Cloudflare challenge, from ending the full research run.

### web_search

`web_search(query, max_results)` uses Tavily to find candidate URLs.

Important behavior:

- Empty queries raise `ResearchToolError`.
- Result count is bounded by `settings.max_sources`.
- Each result is registered in `SourceRegistry`.
- Returned results include `needs_scrape: true`.
- Snippets are not returned to the model as evidence.

Search results are candidates only. Verification will not pass a report that cites a source that was only searched and never scraped. Use this manually for targeted follow-up; use `collect_sources` for ordinary source gathering.

### deep_scrape

`deep_scrape(url)` fetches and registers full source content.

Important behavior:

- Accepts direct URLs, source IDs like `1` or `[1]`, and some malformed URLs.
- Resolves bad scrape targets back to registered search candidates when possible.
- Uses `PlaywrightScraper`.
- Saves source markdown to `source_docs/source_<id>.md`.
- Returns only a compact excerpt to the model to avoid provider token limits.
- Returns `source_usable: false` instead of raising when the page is blocked, low quality, or cannot be fetched. Those records must not be cited.

### write_file and read_file

File tools operate inside the current run directory.

Path behavior:

- Relative paths are resolved under the run directory.
- Leading slash paths like `/research_plan.md` are treated as virtual run-root paths.
- Windows drive paths, UNC paths, and `..` escapes are rejected.
- Missing reads return an explicit `ERROR: file not found` string instead of crashing the graph.

### verify_report_file

`verify_report_file(file_path="report.md")` runs deterministic verification and writes `verification.json`.

If `report.md` does not exist, it returns a failed verification payload instead of raising.

### python_repl

`python_repl(code)` runs Python through `langchain_experimental.utilities.PythonREPL`.

It is exposed to the analyst subagent only. The current dependency emits a deprecation warning but still works.

## 9. Scraping Architecture

Scraping is implemented in `deep_research/scraper.py`.

`PlaywrightScraper.fetch(url)` first tries browser rendering:

1. Launch Chromium headless.
2. Navigate with `wait_until="domcontentloaded"`.
3. Try to wait briefly for `networkidle`.
4. Extract title and page HTML.
5. Convert HTML to markdown.

If Playwright fails, the scraper falls back to HTTP extraction:

1. Fetch with `httpx` and a browser-like user agent.
2. Raise for non-2xx responses.
3. Extract `<title>`.
4. Convert HTML to markdown.
5. Record `extraction_method="httpx"`.

This fallback was added because some pages, such as IBM pages in local tests, can time out under Playwright while still being fetchable over HTTP.

## 10. Source Registry

`deep_research/source_registry.py` owns source identity and persistence.

Each source is represented by `SourceRecord`:

```text
id
url
canonical_url
title
fetched_at
extraction_method
content_hash
content_path
query
snippet
search_score
source_quality_score
source_quality_label
source_quality_type
source_quality_reasons
```

Registry behavior:

- Search candidates are assigned stable source IDs.
- URLs are canonicalized for deduplication.
- Scraped content receives a SHA-256 content hash.
- Scraped markdown is written to `source_docs/source_<id>.md`.
- Registry state is persisted to `sources.jsonl` after updates.
- Duplicate canonical URLs reuse the same source ID.
- Duplicate content hashes reuse the existing scraped source.
- Source quality is scored at search time and refreshed after scrape using URL, title, snippet, search score, extracted text length, and domain/source-type signals.

The registry is the source of truth for citation numbers. The final report must cite source IDs from this registry.

## 11. URL Canonicalization

`deep_research/urls.py` canonicalizes URLs before deduplication and verification.

Canonicalization behavior:

- Lowercases scheme and host.
- Removes default ports.
- Normalizes trailing slash behavior.
- Sorts query parameters.
- Removes fragments.
- Removes common tracking parameters such as `utm_*`, `fbclid`, `gclid`, and similar fields.

This prevents the same source from being counted multiple times due to tracking URLs.

## 12. Artifacts and Run Directory

Run artifact management is implemented in `deep_research/artifacts.py`.

Each run is stored under:

```text
runs/<YYYYMMDDTHHMMSSZ>-<question-slug>/
```

Expected artifacts:

| File or directory | Purpose |
| --- | --- |
| `request.md` | Original research request. |
| `activity.md` / `activity.jsonl` | Visible progress feed and machine-readable progress events. |
| `activity.html` | Auto-refreshing local dashboard for visible progress events. |
| `run_manifest.json` | Redacted settings, runtime metadata, package versions, progress mode, and model routes. |
| `model_routes.json` | Per-role provider/model/key-slot manifest without API key values. |
| `research_plan.md` | Plan written by planner/root agent. |
| `sources.jsonl` | Machine-readable source registry. |
| `source_docs/` | Scraped source markdown. |
| `findings/` | Intermediate researcher/analyst/verifier notes. |
| `findings/verification_repair.md` | Deterministic repair checklist written when final verification fails. |
| `report.md` | Final report. |
| `verification.json` | Deterministic citation verification output. |
| `metrics.json` | Runtime, tool counts, source count, error state. |
| `failure.json` | Present only when the graph raises an error; classifies provider/tool failures. |
| `transcript.log` | Raw streamed graph content captured during the run. |
| `error.txt` | Present only when the graph raises an error. |

`runs/` is gitignored because it contains generated output.

## 13. Verification Architecture

Verification is implemented in `deep_research/verifier.py`.

It checks `report.md` against the in-memory `SourceRegistry`.

Verification inputs:

- Report markdown.
- List of `SourceRecord` objects.
- Verification round count.

Verification checks:

- Inline citations use `[number]` format.
- The `## Sources` or `### Sources` section is parseable.
- Every cited ID exists in `sources.jsonl`.
- Every cited ID appears in the Sources section.
- Every Sources section URL matches the registry canonical URL.
- Sources are sequential without gaps.
- Factual paragraphs have at least one inline citation.
- Cited sources were actually scraped, not merely returned by search.
- Each cited paragraph is compared against the text of its cited source files with a conservative lexical support check.

Verification output is `VerificationResult`:

```text
valid
citation_validity_score
source_support_score
missing_sources
unused_sources
unscraped_sources
unsupported_claims
weakly_supported_claims
support_checks
source_list_errors
cited_source_ids
total_citations
verification_rounds
```

`unsupported_claims` means factual paragraphs with no citation. `weakly_supported_claims` means cited paragraphs whose important terms are not sufficiently present in the cited scraped source text. This is not full semantic proof, but it catches citation laundering and obvious hallucinated claims while keeping the check deterministic and benchmarkable.

A report only passes when it has citations, parseable sources, no uncited factual paragraphs, no weakly supported cited claims, and no cited-but-unscraped sources.

When final verification fails, `deep_research/repair.py` renders
`findings/verification_repair.md` from the same `VerificationResult`. The file
lists required repairs, weakly supported claims, uncited paragraphs, source-list
errors, missing sources, and cited-but-unscraped sources. This artifact is
deterministic and does not depend on a model verifier completing successfully.

## 14. Progress Feed

Progress rendering is implemented in `deep_research/progress.py`.

Progress modes:

| Mode | Behavior |
| --- | --- |
| `live` | Concise progress feed. Default. |
| `raw` | Raw graph messages and tool payloads. Useful for debugging. |
| `quiet` | Suppresses progress and prints final paths only. |

Example `live` output:

```text
[12:11:41] run: created C:\...\runs\...
[12:11:56] run: starting research stream
[12:11:58] search: retrieval-augmented generation definition (top 1)
[12:12:01] search: registered 1 source candidate(s): [1]
[12:12:02] scrape: https://www.ibm.com/think/topics/retrieval-augmented-generation
[12:12:48] scrape: source [1] What is RAG (Retrieval Augmented Generation)? | IBM (2,500 chars)
[12:12:55] run: finalizing artifacts
[12:12:55] run: complete
```

The progress feed shows observable execution state. It does not expose private model chain-of-thought.

Every run also maintains `activity.html`, an auto-refreshing local dashboard
rendered from `activity.jsonl`. The dashboard shows stage counts, the latest
event, recent search/scrape/write/verify/fallback activity, and a clear note
that it is not hidden chain-of-thought. For terminal inspection, use:

```powershell
python -m deep_research.activity --follow
python -m deep_research.activity --latest --out runs --follow
python -m deep_research.activity runs/<run-dir>
python -m deep_research.activity runs/<run-dir> --follow
```

If no run directory is passed, the activity viewer selects the newest run under
`runs/` or the directory provided with `--out`.

## 15. Report Reconstruction

The ideal path is for the agent to write `report.md` directly with `write_file`.

Some models answer in chat instead of writing the file. To preserve output, `run_research()` captures the final model content and reconstructs `report.md` if needed.

Reconstruction behavior:

- Writes the final model text into `report.md`.
- Appends `## Sources` using `SourceRegistry.source_lines()` when no source section exists.
- Marks `report_reconstructed: true` in `metrics.json`.
- Runs deterministic verification on the reconstructed report.

This fallback keeps the run recoverable without hiding the fact that the agent did not follow the ideal artifact workflow.

## 16. Metrics

Metrics are collected in `deep_research/models.py` and written by `run_research()`.

Core counters:

- `search_count`
- `scrape_count`
- `write_count`
- `read_count`
- `python_exec_count`
- `verification_rounds`

Final metrics additions:

- `runtime_seconds`
- `source_count`
- `report_exists`
- `report_reconstructed`
- `verification_valid`
- `avg_source_quality_score`
- `strong_source_count`
- `error`

Metrics are intentionally machine-readable so benchmark runs and external dashboards can consume them later.

## 17. Benchmark Harness

Benchmark execution is implemented in `deep_research/eval.py`.

Dataset schema:

```json
{
  "id": "seed-rag-vs-finetune",
  "question": "What are the main differences between RAG and fine-tuning for LLM applications?",
  "expected_answer": "RAG retrieves external context at inference time, while fine-tuning changes model weights during training.",
  "must_include": ["retrieval", "fine-tuning", "model weights"],
  "source_requirements": ["retrieval", "fine-tuning"],
  "difficulty": "easy",
  "notes": "Basic sanity benchmark for answer completeness and cited sources."
}
```

Evaluation per case:

1. Run the research agent.
2. Read `report.md`, `verification.json`, and `metrics.json`.
3. Compute expected-answer token recall and must-include phrase coverage.
4. Compute source-requirement coverage against the report plus scraped source documents.
5. Run an LLM judge with `settings.fast_model`.
6. Write one JSONL result row.

The harness treats a failed research case as data, not as a process-level abort.
`ResearchRunError` rows keep any generated run directory, report path,
verification, metrics, `failure.json` category, retry metadata, and repair path.
Unexpected runner failures and judge failures are also captured in the result
row so the remaining dataset cases still run.

Summary metrics:

- `accuracy`
- `citation_validity`
- `source_support`
- `must_include_coverage`
- `source_requirement_coverage`
- `llm_judge`
- `avg_runtime_seconds`
- `failures`
- `run_failure_count`
- `failure_categories`

Each result row also records missing must-include phrases, missing source
requirements, source-quality metrics, tool counts, failure category, report
reconstruction status, and repair checklist path. These fields make benchmark
failures diagnosable without re-reading the whole run directory.

`deep_research/eval_report.py` can summarize a completed result JSONL file.

## 18. Error Handling

The system distinguishes recoverable tool feedback from run-ending errors.

Recoverable behavior:

- Missing file reads return `ERROR: file not found`.
- Missing report verification writes a failed `verification.json`.
- Final model text can be reconstructed into `report.md`.
- Mangled scrape targets can be resolved to registered source candidates.
- Playwright scrape failures fall back to HTTP extraction.
- Blocked, bot-protected, low-content, and non-fetchable pages return unusable source payloads and are skipped by `collect_sources`.

Run-ending behavior:

- Provider API errors.
- Search client failures.
- Path safety violations.
- Empty required inputs.

When a run-ending error occurs, the runner still writes:

- `failure.json`
- `error.txt`
- `transcript.log`
- `verification.json`
- `metrics.json`

`failure.json` classifies common provider and tool failures into categories such
as `quota_or_rate_limit`, `token_budget_exceeded`, `tool_call_parse_error`, and
`auth_or_permission`. It also records retryability, provider retry-after seconds
when present, and a suggested action. The same category is mirrored in
`metrics.json` under `error_category`.

The CLI returns exit code `1` for `ResearchRunError` and exit code `2` for configuration or user-input errors.

## 19. Security and Safety Boundaries

The implementation has several local safety boundaries:

- File tools can only operate within the current run directory.
- `..` path escapes are rejected.
- Windows drive paths and UNC paths are rejected.
- Leading slash paths are treated as DeepAgents-style virtual run-root paths, not operating-system root paths.
- `.env`, generated runs, eval outputs, and caches are gitignored.
- API keys are loaded into settings but excluded from dataclass repr output.

Current limitations:

- `python_repl` can execute arbitrary Python and should only be exposed to trusted users.
- There is no sandbox around fetched web content beyond text extraction.
- There is no authenticated browser/session support.
- There is no per-domain allowlist or blocklist yet.

## 20. Testing Strategy

Tests live in `tests/` and currently cover:

- `.env` loading and missing-key validation.
- Provider resolution for Google and Groq.
- URL canonicalization and tracking-parameter removal.
- Source registry dedupe by canonical URL and content hash.
- Run artifact path safety.
- Leading slash virtual-root path behavior.
- Subagent loading and model override.
- Citation parsing and deterministic verification.
- Cited-but-unscraped source detection.
- Mocked search/scrape integration.
- Progress formatting and tool progress events.
- Eval harness JSONL output.
- Report reconstruction source-section insertion.

The standard command is:

```powershell
python -m pytest
```

Live provider checks are done manually with constrained commands because they consume external API quota.

## 21. Known Tradeoffs

### DeepAgents Compliance vs Recovery

The prompt tells the model to write `report.md`, but some providers answer directly. The runner reconstructs a report to preserve user value. This is pragmatic, but `report_reconstructed` remains visible in metrics so benchmark consumers can penalize it if needed.

### Full Source Text vs Provider Limits

The system saves more scraped source text to disk than it returns to the model. This avoids Groq token-per-minute failures while preserving reproducibility. The tradeoff is that the model may reason over excerpts rather than full saved pages unless it explicitly reads source files.

### Deterministic Verification vs Semantic Fact Checking

The verifier can prove citation structure and source availability. It does not fully prove every claim is semantically entailed by the source. The benchmark harness includes an LLM judge path for broader quality scoring, but deterministic checks remain the hard gate.

### Public Web First

The current architecture is intentionally public-web-only. Private sources, local documents, PDF extraction, and authenticated browsing should be added as separate ingestion subsystems instead of being mixed into the current Tavily/Search/Scrape path.

## 22. Extension Points

Likely future extensions:

- Add a document-ingestion subsystem for PDFs, DOCX, CSV, and local corpora.
- Replace `PythonREPL` with a maintained, sandboxed analysis runtime.
- Tune source-quality weights with benchmark outcomes and domain allow/block lists.
- Add per-source extraction metadata such as HTTP status, content type, and final redirect chain.
- Add semantic claim verification against scraped source chunks.
- Add durable pause/resume scheduling for retry windows longer than the configured in-process wait cap.
- Add optional richer browser controls for filtering and comparing activity timelines.
- Add a report renderer for HTML/PDF exports.
- Add benchmark history tracking across runs.

## 23. Current Operational Command Set

Run a normal research task:

```powershell
python -m deep_research "how to build a multi-step web research agent from scratch"
```

Force Groq:

```powershell
python -m deep_research --provider groq "how to build a multi-step web research agent from scratch"
```

Force Google:

```powershell
python -m deep_research --provider google "how to build a multi-step web research agent from scratch"
```

Use raw debug streaming:

```powershell
python -m deep_research --progress raw "question"
```

Use compact output only:

```powershell
python -m deep_research --progress quiet "question"
```

Run seed eval:

```powershell
python -m deep_research.eval --dataset benchmarks/seed.jsonl --out eval_runs --limit 1
```

Summarize eval:

```powershell
python -m deep_research.eval_report eval_runs/<run>/results.jsonl
```

Run tests:

```powershell
python -m pytest
```

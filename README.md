# Deep Research Agent

Graph-enforced deep research engine with v2 artifacts, branch-based source acquisition, evidence cards, semantic verification, resume checkpoints, local document ingestion, MCP manifest ingestion, and Gemini managed Deep Research support.

The default engine is `local_langgraph`. It does not rely on a prompt-led orchestrator to decide whether to delegate. The LangGraph controller owns the lifecycle:

```text
classify_request -> plan -> acquire_sources -> read_sources -> build_evidence
  -> evidence_hygiene -> semantic_enrichment -> check_coverage
  -> synthesize -> verify -> repair_or_finish
```

## Setup

```powershell
pip install -e .[test]
playwright install chromium
```

Create `.env` from `.env.example`.

For local LangGraph web research:

```text
GOOGLE_API_KEY=...
GROQ_API_KEY=...       # optional, depending on provider routing
OPENROUTER_API_KEY=... # optional free-model fallback/provider lane
TAVILY_API_KEY=...
```

For Gemini managed Deep Research:

```text
GOOGLE_API_KEY=...
```

## Run

Local graph research:

```powershell
python -m deep_research run "How do urban heat islands affect public health?"
```

The old shorthand still works:

```powershell
python -m deep_research "How do urban heat islands affect public health?"
```

Resume a checkpointed run:

```powershell
python -m deep_research resume RUN_ID
```

Rerun verification only:

```powershell
python -m deep_research verify RUN_ID
```

Run Gemini managed Deep Research:

```powershell
python -m deep_research managed gemini "How do urban heat islands affect public health?"
```

## Useful Options

```powershell
python -m deep_research run `
  --mode balanced `
  --engine local_langgraph `
  --min-usable-sources 17 `
  --max-search-queries 48 `
  --max-candidates 1000 `
  --input .\docs `
  --mcp-manifest .\mcp_sources.json `
  "How do urban heat islands affect public health?"
```

Key options:

- `--mode fast|balanced|max_quality`
- `--engine local_langgraph|gemini_managed|openai_managed`
- `--input PATH`: ingest local PDF, DOCX, Markdown, TXT, CSV/XLSX, or HTML
- `--mcp-manifest PATH`: ingest connector-sourced content from a JSON manifest
- `--min-usable-sources N`
- `--max-search-queries N`
- `--max-candidates N`
- `--max-followup-queries-per-branch N`: interleave missing-branch follow-up searches fairly
- `--min-source-words N`
- `--provider auto|google|groq|hybrid|ollama|openrouter`
- `--provider openrouter`: use OpenRouter-compatible chat completions; the default model is `openrouter/free`
- `--model-request-timeout-seconds N`: cap individual provider calls
- `--model-max-output-tokens N`: give long synthesis/report calls enough output room
- `--scrape-timeout-ms N`: cap each fallback scrape/fetch attempt
- `--scrape-retries N`: control retry count for fallback fetches
- `--allow-weak-tool-models`: opt out of strict tool-model policy
- `--no-llm-planning`: disable JSON-only semantic plan enrichment
- `--no-llm-synthesis`: use deterministic evidence-card report synthesis
- `--no-semantic-verification`: skip LLM evidence/report semantic gates
- `--progress live|raw|quiet`

## Artifacts

Each run writes `runs/<timestamp-slug>/` with:

- `request.json`
- `plan.json`
- `activity.jsonl`
- `activity.md`
- `activity.html`
- `sources.jsonl`
- `evidence_cards.jsonl`
- `evidence_rejections.jsonl`
- `semantic_judgments.json`
- `coverage.json`
- `report.md`
- `verification.json`
- `metrics.json`
- `manifest.json`
- `model_routes.json`
- `checkpoints/latest.json`
- `source_docs/source_<id>.md`
- `failure.json` and `error.txt` when the run fails

`activity.*` shows observable progress and status only; it does not expose hidden reasoning.

## Verification

The v2 verifier checks:

- every factual paragraph has citations
- cited IDs exist in usable `SourceRecordV2` records
- cited sources appear in the `## Sources` section
- cited paragraphs are supported by source text
- report claims link back to evidence cards
- required branches are covered
- source quality meets the configured minimum
- report structure is sufficient

`verification.json` contains the failure list, weakly supported claims, cited source IDs, and per-gate scores.

## Managed Gemini

Gemini managed mode uses the Gemini Interactions API with:

```text
agent = deep-research-pro-preview-12-2025
background = true
```

The managed report is written into the same v2 artifact surface where possible. Source references parsed from the managed report are marked with `managed_gemini` provenance.

## Evaluate

The existing eval entry points remain available:

```powershell
python -m deep_research.eval --dataset benchmarks/seed.jsonl --out eval_runs --limit 1
python -m deep_research.eval_report eval_runs/<run>/results.jsonl
```

Benchmarks should define topic-specific must-include requirements, source requirements, and coverage expectations without relying on hardcoded planner branches for any single subject.

## OpenRouter Free Models

OpenRouter is supported through model specs like:

```text
DEEP_RESEARCH_PROVIDER=openrouter
DEEP_RESEARCH_MODEL=openrouter:openrouter/free
```

You can also use it for cheaper utility roles while keeping stronger providers for planning/synthesis:

```text
DEEP_RESEARCH_JUDGE_MODEL=openrouter:openrouter/free
DEEP_RESEARCH_FAST_MODEL=openrouter:openrouter/free
```

OpenRouter's free router has lower and changing rate limits, so it is best as an explicit fallback or utility lane rather than the default source-acquisition or final-writing bottleneck. Specific free variants can be used with model IDs ending in `:free`, for example `openrouter:meta-llama/llama-3.2-3b-instruct:free`.

## Tavily Key Pool

Search acquisition rotates numbered Tavily keys:

```text
TAVILY_API_KEY=...
TAVILY_API_KEY1=...
TAVILY_API_KEY2=...
TAVILY_API_KEY_3=...
TAVILY_API_KEYS=key-a,key-b
TAVILY_SEARCH_API_KEY=...
```

If one key hits a usage/rate limit, the search client tries the next configured key before failing the query. The documented names are preferred, but the loader also discovers other environment variables whose names contain `TAVILY`, `API`, and `KEY`.

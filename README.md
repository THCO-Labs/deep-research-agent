# Deep Research Agent

Benchmark-grade public-web research agent built on DeepAgents, Tavily, Playwright,
and provider-selectable Groq or Google Gemini models. It creates reproducible run
artifacts, tracks sources, checks citations deterministically, and includes a
benchmark harness.
Verification now checks both citation structure and a deterministic source-text
support score for cited paragraphs.

## Setup

1. Install dependencies:

   ```powershell
   pip install -e .[test]
   playwright install chromium
   ```

2. Create `.env` from `.env.example` and set:

   ```text
GOOGLE_API_KEY=...
GOOGLE_API_KEY1=...   # optional second Google key
GROQ_API_KEY=...
GROQ_API_KEY1=...     # optional second Groq key
TAVILY_API_KEY=...
```

If both Groq and Google keys are present, `auto` provider mode uses `hybrid` by
default so the graph can use both providers in one run. You can force a provider
when needed:

```powershell
python -m deep_research --provider hybrid "question"
python -m deep_research --provider groq "question"
python -m deep_research --provider google "question"
python -m deep_research --provider groq --model openai/gpt-oss-20b "question"
```

The default Groq model is `openai/gpt-oss-20b` because it supports tool calling
and fits Groq on-demand token limits better than larger models. Use larger Groq
models only when your tier has enough TPM headroom.

For Groq runs, the app registers a DeepAgents runtime profile that hides the
framework's internal `write_todos` tool. The CLI progress feed remains the
source of visible progress, and this avoids Groq JSON-parser failures caused by
malformed internal todo tool calls.

Additional keys can be added as `GROQ_API_KEY1`, `GROQ_API_KEY2`, and
`GOOGLE_API_KEY1`, `GOOGLE_API_KEY2`. The agent treats them as provider key
pools and assigns orchestrator/researcher/planner/verifier/analyst/judge roles
across the available keys. With two Groq keys and two Google keys in hybrid
mode, the active default run uses all four: orchestrator on Groq key 0,
researcher on Groq key 1, planner on Google key 0, and verifier on Google key 1.
It does not print the key values; metrics only record the number of keys
available per provider. Each run also writes `model_routes.json`, which records
role, provider, model, key slot, fallback routes, and safe key labels such as
`GROQ_API_KEY1` without storing API key values. Model fallback is enabled by
default: a rate-limit, TPM/token-budget, or tool-call parse failure tries
same-provider alternate keys first, then cross-provider fallback routes in
hybrid mode. Disable it with `--no-model-fallbacks` or
`DEEP_RESEARCH_MODEL_FALLBACKS=false`.

If every fallback candidate returns a provider retry window, the model layer can
pause and retry once by default. Configure this with
`DEEP_RESEARCH_PROVIDER_RETRY_ATTEMPTS`,
`DEEP_RESEARCH_PROVIDER_RETRY_MAX_WAIT_SECONDS`,
`--provider-retry-attempts`, or `--provider-retry-max-wait-seconds`. Retry waits
are emitted as `model_retry` events in the terminal, `activity.jsonl`, and
`activity.html`.

Groq and hybrid runs keep full scraped source files in the run directory, but
tool responses and `read_file` previews are capped by
`DEEP_RESEARCH_TOOL_EXCERPT_CHAR_LIMIT` to avoid Groq on-demand TPM failures.
Scrapes that resolve to bot checks, Cloudflare pages, very low-content pages, or
fetch failures such as 403s are returned as unusable source results so the
researcher can choose another URL instead of ending the run or saving a bad
citation source.
For normal research branches, the researcher uses `collect_sources`, which
searches and scrapes candidates in one deterministic recovery loop until it has
enough usable, citable sources or reports that more sources are needed.
Candidates and scraped sources include deterministic quality metadata:
`source_quality_score`, `source_quality_label`, `source_quality_type`, and
`source_quality_reasons`. `collect_sources` spends scrape budget on higher
quality candidates first, preferring official documentation, government,
standards, and academic sources over generic blogs or user-content platforms.

You can route individual roles to different model strings:

```powershell
python -m deep_research --provider groq `
  --model openai/gpt-oss-20b `
  --planner-model openai/gpt-oss-20b `
  --researcher-model openai/gpt-oss-20b `
  --verifier-model openai/gpt-oss-20b `
  "question"
```

Hugging Face can be added for compatible roles through LangChain-supported model
strings after installing the optional extra:

```powershell
pip install -e ".[huggingface]"
```

Only use Hugging Face models for tool-using roles when that model/provider
supports chat tool calling. Many free hosted models do not, so Groq remains the
safer default for planner/researcher/verifier agents.

## Run

```powershell
python -m deep_research "What are the main differences between RAG and fine-tuning for LLM applications?"
```

The default progress mode is a concise live feed:

```text
[11:28:19] run: created runs/...
[11:28:19] model: orchestrator=groq:openai/gpt-oss-20b via GROQ_API_KEY; ...
[11:28:20] run: building agent graph
[11:28:25] search: retrieval-augmented generation (top 5)
[11:28:27] search: registered 3 source candidate(s): [1], [2], [3]
[11:28:29] scrape: source [1] Example Title (15,000 chars)
[11:28:33] collect: gathered 2/2 usable source(s), skipped 1
[11:28:45] verify: passed: score 1.00, 0 unsupported paragraph(s)
```

Progress options:

```powershell
python -m deep_research --progress live "question"   # concise progress feed
python -m deep_research --progress raw "question"    # raw agent stream
python -m deep_research --progress quiet "question"  # final artifact paths only
```

Each run also saves a visible progress trail:

- `activity.md`: human-readable activity log for plan/search/scrape/write/verify steps.
- `activity.jsonl`: machine-readable progress events with timestamps and structured data.
- `activity.html`: auto-refreshing local dashboard for the same observable progress events.

These logs show observable actions and status, not hidden chain-of-thought.

To inspect progress from another terminal while a run is active:

```powershell
python -m deep_research.activity --follow
python -m deep_research.activity --latest --out runs --follow
python -m deep_research.activity runs/<run-dir> --follow
python -m deep_research.activity runs/<run-dir> --html
```

When `run_dir` is omitted, the viewer opens the latest run under `runs`.
You can also open `runs/<run-dir>/activity.html` directly in a browser; it refreshes every 5 seconds.

Artifacts are written to `runs/<timestamp-slug>/`:

- `request.md`
- `activity.md`
- `activity.jsonl`
- `activity.html`
- `run_manifest.json`
- `model_routes.json`
- `research_plan.md`
- `sources.jsonl`
- `findings/`
- `source_docs/`
- `report.md`
- `verification.json`
- `findings/verification_repair.md` when final verification fails
- `metrics.json`
- `failure.json` when a run-ending error occurs

Run-ending provider failures are classified in `failure.json` and echoed by the
CLI. Quota, token budget, tool-call parse, and permission errors include a
machine-readable category, retryability flag, retry-after value when present,
and suggested action.

`run_manifest.json` captures the redacted run configuration, model route
manifest, runtime metadata, package versions, key-pool counts, and progress mode
without storing API key values. Use it with `model_routes.json`, `metrics.json`,
and `sources.jsonl` when comparing or reproducing runs.

`verification.json` includes `citation_validity_score`,
`source_support_score`, `unsupported_claims` for uncited factual paragraphs, and
`weakly_supported_claims` for cited paragraphs that do not match the scraped
source text closely enough.

When final verification fails, the runner also writes
`findings/verification_repair.md`, a deterministic checklist generated from
`verification.json`. This gives you the exact citation, source-list,
unscraped-source, or source-support repairs needed before rerunning
verification.

`metrics.json` includes `avg_source_quality_score` and `strong_source_count` for
scraped sources.

## Evaluate

```powershell
python -m deep_research.eval --dataset benchmarks/seed.jsonl --out eval_runs --limit 1
python -m deep_research.eval_report eval_runs/<run>/results.jsonl
```

Each eval row includes deterministic diagnostics alongside the LLM judge score:
`expected_answer_recall`, `must_include_coverage`, `missing_must_include`,
`source_requirement_coverage`, `missing_source_requirements`,
`citation_verifier_score`, `source_support_score`, source-quality metrics,
runtime/tool counts, failure category, retry metadata, judge error, run-failure
state, and repair checklist path when present. A failed research case is written
as a failed row and the remaining benchmark cases continue.

## Test

```powershell
pytest
```

## Architecture

See `ARCHITECTURE.md` for the full system architecture, runtime flow, artifact
model, provider strategy, verification design, and extension points.

# Deep Research Agent

Benchmark-grade public-web research agent built on DeepAgents, Tavily, Playwright,
and provider-selectable Groq or Google Gemini models. It creates reproducible run
artifacts, tracks sources, checks citations deterministically, and includes a
benchmark harness.

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
role, provider, model, key slot, and safe key label such as `GROQ_API_KEY1`
without storing API key values.

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

These logs show observable actions and status, not hidden chain-of-thought.

Artifacts are written to `runs/<timestamp-slug>/`:

- `request.md`
- `activity.md`
- `activity.jsonl`
- `model_routes.json`
- `research_plan.md`
- `sources.jsonl`
- `findings/`
- `source_docs/`
- `report.md`
- `verification.json`
- `metrics.json`
- `failure.json` when a run-ending error occurs

Run-ending provider failures are classified in `failure.json` and echoed by the
CLI. Quota, token budget, tool-call parse, and permission errors include a
machine-readable category, retryability flag, retry-after value when present,
and suggested action.

## Evaluate

```powershell
python -m deep_research.eval --dataset benchmarks/seed.jsonl --out eval_runs --limit 1
python -m deep_research.eval_report eval_runs/<run>/results.jsonl
```

## Test

```powershell
pytest
```

## Architecture

See `ARCHITECTURE.md` for the full system architecture, runtime flow, artifact
model, provider strategy, verification design, and extension points.

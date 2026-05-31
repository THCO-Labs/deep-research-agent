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
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

If `GROQ_API_KEY` is present, `auto` provider mode uses Groq by default to avoid
Gemini free-tier quota failures. You can force a provider when needed:

```powershell
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
[11:28:20] run: building agent graph
[11:28:25] search: retrieval-augmented generation (top 5)
[11:28:27] search: registered 3 source candidate(s): [1], [2], [3]
[11:28:29] scrape: source [1] Example Title (15,000 chars)
[11:28:45] verify: passed: score 1.00, 0 unsupported paragraph(s)
```

Progress options:

```powershell
python -m deep_research --progress live "question"   # concise progress feed
python -m deep_research --progress raw "question"    # raw agent stream
python -m deep_research --progress quiet "question"  # final artifact paths only
```

Artifacts are written to `runs/<timestamp-slug>/`:

- `request.md`
- `research_plan.md`
- `sources.jsonl`
- `findings/`
- `source_docs/`
- `report.md`
- `verification.json`
- `metrics.json`

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

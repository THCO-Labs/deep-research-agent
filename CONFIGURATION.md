# Deep Research Configuration Guide

This guide lists the runtime settings a user can change, what they do, their
defaults, and the practical values to start with.

Most settings can be changed in two ways:

- CLI flags for one run, for example `python -m deep_research run --mode balanced "question"`.
- Environment variables in `.env` for persistent defaults, for example `DEEP_RESEARCH_MODE` is not used, but `DEEP_RESEARCH_PROVIDER=hybrid` is.

CLI flags win over environment values when both exist. Model overrides from the
environment are ignored when `--provider` is passed explicitly, so provider
experiments do not accidentally inherit incompatible model names.

## Quick Start Profiles

### Recommended General Profile

Use this for most public-web research where quality matters and a 10-20 minute
run is acceptable.

```powershell
python -m deep_research run `
  --mode max_quality `
  --provider auto `
  "Your research question"
```

Best `.env`:

```text
GOOGLE_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

Why: `auto` becomes `hybrid` when Google and Groq keys are present. Google is
used for higher-judgment roles, Groq for faster collection roles, and fallback
routing gives the run more ways to recover from quota or timeout failures.

### Faster Draft Profile

Use this when you want a cheaper or faster pass before a final run.

```powershell
python -m deep_research run `
  --mode fast `
  --max-rounds 2 `
  --min-usable-sources 17 `
  "Your research question"
```

Why: the run gathers fewer sources and uses shorter scrape timeouts. This is good
for scoping, but not ideal for final reports on complex or high-stakes topics.

### Local Documents Plus Web Profile

Use this when you have local files that should be considered alongside web
sources.

```powershell
python -m deep_research run `
  --input .\docs `
  --input .\brief.pdf `
  --mode balanced `
  "Your research question"
```

Why: local files are ingested as candidate source material before synthesis, while
the web acquisition stage can still fill gaps.

### Lower-Cost OpenRouter Profile

```powershell
python -m deep_research run `
  --provider openrouter `
  --mode balanced `
  "Your research question"
```

Best `.env`:

```text
OPENROUTER_API_KEY=...
TAVILY_API_KEY=...
```

Why: useful when you need a single provider lane, but free OpenRouter models can
have changing limits and weaker tool-following. Prefer it for exploratory runs or
as a fallback lane, not the best default for final reports.

## Commands

| Command | Purpose | Notes |
| --- | --- | --- |
| `python -m deep_research run "question"` | Run the local LangGraph engine. | The explicit `run` subcommand is preferred. |
| `python -m deep_research "question"` | Shorthand for `run`. | Kept for compatibility. |
| `python -m deep_research resume RUN_ID` | Resume from `runs/RUN_ID/checkpoints/latest.json`. | Uses the current settings for provider routing and output root. |
| `python -m deep_research verify RUN_ID` | Rerun verification for an existing run. | Does not rerun acquisition or synthesis. |
| `python -m deep_research managed gemini "question"` | Use Gemini managed Deep Research. | Requires `GOOGLE_API_KEY`. |
| `python -m deep_research managed openai "question"` | Select the OpenAI managed engine name. | Availability depends on the managed implementation. |
| `python -m deep_research config` | Print this guide. | Shows environment-only settings that normal help omits. |

## Core Runtime Settings

| CLI flag | Environment variable | Default | Recommended | Effect |
| --- | --- | --- | --- | --- |
| `--mode fast\|balanced\|max_quality` | none | `max_quality` | `max_quality` for final work, `balanced` for iteration, `fast` for drafts | Sets source/search depth defaults, rounds, scrape timeout, and retries. |
| `--out PATH` | none | `runs` | Keep default unless separating experiments | Root directory for run artifacts. Relative paths resolve from the project root. |
| `--engine local_langgraph\|gemini_managed\|openai_managed` | `DEEP_RESEARCH_ENGINE` | `local_langgraph` | `local_langgraph` | Chooses the research engine. Managed engines bypass most local graph behavior. |
| `--progress live\|raw\|quiet` | none | `live` | `live` for humans, `raw` for logs, `quiet` for scripts | Controls progress output only. Artifacts are still written. |
| `--live` | none | `false` | Usually leave off | Stores a live-mode setting for runtime surfaces that inspect `Settings.live`. |
| `--writing-guidance TEXT` | none | empty | Use for tone, length, or audience constraints | Adds instructions to the synthesis/writing stage without changing acquisition. |

Mode-derived defaults:

| Mode | `max_sources` | `max_rounds` | `min_usable_sources` | `max_search_queries` | `max_candidates` | `min_source_words` | scrape timeout | scrape retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `fast` | `24` | `2` | `17` | `16` | `160` | `180` | `10000 ms` | `1` |
| `balanced` | `40` | `3` | `24` | `48` | `420` | `250` | `15000 ms` | `1` |
| `max_quality` | `0` | `3` | `40` | `80` | `250` | `350` | `20000 ms` | `2` |

`max_sources=0` means no explicit final source cap. The acquisition process is
still bounded by rounds, candidates, search query counts, and verification needs.

## Acquisition And Source Controls

| CLI flag | Environment variable | Default | Recommended | Effect |
| --- | --- | --- | --- | --- |
| `--max-sources N` | none | mode-derived | `0` for final, `24-40` for faster runs | Caps usable sources. Must be `0` or at least `17`. |
| `--max-rounds N` | none | mode-derived | `3` | Caps acquisition/repair rounds. More rounds can help hard topics but increases state size and runtime. |
| `--min-usable-sources N` | `DEEP_RESEARCH_MIN_USABLE_SOURCES` | mode-derived, floored to `17` | `24-40` | Minimum usable scraped sources before synthesis is considered adequately supplied. |
| `--max-search-queries N` | `DEEP_RESEARCH_MAX_SEARCH_QUERIES` | mode-derived | `48-80` | Upper bound for generated search queries. More queries improve breadth but cost time/search quota. |
| `--max-candidates N` | `DEEP_RESEARCH_MAX_CANDIDATES` | mode-derived | `250` for final, `160` for fast | Caps URL candidates retained for acquisition. Must be at least `min_usable_sources`. |
| `--max-followup-queries-per-branch N` | `DEEP_RESEARCH_MAX_FOLLOWUP_QUERIES_PER_BRANCH` | `12` | `5-12` | Caps follow-up searches for any one missing research branch so one branch does not starve others. |
| `--min-source-words N` | `DEEP_RESEARCH_MIN_SOURCE_WORDS` | mode-derived | `250-350` | Rejects shallow pages below the word threshold. Lower it for sparse official pages; raise it for deep literature reviews. |
| none | `DEEP_RESEARCH_MIN_RELEVANT_CHUNKS` | `1` | `1` | Minimum relevant chunks required when scoring source relevance. Raise only if noisy sources pass too often. |
| none | `DEEP_RESEARCH_SEARCH_DEPTH` | `advanced` | `advanced` | Passed to search providers that support depth. |
| none | `DEEP_RESEARCH_ALLOW_RAW_CONTENT` | `true` | `true` | Lets search responses include raw content when providers support it. Disable only if payload size is a problem. |
| none | `DEEP_RESEARCH_PRECOLLECT_SOURCES` | `true` | `true` | Runs deterministic source precollection before graph synthesis. Disable for debugging graph-only behavior. |
| none | `DEEP_RESEARCH_ACQUISITION_TIMEOUT_SECONDS` | `1500` | `900-1500` | Overall acquisition timeout. Lower values fail faster; higher values help slow sites. |

Best default: leave `max_quality` alone unless you are debugging runtime or quota.
The defaults were tightened to avoid huge acquisition state while still collecting
enough evidence for strong reports.

## Scraping And Fetching Controls

| CLI flag | Environment variable | Default | Recommended | Effect |
| --- | --- | --- | --- | --- |
| `--scrape-char-limit N` | `DEEP_RESEARCH_SCRAPE_CHAR_LIMIT` | `15000` for Google, `6000` for Groq/hybrid/Ollama/OpenRouter | Keep default | Max characters retained per scraped source for downstream tools. Larger values improve context but increase token use. |
| `--scrape-timeout-ms N` | `DEEP_RESEARCH_SCRAPE_TIMEOUT_MS` | mode-derived | `15000-20000` | Timeout for each scrape/fetch attempt. Increase for slow official sites. |
| `--scrape-retries N` | `DEEP_RESEARCH_SCRAPE_RETRIES` | mode-derived | `1-2` | Retry count for fallback fetches. Must be at least `1`. |
| `--max-browser-scrapes-per-query N` | `DEEP_RESEARCH_MAX_BROWSER_SCRAPES_PER_QUERY` | `12` | `8-12`; use `0` to disable browser fallback | Caps Playwright/browser fallback scrapes per query. Higher values improve recovery from JS-heavy sites but slow runs. |
| none | `DEEP_RESEARCH_BLOCKED_SOURCE_PATTERNS` | empty | Add benchmark/leak URLs or known bad domains | Semicolon-separated regex patterns. Matching source URLs are skipped. |
| none | `DEEP_RESEARCH_TOOL_EXCERPT_CHAR_LIMIT` | `2500` for Google, `900` for Groq/hybrid/Ollama/OpenRouter | Keep default | Max excerpt size returned from tools to agents. Lower values protect small-context models. |

Use `DEEP_RESEARCH_BLOCKED_SOURCE_PATTERNS` when you know certain URLs should
never be used, for example:

```text
DEEP_RESEARCH_BLOCKED_SOURCE_PATTERNS=deep[_-]?research[_-]?bench;reference\.jsonl
```

## Local And Connector Inputs

| CLI flag | Environment variable | Default | Recommended | Effect |
| --- | --- | --- | --- | --- |
| `--input PATH` | `DEEP_RESEARCH_LOCAL_INPUTS` | empty | Use repeated `--input` for CLI runs | Ingest local files or directories. Supported formats include PDF, DOCX, Markdown, TXT, CSV/XLSX, and HTML. |
| `--mcp-manifest PATH` | `DEEP_RESEARCH_MCP_MANIFEST` | empty | Use when connector data is prepared | Reads connector-sourced content from a JSON manifest. |

`DEEP_RESEARCH_LOCAL_INPUTS` is semicolon-separated:

```text
DEEP_RESEARCH_LOCAL_INPUTS=.\docs;.\brief.pdf
```

## Provider And Model Routing

| CLI flag | Environment variable | Default | Recommended | Effect |
| --- | --- | --- | --- | --- |
| `--provider auto\|google\|groq\|hybrid\|ollama\|openrouter` | `DEEP_RESEARCH_PROVIDER` | `auto` | `auto` | Chooses model provider routing. `auto` resolves from configured keys. |
| `--model MODEL` | `DEEP_RESEARCH_MODEL` | provider default | Usually keep default | Main/orchestrator model. Short names are prefixed with the selected provider. |
| `--fast-model MODEL` | `DEEP_RESEARCH_FAST_MODEL` | provider default | Keep default unless optimizing cost | Utility model used by faster roles and fallback defaults. |
| `--planner-model MODEL` | `DEEP_RESEARCH_PLANNER_MODEL` | fast model or hybrid role default | Strong model for complex questions | Semantic planning role. |
| `--researcher-model MODEL` | `DEEP_RESEARCH_RESEARCHER_MODEL` | fast model or hybrid role default | Fast, cheap model is usually fine | Research/source collection role. |
| `--analyst-model MODEL` | `DEEP_RESEARCH_ANALYST_MODEL` | fast model or hybrid role default | Keep default | Quantitative analysis role. |
| `--verifier-model MODEL` | `DEEP_RESEARCH_VERIFIER_MODEL` | fast model or hybrid role default | Strong model for final work | Verification role. |
| `--judge-model MODEL` | `DEEP_RESEARCH_JUDGE_MODEL` | fast model or hybrid role default | Strong model for final work | Source/evidence judgment role. |
| none | `DEEP_RESEARCH_SYNTHESIS_MODEL` | falls back to `model` | Set explicitly for final writing if needed | Final synthesis model. Empty means use the main model. |
| `--allow-weak-tool-models` | `DEEP_RESEARCH_STRICT_TOOL_MODELS=false` | strict tool models enabled | Keep strict enabled | Allows weaker models for tool-heavy roles. Useful only when you accept more tool-call risk. |

Provider defaults:

| Provider | Main model | Fast model | Notes |
| --- | --- | --- | --- |
| `google` | `google_genai:gemini-2.5-flash` | `google_genai:gemini-2.5-flash` | Best single-provider default when Google quota is available. |
| `groq` | `groq:openai/gpt-oss-20b` | `groq:openai/gpt-oss-20b` | Fast and cheap, but smaller context/excerpt defaults are used. |
| `hybrid` | Google for orchestration/planning/verifying/judging; Groq for research/analysis/fast | role-specific | Best default when both Google and Groq keys exist. |
| `ollama` | `ollama:qwen2.5:7b` | `ollama:qwen2.5:3b` | Local model option. Quality depends on your local Ollama setup. |
| `openrouter` | `openrouter:meta-llama/llama-3.3-70b-instruct:free` | same | Good fallback or low-cost lane, but free limits can change. |

Provider prefixes accepted in model specs:

```text
google_genai:gemini-2.5-flash
groq:openai/gpt-oss-20b
ollama:qwen2.5:7b
openrouter:meta-llama/llama-3.3-70b-instruct:free
mistral_ai:...
together:...
```

If a model name has no prefix, the selected provider is added automatically:

```powershell
python -m deep_research run --provider groq --model openai/gpt-oss-120b "question"
```

This resolves to `groq:openai/gpt-oss-120b`.

## Fallbacks, Retries, And Timeouts

| CLI flag | Environment variable | Default | Recommended | Effect |
| --- | --- | --- | --- | --- |
| `--no-model-fallbacks` | `DEEP_RESEARCH_MODEL_FALLBACKS=false` | `true` | Keep enabled | Disables same-provider key fallback, hybrid cross-provider fallback, and OpenRouter fallback routes. |
| `--provider-retry-attempts N` | `DEEP_RESEARCH_PROVIDER_RETRY_ATTEMPTS` | `1` | `1-2` | Retries all model candidates after retryable provider failures. |
| `--provider-retry-max-wait-seconds N` | `DEEP_RESEARCH_PROVIDER_RETRY_MAX_WAIT_SECONDS` | `60` | `15-60` | Max wait between retry attempts after rate limits/timeouts. |
| `--model-request-timeout-seconds N` | `DEEP_RESEARCH_MODEL_REQUEST_TIMEOUT_SECONDS` | `120` | `120` for final, `45-90` for debugging | Per-request model timeout. |
| `--model-max-output-tokens N` | `DEEP_RESEARCH_MODEL_MAX_OUTPUT_TOKENS` | `8192` | `8192-12000` | Max model output tokens where providers support it. Increase for long reports. |

Fallback behavior:

- With multiple numbered keys for the same provider, roles are assigned stable key slots and can fall back to other keys.
- In `hybrid`, Google roles can fall back to Groq and Groq roles can fall back to Google when matching keys exist.
- If `OPENROUTER_API_KEY` exists and the active route is not OpenRouter, the default OpenRouter free model can be used as an additional fallback.

## Planning, Synthesis, And Verification Gates

| CLI flag | Environment variable | Default | Recommended | Effect |
| --- | --- | --- | --- | --- |
| `--no-llm-planning` | `DEEP_RESEARCH_LLM_PLANNING=false` | `true` | Keep enabled | Disables LLM semantic plan enrichment. Deterministic planning still runs. |
| `--no-llm-synthesis` | `DEEP_RESEARCH_LLM_SYNTHESIS=false` | `true` | Keep enabled for final writing | Uses deterministic evidence-card report synthesis instead of LLM report writing. |
| `--no-semantic-verification` | `DEEP_RESEARCH_SEMANTIC_VERIFICATION=false` | `true` | Keep enabled | Skips LLM semantic evidence/report gates. Deterministic citation checks still matter. |
| `--semantic-evidence-max-llm-cards N` | `DEEP_RESEARCH_SEMANTIC_EVIDENCE_MAX_LLM_CARDS` | `120` | `120`; use `0` to disable card LLM judging | Caps how many uncached evidence cards receive LLM semantic judgment. |
| `--allow-failed-verification` | `DEEP_RESEARCH_ALLOW_FAILED_VERIFICATION=true` | `false` | Keep false | Allows a run to finish even when verification fails. Use only for debugging. |
| none | `DEEP_RESEARCH_REPORT_QUALITY_GATE` | `true` | Keep enabled | Applies report quality gates before accepting final output. |

Best default: keep all gates enabled. Disable gates only to isolate a failure or
produce a rough draft that you will not treat as verified.

## API Keys

The loader reads `.env` from the project root and also respects shell
environment variables. Values in the shell are not overwritten by `.env`.

| Environment variable | Required for | Notes |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Google local routing and Gemini managed mode | Also supports numbered and pooled forms. |
| `GROQ_API_KEY` | Groq local routing | Optional when using Google-only or OpenRouter-only. |
| `OPENROUTER_API_KEY` | OpenRouter provider or fallback lane | Optional but useful as a free fallback lane. |
| `TOGETHER_API_KEY` | Together model specs | Resolved by provider auto only if no Google/Groq/OpenRouter keys exist. |
| `MISTRAL_API_KEY` | `mistral_ai:` explicit model specs | Not selected by `--provider`, but explicit model specs can use it. |
| `TAVILY_API_KEY` | Tavily search | Optional at validation time because DuckDuckGo fallback can be installed, but recommended. |
| `EXA_API_KEY` | Future/alternate search integrations | Loaded into settings. |
| `BRAVE_SEARCH_API_KEY` | Future/alternate search integrations | Loaded into settings. |
| `FIRECRAWL_API_KEY` | Future/alternate scrape integrations | Loaded into settings. |
| `SERPER_API_KEY` | Future/alternate search integrations | Loaded into settings. |
| `JINA_API_KEY` | Jina scrape helper | Read directly by the scraper when that path is used. |

Key pools support these forms:

```text
GOOGLE_API_KEY=key-a
GOOGLE_API_KEY1=key-b
GOOGLE_API_KEY_2=key-c
GOOGLE_API_KEYS=key-d,key-e
GOOGLE_API_KEY_POOL=key-f;key-g
```

The same numbered/list forms work for `GROQ_API_KEY`, `OPENROUTER_API_KEY`,
`MISTRAL_API_KEY`, `TOGETHER_API_KEY`, and `TAVILY_API_KEY`.

Tavily also discovers environment variable names containing `TAVILY`, `API`, and
`KEY`, such as `TAVILY_SEARCH_API_KEY`. Prefer the documented names for clarity.

## OpenRouter Metadata

| Environment variable | Default | Effect |
| --- | --- | --- |
| `OPENROUTER_HTTP_REFERER` | empty | Adds the HTTP referer header for OpenRouter requests. |
| `OPENROUTER_APP_TITLE` | `Deep Research Agent` | Adds OpenRouter title headers. |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1/chat/completions` | Overrides the OpenRouter chat completions endpoint. |

## Choosing Values

- Increase `--min-usable-sources`, `--max-search-queries`, and `--max-candidates` when reports miss important branches.
- Increase `--scrape-timeout-ms` or `--max-browser-scrapes-per-query` when official or JavaScript-heavy pages are skipped too often.
- Increase `--model-max-output-tokens` when final reports truncate or omit required sections.
- Lower `--max-rounds`, `--max-candidates`, and `--model-request-timeout-seconds` when debugging fast failure paths.
- Keep `--allow-failed-verification` off for anything you plan to trust.
- Prefer provider `auto` with Google plus Groq keys for the best balance of judgment, speed, and fallback recovery.

## Artifact Files To Inspect

Each run writes a directory under `runs/`. These files explain how settings were
applied:

| File | Use |
| --- | --- |
| `run_manifest.json` or `manifest.json` | Redacted runtime settings, package versions, and reproducibility metadata. |
| `model_routes.json` | Resolved role-to-model routing, key slot labels, and fallback routes. |
| `activity.jsonl`, `activity.md`, `activity.html` | Progress events, model fallback events, and retry-window waits. |
| `failure.json` | Failure category, retry advice, and suggested action when a run fails. |
| `verification.json` | Citation, source, coverage, and semantic verification results. |


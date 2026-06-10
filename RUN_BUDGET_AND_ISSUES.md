# Run Budget Notes

Date: 2026-06-09

## Query Quota / Budget Summary

- One research run (default `max_quality` mode) can trigger roughly **30 model calls**.
- It can also trigger up to **192 search calls** in acquisition.
- If you are on a small free quota, treat this as a **hard upper bound** for planning, not a guaranteed minimum.
- In current tests, citation/repair behavior means retries can increase token and call usage.

## Practical Limits to Watch

- Google Gemini free tier can fail quickly on request-count constraints (e.g., 20/day caps observed in logs).
- Some providers enforce token-per-minute limits; one call can request up to `8192` output tokens under default settings.
- For stable runs, use paid/high-usage models or lower per-query budgets when possible.

## What This Means for “one query”

- One user query is not a single API call; it is a small workflow of many steps.
- Use a per-query budget like `>=30` model calls and `>=192` searches as a planning guard.
- If you need a strict budget cap, lower these in config before running.

## Notes from this pass

- 50-query benchmark run scored all zeros due provider quota exhaustion:

| Benchmark | Count | Successful | Errors | Overall | Comprehensiveness | Candidate | Insight | Instruction Following | Readability | Topic Focus | Reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| local-langgraph-50-eng-openrouter-v1 | 50 | 0 | 50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| local-langgraph-50-fallback-fast-v1 | 50 | 0 | 50 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |

- Per-query failure details now live in [`RUN_50_QUERY_FAILURES.md`](RUN_50_QUERY_FAILURES.md), including verification status, fallback-event counts, and repair-event indicators.
- Current update remains documentation of the failure profile while model routing/fallback tuning is applied.

## 2026-06-09 follow-up

- Current status is stable in logging coverage: all 50 runs emit failure + activity artifacts for traceability.
- Operational blocker is still Gemini free-tier daily quota, so next iterations need paid/model-key diversification before score recovery.

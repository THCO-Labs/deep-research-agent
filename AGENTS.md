# Active Workspace Rules

<!-- Maintained by NEXUS's curator. Edits are preserved, but rules
     are de-duplicated and the oldest drop out past the cap. -->

## Consolidated Rules

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

# DeepResearch Bench — Official FACT & RACE Scoring Guide

This document records the exact steps used to score reports from the deep research agent against the official DeepResearch Bench pipeline.

---

## Prerequisites

```powershell
# Clone the benchmark repo
cd C:\Users\Hp\Documents\Codex\deep_research_bench

# API keys (any OpenAI-compatible provider works)
$env:LLM_BACKEND = "openai"
$env:OPENAI_BASE_URL = "https://api.together.xyz/v1"          # or Groq / z.ai
$env:OPENAI_API_KEY = "tgp_v1_xxxxx"                           # your key
$env:FACT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"   # for FACT steps
$env:RACE_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"   # for RACE steps
$env:MAX_OUTPUT_TOKENS = "16000"
```

> **Provider limits**: Together (credit limit), z.ai (balance required), Groq free tier (30K TPM for scout, 12K TPM for 70B). Truncate articles if you hit TPM limits.

---

## Step 0 — Prepare the Submission File

Take the best draft from the agent run and write it as a benchmark raw submission.

```python
import json, os

run_dir = r"C:\Users\Hp\...\runs\YOUR_RUN_DIR"
report = open(os.path.join(run_dir, "best_draft.md"), "r", encoding="utf-8").read()

# Read the benchmark prompt for the task
queries = [json.loads(l) for l in open(
    r"C:\Users\Hp\Documents\Codex\deep_research_bench\data\prompt_data\query.jsonl",
    "r", encoding="utf-8").read().splitlines() if l.strip()]
task = [q for q in queries if q["id"] == 83][0]  # change task ID

row = {"id": task["id"], "prompt": task["prompt"], "article": report.strip() + "\n"}

out_dir = r"C:\Users\Hp\Documents\Codex\deep_research_bench\data\test_data\raw_data"
out_path = os.path.join(out_dir, "your-model-name.jsonl")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

# Also copy to cleaned_data (skip cleaning step)
import shutil
shutil.copy(out_path, out_path.replace("raw_data", "cleaned_data"))
print(f"Written: {out_path}")
```

---

## FACT Pipeline (5 Steps)

The FACT pipeline validates each factual claim in the report against the cited source's actual web content.

### Step 1 — Extract Claims

LLM reads the article and extracts individual factual claims with their citation references.

```powershell
$model = "your-model-name"
$out = "results/fact-official/$model"

python -u -m utils.extract `
    --raw_data_path "data/test_data/raw_data/$model.jsonl" `
    --output_path "$out/extracted.jsonl" `
    --query_data_path "data/prompt_data/query.jsonl"
```

Output: `extracted.jsonl` — contains `citations` list with `{fact, ref_idx, url}` per claim.

### Step 2 — Deduplicate Claims

Groups claims by URL and removes near-duplicates within each URL group.

```powershell
python -u -m utils.deduplicate `
    --raw_data_path "$out/extracted.jsonl" `
    --output_path "$out/deduplicated.jsonl" `
    --query_data_path "data/prompt_data/query.jsonl" `
    --n_total_process 5
```

Output: `deduplicated.jsonl` — contains `citations_deduped` dict: `{url: {facts: [...], url_content: null}}`.

### Step 3 — Scrape Source URLs

Fetches live web content for each cited URL via Jina Reader API.

#### Option A: Live Jina scrape (official method)

```powershell
$env:JINA_API_KEY = "jina_xxxxx"

python -u -m utils.scrape `
    --raw_data_path "$out/deduplicated.jsonl" `
    --output_path "$out/scraped.jsonl" `
    --n_total_process 8
```

#### Option B: Use cached source docs (skip Jina)

If you already have scraped source content in the agent's run directory, inject it directly:

```python
import json, re, os

dedup_path = r"...\results\fact-official\your-model\deduplicated.jsonl"
run_dir = r"...\runs\YOUR_RUN_DIR"

d = json.loads(open(dedup_path, "r", encoding="utf-8").read())

# Load sources.jsonl from the run
by_id = {}
for line in open(os.path.join(run_dir, "sources.jsonl"), "r", encoding="utf-8").read().splitlines():
    if not line.strip(): continue
    r = json.loads(line)
    by_id[int(r["id"])] = r

# Build URL → source_id map
url_to_sid = {}
for sid, rec in by_id.items():
    url_to_sid[rec["url"]] = sid
    url_to_sid[rec["url"].rstrip("/")] = sid

# Parse article Sources section
article = d["article"]
source_line_re = re.compile(r"^\[(\d+)\]\s+(.+?):\s+(https?://\S+)\s*$", re.MULTILINE)
sources_section = re.search(r"(?im)^##\s+Sources\s*$", article)
ref_to_url = {}
if sources_section:
    for m in source_line_re.finditer(article[sources_section.start():]):
        ref_to_url[int(m.group(1))] = m.group(3)

# Strip source file metadata header
header_re = re.compile(r"^(URL:\s|Canonical URL:\s|Branch:\s|Extraction method:\s|Word count:\s)", re.M)

cdd = d["citations_deduped"]
filled = 0
missing = 0
for url in cdd:
    if not url:
        cdd[url]["url_content"] = "no URL provided"
        filled += 1
        continue
    sid = url_to_sid.get(url) or url_to_sid.get(url.rstrip("/"))
    if sid is None:
        for ref_id, ref_url in ref_to_url.items():
            if ref_url.rstrip("/") == url.rstrip("/"):
                sid = url_to_sid.get(ref_url.rstrip("/"))
                break
    if sid is None:
        cdd[url]["url_content"] = "source not found in records"
        missing += 1
        continue
    cp = by_id[sid].get("content_path", "")
    doc_path = os.path.join(run_dir, cp) if cp else ""
    if doc_path and os.path.exists(doc_path):
        raw = open(doc_path, "r", encoding="utf-8").read()
        lines = raw.splitlines(keepends=True)
        saw = False
        for i, line in enumerate(lines):
            if header_re.match(line):
                saw = True
            elif saw and line.strip() == "":
                raw = "".join(lines[i+1:])
                break
        title = by_id[sid]["title"][:120]
        cdd[url]["url_content"] = title + "\n\n" + raw[:30000]
        filled += 1
    else:
        cdd[url]["url_content"] = "source file not found"
        missing += 1

print(f"Filled: {filled}, Missing: {missing}")
out_path = r"...\results\fact-official\your-model\scraped.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(json.dumps(d, ensure_ascii=False) + "\n")
print(f"Wrote: {out_path}")
```

### Step 4 — Validate Claims

LLM judge checks each claim against the scraped source content (supported / unsupported / unknown).

```powershell
python -u -m utils.validate `
    --raw_data_path "$out/scraped.jsonl" `
    --output_path "$out/validated.jsonl" `
    --query_data_path "data/prompt_data/query.jsonl" `
    --n_total_process 1
```

Output: `validated.jsonl` — each URL group now has `validate_res` with per-claim verdicts.

### Step 5 — Compute Score

```powershell
python -u -m utils.stat `
    --input_path "$out/validated.jsonl" `
    --output_path "$out/fact_result.txt"
```

Output: `fact_result.txt`:
```
total_citations: 187.0
total_valid_citations: 116.0
valid_rate: 0.6203
```

**FACT Citation Accuracy** = `valid_rate` = supported / (supported + unsupported)

**FACT Effective Citation** = supported / (supported + unsupported + unknown)

---

## RACE Pipeline (2 Steps)

The RACE pipeline scores the report against a reference report using task-specific criteria.

### Step 1 — Clean Articles (optional but recommended)

The cleaning step compresses long articles so they fit within LLM context windows.

```powershell
# Delete existing cleaned file first to force re-cleaning
Remove-Item "data/test_data/cleaned_data/$model.jsonl" -Force -ErrorAction SilentlyContinue

python -u -m utils.clean_article `
    --target_model $model `
    --language en `
    --raw_data_dir data/test_data/raw_data `
    --cleaned_data_dir data/test_data/cleaned_data
```

> If cleaning fails due to API limits, copy the raw file to cleaned_data instead:
> ```powershell
> Copy-Item "data/test_data/raw_data/$model.jsonl" "data/test_data/cleaned_data/$model.jsonl" -Force
> ```

### Step 2 — Score with RACE Judge

#### Option A: Full pipeline (may fail on large articles)

```powershell
python -u deepresearch_bench_race.py $model `
    --skip_cleaning --only_en --limit 1 --max_workers 1 `
    --output_dir results/race-official --force
```

#### Option B: Direct scoring (works with truncated articles)

If you hit API limits, truncate articles and score directly:

```python
import os, sys, json

# Set env BEFORE importing api module
os.environ['LLM_BACKEND'] = 'openai'
os.environ['OPENAI_BASE_URL'] = 'https://api.groq.com/openai/v1'
os.environ['OPENAI_API_KEY'] = 'gsk_xxxxx'
os.environ['RACE_MODEL'] = 'meta-llama/llama-4-scout-17b-16e-instruct'
os.environ['MAX_OUTPUT_TOKENS'] = '8000'

sys.path.insert(0, r'C:\Users\Hp\Documents\Codex\deep_research_bench')
from utils.io_utils import load_jsonl
from utils.api import AIClient
from utils.score_calculator import calculate_weighted_scores
from utils.json_extractor import extract_json_from_markdown
from prompt.score_prompt_en import generate_merged_score_prompt
from deepresearch_bench_race import format_criteria_list

MAX_CHARS = 14000  # truncate to fit TPM limits

TASK_ID = 83
MODEL = "your-model-name"

task = [t for t in load_jsonl('data/prompt_data/query.jsonl') if t['id'] == TASK_ID][0]
prompt = task['prompt']
target = load_jsonl(f'data/test_data/cleaned_data/{MODEL}.jsonl')[0]
ref = [r for r in load_jsonl('data/test_data/cleaned_data/reference.jsonl') if r['prompt'] == prompt][0]
crit = [c for c in load_jsonl('data/criteria_data/criteria.jsonl') if c['prompt'] == prompt][0]
crit_str = format_criteria_list(crit)

target_article = target['article'][:MAX_CHARS]
ref_article = ref['article'][:MAX_CHARS]

user_prompt = generate_merged_score_prompt.format(
    task_prompt=prompt, article_1=target_article,
    article_2=ref_article, criteria_list=crit_str)
print(f'Prompt: {len(user_prompt)} chars')

client = AIClient()
resp = client.generate(user_prompt=user_prompt, system_prompt='')
json_str = extract_json_from_markdown(resp)
parsed = json.loads(json_str) if isinstance(json_str, str) else json_str
scores = calculate_weighted_scores(parsed, crit, 'en')

t_total = scores['target']['total']
r_total = scores['reference']['total']
overall = t_total / (t_total + r_total) if (t_total + r_total) > 0 else 0

dims = {}
for dim in ['comprehensiveness', 'insight', 'instruction_following', 'readability']:
    dk = f'{dim}_weighted_avg'
    ts = scores['target']['dims'].get(dk, 0)
    rs = scores['reference']['dims'].get(dk, 0)
    dims[dim] = ts / (ts + rs) if (ts + rs) > 0 else 0

print(json.dumps({
    'overall': round(overall, 4),
    'comprehensiveness': round(dims['comprehensiveness'], 4),
    'insight': round(dims['insight'], 4),
    'instruction_following': round(dims['instruction_following'], 4),
    'readability': round(dims['readability'], 4),
}, indent=2))
```

**RACE Overall Score** = target_total / (target_total + reference_total)

---

## Quick Reference — Directory Structure

```
deep_research_bench/
├── data/
│   ├── prompt_data/
│   │   └── query.jsonl              # 100 benchmark tasks (50 EN, 50 ZH)
│   ├── criteria_data/
│   │   └── criteria.jsonl           # per-task scoring criteria
│   └── test_data/
│       ├── raw_data/
│       │   └── your-model.jsonl     # YOUR submission goes here
│       └── cleaned_data/
│           ├── your-model.jsonl     # cleaned/compressed version
│           └── reference.jsonl      # official reference reports
├── utils/
│   ├── extract.py                   # FACT step 1
│   ├── deduplicate.py               # FACT step 2
│   ├── scrape.py                    # FACT step 3
│   ├── validate.py                  # FACT step 4
│   ├── stat.py                      # FACT step 5
│   ├── clean_article.py             # RACE step 1
│   ├── api.py                       # LLM client (OpenAI-compatible)
│   └── score_calculator.py          # RACE score computation
├── prompt/
│   ├── score_prompt_en.py           # RACE scoring prompt (English)
│   └── score_prompt_zh.py           # RACE scoring prompt (Chinese)
├── deepresearch_bench_race.py       # RACE main script
└── results/
    └── fact-official/
        └── your-model/
            ├── extracted.jsonl
            ├── deduplicated.jsonl
            ├── scraped.jsonl
            ├── validated.jsonl
            └── fact_result.txt      # final FACT score
```

---

## Score Interpretation

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **FACT valid_rate** | supported / (supported + unsupported) | % of checkable claims that are actually backed by the cited source |
| **FACT effective** | supported / (supported + unsupported + unknown) | Same but penalizes for unscrapable sources |
| **RACE overall** | target_score / (target_score + reference_score) | Relative quality vs the official reference report |
| RACE comprehensiveness | — | Coverage breadth and depth |
| RACE insight | — | Analytical depth and originality |
| RACE instruction following | — | Adherence to the task prompt |
| RACE readability | — | Structure, clarity, flow |

---

## Tips

1. **Use a 70B+ model for FACT extract** — smaller models (like llama-4-scout 17B) extract fewer claims, inflating FACT accuracy. Llama 3.3 70B extracts 130-200 claims vs 24 from scout.

2. **Cached source docs > Jina** — if your agent already scraped sources during the run, inject cached content instead of re-scraping via Jina. It's faster and avoids rate limits.

3. **Truncate for RACE on free APIs** — Groq free tier has 12K-30K TPM limits. Truncate both articles to ~14K chars each to fit.

4. **Set MAX_OUTPUT_TOKENS correctly** — Together default is 64000, Groq max is 32768. Always set explicitly.

5. **Model name matters** — Each provider has different model IDs. Check available models before assuming a name works.

import os
import sys
import json
sys.path.insert(0, ".")

from deep_research.settings import Settings
from deep_research.model_router import model_for_role
from deep_research.synthesis_refinement import _expand_report_depth_if_needed, _target_depth_hint
from deep_research.schemas import CoverageMatrix, BranchCoverage, EvidenceCard, ResearchBranch, ResearchPlan, SourceRecordV2

os.environ["TOGETHER_API_KEY"] = "tgp_v1_4ZrbdE0dFCK6RnuZxiTtog6wesCBrX3ApQLpUiN3Bxg"
settings = Settings.from_env(project_root=".")
model = model_for_role(settings, "orchestrator", "together:meta-llama/Llama-3.3-70B-Instruct-Turbo")

run_dir = r"runs\20260708T110054Z-from-2020-to-2050-how-many-elderly-people-will-there-be-in-japan"

# Load the inputs
plan_data = json.load(open(os.path.join(run_dir, "plan.json"), "r", encoding="utf-8"))
branches = [ResearchBranch(**b) for b in plan_data.get("branches", [])]
plan = ResearchPlan(
    question=plan_data["question"],
    intent=plan_data["intent"],
    audience=plan_data["audience"],
    report_outline=plan_data["report_outline"],
    branches=branches,
    source_requirements=plan_data.get("source_requirements"),
    acceptance_criteria=plan_data.get("acceptance_criteria"),
    writer_persona=plan_data.get("writer_persona")
)

cov_data = json.load(open(os.path.join(run_dir, "coverage.json"), "r", encoding="utf-8"))
cov_branches = [BranchCoverage(**b) for b in cov_data.get("branches", [])]
coverage = CoverageMatrix(
    branches=cov_branches,
    complete=cov_data["complete"],
    coverage_score=cov_data["coverage_score"],
    missing_branches=cov_data["missing_branches"]
)

sources = []
for line in open(os.path.join(run_dir, "sources.jsonl"), "r", encoding="utf-8"):
    d = json.loads(line)
    sources.append(SourceRecordV2(**d))

evidence_cards = []
for line in open(os.path.join(run_dir, "evidence_cards.jsonl"), "r", encoding="utf-8"):
    d = json.loads(line)
    evidence_cards.append(EvidenceCard(**d))

report = open(os.path.join(run_dir, "draft_report_1.md"), "r", encoding="utf-8").read()

# Let's inspect the target profile
from deep_research.synthesis_formatting import _target_report_profile
profile = _target_report_profile(plan=plan, evidence_cards=evidence_cards, writing_guidance="")
print("Target word count:", profile["target_words"])
print("Minimum word count:", profile["minimum_words"])

# Let's print out what the prompt is and run one step of expansion
from deep_research.synthesis_refinement import _depth_expansion_prompt, _clean_depth_expansion_markdown, _is_degenerate_expansion
prompt = _depth_expansion_prompt(
    plan=plan,
    evidence_cards=evidence_cards,
    coverage=coverage,
    sources=sources,
    current_report=report,
    target_profile=profile,
    verification_failures=[],
    writing_guidance="",
    round_index=0
)

print("Running single expansion call...")
res = model.invoke(prompt)
content = str(res.content)
print("Response length:", len(content))
print("First 500 chars of response:\n", content[:500])

cleaned = _clean_depth_expansion_markdown(content, sources)
print("Cleaned expansion length:", len(cleaned))
print("Is degenerate expansion:", _is_degenerate_expansion(cleaned))

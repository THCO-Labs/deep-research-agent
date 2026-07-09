import sys
sys.path.insert(0, ".")
from deep_research import source_validation as sv
from deep_research.schemas import ResearchBranch

branch = ResearchBranch(
    id="branch_1",
    title="Need for closure and misinformation acceptance",
    objective="Explain the relationship between need for closure and misinformation acceptance.",
    queries=["NFC and misinformation acceptance studies"],
    required_terms=["need for closure", "misinformation acceptance"],
)
content = (
    "Need for cognition is a motivation to engage in effortful thinking. "
    "Need for cognition appears in studies about false memories, cognitive effort, and misinformation. "
    "Some authors briefly mention need for closure as a related but different construct. "
) * 6

title = "Need for cognition and misinformation acceptance"
normalized = sv._normalize(content)
title_core_terms = sv._title_core_terms(title)
protected_phrases = sv._protected_concept_phrases(branch, "What is the role of need for closure on misinformation acceptance?")
protected_phrase_sets = {frozenset(phrase) for phrase in protected_phrases}
source_text = f"{title}\n{normalized}"

print("title_core_terms:", title_core_terms)
print("protected_phrases:", protected_phrases)
for phrase_terms in protected_phrases:
    competes = sv._title_competes_with_phrase(title_core_terms, phrase_terms)
    target = sv._phrase_count(phrase_terms, source_text)
    competing = sv._competing_title_phrase_count(
        title_core_terms, phrase_terms, source_text, protected_phrase_sets=protected_phrase_sets
    )
    print(f"\nPhrase {phrase_terms}:")
    print(f"  competes={competes}, target_count={target}, competing_count={competing}")
    print(f"  condition: {competes and (target == 0 or competing >= max(4, target * 2.5))}")

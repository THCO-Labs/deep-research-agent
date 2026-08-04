from __future__ import annotations

from dataclasses import asdict, dataclass, field

from deep_research.core.schemas import ResearchPlan
from deep_research.core.settings import Settings


@dataclass(frozen=True)
class RaceSelfJudgeResult:
    comprehensiveness_score: float
    insight_score: float
    instruction_following_score: float
    readability_score: float
    overall_score: float
    weaknesses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def race_self_judge(
    *,
    report_markdown: str,
    plan: ResearchPlan,
    settings: Settings,
    writing_guidance: str = "",
) -> RaceSelfJudgeResult:
    """Return a deterministic advisory RACE-shaped self-judge result."""
    body = report_markdown.strip()
    word_count = len(body.split())
    has_sources = "## Sources" in body
    directly_answers = bool(body) and plan.question.strip()[:20].lower() not in body[:300].lower()
    comprehensiveness = _score_range(word_count, low=1200, high=3500)
    readability = 0.75 if word_count else 0.0
    instruction = 0.8 if has_sources and directly_answers else 0.55 if body else 0.0
    insight = min(0.8, 0.45 + (body.count("## ") * 0.04)) if body else 0.0
    overall = round((comprehensiveness + insight + instruction + readability) / 4, 4)
    weaknesses: list[str] = []
    if not has_sources:
        weaknesses.append("missing Sources section")
    if word_count < 1200:
        weaknesses.append("report may be too short for RACE depth")
    if writing_guidance and writing_guidance[:80] not in body:
        weaknesses.append("task-specific guidance coverage not semantically checked")
    return RaceSelfJudgeResult(
        comprehensiveness_score=round(comprehensiveness, 4),
        insight_score=round(insight, 4),
        instruction_following_score=round(instruction, 4),
        readability_score=round(readability, 4),
        overall_score=overall,
        weaknesses=weaknesses,
    )


def _score_range(value: int, *, low: int, high: int) -> float:
    if value <= 0:
        return 0.0
    if value <= low:
        return max(0.25, value / max(low, 1) * 0.6)
    if value >= high:
        return 0.9
    return 0.6 + ((value - low) / (high - low) * 0.3)

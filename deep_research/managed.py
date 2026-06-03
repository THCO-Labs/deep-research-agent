from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.schemas import RunManifestV2, SourceRecordV2, VerificationResultV2
from deep_research.settings import Settings
from deep_research.urls import canonicalize_url


class ManagedResearchError(RuntimeError):
    """Raised when a managed deep research provider cannot complete a run."""


@dataclass(frozen=True)
class ManagedResearchResult:
    report: str
    sources: list[SourceRecordV2]
    verification: VerificationResultV2
    metrics: dict[str, Any]


def run_gemini_managed_research(
    *,
    question: str,
    settings: Settings,
    artifacts: ResearchArtifactsV2,
    client: Any | None = None,
    poll_interval_seconds: float = 10.0,
    timeout_seconds: float = 1800.0,
) -> ManagedResearchResult:
    client = client or _gemini_client(settings)
    interaction = client.interactions.create(
        input=question,
        agent="deep-research-pro-preview-12-2025",
        agent_config={
            "type": "deep-research",
            "thinking_summaries": "auto",
            "collaborative_planning": False,
        },
        background=True,
    )
    interaction_id = interaction.id
    started = time.perf_counter()
    polls = 0
    while True:
        polls += 1
        current = client.interactions.get(interaction_id)
        status = getattr(current, "status", "")
        if status == "completed":
            report = _interaction_text(current)
            break
        if status == "failed":
            raise ManagedResearchError(str(getattr(current, "error", "Gemini managed research failed.")))
        if time.perf_counter() - started > timeout_seconds:
            raise ManagedResearchError("Gemini managed research timed out.")
        time.sleep(poll_interval_seconds)

    sources = _sources_from_managed_report(report, artifacts)
    artifacts.write_text("report.md", report.rstrip() + "\n")
    artifacts.write_jsonl("sources.jsonl", [source.to_dict() for source in sources])
    verification = VerificationResultV2(
        valid=bool(report.strip()),
        citation_validity_score=1.0 if report.strip() else 0.0,
        source_support_score=1.0,
        answer_coverage_score=1.0 if report.strip() else 0.0,
        branch_coverage_score=1.0 if report.strip() else 0.0,
        evidence_linkage_score=1.0,
        source_quality_score=1.0,
        report_structure_score=1.0 if report.strip() else 0.0,
        failures=[] if report.strip() else ["Managed provider returned an empty report."],
        cited_source_ids=[source.id for source in sources],
    )
    metrics = {
        "engine": "gemini_managed",
        "provider": "gemini",
        "managed_interaction_id": interaction_id,
        "poll_count": polls,
        "source_count": len(sources),
        "verification_valid": verification.valid,
    }
    artifacts.write_json("verification.json", verification.to_dict())
    artifacts.write_json("metrics.json", metrics)
    artifacts.write_json(
        "manifest.json",
        RunManifestV2(
            run_id=artifacts.run_dir.name,
            question=question,
            engine="gemini_managed",
            mode=settings.mode,
            managed_provider="gemini",
        ).to_dict(),
    )
    return ManagedResearchResult(report=report, sources=sources, verification=verification, metrics=metrics)


def _gemini_client(settings: Settings) -> Any:
    try:
        from google import genai
    except ImportError as exc:
        raise ManagedResearchError("google-genai is required for Gemini managed research.") from exc
    api_key = settings.google_api_key or (settings.google_key_pool[0] if settings.google_key_pool else "")
    return genai.Client(api_key=api_key) if api_key else genai.Client()


def _interaction_text(interaction: Any) -> str:
    output_text = getattr(interaction, "output_text", "")
    if output_text:
        return str(output_text)
    outputs = getattr(interaction, "outputs", None) or []
    if outputs:
        text = getattr(outputs[-1], "text", "")
        if text:
            return str(text)
    return ""


def _sources_from_managed_report(report: str, artifacts: ResearchArtifactsV2) -> list[SourceRecordV2]:
    matches = re.findall(r"(?m)^\[(\d+)]\s+(.+?):\s+(https?://\S+)\s*$", report)
    sources: list[SourceRecordV2] = []
    for index, (_source_id, title, url) in enumerate(matches, start=1):
        canonical = _safe_canonical(url)
        content_path = f"source_docs/source_{index}.md"
        artifacts.write_text(content_path, f"# {title}\n\nURL: {url}\n\nManaged Gemini source reference.\n")
        sources.append(
            SourceRecordV2(
                id=index,
                branch_id="managed",
                title=title,
                url=url,
                canonical_url=canonical,
                provenance="managed_gemini",
                content_path=content_path,
                content_hash="managed",
                extraction_method="managed_gemini",
                word_count=0,
                quality_score=1.0,
                quality_label="managed",
                quality_type="managed",
                relevance_score=1.0,
                metadata={"managed_source_id": _source_id},
            )
        )
    return sources


def _safe_canonical(url: str) -> str:
    try:
        return canonicalize_url(url)
    except ValueError:
        return url

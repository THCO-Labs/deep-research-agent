import json
from pathlib import Path

import pytest

from deep_research import agent as agent_module
from deep_research.acquisition import AcquisitionMetrics, AcquisitionResult
from deep_research.agent import ResearchRunError, resume_research, run_research, verify_research_run
from deep_research.research_graph import _resume_entry_point, load_latest_checkpoint
from deep_research.schemas import ResearchBranch, SourceCandidate, SourceRecordV2
from deep_research.settings import Settings


def test_run_research_writes_v2_artifacts_and_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("deep_research.research_graph.acquire_sources", fake_acquire_sources)
    settings = _settings(tmp_path)

    result = run_research("How do urban heat islands affect public health?", settings, progress_mode="quiet")

    expected = {
        "request.json",
        "plan.json",
        "sources.jsonl",
        "evidence_cards.jsonl",
        "coverage.json",
        "report_blueprint.json",
        "claim_ledger.json",
        "sentence_plan.json",
        "report.md",
        "verification.json",
        "metrics.json",
        "manifest.json",
    }
    assert expected <= {path.name for path in result.run_dir.iterdir()}
    manifest_text = (result.run_dir / "manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    verification = json.loads(result.verification_path.read_text(encoding="utf-8"))
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["engine"] == "local_langgraph"
    assert manifest["settings"]["research_engine"] == "local_langgraph"
    assert "google-secret" not in manifest_text
    assert verification["valid"] is True
    assert metrics["source_count"] >= 17
    assert metrics["raw_evidence_card_count"] >= 17
    assert metrics["evidence_card_count"] >= 7


def test_verify_research_run_rechecks_existing_v2_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("deep_research.research_graph.acquire_sources", fake_acquire_sources)
    settings = _settings(tmp_path)
    result = run_research("How do urban heat islands affect public health?", settings, progress_mode="quiet")
    run_id = result.run_dir.name

    verified = verify_research_run(run_id, settings)

    verification = json.loads(verified.verification_path.read_text(encoding="utf-8"))
    assert verification["valid"] is True


def test_resume_reuses_checkpointed_sources_without_duplicate_fetches(tmp_path: Path, monkeypatch) -> None:
    calls = {"fresh": 0, "resume": 0}

    def tracked_acquire(**kwargs):
        if kwargs.get("existing_sources"):
            calls["resume"] += 1
        else:
            calls["fresh"] += 1
        return fake_acquire_sources(**kwargs)

    monkeypatch.setattr("deep_research.research_graph.acquire_sources", tracked_acquire)
    settings = _settings(tmp_path)
    result = run_research("How do urban heat islands affect public health?", settings, progress_mode="quiet")

    resume_settings = _settings(tmp_path, max_rounds=12, max_search_queries=22)
    calls["resume"] = 0
    resumed = resume_research(result.run_dir.name, resume_settings, progress_mode="quiet")

    sources = (resumed.run_dir / "sources.jsonl").read_text(encoding="utf-8").splitlines()
    metrics = json.loads(resumed.metrics_path.read_text(encoding="utf-8"))
    assert calls["fresh"] == 1
    assert calls["resume"] == 0
    assert len(sources) == 17
    assert metrics["max_rounds"] == 12
    assert metrics["max_search_queries"] == 22


def test_resume_from_coverage_checkpoint_reloads_source_texts_for_verification(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("deep_research.research_graph.acquire_sources", fake_acquire_sources)
    settings = _settings(tmp_path)
    result = run_research("How do urban heat islands affect public health?", settings, progress_mode="quiet")
    checkpoint_path = result.run_dir / "checkpoints" / "check_coverage.json"
    latest_path = result.run_dir / "checkpoints" / "latest.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["checkpoint_phase"] = "check_coverage"
    checkpoint.pop("draft_report", None)
    checkpoint.pop("verification", None)
    latest_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    resumed = resume_research(result.run_dir.name, settings, progress_mode="quiet")

    verification = json.loads(resumed.verification_path.read_text(encoding="utf-8"))
    assert verification["valid"] is True
    assert verification["source_support_score"] >= 0.35


def test_resume_from_read_sources_checkpoint_reloads_source_texts_for_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("deep_research.research_graph.acquire_sources", fake_acquire_sources)
    settings = _settings(tmp_path)
    result = run_research("How do urban heat islands affect public health?", settings, progress_mode="quiet")
    checkpoint_path = result.run_dir / "checkpoints" / "read_sources.json"
    latest_path = result.run_dir / "checkpoints" / "latest.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["checkpoint_phase"] = "read_sources"
    checkpoint.pop("evidence_cards", None)
    checkpoint.pop("coverage_matrix", None)
    checkpoint.pop("draft_report", None)
    checkpoint.pop("verification", None)
    latest_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    resumed = resume_research(result.run_dir.name, settings, progress_mode="quiet")

    metrics = json.loads(resumed.metrics_path.read_text(encoding="utf-8"))
    evidence_lines = (resumed.run_dir / "evidence_cards.jsonl").read_text(encoding="utf-8").splitlines()
    assert metrics["raw_evidence_card_count"] >= 17
    assert len(evidence_lines) >= 7


def test_resume_entry_point_continues_after_checkpoint_phase(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.json").write_text(json.dumps({"request": {"question": "q"}}), encoding="utf-8")
    hygiene_checkpoint = checkpoint_dir / "evidence_hygiene.json"
    hygiene_checkpoint.write_text(json.dumps({"request": {"question": "q"}}), encoding="utf-8")

    state = load_latest_checkpoint(run_dir)

    assert state["checkpoint_phase"] == "evidence_hygiene"
    assert _resume_entry_point(state) == "semantic_enrichment"


def test_resume_entry_point_recovers_from_corrupt_latest_and_rich_replay_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "latest.json").write_text("", encoding="utf-8")
    (checkpoint_dir / "classify_request.json").write_text(
        json.dumps(
            {
                "request": {"question": "q"},
                "source_records": [{"source_id": 1}],
                "evidence_cards": [{"id": 1}],
            }
        ),
        encoding="utf-8",
    )

    state = load_latest_checkpoint(run_dir)

    assert state["checkpoint_phase"] == "classify_request"
    assert _resume_entry_point(state) == "semantic_enrichment"


def test_resume_research_writes_structured_failure_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("deep_research.research_graph.acquire_sources", fake_acquire_sources)
    settings = _settings(tmp_path)
    result = run_research("How do urban heat islands affect public health?", settings, progress_mode="quiet")

    def fail_graph(**_kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 31.25s.")

    monkeypatch.setattr(agent_module, "run_local_research_graph", fail_graph)

    with pytest.raises(ResearchRunError) as raised:
        resume_research(result.run_dir.name, settings, progress_mode="quiet")

    failure = json.loads((raised.value.result.run_dir / "failure.json").read_text(encoding="utf-8"))
    metrics = json.loads(raised.value.result.metrics_path.read_text(encoding="utf-8"))
    assert failure["category"] == "quota_or_rate_limit"
    assert failure["retry_after_seconds"] == 31
    assert metrics["error_category"] == "quota_or_rate_limit"


def test_run_research_writes_structured_failure_artifacts(tmp_path: Path, monkeypatch) -> None:
    def fail_graph(**_kwargs):
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 42.156s.")

    monkeypatch.setattr(agent_module, "run_local_research_graph", fail_graph)
    settings = _settings(tmp_path)

    with pytest.raises(ResearchRunError) as raised:
        run_research("quota failure", settings, progress_mode="quiet")

    failure = json.loads((raised.value.result.run_dir / "failure.json").read_text(encoding="utf-8"))
    metrics = json.loads(raised.value.result.metrics_path.read_text(encoding="utf-8"))
    assert failure["category"] == "quota_or_rate_limit"
    assert failure["retry_after_seconds"] == 42
    assert metrics["error_category"] == "quota_or_rate_limit"


def test_run_research_treats_failed_verification_as_failed_run(tmp_path: Path, monkeypatch) -> None:
    def invalid_graph(**kwargs):
        artifacts = kwargs["artifacts"]
        artifacts.write_text("report.md", "# Bad Report\n\nNo cited evidence.\n")
        artifacts.write_json("verification.json", {"schema_version": 2, "valid": False, "failures": ["Report does not cite any sources."]})
        return {
            "metrics": {"engine": "local_langgraph", "verification_valid": False},
            "verification": {"valid": False, "failures": ["Report does not cite any sources."]},
        }

    monkeypatch.setattr(agent_module, "run_local_research_graph", invalid_graph)
    settings = _settings(tmp_path)

    with pytest.raises(ResearchRunError) as raised:
        run_research("invalid verification", settings, progress_mode="quiet")

    failure = json.loads((raised.value.result.run_dir / "failure.json").read_text(encoding="utf-8"))
    report = (raised.value.result.run_dir / "report.md").read_text(encoding="utf-8")
    failed_report = (raised.value.result.run_dir / "failed_report.md").read_text(encoding="utf-8")
    assert failure["category"] == "verification_failed"
    assert "Research Run Failed Verification" in report
    assert "## Run Summary" in report
    assert "No cited evidence" in failed_report


def test_failed_run_archives_best_draft_instead_of_latest_regression(tmp_path: Path, monkeypatch) -> None:
    def invalid_graph(**kwargs):
        artifacts = kwargs["artifacts"]
        artifacts.write_text("draft_report.md", "# Latest Draft\n\nRegressed.\n")
        artifacts.write_text("best_draft.md", "# Best Draft\n\nLower issue count.\n")
        artifacts.write_json("verification.json", {"schema_version": 2, "valid": False, "failures": ["latest regressed"]})
        artifacts.write_json(
            "run_health.json",
            {
                "schema_version": 1,
                "status": "failed_verification",
                "best_draft_path": "best_draft.md",
                "best_draft_index": 1,
                "best_draft_failure_count": 2,
            },
        )
        return {
            "metrics": {
                "engine": "local_langgraph",
                "verification_valid": False,
                "verification_rounds": 2,
                "best_draft_index": 1,
                "best_draft_failure_count": 2,
            },
            "verification": {"valid": False, "failures": ["latest regressed"]},
        }

    monkeypatch.setattr(agent_module, "run_local_research_graph", invalid_graph)
    settings = _settings(tmp_path)

    with pytest.raises(ResearchRunError) as raised:
        run_research("invalid verification", settings, progress_mode="quiet")

    failed_report = (raised.value.result.run_dir / "failed_report.md").read_text(encoding="utf-8")
    draft_report = (raised.value.result.run_dir / "draft_report.md").read_text(encoding="utf-8")
    health = json.loads((raised.value.result.run_dir / "run_health.json").read_text(encoding="utf-8"))
    assert "Best Draft" in failed_report
    assert "Latest Draft" in draft_report
    assert health["best_draft_index"] == 1
    assert health["failed_report_path"] == "failed_report.md"


def test_run_research_stops_without_draft_when_acquisition_returns_no_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def empty_acquire_sources(**_kwargs):
        return AcquisitionResult(
            candidates=[],
            sources=[],
            source_texts={},
            metrics=AcquisitionMetrics(
                search_count=2,
                failures=["Search failed for 'query': This request exceeds your plan's set usage limit."],
            ),
        )

    monkeypatch.setattr("deep_research.research_graph.acquire_sources", empty_acquire_sources)
    settings = _settings(tmp_path)

    with pytest.raises(ResearchRunError) as raised:
        run_research("Why does the body of a report drift away from the topic?", settings, progress_mode="quiet")

    report = (raised.value.result.run_dir / "report.md").read_text(encoding="utf-8")
    verification = json.loads(raised.value.result.verification_path.read_text(encoding="utf-8"))
    assert "Research Run Failed Verification" in report
    assert "No evidence cards were retrieved" in report
    assert "No evidence cards were retrieved" in verification["failures"][0]
    assert not (raised.value.result.run_dir / "draft_report.md").exists()


def test_gemini_managed_mode_writes_v2_artifacts(tmp_path: Path, monkeypatch) -> None:
    def fake_managed(question, settings, artifacts):
        artifacts.write_text(
            "report.md",
            "# Managed\n\nGemini managed report. [1]\n\n## Sources\n\n[1] Gemini Source: https://example.com/source\n",
        )
        artifacts.write_jsonl("sources.jsonl", [])
        artifacts.write_json("verification.json", {"schema_version": 2, "valid": True})
        artifacts.write_json("metrics.json", {"engine": "gemini_managed", "verification_valid": True})

    monkeypatch.setattr(agent_module, "run_gemini_managed_research", fake_managed)
    settings = _settings(tmp_path, research_engine="gemini_managed", tavily_api_key="")

    result = run_research("managed question", settings, progress_mode="quiet")

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    verification = json.loads(result.verification_path.read_text(encoding="utf-8"))
    assert manifest["engine"] == "gemini_managed"
    assert manifest["managed_provider"] == "gemini"
    assert verification["valid"] is True


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "project_root": tmp_path,
        "out_dir": tmp_path,
        "provider": "google",
        "google_api_key": "google-secret",
        "google_api_keys": ("google-secret",),
        "tavily_api_key": "tavily-secret",
        "strict_tool_models": True,
        "min_source_words": 40,
        "min_usable_sources": 17,
        "max_search_queries": 16,
        "max_candidates": 80,
        "llm_planning": False,
        "llm_synthesis": False,
        "semantic_verification": False,
    }
    values.update(overrides)
    return Settings(**values)


def fake_acquire_sources(
    *,
    branches: list[ResearchBranch],
    artifacts,
    existing_candidates=None,
    existing_sources=None,
    existing_source_texts=None,
    **_kwargs,
) -> AcquisitionResult:
    if existing_sources:
        return AcquisitionResult(
            candidates=list(existing_candidates or []),
            sources=list(existing_sources),
            source_texts=dict(existing_source_texts or {}),
            metrics=AcquisitionMetrics(),
        )

    candidates: list[SourceCandidate] = []
    sources: list[SourceRecordV2] = []
    source_texts: dict[int, str] = {}
    next_id = 1
    for branch in branches:
        for _index in range(branch.min_sources):
            title = f"{branch.title} Evidence {next_id}"
            url = f"https://example.com/{branch.id}/{next_id}"
            text = _branch_text(branch, next_id)
            path = f"source_docs/source_{next_id}.md"
            artifacts.write_text(path, text)
            candidates.append(
                SourceCandidate(
                    id=next_id,
                    branch_id=branch.id,
                    title=title,
                    url=url,
                    query=branch.queries[0],
                    snippet=text[:120],
                    search_score=0.9,
                )
            )
            sources.append(
                SourceRecordV2(
                    id=next_id,
                    branch_id=branch.id,
                    title=title,
                    url=url,
                    canonical_url=url,
                    provenance="web",
                    content_path=path,
                    content_hash=f"hash-{next_id}",
                    extraction_method="test",
                    word_count=len(text.split()),
                    quality_score=0.9,
                    quality_label="high",
                    quality_type="official_docs",
                    relevance_score=1.0,
                )
            )
            source_texts[next_id] = text
            next_id += 1
    return AcquisitionResult(
        candidates=candidates,
        sources=sources,
        source_texts=source_texts,
        metrics=AcquisitionMetrics(
            search_count=len(branches) * 2,
            candidate_count=len(candidates),
            scrape_count=len(candidates),
            usable_source_count=len(sources),
        ),
    )


def _branch_text(branch: ResearchBranch, source_id: int) -> str:
    required = ", ".join(branch.required_terms)
    source_context = f"neighborhood area{source_id}"
    return (
        f"{branch.title} evidence explains {required} for urban heat islands and public health in {source_context}. "
        f"Urban heat islands affect public health through {required}, heat exposure, vulnerable populations, "
        f"and higher local temperatures in dense built environments in {source_context}. "
        f"Researchers connect {required} with public health planning, cooling strategies, tree canopy, reflective surfaces, "
        f"emergency alerts, adaptation, and mitigation decisions in {source_context}. "
        f"The evidence on {branch.title} describes why urban heat islands matter for health outcomes, exposure, "
        f"risk reduction, implementation constraints, and local decision-making in {source_context}."
    )

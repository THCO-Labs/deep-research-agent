from pathlib import Path

import pytest

from deep_research.artifacts import PathSafetyError, RunArtifacts
from deep_research.source_registry import SourceRegistry


def test_run_artifacts_rejects_path_escape(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "path safety")

    with pytest.raises(PathSafetyError):
        artifacts.write_text("../escape.txt", "bad")


def test_run_artifacts_treats_leading_slash_as_run_root(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "virtual root")

    path = artifacts.write_text("/research_plan.md", "plan")

    assert path == artifacts.run_dir / "research_plan.md"
    assert path.read_text(encoding="utf-8") == "plan"


def test_source_registry_dedupes_by_canonical_url_and_content(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "registry")
    registry = SourceRegistry(artifacts)

    first = registry.upsert_search_result(
        url="https://example.com/page?utm_source=x",
        title="Example",
        query="example",
    )
    second = registry.upsert_search_result(
        url="https://example.com/page",
        title="Example Duplicate",
        query="example",
    )
    scraped = registry.record_scrape(
        url="https://mirror.example/page",
        title="Mirror",
        markdown="same content",
        extraction_method="playwright",
    )
    scraped_duplicate = registry.record_scrape(
        url="https://another.example/page",
        title="Another",
        markdown="same content",
        extraction_method="playwright",
    )

    assert first.id == second.id
    assert scraped.id == scraped_duplicate.id
    assert len(registry.records) == 2


def test_source_registry_persists_quality_metadata(tmp_path: Path) -> None:
    artifacts = RunArtifacts.create(tmp_path, "quality registry")
    registry = SourceRegistry(artifacts)

    record = registry.upsert_search_result(
        url="https://docs.example.com/guide",
        title="Example documentation",
        query="docs",
        snippet="Official documentation",
        search_score=0.5,
    )
    scraped = registry.record_scrape(
        url=record.url,
        title=record.title,
        markdown="Official documentation includes detailed implementation guidance. " * 20,
        extraction_method="playwright",
    )

    source_doc = (artifacts.run_dir / scraped.content_path).read_text(encoding="utf-8")
    assert scraped.source_quality_label in {"strong", "excellent"}
    assert scraped.source_relevance_score > 0
    assert "Source quality:" in source_doc
    assert "Source relevance:" in source_doc
    assert "source_quality_score" in (artifacts.run_dir / "sources.jsonl").read_text(encoding="utf-8")
    assert "source_relevance_score" in (artifacts.run_dir / "sources.jsonl").read_text(encoding="utf-8")

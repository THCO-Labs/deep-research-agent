import json
from pathlib import Path

from deep_research.agent import ResearchRunError, ResearchRunResult
from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.deepresearch_bench import (
    audit_benchmark_planning,
    evaluate_raw_submission_fact_proxy,
    evaluate_raw_submission_proxy,
    generate_raw_submission,
    load_benchmark_tasks,
)
from deep_research.guidance import CRITERIA_BLOCK_END, CRITERIA_BLOCK_START, criteria_acceptance_lines, extract_structured_criteria
from deep_research.settings import Settings


def test_load_benchmark_tasks_filters_language_and_limit(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)

    tasks = load_benchmark_tasks(bench, language="en", limit=1)

    assert len(tasks) == 1
    assert tasks[0].id == 2
    assert tasks[0].language == "en"
    assert "English task" in tasks[0].prompt


def test_generate_raw_submission_writes_deepresearch_bench_format(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")

    output = generate_raw_submission(
        benchmark_dir=bench,
        model_name="local-test",
        settings=settings,
        language="en",
        runner=_fake_runner,
    )

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert output.name == "local-test.jsonl"
    assert rows == [
        {
            "article": "# Report\n\nAnswer for English task. [1]\n",
            "id": 2,
            "prompt": "English task",
        }
    ]


def test_generate_raw_submission_resumes_without_duplicate_rows(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")

    first = generate_raw_submission(
        benchmark_dir=bench,
        model_name="local-test",
        settings=settings,
        language="en",
        runner=_fake_runner,
    )
    second = generate_raw_submission(
        benchmark_dir=bench,
        model_name="local-test",
        settings=settings,
        language="en",
        runner=_fake_runner,
    )

    assert first == second
    assert len(second.read_text(encoding="utf-8").splitlines()) == 1


def test_generate_raw_submission_records_internal_failure(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    settings = Settings(project_root=tmp_path, out_dir=tmp_path / "runs")

    output = generate_raw_submission(
        benchmark_dir=bench,
        model_name="local-failure",
        settings=settings,
        ids=[2],
        runner=_failing_runner,
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["id"] == 2
    assert "Benchmark Run Failure" in row["article"]
    assert "verification_failed" in row["article"]


def test_generate_raw_submission_can_enrich_agent_prompt_with_criteria(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    criteria_dir = bench / "data" / "criteria_data"
    criteria_dir.mkdir(parents=True)
    (criteria_dir / "criteria.jsonl").write_text(
        json.dumps(
            {
                "id": 2,
                "prompt": "English task",
                "dimension_weight": {"comprehensiveness": 0.5},
                "criterions": {
                    "comprehensiveness": [
                        {
                            "criterion": "Cover the core mechanism",
                            "explanation": "Explain causes, evidence, and implications.",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    seen_questions: list[str] = []
    seen_guidance: list[str] = []

    def runner(question: str, settings: Settings, *, writing_guidance: str = "") -> ResearchRunResult:
        seen_questions.append(question)
        seen_guidance.append(writing_guidance)
        return _fake_runner("English task", settings)

    output = generate_raw_submission(
        benchmark_dir=bench,
        model_name="criteria-test",
        settings=Settings(project_root=tmp_path, out_dir=tmp_path / "runs"),
        ids=[2],
        runner=runner,
    )

    row = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert row["prompt"] == "English task"
    assert seen_questions[0] == "English task"
    assert "DeepResearch Bench evaluation guidance" in seen_guidance[0]
    assert "Cover the core mechanism" in seen_guidance[0]
    assert "Convert each criterion into visible analysis" in seen_guidance[0]
    assert "do not paste the criteria as headings or a checklist" in seen_guidance[0]
    assert CRITERIA_BLOCK_START in seen_guidance[0]
    assert CRITERIA_BLOCK_END in seen_guidance[0]
    structured = extract_structured_criteria(seen_guidance[0])
    assert structured[0]["dimension"] == "comprehensiveness"
    assert structured[0]["criterion"] == "Cover the core mechanism"
    acceptance = criteria_acceptance_lines(seen_guidance[0])
    assert acceptance == [
        "Task-specific comprehensiveness criterion: Cover the core mechanism - Explain causes, evidence, and implications."
    ]


def test_planning_audit_checks_criteria_and_source_floor(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    criteria_dir = bench / "data" / "criteria_data"
    criteria_dir.mkdir(parents=True)
    (criteria_dir / "criteria.jsonl").write_text(
        json.dumps(
            {
                "id": 2,
                "prompt": "English task",
                "dimension_weight": {"comprehensiveness": 1.0},
                "criterions": {
                    "comprehensiveness": [
                        {
                            "criterion": "English task evidence coverage",
                            "explanation": "Cover direct evidence, limitations, and implications for the English task.",
                            "weight": 1.0,
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paths = audit_benchmark_planning(
        benchmark_dir=bench,
        model_name="planning-audit",
        settings=Settings(
            project_root=tmp_path,
            out_dir=tmp_path / "runs",
            llm_planning=False,
            llm_synthesis=False,
            semantic_verification=False,
        ),
        ids=[2],
    )

    row = json.loads(paths.raw_results_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(paths.summary_path.read_text(encoding="utf-8"))
    assert row["id"] == 2
    assert row["min_source_total"] >= 17
    assert row["criteria_acceptance_coverage"] == 1.0
    assert row["criteria_branch_coverage"] >= 0.55
    assert row["passed"]
    assert summary["count"] == 1
    assert summary["passed_count"] == 1
    assert paths.result_path.read_text(encoding="utf-8").startswith("Planning Audit")


def test_proxy_evaluation_scores_existing_raw_submission_against_reference(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    _write_proxy_inputs(bench)

    paths = evaluate_raw_submission_proxy(
        benchmark_dir=bench,
        model_name="local-proxy",
        settings=Settings(project_root=tmp_path, out_dir=tmp_path / "runs"),
        ids=[2],
        use_llm=False,
    )

    row = json.loads(paths.raw_results_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(paths.summary_path.read_text(encoding="utf-8"))
    assert row["id"] == 2
    assert row["method"] == "deterministic"
    assert 0.0 < row["overall_score"] <= 1.0
    assert row["race_relative_score"] == row["overall_score"]
    assert 0.0 < row["candidate_absolute_score"] <= 1.0
    assert 0.0 < row["reference_absolute_score"] <= 1.0
    assert row["benchmark_scoring_note"]
    assert "comprehensiveness" in row["dimension_scores"]
    assert summary["successful_count"] == 1
    assert "candidate_absolute_score" in summary
    assert paths.result_path.read_text(encoding="utf-8").startswith("Count: 1")


def test_proxy_evaluation_accepts_custom_judge_payload(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    _write_proxy_inputs(bench)

    def judge(payload: dict) -> dict:
        criteria = payload["criteria"]["criterions"]
        return {
            "dimensions": {
                dimension: [
                    {
                        "criterion": row["criterion"],
                        "candidate_score": 8,
                        "reference_score": 10,
                    }
                    for row in rows
                ]
                for dimension, rows in criteria.items()
            }
        }

    paths = evaluate_raw_submission_proxy(
        benchmark_dir=bench,
        model_name="local-proxy",
        settings=Settings(project_root=tmp_path, out_dir=tmp_path / "runs"),
        ids=[2],
        judge=judge,
    )

    row = json.loads(paths.raw_results_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["method"] == "custom_proxy_judge"
    assert row["overall_score"] == 0.4444
    assert row["candidate_absolute_score"] == 0.8
    assert row["reference_absolute_score"] == 1.0


def test_proxy_evaluation_scores_failed_run_notice_as_zero(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    _write_proxy_inputs(bench)
    raw_dir = bench / "data" / "test_data" / "raw_data"
    raw = {
        "id": 2,
        "prompt": "English task",
        "article": (
            "# Research Run Failed Verification\n\n"
            "This run did not produce an accepted final report.\n\n"
            "## Benchmark Run Failure\n\n"
            "Failure category: verification_failed.\n"
        ),
    }
    (raw_dir / "local-failed-proxy.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")

    paths = evaluate_raw_submission_proxy(
        benchmark_dir=bench,
        model_name="local-failed-proxy",
        settings=Settings(project_root=tmp_path, out_dir=tmp_path / "runs"),
        ids=[2],
        use_llm=False,
    )

    row = json.loads(paths.raw_results_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(paths.summary_path.read_text(encoding="utf-8"))
    assert row["overall_score"] == 0.0
    assert row["candidate_absolute_score"] == 0.0
    assert summary["low_score_ids"] == [2]


def test_fact_proxy_scores_citations_against_run_source_text(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    _write_fact_proxy_inputs(bench, supported=True)
    run_dir = _write_fact_run_dir(tmp_path, source_text="Need for closure increases misinformation acceptance through quick certainty seeking.")

    paths = evaluate_raw_submission_fact_proxy(
        benchmark_dir=bench,
        model_name="local-fact",
        ids=[2],
        run_dir=run_dir,
    )

    row = json.loads(paths.raw_results_path.read_text(encoding="utf-8").splitlines()[0])
    summary = json.loads(paths.summary_path.read_text(encoding="utf-8"))
    assert row["method"] == "deterministic_fact_proxy"
    assert row["supported_citation_count"] == 1
    assert row["valid_rate"] == 1.0
    assert row["source_breadth_score"] < 1.0
    assert row["overall_score"] > 0.55
    assert summary["supported_citation_count"] == 1


def test_fact_proxy_penalizes_unsupported_wrong_topic_report(tmp_path: Path) -> None:
    bench = _bench_dir(tmp_path)
    _write_fact_proxy_inputs(bench, supported=False)
    run_dir = _write_fact_run_dir(tmp_path, source_text="Need for closure increases misinformation acceptance through quick certainty seeking.")

    paths = evaluate_raw_submission_fact_proxy(
        benchmark_dir=bench,
        model_name="local-fact",
        ids=[2],
        run_dir=run_dir,
    )

    row = json.loads(paths.raw_results_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["unsupported_citation_count"] == 1
    assert row["topic_drift_examples"]
    assert row["topic_consistency_score"] < 1.0
    assert row["valid_rate"] == 0.0
    assert row["overall_score"] < 0.50
    assert row["unsupported_examples"]


def _bench_dir(tmp_path: Path) -> Path:
    bench = tmp_path / "deep_research_bench"
    query_dir = bench / "data" / "prompt_data"
    query_dir.mkdir(parents=True)
    raw_dir = bench / "data" / "test_data" / "raw_data"
    raw_dir.mkdir(parents=True)
    rows = [
        {"id": 1, "topic": "Finance", "language": "zh", "prompt": "Chinese task"},
        {"id": 2, "topic": "Science", "language": "en", "prompt": "English task"},
    ]
    (query_dir / "query.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return bench


def _write_proxy_inputs(bench: Path) -> None:
    criteria_dir = bench / "data" / "criteria_data"
    criteria_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir = bench / "data" / "test_data" / "cleaned_data"
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = bench / "data" / "test_data" / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    criteria = {
        "id": 2,
        "prompt": "English task",
        "dimension_weight": {
            "comprehensiveness": 0.3,
            "insight": 0.3,
            "instruction_following": 0.25,
            "readability": 0.15,
        },
        "criterions": {
            "comprehensiveness": [
                {
                    "criterion": "Core mechanism coverage",
                    "explanation": "Covers the mechanism, evidence, and scope.",
                    "weight": 1.0,
                }
            ],
            "insight": [
                {
                    "criterion": "Analytical synthesis",
                    "explanation": "Connects sources and explains implications.",
                    "weight": 1.0,
                }
            ],
            "instruction_following": [
                {
                    "criterion": "Direct answer to the prompt",
                    "explanation": "Answers the original task without drift.",
                    "weight": 1.0,
                }
            ],
            "readability": [
                {
                    "criterion": "Clear structure and prose",
                    "explanation": "Uses organized, readable report prose.",
                    "weight": 1.0,
                }
            ],
        },
    }
    (criteria_dir / "criteria.jsonl").write_text(json.dumps(criteria) + "\n", encoding="utf-8")
    reference = {
        "id": 2,
        "prompt": "English task",
        "article": "# Reference\n\nThis reference covers the core mechanism, evidence, scope, analytical synthesis, implications, direct answer, and clear structure. [1]\n",
    }
    (cleaned_dir / "reference.jsonl").write_text(json.dumps(reference) + "\n", encoding="utf-8")
    raw = {
        "id": 2,
        "prompt": "English task",
        "article": "# Candidate\n\nThis candidate gives a direct answer with core mechanism coverage and clear structure. [1]\n",
    }
    (raw_dir / "local-proxy.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")


def _write_fact_proxy_inputs(bench: Path, *, supported: bool) -> None:
    raw_dir = bench / "data" / "test_data" / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if supported:
        article = (
            "# Candidate\n\n"
            "Need for closure increases misinformation acceptance through quick certainty seeking. [1]\n\n"
            "## Sources\n\n"
            "[1] Need for Closure Source: https://example.com/nfc\n"
        )
    else:
        article = (
            "# Candidate\n\n"
            "COVID-19 vaccination policy determines hospital staffing and public health logistics. [1]\n\n"
            "## Sources\n\n"
            "[1] Need for Closure Source: https://example.com/nfc\n"
        )
    raw = {"id": 2, "prompt": "What is the role of need for closure on misinformation acceptance?", "article": article}
    (raw_dir / "local-fact.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")


def _write_fact_run_dir(tmp_path: Path, *, source_text: str) -> Path:
    run_dir = tmp_path / "run"
    source_dir = run_dir / "source_docs"
    source_dir.mkdir(parents=True)
    (source_dir / "source_1.md").write_text(source_text, encoding="utf-8")
    source = {
        "id": 1,
        "branch_id": "branch_1",
        "title": "Need for Closure Source",
        "url": "https://example.com/nfc",
        "canonical_url": "https://example.com/nfc",
        "provenance": "web",
        "content_path": "source_docs/source_1.md",
        "content_hash": "hash",
        "extraction_method": "test",
        "word_count": len(source_text.split()),
        "quality_score": 0.9,
        "quality_label": "high",
        "quality_type": "academic",
        "relevance_score": 0.9,
    }
    (run_dir / "sources.jsonl").write_text(json.dumps(source) + "\n", encoding="utf-8")
    return run_dir


def _fake_runner(question: str, settings: Settings) -> ResearchRunResult:
    artifacts = ResearchArtifactsV2.create(settings.out_dir, question)
    artifacts.write_text("report.md", f"# Report\n\nAnswer for {question}. [1]\n")
    artifacts.write_json("verification.json", {"valid": True})
    artifacts.write_json("metrics.json", {})
    return ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )


def _failing_runner(question: str, settings: Settings) -> ResearchRunResult:
    artifacts = ResearchArtifactsV2.create(settings.out_dir, question)
    artifacts.write_text("report.md", "# Failed Draft\n")
    artifacts.write_json("failure.json", {"category": "verification_failed"})
    result = ResearchRunResult(
        run_dir=artifacts.run_dir,
        report_path=artifacts.resolve_path("report.md"),
        verification_path=artifacts.resolve_path("verification.json"),
        metrics_path=artifacts.resolve_path("metrics.json"),
    )
    raise ResearchRunError("failed", result)

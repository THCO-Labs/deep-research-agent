import json
import time
from pathlib import Path
from types import SimpleNamespace

from deep_research.artifacts_v2 import ResearchArtifactsV2
from deep_research.evidence import build_evidence_cards
from deep_research.evidence_hygiene import apply_evidence_hygiene, report_quality_issues
from deep_research.acquisition import TavilySearchClientPool, acquire_sources, _branch_queries, _trim_search_query
from deep_research.ingestion import ingest_local_paths, ingest_mcp_manifest
from deep_research.managed import run_gemini_managed_research
from deep_research.planning import build_research_plan
from deep_research.coverage import build_coverage_matrix
from deep_research.guidance import format_criteria_guidance_block
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchBranch, ResearchPlan, ResearchState, SourceRecordV2, VerificationResultV2
from deep_research.semantic import (
    _load_json_object,
    apply_semantic_report_result,
    enrich_evidence_cards_with_semantics,
    verify_report_with_semantics,
)
from deep_research.semantic_planning import build_or_enrich_research_plan
from deep_research.semantic_planning import _loads_json_object
from deep_research.settings import Settings
from deep_research.scraper import ScrapeQualityError
from deep_research.source_validation import validate_source_content
from deep_research.synthesis import (
    _append_evidence_coverage_if_needed,
    _cards_for_synthesis,
    _coverage_repair_labels,
    _evidence_backed_sources,
    _normalize_report_markdown,
    _repair_weak_citation_support,
    _synthesis_model_spec,
    _synthesis_prompt,
    _synthesis_request_kwargs,
    _target_report_profile,
    build_claim_ledger,
    build_report_blueprint,
    build_sentence_plan,
    synthesize_report,
    synthesize_report_with_model,
)
from deep_research.verifier_v2 import _report_depth_score, _report_level_criteria, verify_report_v2
from deep_research.research_graph import (
    _acquire_route,
    _coverage_route,
    _focus_terms_from_state,
    _publish_best_draft,
    _select_best_draft,
    _selected_failed_draft,
    _semantic_gate_collapsed_coverage,
    _verification_route,
    _write_run_health,
)


from tests.test_v2_fakes import FakeGeminiClient, FakeSemanticJudge, InvalidSemanticJudge, QuotaSemanticJudge, RaisingSemanticJudge

def test_generic_source_validation_accepts_medical_topic() -> None:
    plan = build_research_plan(
        "Do Mediterranean diets help adults with hypertension? Cover blood pressure mechanisms and evidence."
    )
    branch = next(branch for branch in plan.branches if "pressure" in " ".join(branch.required_terms + branch.queries).lower())
    content = (
        "Mediterranean diets emphasize vegetables, fruits, legumes, whole grains, nuts, olive oil, "
        "and moderate fish intake. For adults with hypertension, the mechanism may involve lower "
        "sodium intake, higher potassium and fiber intake, improved endothelial function, and better "
        "weight control. Clinical nutrition studies often evaluate systolic and diastolic blood pressure, "
        "adherence, medication use, and cardiovascular risk. This dietary pattern can help some patients, "
        "but it does not replace medical evaluation, antihypertensive medication when indicated, or monitoring."
    )

    result = validate_source_content(
        title="Mediterranean diet and hypertension",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
    )

    assert result.usable is True
    assert result.relevance_score >= 0.30


def test_source_validation_accepts_chinese_sentence_chunks() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="中性粒细胞与脑缺血急性期",
        objective="分析中性粒细胞在脑缺血急性期的功能变化。",
        queries=["中性粒细胞 脑缺血急性期 功能变化"],
        min_sources=1,
        required_terms=["中性粒细胞", "脑缺血急性期", "功能变化"],
    )
    content = (
        "近年研究显示，中性粒细胞在脑缺血急性期会快速募集到损伤区域，"
        "并通过炎症因子释放、血脑屏障影响、微血管阻塞和免疫细胞互作改变局部组织环境。"
        "这些功能变化与梗死扩大、神经炎症强度以及后续修复窗口密切相关。"
        "慢性期研究还提示，中性粒细胞亚群可能参与免疫调节和组织重塑。"
    )

    result = validate_source_content(
        title="中性粒细胞在脑缺血急性期的功能变化",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="请整合中性粒细胞在脑缺血急性期和慢性期的功能变化研究。",
    )

    assert result.usable is True
    assert result.relevant_chunk_count >= 1


def test_source_validation_accepts_translated_branch_source_context() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Top global insurers by assets",
        objective="Identify the strongest global insurance companies by assets and financial strength.",
        queries=["largest insurance companies worldwide by assets"],
        min_sources=1,
        required_terms=["global insurers", "assets", "financial strength"],
    )
    content = (
        "A ranking of the largest insurance companies worldwide by total assets identifies major global insurers "
        "and explains how asset scale, financial strength, and market position differ across companies. "
        "The report compares life insurers and diversified insurance groups across regions. "
    ) * 6

    result = validate_source_content(
        title="Largest insurance companies worldwide by assets",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question=(
            "\u6536\u96c6\u6574\u7406\u76ee\u524d\u56fd\u9645\u7efc\u5408\u5b9e\u529b\u524d\u5341"
            "\u7684\u4fdd\u9669\u516c\u53f8\u7684\u76f8\u5173\u8d44\u6599\uff0c\u5e76\u6a2a"
            "\u5411\u6bd4\u8f83\u878d\u8d44\u3001\u4fe1\u8a89\u5ea6\u3001\u589e\u957f"
            "\u3001\u5206\u7ea2\u548c\u4e2d\u56fd\u53d1\u5c55\u6f5c\u529b\u3002"
        ),
    )

    assert result.usable is True
    assert result.relevance_score >= 0.30


def test_build_evidence_cards_splits_chinese_sentences() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="中性粒细胞与脑缺血急性期",
        objective="分析中性粒细胞在脑缺血急性期的功能变化。",
        queries=["中性粒细胞 脑缺血急性期 功能变化"],
        min_sources=1,
        required_terms=["中性粒细胞", "脑缺血急性期", "功能变化"],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="中性粒细胞脑缺血研究",
        url="https://example.com/neutrophil-stroke",
        canonical_url="https://example.com/neutrophil-stroke",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    text = (
        "背景信息介绍研究设计。"
        "中性粒细胞在脑缺血急性期会快速募集到缺血区域，并通过炎症因子释放、"
        "血脑屏障损伤、微血管阻塞和免疫细胞互作影响神经炎症强度与临床结局。"
        "其他段落讨论统计方法。"
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question="中性粒细胞在脑缺血急性期的功能变化是什么？",
    )

    assert cards
    assert "中性粒细胞" in cards[0].claim
    assert "脑缺血急性期" in cards[0].claim


def test_evidence_builder_accepts_translated_branch_context() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Global insurance companies by assets",
        objective="Identify the largest insurance companies worldwide by assets.",
        queries=["top insurance companies by assets"],
        min_sources=1,
        required_terms=["global insurance companies", "insurance companies assets"],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Largest global insurers by assets",
        url="https://example.com/insurers-assets",
        canonical_url="https://example.com/insurers-assets",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="industry",
        relevance_score=0.8,
    )
    text = (
        "A ranking of global insurance companies by total assets identifies major insurers worldwide "
        "and compares their asset scale, market position, and balance sheet strength across regions."
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question=(
            "\u6536\u96c6\u6574\u7406\u76ee\u524d\u56fd\u9645\u7efc\u5408\u5b9e\u529b"
            "\u524d\u5341\u7684\u4fdd\u9669\u516c\u53f8\u7684\u76f8\u5173\u8d44\u6599"
        ),
    )

    assert cards
    assert cards[0].branch_id == branch.id
    assert "global insurance companies" in cards[0].claim.lower()


def test_source_validation_rejects_generic_content_that_misses_original_question() -> None:
    plan = build_research_plan("What is the role of need for closure on misinformation acceptance?")
    branch = plan.branches[0]
    content = (
        "A purchase approval workflow starts with request submission, budget validation, manager approval, "
        "vendor onboarding, payment release, exception handling, and audit trails. The workflow improves "
        "budget control and operational efficiency for finance teams. "
    ) * 4

    result = validate_source_content(
        title="Operations workflow guide",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question=plan.question,
    )

    assert result.usable is False
    assert any("original question" in reason for reason in result.reasons)


def test_source_validation_rejects_near_neighbor_question_anchor() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    content = (
        "Need for cognition is a motivation to engage in effortful thinking. "
        "Researchers sometimes examine need for cognition and misinformation acceptance, "
        "but this passage discusses effortful cognition rather than certainty seeking. "
    ) * 4

    result = validate_source_content(
        title="Need for cognition and misinformation acceptance",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is False
    assert any("complete phrase" in reason for reason in result.reasons)


def test_source_validation_rejects_neighboring_concept_dominance_with_incidental_target_mentions() -> None:
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

    result = validate_source_content(
        title="Need for cognition and misinformation acceptance",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is False
    assert any("neighboring concept" in reason for reason in result.reasons)


def test_source_validation_rejects_acronym_expansion_collision() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how Need for Closure (NFC) affects misinformation acceptance.",
        queries=["NFC and systematic analysis"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    content = (
        "Near Field Communication (NFC) enables short range wireless communication. "
        "Near field communication payment systems analyze cyber threats, transactions, tags, and devices. "
        "The source discusses systematic analysis, security mitigation, and communication protocols. "
    ) * 8

    result = validate_source_content(
        title="Near-Field Communication (NFC) Cyber Threats and Mitigation Solutions",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is False
    assert any("acronym" in reason for reason in result.reasons)


def test_source_validation_does_not_treat_protected_aliases_as_neighbors() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and cognitive closure",
        objective="Define need for closure as a form of cognitive closure.",
        queries=["need for closure cognitive closure"],
        required_terms=["need for closure", "cognitive closure"],
    )
    content = (
        "Need for closure is a desire for definite cognitive closure instead of prolonged ambiguity. "
        "The need for closure framework explains why people seek certainty and stable answers. "
        "Cognitive closure is therefore part of the same construct rather than a competing topic. "
    ) * 4

    result = validate_source_content(
        title="Need for Closure: influence user behaviour",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is True
    assert not any("neighboring concept" in reason for reason in result.reasons)


def test_source_validation_accepts_source_that_substantively_compares_neighboring_constructs() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["need for closure need for cognition misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    content = (
        "Need for closure is a desire for definite answers and reduced ambiguity. "
        "Need for closure can increase misinformation acceptance when quick certainty displaces careful checking. "
        "The article contrasts need for closure with need for cognition, explaining that need for cognition concerns effortful thinking. "
        "This comparison clarifies why need for closure and misinformation acceptance form a distinct pathway. "
    ) * 4

    result = validate_source_content(
        title="Need for closure, need for cognition, and misinformation acceptance",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is True


def test_source_validation_accepts_direct_single_concept_context_source() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Misinformation acceptance definitions",
        objective="Define misinformation acceptance and belief formation.",
        queries=["misinformation acceptance definition psychology"],
        required_terms=["misinformation acceptance", "belief formation"],
    )
    content = (
        "Misinformation acceptance describes the process by which people endorse inaccurate claims. "
        "The psychology of belief formation includes source credibility, repetition, prior attitudes, "
        "and cognitive shortcuts that shape whether people accept false information. "
    ) * 4

    result = validate_source_content(
        title="Misinformation acceptance and belief formation",
        content=content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert result.usable is True


def test_coverage_allows_partial_soft_required_term_coverage() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mechanisms and evidence",
        objective="Explain the mechanisms and evidence for a relationship.",
        queries=["mechanisms evidence relationship"],
        min_sources=1,
        required_terms=[
            "mechanisms",
            "empirical evidence",
            "mediating factors",
            "moderating factors",
            "boundary conditions",
            "limitations",
            "future research",
            "source credibility",
            "information processing",
            "uncertainty",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Mechanism Source",
        url="https://example.com/mechanism",
        canonical_url="https://example.com/mechanism",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim=(
            "The evidence explains mechanisms, empirical evidence, mediating factors, "
            "source credibility, information processing, and uncertainty."
        ),
        supporting_excerpt=(
            "The evidence explains mechanisms, empirical evidence, mediating factors, "
            "source credibility, information processing, and uncertainty."
        ),
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=[card], sources=[source])

    assert coverage.complete is True
    assert coverage.missing_branches == []
    assert any("required term coverage" in point for point in coverage.branches[0].covered_points)


def test_coverage_keeps_sparse_required_term_coverage_incomplete() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mechanisms and evidence",
        objective="Explain the mechanisms and evidence for a relationship.",
        queries=["mechanisms evidence relationship"],
        min_sources=1,
        required_terms=[
            "mechanisms",
            "empirical evidence",
            "mediating factors",
            "moderating factors",
            "boundary conditions",
            "limitations",
            "future research",
            "source credibility",
            "information processing",
            "uncertainty",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Sparse Source",
        url="https://example.com/sparse",
        canonical_url="https://example.com/sparse",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    card = EvidenceCard(
        id=1,
        source_id=1,
        branch_id=branch.id,
        claim="The evidence mentions mechanisms only.",
        supporting_excerpt="The evidence mentions mechanisms only.",
        source_url=source.url,
        source_title=source.title,
        quality_score=0.9,
        relevance_score=0.9,
        confidence=0.9,
    )

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=[card], sources=[source])

    assert coverage.complete is False
    assert coverage.missing_branches == [branch.id]
    assert any("actual" in point for point in coverage.branches[0].missing_points)


def test_coverage_accepts_strong_semantic_evidence_without_exact_phrase_matches() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mediators and boundary conditions",
        objective="Explain factors that mediate and moderate the relationship.",
        queries=["mediators moderators relationship"],
        min_sources=3,
        required_terms=[
            "source credibility reliance",
            "emotional states",
            "cognitive load",
            "situational urgency",
            "message complexity",
            "prior knowledge",
        ],
    )
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=branch.id,
            title=f"Semantic Source {index}",
            url=f"https://example.com/semantic/{index}",
            canonical_url=f"https://example.com/semantic/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash=f"hash-{index}",
            extraction_method="test",
            word_count=400,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        )
        for index in range(1, 4)
    ]
    cards = [
        EvidenceCard(
            id=index,
            source_id=index,
            branch_id=branch.id,
            claim=f"Study {index} describes a distinct pathway that shapes the relationship.",
            supporting_excerpt=f"Study {index} describes a distinct pathway that shapes the relationship.",
            source_url=sources[index - 1].url,
            source_title=sources[index - 1].title,
            quality_score=0.9,
            relevance_score=0.9,
            confidence=0.9,
            semantic_score=0.86,
            semantic_notes=[f"semantic pathway {index}", "boundary condition"],
        )
        for index in range(1, 4)
    ]

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=cards, sources=sources)

    assert coverage.complete is True
    assert coverage.missing_branches == []
    assert any("semantic evidence sufficiency" in point for point in coverage.branches[0].covered_points)


def test_coverage_allows_evidence_limited_synthesis_when_direct_cards_are_sparse() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Nuanced relationship and evidence limits",
        objective="Synthesize what available evidence can and cannot establish about the relationship.",
        queries=["relationship evidence limitations"],
        min_sources=2,
        required_terms=[
            "relationship evidence",
            "methodological limitations",
            "future studies",
            "boundary conditions",
            "alternative explanations",
            "context dependence",
        ],
    )
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=branch.id,
            title=f"Limited Source {index}",
            url=f"https://example.com/limited/{index}",
            canonical_url=f"https://example.com/limited/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash=f"hash-{index}",
            extraction_method="test",
            word_count=500,
            quality_score=0.88,
            quality_label="good",
            quality_type="academic",
            relevance_score=0.72,
        )
        for index in range(1, 7)
    ]
    cards = [
        EvidenceCard(
            id=1,
            source_id=1,
            branch_id=branch.id,
            claim="The evidence identifies relationship evidence and methodological limitations without proving a direct pathway.",
            supporting_excerpt="The evidence identifies relationship evidence and methodological limitations without proving a direct pathway.",
            source_url=sources[0].url,
            source_title=sources[0].title,
            quality_score=0.88,
            relevance_score=0.62,
            confidence=0.62,
            semantic_score=0.50,
        ),
        EvidenceCard(
            id=2,
            source_id=2,
            branch_id=branch.id,
            claim="The review calls for future studies because boundary conditions and context dependence remain unresolved.",
            supporting_excerpt="The review calls for future studies because boundary conditions and context dependence remain unresolved.",
            source_url=sources[1].url,
            source_title=sources[1].title,
            quality_score=0.88,
            relevance_score=0.62,
            confidence=0.62,
            semantic_score=0.50,
        ),
    ]

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=cards, sources=sources)

    assert coverage.complete is True
    assert coverage.missing_branches == []
    assert "evidence-limited synthesis readiness" in coverage.branches[0].covered_points
    assert not any("strong semantic evidence cards" in point for point in coverage.branches[0].covered_points)


def test_coverage_accepts_evidence_rich_branch_with_planner_phrase_drift() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Five-year growth analysis",
        objective="Compare growth patterns across sources.",
        queries=["growth analysis"],
        min_sources=3,
        required_terms=[
            "growth analysis calculate",
            "analysis calculate compare",
            "calculate compare revenue",
            "growth analysis",
        ],
    )
    sources = [
        SourceRecordV2(
            id=index,
            branch_id=branch.id,
            title=f"Growth Source {index}",
            url=f"https://example.com/growth/{index}",
            canonical_url=f"https://example.com/growth/{index}",
            provenance="web",
            content_path=f"source_docs/source_{index}.md",
            content_hash=f"hash-{index}",
            extraction_method="test",
            word_count=500,
            quality_score=0.72,
            quality_label="usable",
            quality_type="industry",
            relevance_score=0.3,
        )
        for index in range(1, 5)
    ]
    cards = [
        EvidenceCard(
            id=index,
            source_id=((index - 1) % 4) + 1,
            branch_id=branch.id,
            claim="Revenue and asset growth are compared across insurers over several years.",
            supporting_excerpt="Revenue and asset growth are compared across insurers over several years.",
            source_url=sources[((index - 1) % 4)].url,
            source_title=sources[((index - 1) % 4)].title,
            quality_score=0.72,
            relevance_score=0.3,
            confidence=0.63,
        )
        for index in range(1, 13)
    ]

    coverage = build_coverage_matrix(branches=[branch], evidence_cards=cards, sources=sources)

    assert coverage.complete is True
    assert "evidence-limited synthesis readiness" in coverage.branches[0].covered_points


def test_evidence_builder_uses_heading_windows_for_ranking_sources() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Future asset ranking prediction",
        objective="Forecast likely future asset ranking among major insurers.",
        queries=["predict insurance industry asset rankings"],
        min_sources=1,
        required_terms=["asset ranking"],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Best's Rankings and World's Largest Insurance Companies",
        url="https://example.com/rankings",
        canonical_url="https://example.com/rankings",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.8,
        quality_label="strong",
        quality_type="industry",
        relevance_score=0.7,
    )
    text = "\n".join(
        [
            "# Best's Rankings and World's Largest Insurance Companies",
            "## By assets",
            "The annual ranking compares insurance groups by assets and premium scale.",
            "The table gives a basis for assessing future asset leadership.",
        ]
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question="Which insurers are likely to lead future asset ranking?",
    )

    assert cards
    assert any("assets" in card.claim.lower() for card in cards)


def test_evidence_builder_uses_strong_branch_overlap_without_exact_anchor_phrase() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Mediating and moderating factors",
        objective="Identify factors that influence a misinformation acceptance relationship.",
        queries=["source credibility information complexity misinformation acceptance"],
        min_sources=1,
        required_terms=[
            "source cues",
            "information processing depth",
            "emotional responses",
            "mediating factors",
            "moderating factors",
        ],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Source Credibility Study",
        url="https://example.com/source-credibility",
        canonical_url="https://example.com/source-credibility",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=200,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    text = (
        "Credibility, information complexity, and platform context can influence misinformation "
        "acceptance by changing how much people scrutinize false claims before accepting them. "
        "This sentence intentionally does not repeat the planned anchor phrases verbatim."
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question="What factors influence misinformation acceptance?",
    )

    assert cards
    assert "misinformation acceptance" in cards[0].claim


def test_evidence_builder_skips_stale_neighboring_concept_source() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain how Need for Closure (NFC) affects misinformation acceptance.",
        queries=["NFC misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    sources = [
        SourceRecordV2(
            id=1,
            branch_id=branch.id,
            title="Need for Closure Source",
            url="https://example.com/nfc",
            canonical_url="https://example.com/nfc",
            provenance="web",
            content_path="source_docs/source_1.md",
            content_hash="hash-1",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
        SourceRecordV2(
            id=2,
            branch_id=branch.id,
            title="Near-Field Communication (NFC) Cyber Threats and Mitigation Solutions",
            url="https://example.com/near-field",
            canonical_url="https://example.com/near-field",
            provenance="web",
            content_path="source_docs/source_2.md",
            content_hash="hash-2",
            extraction_method="test",
            word_count=120,
            quality_score=0.9,
            quality_label="excellent",
            quality_type="academic",
            relevance_score=0.9,
        ),
    ]

    cards = build_evidence_cards(
        branches=[branch],
        sources=sources,
        source_texts={
            1: (
                "Need for closure can increase misinformation acceptance when people seek quick certainty. "
                "Need for closure encourages premature judgment when misinformation acceptance offers a simple answer."
            ),
            2: (
                "Near Field Communication (NFC) enables short range wireless communication. "
                "Near field communication payment systems analyze cyber threats, transactions, tags, and devices. "
            )
            * 4,
        },
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert cards
    assert {card.source_id for card in cards} == {1}


def test_evidence_extraction_prefers_branch_anchor_sentences() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Need for closure and misinformation acceptance",
        objective="Explain the relationship between need for closure and misinformation acceptance.",
        queries=["need for closure misinformation acceptance"],
        required_terms=["need for closure", "misinformation acceptance"],
    )
    source = SourceRecordV2(
        id=1,
        branch_id=branch.id,
        title="Mixed psychological constructs",
        url="https://example.com/mixed",
        canonical_url="https://example.com/mixed",
        provenance="web",
        content_path="source_docs/source_1.md",
        content_hash="hash",
        extraction_method="test",
        word_count=120,
        quality_score=0.9,
        quality_label="excellent",
        quality_type="academic",
        relevance_score=0.9,
    )
    text = (
        "Need for cognition is associated with effortful thinking and may affect how people evaluate misinformation acceptance. "
        "Need for closure is linked to misinformation acceptance when people seek quick certainty and stop evaluating alternatives."
    )

    cards = build_evidence_cards(
        branches=[branch],
        sources=[source],
        source_texts={1: text},
        question="What is the role of need for closure on misinformation acceptance?",
    )

    assert cards
    assert "need for closure" in cards[0].claim.lower()


def test_stanford_hai_bad_scrape_is_rejected() -> None:
    branch = build_research_plan("What are urban heat islands and how do they affect public health?").branches[0]
    bad_content = (
        "Explore Similar Terms. Stanford HAI. Your browser does not support the video tag. "
        "Subscribe to newsletter. Main navigation. Search. Related content. "
    ) * 8

    result = validate_source_content(
        title="Stanford HAI",
        content=bad_content,
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
    )

    assert result.usable is False
    assert any("boilerplate" in reason or "relevance" in reason for reason in result.reasons)

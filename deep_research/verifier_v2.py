from __future__ import annotations

import re

from deep_research.evidence_hygiene import report_quality_issues
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchPlan, SourceRecordV2, VerificationResultV2
from deep_research.source_validation import branch_terms, content_terms
from deep_research.verifier import parse_inline_citations, parse_source_list

SUPPORT_THRESHOLD_V2 = 0.35
ANSWER_COVERAGE_THRESHOLD = 0.50
BRANCH_COVERAGE_THRESHOLD = 0.80
EVIDENCE_LINKAGE_THRESHOLD = 0.80
SOURCE_QUALITY_THRESHOLD = 0.55
STRUCTURE_THRESHOLD = 0.60
REPORT_CLEANLINESS_THRESHOLD = 1.0


def verify_report_v2(
    *,
    report_markdown: str,
    plan: ResearchPlan,
    sources: list[SourceRecordV2],
    evidence_cards: list[EvidenceCard],
    coverage: CoverageMatrix,
    source_texts: dict[int, str],
) -> VerificationResultV2:
    failures: list[str] = []
    source_by_id = {source.id: source for source in sources}
    evidence_source_ids = {card.source_id for card in evidence_cards}
    source_list = parse_source_list(report_markdown)
    cited_ids = sorted(set(parse_inline_citations(report_markdown)))

    citation_failures = 0
    if not cited_ids:
        failures.append("Report does not cite any sources.")
        citation_failures += 1
    if not source_list:
        failures.append("Report is missing a parseable Sources section.")
        citation_failures += 1
    for source_id in cited_ids:
        if source_id not in source_by_id:
            failures.append(f"Cited source [{source_id}] is not a usable source record.")
            citation_failures += 1
        if source_id not in source_list:
            failures.append(f"Cited source [{source_id}] is missing from the Sources section.")
            citation_failures += 1
        if source_id not in evidence_source_ids:
            failures.append(f"Cited source [{source_id}] has no evidence card.")
            citation_failures += 1
    for source_id in source_list:
        if source_id not in cited_ids:
            failures.append(f"Sources section lists uncited source [{source_id}].")
            citation_failures += 1
    citation_validity_score = max(0.0, round(1.0 - citation_failures / max(len(cited_ids) + len(source_list) + 1, 1), 4))

    unsupported_claims = _paragraphs_without_citations(report_markdown)
    for paragraph in unsupported_claims:
        failures.append(f"Uncited factual paragraph: {paragraph[:120]}")

    weak_claims, support_score = _source_support_checks(report_markdown, source_texts)
    for claim in weak_claims:
        failures.append(f"Weakly supported cited paragraph: {claim['paragraph'][:120]}")

    answer_coverage_score = _answer_coverage_score(report_markdown, plan)
    if answer_coverage_score < ANSWER_COVERAGE_THRESHOLD:
        failures.append(f"Answer coverage below threshold: {answer_coverage_score}")

    branch_coverage_score = coverage.coverage_score
    if branch_coverage_score < BRANCH_COVERAGE_THRESHOLD:
        failures.append(f"Branch coverage below threshold: {branch_coverage_score}")

    evidence_linkage_score = _evidence_linkage_score(report_markdown, evidence_cards)
    if evidence_linkage_score < EVIDENCE_LINKAGE_THRESHOLD:
        failures.append(f"Evidence linkage below threshold: {evidence_linkage_score}")

    source_quality_score = _source_quality_score([source_by_id[source_id] for source_id in cited_ids if source_id in source_by_id])
    if source_quality_score < SOURCE_QUALITY_THRESHOLD:
        failures.append(f"Average cited source quality below threshold: {source_quality_score}")

    structure_score = _report_structure_score(report_markdown, plan)
    if structure_score < STRUCTURE_THRESHOLD:
        failures.append(f"Report structure below threshold: {structure_score}")

    report_artifacts = report_quality_issues(report_markdown)
    for issue in report_artifacts:
        failures.append(issue)
    report_cleanliness_score = max(
        0.0,
        round(1.0 - len(report_artifacts) / max(len(_report_body_lines(report_markdown)), 1), 4),
    )

    valid = (
        citation_validity_score >= 1.0
        and not unsupported_claims
        and support_score >= SUPPORT_THRESHOLD_V2
        and answer_coverage_score >= ANSWER_COVERAGE_THRESHOLD
        and branch_coverage_score >= BRANCH_COVERAGE_THRESHOLD
        and evidence_linkage_score >= EVIDENCE_LINKAGE_THRESHOLD
        and source_quality_score >= SOURCE_QUALITY_THRESHOLD
        and structure_score >= STRUCTURE_THRESHOLD
        and report_cleanliness_score >= REPORT_CLEANLINESS_THRESHOLD
        and not weak_claims
        and not report_artifacts
    )
    return VerificationResultV2(
        valid=valid,
        citation_validity_score=citation_validity_score,
        source_support_score=support_score,
        answer_coverage_score=answer_coverage_score,
        branch_coverage_score=branch_coverage_score,
        evidence_linkage_score=evidence_linkage_score,
        source_quality_score=source_quality_score,
        report_structure_score=structure_score,
        report_cleanliness_score=report_cleanliness_score,
        failures=failures,
        cited_source_ids=cited_ids,
        unsupported_claims=unsupported_claims,
        weakly_supported_claims=weak_claims,
    )


def _source_support_checks(report: str, source_texts: dict[int, str]) -> tuple[list[dict[str, object]], float]:
    checks: list[float] = []
    weak: list[dict[str, object]] = []
    for paragraph in _paragraphs_with_citations(report):
        cited_ids = sorted(set(parse_inline_citations(paragraph)))
        claim_terms = content_terms(_strip_citations(paragraph))
        if len(claim_terms) < 5:
            continue
        source_terms = content_terms(" ".join(source_texts.get(source_id, "") for source_id in cited_ids))
        score = round(len(claim_terms & source_terms) / max(len(claim_terms), 1), 4)
        checks.append(score)
        if score < SUPPORT_THRESHOLD_V2:
            weak.append(
                {
                    "paragraph": paragraph[:240],
                    "cited_source_ids": cited_ids,
                    "support_score": score,
                    "missing_terms": sorted(claim_terms - source_terms)[:20],
                }
            )
    if not checks:
        return weak, 0.0
    return weak, round(sum(checks) / len(checks), 4)


def _answer_coverage_score(report: str, plan: ResearchPlan) -> float:
    required_terms: set[str] = set()
    for branch in plan.branches:
        required_terms.update(term.lower() for term in branch.required_terms)
        required_terms.update(branch_terms(branch))
    if not required_terms:
        return 1.0
    normalized_report = report.lower().replace("-", " ")
    hits = 0
    for term in required_terms:
        normalized = term.lower().replace("-", " ")
        if normalized in normalized_report:
            hits += 1
    return round(hits / len(required_terms), 4)


def _evidence_linkage_score(report: str, cards: list[EvidenceCard]) -> float:
    cited_ids = sorted(set(parse_inline_citations(report)))
    if not cited_ids:
        return 0.0
    card_source_ids = {card.source_id for card in cards}
    return round(len([source_id for source_id in cited_ids if source_id in card_source_ids]) / len(cited_ids), 4)


def _source_quality_score(sources: list[SourceRecordV2]) -> float:
    if not sources:
        return 0.0
    return round(sum(source.quality_score for source in sources) / len(sources), 4)


def _report_structure_score(report: str, plan: ResearchPlan) -> float:
    score = 0.0
    body = _without_sources(report)
    headings = re.findall(r"(?m)^##\s+(.+?)\s*$", body)
    cited_paragraphs = _paragraphs_with_citations(report)
    if re.search(r"(?im)^#\s+\S", report):
        score += 0.10
    if cited_paragraphs and _first_substantive_paragraph_answers_question(cited_paragraphs[0], plan):
        score += 0.20
    elif cited_paragraphs:
        score += 0.10
    if headings:
        score += 0.15
    if _branch_topic_heading_coverage(headings, plan) >= 0.35:
        score += 0.15
    if _has_synthesis_language(body):
        score += 0.15
    if _has_uncertainty_or_limits(body):
        score += 0.10
    if re.search(r"(?im)^##\s+Sources\s*$", report):
        score += 0.15
    section_count = len(headings)
    if section_count >= min(4, max(2, len(plan.branches) // 2)):
        score += 0.10
    if "|" in report and "---" in report:
        score += 0.05
    if "Verification Notes" not in report:
        score += 0.05
    return round(min(score, 1.0), 4)


def _first_substantive_paragraph_answers_question(paragraph: str, plan: ResearchPlan) -> bool:
    question_terms = content_terms(plan.question)
    paragraph_terms = content_terms(paragraph)
    if not question_terms:
        return True
    return len(question_terms & paragraph_terms) / max(len(question_terms), 1) >= 0.25


def _branch_topic_heading_coverage(headings: list[str], plan: ResearchPlan) -> float:
    if not plan.branches or not headings:
        return 0.0
    heading_terms = content_terms(" ".join(headings))
    covered = 0
    for branch in plan.branches:
        terms = content_terms(branch.title + " " + branch.objective)
        if terms and len(terms & heading_terms) / max(len(terms), 1) >= 0.15:
            covered += 1
    return round(covered / len(plan.branches), 4)


def _has_synthesis_language(body: str) -> bool:
    return bool(
        re.search(
            r"\b(?:taken together|across sources|the evidence|sources agree|sources differ|trade[- ]?off|"
            r"tension|pattern|overall|in contrast|compared with|suggests that|indicates that)\b",
            body,
            flags=re.I,
        )
    )


def _has_uncertainty_or_limits(body: str) -> bool:
    return bool(
        re.search(
            r"\b(?:limitation|limits|uncertain|uncertainty|confidence|evidence gap|gap|caveat|"
            r"cannot determine|not enough evidence|mixed evidence|correlational|causal)\b",
            body,
            flags=re.I,
        )
    )


def _report_body_lines(markdown: str) -> list[str]:
    return [line for line in _without_sources(markdown).splitlines() if line.strip()]


def _paragraphs_without_citations(markdown: str) -> list[str]:
    body = _without_sources(markdown)
    unsupported: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not text or text.startswith("#") or text.startswith("|"):
            continue
        if len(text) < 80:
            continue
        if not parse_inline_citations(text):
            unsupported.append(text[:240])
    return unsupported


def _paragraphs_with_citations(markdown: str) -> list[str]:
    body = _without_sources(markdown)
    cited: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not text or text.startswith("#") or text.startswith("|"):
            continue
        if parse_inline_citations(text):
            cited.append(text)
    return cited


def _without_sources(markdown: str) -> str:
    match = re.search(r"(?ims)^#{2,3}\s+sources\s*$", markdown)
    return markdown if not match else markdown[: match.start()]


def _strip_citations(text: str) -> str:
    return re.sub(r"\[([0-9][0-9,\s]*)\]", "", text)

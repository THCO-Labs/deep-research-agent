from __future__ import annotations

import re
from typing import Any

from deep_research.evidence_hygiene import report_quality_issues
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchPlan, SourceRecordV2, VerificationResultV2
from deep_research.source_limits import MINIMUM_SOURCE_TARGET
from deep_research.source_validation import anchor_groups_for_question, branch_terms, content_terms
from deep_research.verifier import parse_inline_citations, parse_source_list

SUPPORT_THRESHOLD_V2 = 0.35
ANSWER_COVERAGE_THRESHOLD = 0.50
BRANCH_COVERAGE_THRESHOLD = 0.80
EVIDENCE_LINKAGE_THRESHOLD = 0.80
SOURCE_QUALITY_THRESHOLD = 0.55
STRUCTURE_THRESHOLD = 0.60
REPORT_CLEANLINESS_THRESHOLD = 1.0
REQUEST_ALIGNMENT_THRESHOLD = 0.45
CRITERIA_COVERAGE_THRESHOLD = 0.65
OPENING_ALIGNMENT_THRESHOLD = 0.62


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

    criteria_coverage_score, undercovered_criteria = _criteria_coverage(report_markdown, plan)
    if criteria_coverage_score < CRITERIA_COVERAGE_THRESHOLD:
        failures.append(f"Acceptance criteria coverage below threshold: {criteria_coverage_score}")
    for row in undercovered_criteria[:8]:
        failures.append(f"Under-covered acceptance criterion: {row['criterion'][:120]}")

    request_alignment_score = _request_alignment_score(report_markdown, plan)
    if request_alignment_score < _request_alignment_threshold(plan):
        failures.append(f"Report topic alignment below threshold: {request_alignment_score}")

    opening_alignment_score = _opening_alignment_score(report_markdown, plan)
    if opening_alignment_score < _opening_alignment_threshold(plan):
        failures.append(f"Opening answer topic alignment below threshold: {opening_alignment_score}")

    topic_drift_paragraphs = _topic_drift_paragraphs(report_markdown, plan)
    for paragraph in topic_drift_paragraphs:
        failures.append(f"Topic-drift cited paragraph: {paragraph[:120]}")

    branch_coverage_score = coverage.coverage_score
    if not coverage.complete:
        failures.append(f"Branch coverage incomplete: {', '.join(coverage.missing_branches) or 'unknown'}")
    if branch_coverage_score < BRANCH_COVERAGE_THRESHOLD:
        failures.append(f"Branch coverage below threshold: {branch_coverage_score}")

    evidence_linkage_score = _evidence_linkage_score(report_markdown, evidence_cards)
    if evidence_linkage_score < EVIDENCE_LINKAGE_THRESHOLD:
        failures.append(f"Evidence linkage below threshold: {evidence_linkage_score}")

    source_breadth_score = _source_breadth_score(cited_ids, evidence_cards)
    if source_breadth_score < 1.0:
        required = _required_cited_source_count(evidence_cards)
        cited_evidence_sources = len(set(cited_ids) & {card.source_id for card in evidence_cards})
        failures.append(f"Cited evidence-backed source count below threshold: {cited_evidence_sources} < {required}")

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
        and criteria_coverage_score >= CRITERIA_COVERAGE_THRESHOLD
        and request_alignment_score >= _request_alignment_threshold(plan)
        and opening_alignment_score >= _opening_alignment_threshold(plan)
        and not topic_drift_paragraphs
        and coverage.complete
        and branch_coverage_score >= BRANCH_COVERAGE_THRESHOLD
        and evidence_linkage_score >= EVIDENCE_LINKAGE_THRESHOLD
        and source_breadth_score >= 1.0
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
        request_alignment_score=request_alignment_score,
        source_breadth_score=source_breadth_score,
        report_cleanliness_score=report_cleanliness_score,
        criteria_coverage_score=criteria_coverage_score,
        failures=failures,
        cited_source_ids=cited_ids,
        unsupported_claims=unsupported_claims,
        weakly_supported_claims=weak_claims,
        undercovered_criteria=undercovered_criteria,
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
    report_terms = content_terms(_without_sources(report))
    hits = 0
    for term in required_terms:
        if _coverage_point_is_present(term, report_terms):
            hits += 1
    return round(hits / len(required_terms), 4)


def _criteria_coverage(report: str, plan: ResearchPlan) -> tuple[float, list[dict[str, Any]]]:
    criteria = _report_level_criteria(plan.acceptance_criteria)
    if not criteria:
        return 1.0, []
    report_terms = content_terms(_without_sources(report))
    scores: list[float] = []
    undercovered: list[dict[str, Any]] = []
    for criterion in criteria:
        terms = _criterion_terms(criterion)
        if len(terms) < 2:
            continue
        score = round(len(terms & report_terms) / len(terms), 4)
        scores.append(score)
        threshold = 0.42 if len(terms) <= 5 else 0.35
        if score < threshold:
            undercovered.append(
                {
                    "criterion": criterion,
                    "coverage_score": score,
                    "missing_terms": sorted(terms - report_terms)[:18],
                }
            )
    if not scores:
        return 1.0, []
    return round(sum(scores) / len(scores), 4), undercovered


def _report_level_criteria(criteria: list[str]) -> list[str]:
    selected: list[str] = []
    for criterion in criteria:
        cleaned = _clean_criterion(criterion)
        if not cleaned:
            continue
        terms = content_terms(cleaned)
        if len(terms) < 2:
            continue
        if _looks_like_internal_runtime_criterion(cleaned):
            continue
        selected.append(cleaned)
    return selected


def _clean_criterion(criterion: str) -> str:
    cleaned = re.sub(r"\s+", " ", criterion).strip()
    cleaned = re.sub(r"(?i)^cover this task-specific criterion in synthesis:\s*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\(\s*weight\s*:\s*[^)]*\)", "", cleaned).strip()
    return cleaned.strip(" .:")


def _looks_like_internal_runtime_criterion(criterion: str) -> bool:
    return bool(
        re.search(
            r"\b(?:evidence cards?|inline citations?|factual paragraphs?|verification passes?|quality gates?|"
            r"citation|citations|source list|sources section|report answers?|answers? the question)\b",
            criterion,
            flags=re.I,
        )
    )


def _criterion_terms(criterion: str) -> set[str]:
    return content_terms(criterion)


def _coverage_point_is_present(term: str, report_terms: set[str]) -> bool:
    term_terms = content_terms(term)
    if not term_terms:
        return False
    hits = len(term_terms & report_terms)
    if len(term_terms) <= 2:
        return hits == len(term_terms)
    required_overlap = 0.55 if len(term_terms) <= 5 else 0.45
    return hits / len(term_terms) >= required_overlap


def _request_alignment_score(report: str, plan: ResearchPlan) -> float:
    question_terms = content_terms(plan.question)
    if not question_terms:
        return 1.0
    body = _without_sources(report)
    opening = _opening_answer_text(body)
    body_terms = content_terms(body)
    opening_terms = content_terms(opening)
    branch_terms_text = " ".join(
        branch.title + " " + branch.objective + " " + " ".join(branch.required_terms)
        for branch in plan.branches
    )
    plan_terms = content_terms(plan.question + " " + branch_terms_text)
    evidence_terms = body_terms & plan_terms
    body_score = len(question_terms & body_terms) / max(len(question_terms), 1)
    opening_score = len(question_terms & opening_terms) / max(len(question_terms), 1)
    evidence_anchor_score = len(question_terms & evidence_terms) / max(len(question_terms), 1)
    return round((body_score * 0.50) + (opening_score * 0.30) + (evidence_anchor_score * 0.20), 4)


def _opening_alignment_score(report: str, plan: ResearchPlan) -> float:
    question_terms = content_terms(plan.question)
    if not question_terms:
        return 1.0
    opening = _opening_answer_text(_without_sources(report))
    if not opening:
        return 0.0
    opening_terms = content_terms(opening)
    term_score = len(question_terms & opening_terms) / max(len(question_terms), 1)
    anchor_groups = anchor_groups_for_question(plan.question)
    if not anchor_groups:
        phrase_score = term_score
    else:
        matches = 0
        for group in anchor_groups[:18]:
            if group <= opening_terms:
                matches += 1
        phrase_score = matches / max(min(len(anchor_groups), 18), 1)
    return round((term_score * 0.45) + (phrase_score * 0.55), 4)


def _request_alignment_threshold(plan: ResearchPlan) -> float:
    term_count = len(content_terms(plan.question))
    if term_count <= 3:
        return 0.60
    if term_count <= 8:
        return REQUEST_ALIGNMENT_THRESHOLD
    return 0.35


def _opening_alignment_threshold(plan: ResearchPlan) -> float:
    term_count = len(content_terms(plan.question))
    if term_count <= 3:
        return 0.70
    if term_count <= 8:
        return OPENING_ALIGNMENT_THRESHOLD
    if term_count <= 20:
        return 0.48
    return 0.35


def _topic_drift_paragraphs(report: str, plan: ResearchPlan) -> list[str]:
    topic_terms = _topic_guard_terms(plan)
    if len(topic_terms) < 2:
        return []
    drifted: list[str] = []
    for paragraph in _paragraphs_with_citations(report):
        paragraph_terms = content_terms(_strip_citations(paragraph))
        if len(paragraph_terms) < 8:
            continue
        hits = len(paragraph_terms & topic_terms)
        local_relevance = hits / max(len(paragraph_terms), 1)
        anchor_relevance = hits / max(min(len(topic_terms), 16), 1)
        if hits < 2 and local_relevance < 0.08:
            drifted.append(paragraph[:240])
            continue
        if hits < 3 and local_relevance < 0.045 and anchor_relevance < 0.12:
            drifted.append(paragraph[:240])
    return drifted


def _topic_guard_terms(plan: ResearchPlan) -> set[str]:
    seed = [plan.question]
    for branch in plan.branches:
        seed.extend(
            [
                branch.title,
                branch.objective,
                " ".join(branch.required_terms),
                " ".join(branch.completion_criteria),
            ]
        )
    return content_terms(" ".join(seed))


def _opening_answer_text(body: str) -> str:
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", body):
        text = " ".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if not text:
            continue
        if text.startswith("#") or text.startswith("|"):
            continue
        paragraphs.append(text)
        if len(paragraphs) >= 1:
            break
    return " ".join(paragraphs)


def _evidence_linkage_score(report: str, cards: list[EvidenceCard]) -> float:
    cited_ids = sorted(set(parse_inline_citations(report)))
    if not cited_ids:
        return 0.0
    card_source_ids = {card.source_id for card in cards}
    return round(len([source_id for source_id in cited_ids if source_id in card_source_ids]) / len(cited_ids), 4)


def _source_breadth_score(cited_ids: list[int], cards: list[EvidenceCard]) -> float:
    required = _required_cited_source_count(cards)
    if required == 0:
        return 1.0
    evidence_source_ids = {card.source_id for card in cards}
    cited_evidence_sources = set(cited_ids) & evidence_source_ids
    return round(min(1.0, len(cited_evidence_sources) / required), 4)


def _required_cited_source_count(cards: list[EvidenceCard]) -> int:
    evidence_source_count = len({card.source_id for card in cards})
    return min(MINIMUM_SOURCE_TARGET, evidence_source_count)


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

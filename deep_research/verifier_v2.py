from __future__ import annotations

import re
from typing import Any

from deep_research.evidence_hygiene import report_quality_issues
from deep_research.schemas import CoverageMatrix, EvidenceCard, ResearchPlan, SourceRecordV2, VerificationResultV2
from deep_research.source_limits import MINIMUM_SOURCE_TARGET
from deep_research.source_validation import anchor_groups_for_question, branch_terms, content_terms, validate_source_content
from deep_research.text_terms import cjk_char_count, latin_letter_count, preferred_output_language
from deep_research.verifier import parse_inline_citations, parse_source_list

SUPPORT_THRESHOLD_V2 = 0.35
INDIVIDUAL_CITATION_SUPPORT_THRESHOLD = 0.22
ANSWER_COVERAGE_THRESHOLD = 0.50
BRANCH_COVERAGE_THRESHOLD = 0.80
EVIDENCE_LINKAGE_THRESHOLD = 0.80
SOURCE_QUALITY_THRESHOLD = 0.55
STRUCTURE_THRESHOLD = 0.60
REPORT_CLEANLINESS_THRESHOLD = 1.0
REQUEST_ALIGNMENT_THRESHOLD = 0.45
CRITERIA_COVERAGE_THRESHOLD = 0.65
OPENING_ALIGNMENT_THRESHOLD = 0.62
LANGUAGE_ALIGNMENT_THRESHOLD = 0.80
REPORT_DEPTH_THRESHOLD = 0.45
CRITERIA_RICH_REPORT_DEPTH_THRESHOLD = 0.90
CITED_SOURCE_ALIGNMENT_THRESHOLD = 1.0


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

    source_alignment_issues, cited_source_alignment_score = _cited_source_alignment_checks(
        cited_ids=cited_ids,
        source_by_id=source_by_id,
        source_texts=source_texts,
        plan=plan,
    )
    for issue in source_alignment_issues:
        reason_text = "; ".join(str(reason) for reason in issue["reasons"][:3])
        failures.append(f"Cited source [{issue['source_id']}] fails current branch/request alignment: {reason_text}")

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

    language_alignment_score = _language_alignment_score(report_markdown, plan)
    if language_alignment_score < LANGUAGE_ALIGNMENT_THRESHOLD:
        failures.append(f"Report language alignment below threshold: {language_alignment_score}")

    report_depth_score = _report_depth_score(report_markdown, plan, evidence_cards)
    report_depth_threshold = _report_depth_threshold(plan)
    if report_depth_score < report_depth_threshold:
        failures.append(f"Report depth below threshold: {report_depth_score}")

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
        and cited_source_alignment_score >= CITED_SOURCE_ALIGNMENT_THRESHOLD
        and not source_alignment_issues
        and structure_score >= STRUCTURE_THRESHOLD
        and report_cleanliness_score >= REPORT_CLEANLINESS_THRESHOLD
        and language_alignment_score >= LANGUAGE_ALIGNMENT_THRESHOLD
        and report_depth_score >= report_depth_threshold
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
        language_alignment_score=language_alignment_score,
        report_depth_score=report_depth_score,
        cited_source_alignment_score=cited_source_alignment_score,
        failures=failures,
        cited_source_ids=cited_ids,
        unsupported_claims=unsupported_claims,
        weakly_supported_claims=weak_claims,
        undercovered_criteria=undercovered_criteria,
    )


def _source_support_checks(report: str, source_texts: dict[int, str]) -> tuple[list[dict[str, object]], float]:
    checks: list[float] = []
    weak: list[dict[str, object]] = []
    source_terms_by_id = {
        source_id: content_terms(text)
        for source_id, text in source_texts.items()
    }
    for paragraph in _paragraphs_with_citations(report):
        cited_ids = sorted(set(parse_inline_citations(paragraph)))
        claim_terms = content_terms(_strip_citations(paragraph))
        if len(claim_terms) < 5:
            continue
        source_terms: set[str] = set()
        for source_id in cited_ids:
            source_terms.update(source_terms_by_id.get(source_id, set()))
        score = round(len(claim_terms & source_terms) / max(len(claim_terms), 1), 4)
        checks.append(score)
        if score < SUPPORT_THRESHOLD_V2:
            weak.append(
                {
                    "paragraph": paragraph[:240],
                    "cited_source_ids": cited_ids,
                    "support_kind": "citation_group",
                    "support_score": score,
                    "missing_terms": sorted(claim_terms - source_terms)[:20],
                }
            )
        for source_id in cited_ids:
            individual_source_terms = source_terms_by_id.get(source_id, set())
            individual_score = round(len(claim_terms & individual_source_terms) / max(len(claim_terms), 1), 4)
            checks.append(individual_score)
            if individual_score < INDIVIDUAL_CITATION_SUPPORT_THRESHOLD:
                weak.append(
                    {
                        "paragraph": paragraph[:240],
                        "cited_source_ids": [source_id],
                        "support_kind": "individual_citation",
                        "support_score": individual_score,
                        "missing_terms": sorted(claim_terms - individual_source_terms)[:20],
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


def _language_alignment_score(report: str, plan: ResearchPlan) -> float:
    expected = preferred_output_language(plan.question)
    body = _without_sources(report)
    cjk_count = cjk_char_count(body)
    latin_count = latin_letter_count(body)
    total = cjk_count + latin_count
    if total < 120:
        return 1.0
    if expected == "zh":
        ratio = cjk_count / max(total, 1)
        return round(min(1.0, ratio / 0.35), 4)
    if cjk_count < 40:
        return 1.0
    ratio = latin_count / max(total, 1)
    return round(min(1.0, ratio / 0.55), 4)


def _report_depth_score(report: str, plan: ResearchPlan, evidence_cards: list[EvidenceCard]) -> float:
    body = _without_sources(report)
    cited_paragraphs = _paragraphs_with_citations(report)
    if not cited_paragraphs:
        return 0.0
    evidence_source_count = len({card.source_id for card in evidence_cards})
    criteria_rich = _criteria_rich_plan(plan)
    expected_paragraphs = min(
        40 if criteria_rich else 12,
        max(
            3,
            len(plan.branches) * 2 if criteria_rich else len(plan.branches),
            evidence_source_count // 4,
            len(plan.acceptance_criteria) // 3,
        ),
    )
    paragraph_score = min(1.0, len(cited_paragraphs) / max(expected_paragraphs, 1))
    paragraph_term_counts = [len(content_terms(_strip_citations(paragraph))) for paragraph in cited_paragraphs]
    avg_terms = sum(paragraph_term_counts) / max(len(paragraph_term_counts), 1)
    term_depth_score = min(1.0, avg_terms / 18)
    headings = re.findall(r"(?m)^##\s+(.+?)\s*$", body)
    heading_score = min(1.0, len(headings) / _target_heading_count(plan, criteria_rich=criteria_rich))
    synthesis_score = 1.0 if _has_synthesis_language(body) else 0.0
    limits_score = 1.0 if _has_uncertainty_or_limits(body) else 0.0
    word_score = min(1.0, _report_word_count(body) / max(_required_report_word_count(plan, evidence_cards), 1))
    return round(
        min(
            1.0,
            (paragraph_score * 0.25)
            + (term_depth_score * 0.20)
            + (heading_score * 0.20)
            + (word_score * 0.25)
            + (synthesis_score * 0.05)
            + (limits_score * 0.05),
        ),
        4,
    )


def _report_depth_threshold(plan: ResearchPlan) -> float:
    return CRITERIA_RICH_REPORT_DEPTH_THRESHOLD if _criteria_rich_plan(plan) else REPORT_DEPTH_THRESHOLD


def _target_heading_count(plan: ResearchPlan, *, criteria_rich: bool) -> int:
    if criteria_rich:
        return min(
            30,
            max(
                16,
                len(plan.branches) + 7,
                len(_report_level_criteria(plan.acceptance_criteria)) // 2,
            ),
        )
    return max(min(6, max(2, len(plan.branches) // 2)), 1)


def _criteria_rich_plan(plan: ResearchPlan) -> bool:
    return len(_report_level_criteria(plan.acceptance_criteria)) >= 8 or any(
        "task-specific" in criterion.lower() and "criterion" in criterion.lower()
        for criterion in plan.acceptance_criteria
    )


def _required_report_word_count(plan: ResearchPlan, evidence_cards: list[EvidenceCard]) -> int:
    criteria_count = len(_report_level_criteria(plan.acceptance_criteria))
    branch_count = len(plan.branches)
    evidence_source_count = len({card.source_id for card in evidence_cards})
    if _criteria_rich_plan(plan):
        return min(
            9000,
            max(
                6500,
                criteria_count * 220,
                branch_count * 520,
                evidence_source_count * 150,
            ),
        )
    if evidence_source_count >= 30 or branch_count >= 8:
        return 3200
    if evidence_source_count >= MINIMUM_SOURCE_TARGET or branch_count >= 5:
        return 2600
    return 1400


def _report_word_count(text: str) -> int:
    cjk_count = cjk_char_count(text)
    latin_words = len(re.findall(r"\b[A-Za-z0-9][A-Za-z0-9+.-]*\b", text))
    return latin_words + (cjk_count // 2)


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


def _cited_source_alignment_checks(
    *,
    cited_ids: list[int],
    source_by_id: dict[int, SourceRecordV2],
    source_texts: dict[int, str],
    plan: ResearchPlan,
) -> tuple[list[dict[str, Any]], float]:
    if not cited_ids:
        return [], 0.0
    branch_by_id = {branch.id: branch for branch in plan.branches}
    checks: list[float] = []
    issues: list[dict[str, Any]] = []
    for source_id in cited_ids:
        source = source_by_id.get(source_id)
        if source is None:
            continue
        branch = branch_by_id.get(source.branch_id)
        if branch is None:
            checks.append(0.0)
            issues.append(
                {
                    "source_id": source_id,
                    "branch_id": source.branch_id,
                    "reasons": ["source branch is not present in the active research plan"],
                    "relevance_score": 0.0,
                }
            )
            continue
        validation = validate_source_content(
            title=source.title,
            content=source_texts.get(source_id, ""),
            branch=branch,
            min_words=0,
            min_relevant_chunks=0,
            question=plan.question,
        )
        aligned = validation.usable
        checks.append(1.0 if aligned else 0.0)
        if not aligned:
            issues.append(
                {
                    "source_id": source_id,
                    "branch_id": source.branch_id,
                    "reasons": validation.reasons,
                    "relevance_score": validation.relevance_score,
                }
            )
    if not checks:
        return issues, 0.0
    return issues, round(sum(checks) / len(checks), 4)


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
    if re.search(
        r"\b(?:taken together|across sources|the evidence|sources agree|sources differ|trade[- ]?off|"
        r"tension|pattern|overall|in contrast|compared with|suggests that|indicates that)\b",
        body,
        flags=re.I,
    ):
        return True
    return bool(
        re.search(
            (
                "\u7efc\u5408|\u8bc1\u636e|\u6765\u6e90|\u7814\u7a76|\u663e\u793a|\u8868\u660e|"
                "\u6bd4\u8f83|\u5dee\u5f02|\u4e00\u81f4|\u6743\u8861|\u603b\u4f53|"
                "\u5171\u540c|\u77db\u76fe|\u5f20\u529b"
            ),
            body,
        )
    )


def _has_uncertainty_or_limits(body: str) -> bool:
    if re.search(
        r"\b(?:limitation|limits|uncertain|uncertainty|confidence|evidence gap|gap|caveat|"
        r"cannot determine|not enough evidence|mixed evidence|correlational|causal)\b",
        body,
        flags=re.I,
    ):
        return True
    return bool(
        re.search(
            (
                "\u5c40\u9650|\u9650\u5236|\u4e0d\u786e\u5b9a|\u4fe1\u5fc3|"
                "\u8bc1\u636e\u7f3a\u53e3|\u7f3a\u53e3|\u8c28\u614e|"
                "\u4e0d\u80fd\u786e\u5b9a|\u8bc1\u636e\u4e0d\u8db3|\u76f8\u5173\u6027|\u56e0\u679c"
            ),
            body,
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

from __future__ import annotations

from collections import defaultdict
import re

from deep_research.core.schemas import EvidenceCard, ResearchPlan, SourceRecordV2
from deep_research.evidence.source_validation import content_terms

# Raised from 8/36: most runs gather well under 150 evidence cards total, and
# capping the writer to a small pre-filtered subset means it never sees most of
# what was actually found. These act as a safety ceiling for pathologically
# large evidence sets, not the default filter -- comparable to open_deep_research
# handing its writer all compressed notes, unfiltered, bounded only by context.
MAX_SYNTHESIS_CARDS_PER_BRANCH = 25
MAX_SYNTHESIS_CARDS_TOTAL = 150


def _cards_by_branch(evidence_cards: list[EvidenceCard]) -> dict[str, list[EvidenceCard]]:
    cards_by_branch: dict[str, list[EvidenceCard]] = defaultdict(list)
    for card in evidence_cards:
        cards_by_branch[card.branch_id].append(card)
    return cards_by_branch


def _evidence_backed_sources(
    sources: list[SourceRecordV2],
    evidence_cards: list[EvidenceCard],
) -> list[SourceRecordV2]:
    evidence_source_ids = {card.source_id for card in evidence_cards}
    return [source for source in sources if source.id in evidence_source_ids]


def _blocked_source_ids_from_failures(failures: list[str]) -> set[int]:
    blocked: set[int] = set()
    for failure in failures:
        if not re.search(r"\bfails current branch/request alignment\b", failure, flags=re.I):
            continue
        blocked.update(int(value) for value in re.findall(r"Cited source \[([0-9]+)]", failure, flags=re.I))
    return blocked


def _without_blocked_sources(evidence_cards: list[EvidenceCard], blocked_source_ids: set[int]) -> list[EvidenceCard]:
    if not blocked_source_ids:
        return evidence_cards
    return [card for card in evidence_cards if card.source_id not in blocked_source_ids]


def _cards_for_synthesis(
    plan: ResearchPlan,
    evidence_cards: list[EvidenceCard],
    *,
    max_total: int = MAX_SYNTHESIS_CARDS_TOTAL,
    max_per_branch: int = MAX_SYNTHESIS_CARDS_PER_BRANCH,
) -> list[EvidenceCard]:
    if not evidence_cards:
        return []
    cards_by_branch = _cards_by_branch(evidence_cards)
    selected: list[EvidenceCard] = []
    selected_ids: set[int] = set()
    branch_count = max(len(plan.branches), 1)
    per_branch_limit = max(2, min(max_per_branch, max_total // branch_count))
    for branch in plan.branches:
        branch_cards = sorted(
            cards_by_branch.get(branch.id, []),
            key=lambda item: _card_rank_key(item, question=plan.question),
        )
        for card in _source_diverse_cards(branch_cards, limit=per_branch_limit):
            if len(selected) >= max_total:
                return selected
            if card.id in selected_ids:
                continue
            selected.append(card)
            selected_ids.add(card.id)

    if len(selected) >= max_total:
        return selected[:max_total]

    remaining = [card for card in evidence_cards if card.id not in selected_ids]
    selected.extend(
        _marginal_coverage_cards(
            plan=plan,
            candidates=remaining,
            selected=selected,
            limit=max_total - len(selected),
        )
    )
    return selected[:max_total]


def _marginal_coverage_cards(
    *,
    plan: ResearchPlan,
    candidates: list[EvidenceCard],
    selected: list[EvidenceCard],
    limit: int,
) -> list[EvidenceCard]:
    if limit <= 0 or not candidates:
        return []

    branch_terms = _plan_branch_terms(plan)
    global_terms = _plan_global_terms(plan, branch_terms)
    card_terms = {card.id: _card_terms(card) for card in candidates + selected}
    covered_global_terms: set[str] = set()
    covered_branch_terms: dict[str, set[str]] = defaultdict(set)
    selected_term_sets: list[set[str]] = []
    branch_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[int, int] = defaultdict(int)

    for card in selected:
        terms = card_terms[card.id]
        covered_global_terms.update(terms & global_terms)
        covered_branch_terms[card.branch_id].update(terms & branch_terms.get(card.branch_id, set()))
        selected_term_sets.append(terms)
        branch_counts[card.branch_id] += 1
        source_counts[card.source_id] += 1

    remaining = list(candidates)
    chosen: list[EvidenceCard] = []
    while remaining and len(chosen) < limit:
        best_index = 0
        best_score: tuple[float, float, float, int] | None = None
        for index, card in enumerate(remaining):
            terms = card_terms[card.id]
            global_gain = len(terms & (global_terms - covered_global_terms))
            branch_gain = len(terms & (branch_terms.get(card.branch_id, set()) - covered_branch_terms[card.branch_id]))
            new_source_bonus = 1.0 if source_counts[card.source_id] == 0 else 1.0 / (source_counts[card.source_id] + 2)
            branch_balance_bonus = 1.0 / (branch_counts[card.branch_id] + 1)
            redundancy_penalty = _max_term_overlap(terms, selected_term_sets)
            relevance = _card_relevance_score(card, question=plan.question)
            score = (
                (branch_gain * 4.0)
                + (global_gain * 3.0)
                + (new_source_bonus * 1.25)
                + (branch_balance_bonus * 0.75)
                + relevance
                - (redundancy_penalty * 1.5),
                relevance,
                card.confidence,
                -card.id,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_index = index

        card = remaining.pop(best_index)
        terms = card_terms[card.id]
        chosen.append(card)
        covered_global_terms.update(terms & global_terms)
        covered_branch_terms[card.branch_id].update(terms & branch_terms.get(card.branch_id, set()))
        selected_term_sets.append(terms)
        branch_counts[card.branch_id] += 1
        source_counts[card.source_id] += 1

    return chosen


def _plan_branch_terms(plan: ResearchPlan) -> dict[str, set[str]]:
    return {
        branch.id: content_terms(
            " ".join(
                [
                    branch.title,
                    branch.objective,
                    " ".join(branch.queries),
                    " ".join(branch.required_terms),
                    " ".join(branch.completion_criteria),
                ]
            )
        )
        for branch in plan.branches
    }


def _plan_global_terms(plan: ResearchPlan, branch_terms: dict[str, set[str]]) -> set[str]:
    terms = content_terms(
        " ".join(
            [
                plan.question,
                " ".join(plan.report_outline),
                " ".join(plan.acceptance_criteria),
                " ".join(requirement.source_type + " " + requirement.rationale for requirement in plan.source_requirements),
            ]
        )
    )
    for values in branch_terms.values():
        terms.update(values)
    return terms


def _card_terms(card: EvidenceCard) -> set[str]:
    return content_terms(" ".join([card.claim, card.supporting_excerpt, card.source_title, " ".join(card.limitations)]))


def _card_relevance_score(card: EvidenceCard, *, question: str) -> float:
    return (
        _question_phrase_score(question, f"{card.claim} {card.supporting_excerpt} {card.source_title}") * 2.0
        + _question_term_score(question, f"{card.claim} {card.supporting_excerpt} {card.source_title}")
        + card.confidence
        + card.quality_score
        + card.relevance_score
        + (card.semantic_score or 0.0)
    )


def _max_term_overlap(terms: set[str], selected_term_sets: list[set[str]]) -> float:
    if not terms or not selected_term_sets:
        return 0.0
    return max(
        (len(terms & selected_terms) / max(len(terms | selected_terms), 1) for selected_terms in selected_term_sets),
        default=0.0,
    )


def _source_diverse_cards(cards: list[EvidenceCard], *, limit: int) -> list[EvidenceCard]:
    selected: list[EvidenceCard] = []
    seen_sources: set[int] = set()
    for card in cards:
        if len(selected) >= limit:
            break
        if card.source_id in seen_sources:
            continue
        selected.append(card)
        seen_sources.add(card.source_id)
    if len(selected) < limit:
        selected_ids = {card.id for card in selected}
        for card in cards:
            if len(selected) >= limit:
                break
            if card.id in selected_ids:
                continue
            selected.append(card)
    return selected


def _rank_cards(cards: list[EvidenceCard], *, question: str = "") -> list[EvidenceCard]:
    return sorted(
        cards,
        key=lambda item: _card_rank_key(item, question=question),
    )


def _card_rank_key(card: EvidenceCard, *, question: str = "") -> tuple[float, float, float, float, float, int]:
    card_text = f"{card.claim} {card.supporting_excerpt} {card.source_title}"
    return (
        -_question_phrase_score(question, card_text),
        -_question_term_score(question, card_text),
        -card.confidence,
        -(card.semantic_score or 0.0),
        -card.quality_score,
        card.id,
    )


def _question_phrase_score(question: str, text: str) -> float:
    question_terms = content_terms(question)
    if len(question_terms) < 2:
        return _question_term_score(question, text)
    ordered_question_terms = [term for term in question.lower().replace("-", " ").split() if term in question_terms]
    if len(ordered_question_terms) < 2:
        ordered_question_terms = sorted(question_terms)
    text_terms = content_terms(text)
    phrase_terms: set[str] = set()
    for size in (3, 2):
        for index in range(0, max(0, len(ordered_question_terms) - size + 1)):
            window = ordered_question_terms[index : index + size]
            if all(term in text_terms for term in window):
                phrase_terms.update(window)
    return round(len(phrase_terms) / max(len(question_terms), 1), 4)


def _question_term_score(question: str, text: str) -> float:
    question_terms = content_terms(question)
    if not question_terms:
        return 1.0
    text_terms = content_terms(text)
    return round(len(question_terms & text_terms) / max(len(question_terms), 1), 4)


def _opening_cards(plan: ResearchPlan, evidence_cards: list[EvidenceCard]) -> list[EvidenceCard]:
    ranked = _cards_for_synthesis(plan, evidence_cards)
    if not ranked:
        return []
    question_terms = content_terms(plan.question)
    if not question_terms:
        return ranked
    if len(question_terms) <= 3:
        threshold = 0.67
    elif len(question_terms) <= 8:
        threshold = 0.50
    else:
        threshold = 0.35
    direct = [
        card
        for card in ranked
        if _question_term_score(plan.question, f"{card.claim} {card.supporting_excerpt} {card.source_title}") >= threshold
    ]
    return direct + [card for card in ranked if card not in direct]



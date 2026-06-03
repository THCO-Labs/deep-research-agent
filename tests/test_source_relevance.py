from deep_research.source_relevance import score_source_relevance


def test_source_relevance_scores_matching_source_higher_than_irrelevant_source() -> None:
    relevant = score_source_relevance(
        query="fine tuning model adaptation",
        title="Fine tuning model adaptation guide",
        snippet="Fine tuning adapts pretrained models to new downstream tasks.",
        url="https://example.com/fine-tuning-adaptation",
    )
    irrelevant = score_source_relevance(
        query="fine tuning model adaptation",
        title="Billing API documentation",
        snippet="Invoices, payments, plans, and subscription lifecycle documentation.",
        url="https://docs.example.com/billing",
    )

    assert relevant.score > irrelevant.score
    assert relevant.matched_terms == ["adaptation", "fine", "model", "tuning"]
    assert "adaptation" in irrelevant.missing_terms


def test_source_relevance_can_use_scraped_markdown_after_search() -> None:
    relevance = score_source_relevance(
        query="retrieval augmented generation grounding",
        title="RAG overview",
        markdown="Retrieval augmented generation grounds model responses in retrieved documents.",
    )

    assert relevance.score > 0.5
    assert "retrieval" in relevance.matched_terms

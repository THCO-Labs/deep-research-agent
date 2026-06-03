from deep_research.source_relevance import score_source_relevance


def test_source_relevance_scores_matching_source_higher_than_irrelevant_source() -> None:
    relevant = score_source_relevance(
        query="urban heat island public health",
        title="Urban heat island public health guide",
        snippet="Urban heat islands increase heat exposure and can affect vulnerable populations.",
        url="https://example.com/urban-heat-public-health",
    )
    irrelevant = score_source_relevance(
        query="urban heat island public health",
        title="Billing API documentation",
        snippet="Invoices, payments, plans, and subscription lifecycle documentation.",
        url="https://docs.example.com/billing",
    )

    assert relevant.score > irrelevant.score
    assert relevant.matched_terms == ["health", "heat", "island", "public", "urban"]
    assert "urban" in irrelevant.missing_terms


def test_source_relevance_can_use_scraped_markdown_after_search() -> None:
    relevance = score_source_relevance(
        query="retrieval augmented generation grounding",
        title="RAG overview",
        markdown="Retrieval augmented generation grounds model responses in retrieved documents.",
    )

    assert relevance.score > 0.5
    assert "retrieval" in relevance.matched_terms

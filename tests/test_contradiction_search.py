from deep_research.contradiction_search import generate_contradiction_queries


def test_generate_contradiction_queries_is_bounded_and_generic() -> None:
    claims = [
        {
            "id": "claim_1",
            "branch_id": "branch_1",
            "claim": "Spring Boot improves deployment speed for Java teams.",
            "high_impact": True,
            "weak": False,
            "support_count": 2,
            "average_confidence": 0.8,
        }
    ]

    queries = generate_contradiction_queries(claims, question="Java architecture evolution", limit=3)

    assert len(queries) == 2
    assert queries[0].branch_id == "branch_1"
    assert queries[0].purpose == "limitation"
    assert all("DMG" not in query.query for query in queries)


def test_generate_contradiction_queries_uses_lexical_expansion(monkeypatch) -> None:
    monkeypatch.setattr(
        "deep_research.contradiction_search.expand_terms",
        lambda _terms, max_per_term=4: {"constraint"},
    )
    claims = [
        {
            "id": "claim_1",
            "branch_id": "branch_1",
            "claim": "The architecture improves maintainability.",
            "high_impact": True,
            "weak": False,
            "support_count": 2,
            "average_confidence": 0.8,
        }
    ]

    queries = generate_contradiction_queries(claims, question="Architecture review", limit=1)

    assert "constraint" in queries[0].query.lower()

from deep_research import text_terms


def test_stopwords_use_sklearn_when_available() -> None:
    text_terms.english_stopwords.cache_clear()

    stopwords = text_terms.english_stopwords()

    assert text_terms.stopword_source() == "sklearn"
    assert len(stopwords) > 100
    assert "the" in stopwords
    assert "and" in stopwords


def test_ordered_terms_remove_library_stopwords() -> None:
    terms = text_terms.ordered_terms(
        "What is the relationship between Mediterranean diets and hypertension?"
    )

    assert terms == ["relationship", "mediterranean", "diets", "hypertension"]

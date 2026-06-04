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


def test_ordered_terms_support_chinese_prompts_without_collapsing_to_empty() -> None:
    terms = text_terms.ordered_terms("请为我调研中性粒细胞在脑缺血急性期和慢性期的功能变化")

    assert terms
    assert any("中性" in term or "粒细胞" in term or "脑缺血" in term for term in terms)
    assert text_terms.contains_cjk("中性粒细胞")

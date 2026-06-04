from deep_research import text_terms


def test_stopwords_use_sklearn_when_available() -> None:
    text_terms.english_stopwords.cache_clear()

    stopwords = text_terms.english_stopwords()

    assert text_terms.stopword_source() == "sklearn"
    assert len(stopwords) > 100
    assert "the" in stopwords
    assert "and" in stopwords


def test_preferred_output_language_uses_script_balance_not_topic_keywords() -> None:
    chinese_question = "\u8bf7\u5206\u6790\u57ce\u5e02\u70ed\u5c9b\u5982\u4f55\u5f71\u54cd\u516c\u5171\u5065\u5eb7"

    assert text_terms.preferred_output_language(chinese_question) == "zh"
    assert text_terms.preferred_output_language("How do urban heat islands affect public health?") == "en"


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


def test_chinese_fallback_keeps_content_chunks_before_recall_ngrams() -> None:
    chunks = text_terms.cjk_content_chunks("请为我提供一份详尽的报告，分析中性粒细胞在脑缺血急性期的功能变化")

    assert "中性粒细胞" in chunks
    assert any("脑缺血急性期" in chunk for chunk in chunks)
    assert "请为我" not in chunks
    assert "报告" not in chunks

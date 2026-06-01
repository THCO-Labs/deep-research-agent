from deep_research.source_quality import score_source


def test_source_quality_prefers_government_and_academic_sources() -> None:
    gov = score_source(url="https://nist.gov/report", title="Technical report")
    blog = score_source(url="https://medium.com/@writer/what-is-ai", title="What is AI?")

    assert gov.score > blog.score
    assert gov.label in {"strong", "excellent"}
    assert blog.source_type == "user_content"


def test_source_quality_marks_official_docs() -> None:
    quality = score_source(
        url="https://docs.python.org/3/tutorial/",
        title="Python documentation",
        snippet="Official documentation",
    )

    assert quality.source_type == "official_docs"
    assert quality.score >= 0.7

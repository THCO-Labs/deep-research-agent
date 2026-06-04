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


def test_source_quality_recognizes_multilateral_and_public_sector_sources() -> None:
    who = score_source(url="https://www.who.int/news-room/fact-sheets/detail/hypertension")
    gov_uk = score_source(url="https://www.gov.uk/government/publications/example")

    assert who.source_type == "government"
    assert gov_uk.source_type == "government"
    assert who.label in {"strong", "excellent"}
    assert gov_uk.label in {"strong", "excellent"}


def test_source_quality_recognizes_expanded_scholarly_publishers_and_repositories() -> None:
    sage = score_source(
        url="https://journals.sagepub.com/doi/10.1177/example",
        title="Peer-reviewed article",
    )
    medrxiv = score_source(url="https://www.medrxiv.org/content/10.1101/example")
    academic_suffix = score_source(url="https://www.ox.ac.uk/research/example")

    assert sage.source_type == "academic"
    assert sage.label in {"strong", "excellent"}
    assert medrxiv.source_type == "academic"
    assert medrxiv.label == "strong"
    assert any("preprint" in reason for reason in medrxiv.reasons)
    assert academic_suffix.source_type == "academic"


def test_source_quality_treats_pubmed_and_pmc_records_as_scholarly_not_government() -> None:
    pubmed = score_source(
        url="https://pubmed.ncbi.nlm.nih.gov/24094278",
        title="Need for closure and heuristic information processing",
    )
    pmc = score_source(
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11411052",
        title="Cogn Res Princ Implic. doi: 10.1186/s41235-024-00595-1",
    )

    assert pubmed.source_type == "academic"
    assert pmc.source_type == "academic"


def test_source_quality_recognizes_more_standards_and_docs_hosts() -> None:
    standard = score_source(url="https://tc39.es/ecma262/", title="ECMAScript specification")
    docs = score_source(
        url="https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
        title="Amazon S3 documentation",
    )

    assert standard.source_type == "standards_or_government"
    assert standard.label == "excellent"
    assert docs.source_type == "official_docs"
    assert docs.label in {"strong", "excellent"}


def test_source_quality_keeps_user_content_and_software_repositories_distinct() -> None:
    repository = score_source(url="https://github.com/langchain-ai/langchain")
    answer = score_source(url="https://stackoverflow.com/questions/1/example")

    assert repository.source_type == "software_repository"
    assert repository.label == "usable"
    assert answer.source_type == "user_content"
    assert answer.label == "weak"

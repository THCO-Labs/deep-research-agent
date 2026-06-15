from deep_research.schemas import ResearchBranch
from deep_research.source_validation import validate_source_content, validation_policy_for_source


def test_product_spec_policy_accepts_short_official_spec_content() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Machine specifications and integration",
        objective="Compare the models for spindle power, control software, and integration.",
        queries=["compare model 5000 model 7000 technical specifications"],
        min_sources=1,
        required_terms=["spindle power", "control software"],
    )
    content = (
        "Model 5000\n"
        "Max spindle speed: 12,000 rpm\n"
        "Main motor: 26 kW\n"
        "X-axis stroke: 500 mm\n"
        "Control software: OPC UA connectivity and production monitoring.\n"
        "The machine page lists bar feeder compatibility, tool capacity, coolant options, "
        "machine monitoring, and Ethernet integration for shop-floor data collection.\n"
    )

    result = validate_source_content(
        title="Model 5000 technical specifications",
        content=content,
        branch=branch,
        min_words=350,
        min_relevant_chunks=1,
        question="Compare Model 5000 and Model 7000 for hardware specifications and integration.",
        source_type="product_page",
        url="https://manufacturer.example.com/products/machines/model-5000",
        extraction_method="httpx",
    )

    assert result.usable
    assert result.word_count < 350
    assert result.relevant_chunk_count >= 1


def test_literature_review_policy_keeps_short_vendor_content_strict() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Clinical evidence",
        objective="Review randomized clinical evidence and meta-analyses.",
        queries=["clinical trial evidence systematic review"],
        min_sources=1,
        required_terms=["clinical trial", "systematic review"],
    )
    content = (
        "Model 5000\n"
        "Max spindle speed: 12,000 rpm\n"
        "Main motor: 26 kW\n"
        "X-axis stroke: 500 mm\n"
    )

    result = validate_source_content(
        title="Model 5000 technical specifications",
        content=content,
        branch=branch,
        min_words=350,
        min_relevant_chunks=1,
        question="Write a literature review of randomized clinical evidence.",
        source_type="product_page",
        url="https://manufacturer.example.com/products/machines/model-5000",
        extraction_method="httpx",
    )

    assert not result.usable
    assert validation_policy_for_source(
        title="Model 5000 technical specifications",
        content=content,
        branch=branch,
        question="Write a literature review of randomized clinical evidence.",
        source_type="product_page",
        url="https://manufacturer.example.com/products/machines/model-5000",
        extraction_method="httpx",
    ) == "default"


def test_product_spec_policy_rejects_typed_source_without_technical_evidence() -> None:
    branch = ResearchBranch(
        id="branch_1",
        title="Device comparison",
        objective="Compare models for features and compatibility.",
        queries=["compare model 5000 model 7000 features"],
        min_sources=1,
        required_terms=["features", "compatibility"],
    )

    result = validate_source_content(
        title="Model 5000 overview",
        content="Welcome to our product page. Contact sales for details. Related articles and promotions.",
        branch=branch,
        min_words=40,
        min_relevant_chunks=1,
        question="Compare Model 5000 and Model 7000 features and compatibility.",
        source_type="product_page",
        url="https://manufacturer.example.com/products/model-5000",
        extraction_method="httpx",
    )

    assert not result.usable
    assert "product/spec source lacks matching entity and technical field evidence" in result.reasons

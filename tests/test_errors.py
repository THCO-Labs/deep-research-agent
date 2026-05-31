from deep_research.errors import classify_exception


def test_classify_google_quota_error_extracts_retry_after() -> None:
    exc = RuntimeError(
        "429 RESOURCE_EXHAUSTED. {'error': {'message': 'Quota exceeded. Please retry in 42.156s.'}}"
    )

    failure = classify_exception(exc)

    assert failure.category == "quota_or_rate_limit"
    assert failure.retryable is True
    assert failure.retry_after_seconds == 42


def test_classify_google_retry_info_extracts_retry_delay_field() -> None:
    exc = RuntimeError("{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '42s'}")

    failure = classify_exception(exc)

    assert failure.retry_after_seconds == 42


def test_classify_groq_tpm_error_as_token_budget() -> None:
    exc = RuntimeError(
        "Error code: 413 - Request too large on tokens per minute (TPM): Limit 8000, Requested 8184"
    )

    failure = classify_exception(exc)

    assert failure.category == "token_budget_exceeded"
    assert failure.retryable is False
    assert "DEEP_RESEARCH_TOOL_EXCERPT_CHAR_LIMIT" in failure.suggested_action


def test_classify_tool_call_parse_error() -> None:
    exc = RuntimeError("Failed to parse tool call arguments as JSON: tool_use_failed")

    failure = classify_exception(exc)

    assert failure.category == "tool_call_parse_error"
    assert failure.retryable is False

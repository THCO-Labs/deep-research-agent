from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FailureClassification:
    category: str
    retryable: bool
    retry_after_seconds: int | None
    suggested_action: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_exception(exc: BaseException) -> FailureClassification:
    message = str(exc)
    lower = message.lower()
    retry_after = _extract_retry_after_seconds(message)

    if _is_token_budget_error(lower):
        return FailureClassification(
            category="token_budget_exceeded",
            retryable=False,
            retry_after_seconds=retry_after,
            suggested_action=(
                "Reduce --max-sources, lower DEEP_RESEARCH_TOOL_EXCERPT_CHAR_LIMIT, "
                "or use a model/provider tier with a larger token-per-minute budget."
            ),
            error_type=type(exc).__name__,
            message=message,
        )
    if _is_quota_error(lower):
        return FailureClassification(
            category="quota_or_rate_limit",
            retryable=retry_after is not None,
            retry_after_seconds=retry_after,
            suggested_action=(
                "Wait for the provider retry window or route affected roles to a different provider/key pool."
            ),
            error_type=type(exc).__name__,
            message=message,
        )
    if _is_provider_timeout_error(lower):
        return FailureClassification(
            category="provider_timeout",
            retryable=True,
            retry_after_seconds=retry_after,
            suggested_action="Retry the request or route affected roles to a different provider/key pool.",
            error_type=type(exc).__name__,
            message=message,
        )
    if _is_tool_call_error(lower):
        return FailureClassification(
            category="tool_call_parse_error",
            retryable=False,
            retry_after_seconds=None,
            suggested_action=(
                "Use a tool-call-capable model for tool-using roles or switch to a provider profile "
                "that suppresses incompatible internal tools."
            ),
            error_type=type(exc).__name__,
            message=message,
        )
    if _is_auth_error(lower):
        return FailureClassification(
            category="auth_or_permission",
            retryable=False,
            retry_after_seconds=None,
            suggested_action="Check API keys, provider permissions, and requested model access.",
            error_type=type(exc).__name__,
            message=message,
        )
    return FailureClassification(
        category="unknown",
        retryable=False,
        retry_after_seconds=retry_after,
        suggested_action="Inspect error.txt, transcript.log, activity.md, and provider dashboards.",
        error_type=type(exc).__name__,
        message=message,
    )


def _is_token_budget_error(lower_message: str) -> bool:
    tokens = (
        "request too large",
        "tokens per minute",
        "tpm",
        "context length",
        "maximum context",
        "413",
    )
    return any(token in lower_message for token in tokens)


def _is_quota_error(lower_message: str) -> bool:
    tokens = (
        "resource_exhausted",
        "rate_limit",
        "rate limit",
        "quota",
        "usage limit",
        "set usage limit",
        "plan's set usage limit",
        "429",
    )
    return any(token in lower_message for token in tokens)


def _is_provider_timeout_error(lower_message: str) -> bool:
    tokens = (
        "deadline_exceeded",
        "deadline exceeded",
        "request timed out",
        "timed out",
        "timeout",
        "504",
        "503",
        "service unavailable",
        "temporarily unavailable",
    )
    return any(token in lower_message for token in tokens)


def _is_tool_call_error(lower_message: str) -> bool:
    tokens = (
        "tool_use_failed",
        "failed to parse tool call",
        "invalid tool",
        "tool call",
    )
    return any(token in lower_message for token in tokens)


def _is_auth_error(lower_message: str) -> bool:
    tokens = (
        "invalid api key",
        "unauthorized",
        "permission denied",
        "401",
        "403",
    )
    return any(token in lower_message for token in tokens)


def _extract_retry_after_seconds(message: str) -> int | None:
    patterns = (
        r"retry(?:Delay| delay)?['\"]?\s*[:=]\s*['\"]?(\d+)s",
        r"retry in\s+(\d+(?:\.\d+)?)s",
        r"retry after\s+(\d+(?:\.\d+)?)\s*seconds?",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return max(1, round(float(match.group(1))))
    return None

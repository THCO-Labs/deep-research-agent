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
        if retry_after is not None and _is_retry_window_error(lower):
            return FailureClassification(
                category="quota_or_rate_limit",
                retryable=True,
                retry_after_seconds=retry_after,
                suggested_action=(
                    "Wait for the provider retry window or route affected roles to a different provider/key pool."
                ),
                error_type=type(exc).__name__,
                message=message,
            )
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
    if _is_network_error(exc, lower):
        return FailureClassification(
            category="network_error",
            retryable=True,
            retry_after_seconds=retry_after,
            suggested_action=(
                "Transient network failure (DNS/connection). The fallback chain will try other keys/providers."
            ),
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


def _is_retry_window_error(lower_message: str) -> bool:
    tokens = (
        "rate limit reached",
        "please try again",
        "try again in",
        "retry in",
        "retry after",
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
        # Our own wall-clock watchdogs (semantic.JudgeTimeoutError,
        # synthesis.SynthesisTimeoutError) raise messages containing these.
        "did not return within",
        "wall-clock budget",
        "abandoning thread",
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


def _is_network_error(exc: BaseException, lower_message: str) -> bool:
    """Detect transient network/DNS/connection failures that should trigger fallback.

    Catches: DNS resolution (getaddrinfo, gaierror, NXDOMAIN), connection reset/refused,
    SSL handshake failures, socket errors, httpx ConnectError, and Windows error 11001
    (DNS) / 10054 (reset) / 10061 (refused).
    """
    tokens = (
        "getaddrinfo failed",
        "name or service not known",
        "nodename nor servname",
        "temporary failure in name resolution",
        "no address associated with hostname",
        "connection reset",
        "connection refused",
        "connection aborted",
        "connection error",
        "connecterror",
        "remote end closed connection",
        "ssl handshake",
        "ssl: certificate",
        "certificate verify failed",
        "max retries exceeded",
        "errno 11001",
        "errno 10054",
        "errno 10061",
        "winerror 11001",
        "winerror 10061",
        "winerror 10054",
        "actively refused",
        "target machine actively refused",
        "[errno -2]",  # POSIX gaierror
        "[errno -3]",
    )
    if any(token in lower_message for token in tokens):
        return True
    # Type-based detection as backup (covers cases where the str() repr is opaque).
    exc_type_name = type(exc).__name__.lower()
    type_tokens = ("gaierror", "connecterror", "connectionerror", "sslerror", "remotedisconnected")
    return any(token in exc_type_name for token in type_tokens)


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
        r"try again in\s+(\d+(?:\.\d+)?)s",
        r"retry in\s+(\d+(?:\.\d+)?)s",
        r"retry after\s+(\d+(?:\.\d+)?)\s*seconds?",
        # OpenRouter and other providers return retry_after_seconds in JSON error bodies.
        r'"retry_after_seconds"\s*:\s*(\d+(?:\.\d+)?)',
        r'"retry_after_seconds_raw"\s*:\s*(\d+(?:\.\d+)?)',
        r'"retry-after"\s*:\s*"?(\d+(?:\.\d+)?)"?',
        r'"Retry-After"\s*:\s*"?(\d+(?:\.\d+)?)"?',
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return max(1, round(float(match.group(1))))
    return None

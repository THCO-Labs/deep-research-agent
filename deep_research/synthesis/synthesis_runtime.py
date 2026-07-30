from __future__ import annotations

import threading
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from deep_research.core.schemas import ResearchPlan
from deep_research.core.settings import GOOGLE_DEFAULT_MODEL, Settings
from deep_research.synthesis.synthesis_formatting import _criteria_rich_plan

LOW_TPM_PROVIDER_BUDGET_TOKENS = 7_600
MIN_SYNTHESIS_COMPLETION_TOKENS = 768


def _synthesis_model_spec(settings: Settings, plan: ResearchPlan, writing_guidance: str = "") -> str:
    # Explicit synthesis_model override takes priority — allows routing the writer
    # to a stronger model than the general orchestrator.
    if getattr(settings, "synthesis_model", ""):
        return settings.synthesis_model
    if not _criteria_rich_plan(plan, writing_guidance=writing_guidance):
        return settings.model
    provider, _, _model = settings.model.partition(":")
    if provider == "google_genai":
        return settings.model
    if settings.provider == "hybrid" and settings.google_key_pool:
        return GOOGLE_DEFAULT_MODEL
    return settings.model


class SynthesisTimeoutError(RuntimeError):
    """Raised when a synthesis LLM call exceeds the wall-clock budget."""


def _invoke_with_synthesis_budget(
    model: BaseChatModel,
    *,
    prompt: str,
    settings: Settings,
    model_spec: str,
) -> Any:
    kwargs = _synthesis_request_kwargs(settings=settings, prompt=prompt, model_spec=model_spec)
    timeout_s = _synthesis_wall_clock_timeout(settings)

    def _invoke() -> Any:
        try:
            return model.invoke([HumanMessage(content=prompt)], **kwargs)
        except TypeError as exc:
            if kwargs and "unexpected" in str(exc).lower() and "keyword" in str(exc).lower():
                return model.invoke([HumanMessage(content=prompt)])
            raise

    return _call_with_wall_clock_timeout(_invoke, timeout_s=timeout_s, label=f"synthesis ({model_spec})")


def _synthesis_wall_clock_timeout(settings: Settings) -> float:
    """Wall-clock timeout for a single synthesis LLM call.

    Library-level timeouts (httpx, langchain) are not always honoured for
    streaming responses or hung TCP sockets. This independent wall clock
    guarantees that a frozen call surfaces as an exception, which the
    synthesize() node catches and degrades to deterministic synthesis.
    """
    base = float(getattr(settings, "model_request_timeout_seconds", 120) or 120)
    # Give synthesis 1.5x the per-call timeout — large reports legitimately
    # take 2-5 minutes on Mistral Large. Hard cap at 15 minutes regardless.
    return min(900.0, max(180.0, base * 1.5))


def _call_with_wall_clock_timeout(
    fn: Any,
    *,
    timeout_s: float,
    label: str,
) -> Any:
    """Run fn() in a daemon thread; raise SynthesisTimeoutError if it doesn't
    return within timeout_s. The hung thread is abandoned — it dies with the
    process. This is correct for HTTP calls: the worst case is one leaked
    socket per timeout, recovered when the process exits.
    """
    result_holder: list[Any] = []
    error_holder: list[BaseException] = []

    def _runner() -> None:
        try:
            result_holder.append(fn())
        except BaseException as exc:
            error_holder.append(exc)

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    if thread.is_alive():
        raise SynthesisTimeoutError(
            f"{label} did not return within {timeout_s:.0f}s wall-clock budget; "
            f"abandoning thread and falling back."
        )
    if error_holder:
        raise error_holder[0]
    if result_holder:
        return result_holder[0]
    raise SynthesisTimeoutError(f"{label} returned no result and no error.")


def _synthesis_request_kwargs(*, settings: Settings, prompt: str, model_spec: str) -> dict[str, int]:
    provider, _separator, _model_name = model_spec.partition(":")
    if provider == "groq":
        prompt_tokens = _rough_token_count(prompt)
        requested_completion_tokens = min(
            settings.model_max_output_tokens,
            max(MIN_SYNTHESIS_COMPLETION_TOKENS, LOW_TPM_PROVIDER_BUDGET_TOKENS - prompt_tokens),
        )
        return {"max_tokens": requested_completion_tokens}
    if provider == "mistral_ai":
        # Cap Mistral output so the streaming response can't hang indefinitely.
        # 8000 tokens ≈ 6000 words — plenty for a long structured report.
        return {"max_tokens": min(settings.model_max_output_tokens or 8000, 8000)}
    return {}


def _rough_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)



from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeAlias

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
import httpx

from deep_research.core.errors import classify_exception
from deep_research.core.settings import OPENROUTER_DEFAULT_MODEL, Settings

ModelLike: TypeAlias = str | BaseChatModel
FallbackCallback: TypeAlias = Callable[[str], None]
RetryCallback: TypeAlias = Callable[[str], None]

FALLBACK_ERROR_CATEGORIES = {
    "quota_or_rate_limit",
    "provider_timeout",
    "token_budget_exceeded",
    "tool_call_parse_error",
    "network_error",
}

_ROLE_KEY_INDEX = {
    "orchestrator": 0,
    "researcher": 1,
    "planner": 2,
    "verifier": 3,
    "analyst": 4,
    "judge": 5,
    "fast": 6,
    "citation": 3,
    "synthesis": 0,
}

_ROLE_MODEL_ATTRS = {
    "orchestrator": "model",
    "planner": "planner_model",
    "researcher": "researcher_model",
    "analyst": "analyst_model",
    "verifier": "verifier_model",
    "judge": "judge_model",
    "citation": "citation_model",
    "fast": "fast_model",
    "synthesis": "synthesis_model",
}

_usage_lock = threading.Lock()
_run_usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0}


def get_and_reset_token_usage() -> dict[str, int]:
    """Return accumulated token usage for the current run and reset to zero."""
    with _usage_lock:
        snapshot = dict(_run_usage)
        _run_usage.update({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "llm_calls": 0})
    return snapshot


def _accumulate_usage(result: Any) -> None:
    usage: dict | None = None
    if hasattr(result, "llm_output") and isinstance(result.llm_output, dict):
        usage = result.llm_output.get("usage") or result.llm_output.get("usage_metadata")
    if usage is None and hasattr(result, "generations"):
        for gen in result.generations or []:
            msg = getattr(gen, "message", None)
            if msg and getattr(msg, "usage_metadata", None):
                usage = msg.usage_metadata
                break
    if not isinstance(usage, dict):
        return
    with _usage_lock:
        _run_usage["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0))
        _run_usage["completion_tokens"] += int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0))
        _run_usage["total_tokens"] += int(usage.get("total_tokens", 0))
        _run_usage["llm_calls"] += 1


_KEY_ENV_BASE = {
    "groq": "GROQ_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral_ai": "MISTRAL_API_KEY",
    "together": "TOGETHER_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
}


@dataclass(frozen=True)
class RoutedAgentModels:
    orchestrator: ModelLike
    fast: ModelLike
    planner: ModelLike
    researcher: ModelLike
    analyst: ModelLike
    verifier: ModelLike


@dataclass(frozen=True)
class ModelRoute:
    role: str
    provider: str
    model: str
    model_spec: str
    key_count: int
    key_slot: int | None
    key_label: str | None
    fallback_routes: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FallbackChatModel(BaseChatModel):
    primary: Any
    fallbacks: tuple[Any, ...] = ()
    route_label: str = ""
    on_fallback: FallbackCallback | None = None
    on_retry: RetryCallback | None = None
    retry_attempts: int = 0
    retry_max_wait_seconds: int = 0

    @property
    def _llm_type(self) -> str:
        return "deep_research_fallback"

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        getter = getattr(self.primary, "_get_ls_params", None)
        if callable(getter):
            return getter(stop=stop, **kwargs)
        return super()._get_ls_params(stop=stop, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return _call_with_classified_fallbacks(
            self.primary,
            self.fallbacks,
            lambda model: model._generate(messages, stop=stop, run_manager=run_manager, **kwargs),
            route_label=self.route_label,
            on_fallback=self.on_fallback,
            on_retry=self.on_retry,
            retry_attempts=self.retry_attempts,
            retry_max_wait_seconds=self.retry_max_wait_seconds,
        )

    def bind_tools(
        self,
        tools: list[dict[str, Any] | type | Callable | Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, Any]:
        bound_primary = self.primary.bind_tools(tools, tool_choice=tool_choice, **kwargs)
        bound_fallbacks = tuple(
            fallback.bind_tools(tools, tool_choice=tool_choice, **kwargs)
            for fallback in self.fallbacks
        )
        return ClassifiedFallbackRunnable(
            primary=bound_primary,
            fallbacks=bound_fallbacks,
            route_label=self.route_label,
            on_fallback=self.on_fallback,
            on_retry=self.on_retry,
            retry_attempts=self.retry_attempts,
            retry_max_wait_seconds=self.retry_max_wait_seconds,
        )


class ChatOpenRouter(BaseChatModel):
    model: str
    api_key: str
    referer: str = ""
    app_title: str = "Deep Research Agent"
    timeout_seconds: float = 120.0
    max_tokens: int | None = None
    base_url: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

    @property
    def _llm_type(self) -> str:
        return "openrouter"

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {"ls_provider": "openrouter", "ls_model_name": self.model}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_openrouter(message) for message in messages],
        }
        if stop:
            payload["stop"] = stop
        if self.max_tokens and "max_tokens" not in kwargs:
            payload["max_tokens"] = self.max_tokens
        for key in ("temperature", "max_tokens", "top_p"):
            if key in kwargs and kwargs[key] is not None:
                payload[key] = kwargs[key]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer
        if self.app_title:
            headers["X-Title"] = self.app_title
            headers["X-OpenRouter-Title"] = self.app_title
        base_url = os.environ.get("OPENROUTER_BASE_URL") or self.base_url
        response = httpx.post(base_url, headers=headers, json=payload, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter request failed with HTTP {response.status_code}: {response.text[:1000]}")
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter response did not include choices: {str(data)[:1000]}")
        message = choices[0].get("message") or {}
        content = _openrouter_content_to_text(message.get("content", ""))
        response_metadata = {
            "model": data.get("model") or self.model,
            "id": data.get("id"),
            "finish_reason": choices[0].get("finish_reason"),
        }
        generation = ChatGeneration(message=AIMessage(content=content, response_metadata=response_metadata))
        return ChatResult(generations=[generation], llm_output={"usage": data.get("usage"), "model": data.get("model")})


class ClassifiedFallbackRunnable(Runnable[Any, Any]):
    def __init__(
        self,
        *,
        primary: Runnable[Any, Any],
        fallbacks: tuple[Runnable[Any, Any], ...],
        route_label: str,
        on_fallback: FallbackCallback | None = None,
        on_retry: RetryCallback | None = None,
        retry_attempts: int = 0,
        retry_max_wait_seconds: int = 0,
    ) -> None:
        self.primary = primary
        self.fallbacks = fallbacks
        self.route_label = route_label
        self.on_fallback = on_fallback
        self.on_retry = on_retry
        self.retry_attempts = retry_attempts
        self.retry_max_wait_seconds = retry_max_wait_seconds

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return _call_with_classified_fallbacks(
            self.primary,
            self.fallbacks,
            lambda runnable: runnable.invoke(input, config=config, **kwargs),
            route_label=self.route_label,
            on_fallback=self.on_fallback,
            on_retry=self.on_retry,
            retry_attempts=self.retry_attempts,
            retry_max_wait_seconds=self.retry_max_wait_seconds,
        )

    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await _acall_with_classified_fallbacks(
            self.primary,
            self.fallbacks,
            lambda runnable: runnable.ainvoke(input, config=config, **kwargs),
            route_label=self.route_label,
            on_fallback=self.on_fallback,
            on_retry=self.on_retry,
            retry_attempts=self.retry_attempts,
            retry_max_wait_seconds=self.retry_max_wait_seconds,
        )


def build_agent_models(
    settings: Settings,
    *,
    on_fallback: FallbackCallback | None = None,
    on_retry: RetryCallback | None = None,
) -> RoutedAgentModels:
    return RoutedAgentModels(
        orchestrator=model_for_role(settings, "orchestrator", settings.model, on_fallback=on_fallback, on_retry=on_retry),
        fast=model_for_role(settings, "fast", settings.fast_model, on_fallback=on_fallback, on_retry=on_retry),
        planner=model_for_role(settings, "planner", settings.planner_model, on_fallback=on_fallback, on_retry=on_retry),
        researcher=model_for_role(settings, "researcher", settings.researcher_model, on_fallback=on_fallback, on_retry=on_retry),
        analyst=model_for_role(settings, "analyst", settings.analyst_model, on_fallback=on_fallback, on_retry=on_retry),
        verifier=model_for_role(settings, "verifier", settings.verifier_model, on_fallback=on_fallback, on_retry=on_retry),
    )


def model_for_role(
    settings: Settings,
    role: str,
    model_spec: str,
    *,
    on_fallback: FallbackCallback | None = None,
    on_retry: RetryCallback | None = None,
) -> ModelLike:
    provider, model_name = _split_model_spec(model_spec)
    primary = _chat_model_for_role(settings, role, provider, model_name)
    if primary is not None:
        fallback_routes = _fallback_routes(settings, role, provider, model_name) if settings.model_fallbacks else []
        fallbacks = tuple(_chat_model_for_route(route) for route in fallback_routes)
        if fallbacks or settings.provider_retry_attempts > 0:
            return FallbackChatModel(
                primary=primary,
                fallbacks=fallbacks,
                route_label=f"{role} {provider}:{model_name}",
                on_fallback=on_fallback,
                on_retry=on_retry,
                retry_attempts=settings.provider_retry_attempts,
                retry_max_wait_seconds=settings.provider_retry_max_wait_seconds,
            )
        return primary
    return model_spec


def describe_model_routes(settings: Settings) -> dict[str, object]:
    routes = [_describe_role(settings, role) for role in _ROLE_MODEL_ATTRS]
    return {
        "provider": settings.provider,
        "model_fallbacks": settings.model_fallbacks,
        "provider_retry_attempts": settings.provider_retry_attempts,
        "provider_retry_max_wait_seconds": settings.provider_retry_max_wait_seconds,
        "model_request_timeout_seconds": settings.model_request_timeout_seconds,
        "model_max_output_tokens": settings.model_max_output_tokens,
        "google_key_count": len(settings.google_key_pool),
        "groq_key_count": len(settings.groq_key_pool),
        "openrouter_key_count": len(settings.openrouter_key_pool),
        "mistral_key_count": len(settings.mistral_key_pool),
        "roles": [route.to_dict() for route in routes],
    }


def route_summary(settings: Settings) -> str:
    visible_roles = ("orchestrator", "planner", "researcher", "analyst", "verifier", "judge", "citation", "synthesis")
    routes = [_describe_role(settings, role) for role in visible_roles]
    parts = []
    for route in routes:
        key = route.key_label or "no-key"
        fallback_count = len(route.fallback_routes) if settings.model_fallbacks else 0
        fallback_text = f" +{fallback_count} fallback(s)" if fallback_count else ""
        parts.append(f"{route.role}={route.provider}:{route.model} via {key}{fallback_text}")
    return "; ".join(parts)


def _key_for_role(keys: tuple[str, ...], role: str) -> str:
    if not keys:
        raise ValueError("API key pool cannot be empty.")
    return keys[_key_slot_for_role(keys, role)]


def _describe_role(settings: Settings, role: str) -> ModelRoute:
    model_spec = _model_spec_for_manifest_role(settings, role)
    provider, model_name = _split_model_spec(model_spec)
    keys = _key_pool_for_provider(settings, provider)
    key_slot = _key_slot_for_role(keys, role) if keys else None
    return ModelRoute(
        role=role,
        provider=provider or "unresolved",
        model=model_name,
        model_spec=model_spec,
        key_count=len(keys),
        key_slot=key_slot,
        key_label=_key_label(provider, key_slot),
        fallback_routes=[
            _route_to_dict(route)
            for route in _fallback_routes(settings, role, provider, model_name)
        ]
        if settings.model_fallbacks
        else [],
    )


def _model_spec_for_manifest_role(settings: Settings, role: str) -> str:
    attr = _ROLE_MODEL_ATTRS[role]
    model_spec = str(getattr(settings, attr) or "")
    if role == "citation" and not model_spec:
        return settings.verifier_model
    if role == "synthesis" and not model_spec:
        return settings.model
    return model_spec


def _key_pool_for_provider(settings: Settings, provider: str) -> tuple[str, ...]:
    if provider == "groq":
        return settings.groq_key_pool
    if provider == "google_genai":
        return settings.google_key_pool
    if provider == "openrouter":
        return settings.openrouter_key_pool
    if provider == "mistral_ai":
        return settings.mistral_key_pool
    if provider == "together":
        return settings.together_key_pool
    if provider == "deepseek":
        return settings.deepseek_key_pool
    if provider == "ollama":
        return ("ollama-local",)
    if provider == "azure_openai":
        return settings.azure_openai_key_pool
    return ()


def _key_slot_for_role(keys: tuple[str, ...], role: str) -> int:
    if not keys:
        raise ValueError("API key pool cannot be empty.")
    index = _ROLE_KEY_INDEX.get(role, 0)
    return index % len(keys)


def _key_label(provider: str, key_slot: int | None) -> str | None:
    if key_slot is None:
        return None
    base = _KEY_ENV_BASE.get(provider)
    if base is None:
        return None
    return base if key_slot == 0 else f"{base}{key_slot}"


def _chat_model_for_role(
    settings: Settings,
    role: str,
    provider: str,
    model_name: str,
) -> BaseChatModel | None:
    keys = _key_pool_for_provider(settings, provider)
    if not keys:
        return None
    key_slot = _key_slot_for_role(keys, role)
    return _chat_model_for_route(
        {
            "provider": provider,
            "model": model_name,
            "api_key": keys[key_slot],
            "referer": settings.openrouter_http_referer,
            "app_title": settings.openrouter_app_title,
            "request_timeout_seconds": settings.model_request_timeout_seconds,
            "max_output_tokens": settings.model_max_output_tokens,
            "azure_endpoint": settings.azure_openai_endpoint,
            "azure_api_version": settings.azure_openai_api_version,
        }
    )


def _chat_model_for_route(route: dict[str, object]) -> BaseChatModel:
    provider = str(route["provider"])
    model_name = str(route["model"])
    api_key = str(route["api_key"])
    timeout_seconds = float(route.get("request_timeout_seconds") or 120)
    max_output_tokens = int(route.get("max_output_tokens") or 0)
    if provider == "groq":
        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if max_output_tokens > 0:
            kwargs["max_tokens"] = max_output_tokens
        return ChatGroq(**kwargs)
    if provider == "google_genai":
        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "request_timeout": timeout_seconds,
            "retries": 0,
        }
        if max_output_tokens > 0:
            kwargs["max_output_tokens"] = max_output_tokens
        return ChatGoogleGenerativeAI(**kwargs)
    if provider == "openrouter":
        return ChatOpenRouter(
            model=model_name,
            api_key=api_key,
            referer=str(route.get("referer") or ""),
            app_title=str(route.get("app_title") or "Deep Research Agent"),
            timeout_seconds=timeout_seconds,
            max_tokens=max_output_tokens if max_output_tokens > 0 else None,
        )
    if provider == "mistral_ai":
        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if max_output_tokens > 0:
            kwargs["max_tokens"] = max_output_tokens
        return ChatMistralAI(**kwargs)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name)
    if provider == "together":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": model_name,
            "api_key": api_key,
            "base_url": "https://api.together.xyz/v1",
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if max_output_tokens > 0:
            kwargs["max_tokens"] = max_output_tokens
        return ChatOpenAI(**kwargs)
    if provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        kwargs = {
            "azure_deployment": model_name,
            "api_key": api_key,
            "azure_endpoint": str(route.get("azure_endpoint") or ""),
            "api_version": str(route.get("azure_api_version") or "2024-08-01-preview"),
            "timeout": timeout_seconds,
            "max_retries": 0,
        }
        if max_output_tokens > 0:
            kwargs["max_tokens"] = max_output_tokens
        return AzureChatOpenAI(**kwargs)
    raise ValueError(f"Unsupported model provider: {provider}")


def _fallback_routes(
    settings: Settings,
    role: str,
    provider: str,
    model_name: str,
) -> list[dict[str, object]]:
    if not settings.model_fallbacks:
        return []
    routes: list[dict[str, object]] = []
    keys = _key_pool_for_provider(settings, provider)
    primary_slot = _key_slot_for_role(keys, role) if keys else None
    for slot, api_key in enumerate(keys):
        if slot == primary_slot:
            continue
        routes.append(
            {
                "provider": provider,
                "model": model_name,
                "api_key": api_key,
                "key_slot": slot,
                "key_label": _key_label(provider, slot),
                "fallback_type": "same_provider_key",
                "referer": settings.openrouter_http_referer,
                "app_title": settings.openrouter_app_title,
                "request_timeout_seconds": settings.model_request_timeout_seconds,
                "max_output_tokens": settings.model_max_output_tokens,
            }
        )

    if settings.provider == "hybrid":
        alternate_provider, alternate_model = _alternate_hybrid_model(provider)
        alternate_keys = _key_pool_for_provider(settings, alternate_provider)
        if alternate_keys:
            alternate_slot = _key_slot_for_role(alternate_keys, role)
            routes.append(
                {
                    "provider": alternate_provider,
                    "model": alternate_model,
                    "api_key": alternate_keys[alternate_slot],
                    "key_slot": alternate_slot,
                    "key_label": _key_label(alternate_provider, alternate_slot),
                    "fallback_type": "cross_provider",
                    "request_timeout_seconds": settings.model_request_timeout_seconds,
                    "max_output_tokens": settings.model_max_output_tokens,
                }
            )
    if settings.openrouter_key_pool and provider != "openrouter":
        openrouter_slot = _key_slot_for_role(settings.openrouter_key_pool, role)
        openrouter_provider, openrouter_model = _split_model_spec(OPENROUTER_DEFAULT_MODEL)
        routes.append(
            {
                "provider": openrouter_provider,
                "model": openrouter_model,
                "api_key": settings.openrouter_key_pool[openrouter_slot],
                "key_slot": openrouter_slot,
                "key_label": _key_label("openrouter", openrouter_slot),
                "fallback_type": "free_provider",
                "referer": settings.openrouter_http_referer,
                "app_title": settings.openrouter_app_title,
                "request_timeout_seconds": settings.model_request_timeout_seconds,
                "max_output_tokens": settings.model_max_output_tokens,
            }
        )
    return routes


def _alternate_hybrid_model(provider: str) -> tuple[str, str]:
    if provider == "groq":
        return "google_genai", "gemini-2.5-flash"
    if provider == "google_genai":
        return "groq", "openai/gpt-oss-20b"
    if provider == "openrouter":
        return "google_genai", "gemini-2.5-flash"
    return provider, ""


def _route_to_dict(route: dict[str, object]) -> dict[str, object]:
    return {
        "provider": route["provider"],
        "model": route["model"],
        "key_slot": route["key_slot"],
        "key_label": route["key_label"],
        "fallback_type": route["fallback_type"],
    }


def _call_with_classified_fallbacks(
    primary: Any,
    fallbacks: tuple[Any, ...],
    call: Callable[[Any], Any],
    *,
    route_label: str,
    on_fallback: FallbackCallback | None,
    on_retry: RetryCallback | None,
    retry_attempts: int,
    retry_max_wait_seconds: int,
) -> Any:
    candidates = (primary, *fallbacks)
    last_error: Exception | None = None
    for retry_index in range(retry_attempts + 1):
        failures = []
        for index, candidate in enumerate(candidates):
            try:
                result = call(candidate)
                _accumulate_usage(result)
                return result
            except Exception as exc:
                last_error = exc
                failures.append(classify_exception(exc))
                if index == len(candidates) - 1:
                    break
                if not _should_try_fallback(exc):
                    raise
                _emit_fallback(on_fallback, route_label, exc, index + 1)

        wait_seconds = _retry_wait_seconds(
            failures,
            retry_index=retry_index + 1,
            retry_max_wait_seconds=retry_max_wait_seconds,
        )
        if retry_index < retry_attempts and wait_seconds is not None:
            _emit_retry(on_retry, route_label, wait_seconds, retry_index + 1, retry_attempts)
            time.sleep(wait_seconds)
            continue
        break
    if last_error is not None:
        raise last_error
    raise RuntimeError("No model candidates were available.")


async def _acall_with_classified_fallbacks(
    primary: Any,
    fallbacks: tuple[Any, ...],
    call: Callable[[Any], Any],
    *,
    route_label: str,
    on_fallback: FallbackCallback | None,
    on_retry: RetryCallback | None,
    retry_attempts: int,
    retry_max_wait_seconds: int,
) -> Any:
    candidates = (primary, *fallbacks)
    last_error: Exception | None = None
    for retry_index in range(retry_attempts + 1):
        failures = []
        for index, candidate in enumerate(candidates):
            try:
                result = await call(candidate)
                _accumulate_usage(result)
                return result
            except Exception as exc:
                last_error = exc
                failures.append(classify_exception(exc))
                if index == len(candidates) - 1:
                    break
                if not _should_try_fallback(exc):
                    raise
                _emit_fallback(on_fallback, route_label, exc, index + 1)

        wait_seconds = _retry_wait_seconds(
            failures,
            retry_index=retry_index + 1,
            retry_max_wait_seconds=retry_max_wait_seconds,
        )
        if retry_index < retry_attempts and wait_seconds is not None:
            _emit_retry(on_retry, route_label, wait_seconds, retry_index + 1, retry_attempts)
            await asyncio.sleep(wait_seconds)
            continue
        break
    if last_error is not None:
        raise last_error
    raise RuntimeError("No model candidates were available.")


def _should_try_fallback(exc: Exception) -> bool:
    return classify_exception(exc).category in FALLBACK_ERROR_CATEGORIES


def _emit_fallback(
    on_fallback: FallbackCallback | None,
    route_label: str,
    exc: Exception,
    next_index: int,
) -> None:
    if on_fallback is None:
        return
    failure = classify_exception(exc)
    on_fallback(
        f"{route_label} failed with {failure.category}; trying fallback candidate {next_index}"
    )


def _retry_wait_seconds(
    failures: list[Any],
    *,
    retry_index: int,
    retry_max_wait_seconds: int,
) -> int | None:
    if retry_max_wait_seconds <= 0 or not failures:
        return None
    if any(failure.category not in FALLBACK_ERROR_CATEGORIES for failure in failures):
        return None
    retryable_failures = [failure for failure in failures if failure.retryable]
    if not retryable_failures:
        return None
    wait_seconds = None
    explicit_waits: list[int] = []
    for failure in retryable_failures:
        if failure.retry_after_seconds is not None:
            explicit_waits.append(int(failure.retry_after_seconds))
    if explicit_waits:
        wait_seconds = min(explicit_waits)
        if wait_seconds <= 0:
            return None
        if wait_seconds > retry_max_wait_seconds:
            wait_seconds = retry_max_wait_seconds
    else:
        wait_seconds = min(2**retry_index, retry_max_wait_seconds)
        if wait_seconds <= 0:
            return None
    return min(int(wait_seconds), retry_max_wait_seconds)


def _emit_retry(
    on_retry: RetryCallback | None,
    route_label: str,
    wait_seconds: int,
    attempt: int,
    max_attempts: int,
) -> None:
    if on_retry is None:
        return
    on_retry(
        f"{route_label} all candidates hit retryable provider limits; "
        f"waiting {wait_seconds}s before retry {attempt}/{max_attempts}"
    )


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    provider, separator, model_name = model_spec.partition(":")
    if not separator:
        return "", model_spec
    return provider, model_name


def _message_to_openrouter(message: BaseMessage) -> dict[str, Any]:
    role_by_type = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    role = role_by_type.get(message.type, message.type)
    content = message.content
    if not isinstance(content, str | list):
        content = str(content)
    return {"role": role, "content": content}


def _openrouter_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
        return "\n".join(part for part in parts if part)
    return str(content)

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypeAlias

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from deep_research.settings import Settings

ModelLike: TypeAlias = str | BaseChatModel

_ROLE_KEY_INDEX = {
    "orchestrator": 0,
    "researcher": 1,
    "planner": 2,
    "verifier": 3,
    "analyst": 4,
    "judge": 5,
    "fast": 6,
}

_ROLE_MODEL_ATTRS = {
    "orchestrator": "model",
    "planner": "planner_model",
    "researcher": "researcher_model",
    "analyst": "analyst_model",
    "verifier": "verifier_model",
    "judge": "judge_model",
    "fast": "fast_model",
}

_KEY_ENV_BASE = {
    "groq": "GROQ_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_agent_models(settings: Settings) -> RoutedAgentModels:
    return RoutedAgentModels(
        orchestrator=model_for_role(settings, "orchestrator", settings.model),
        fast=model_for_role(settings, "fast", settings.fast_model),
        planner=model_for_role(settings, "planner", settings.planner_model),
        researcher=model_for_role(settings, "researcher", settings.researcher_model),
        analyst=model_for_role(settings, "analyst", settings.analyst_model),
        verifier=model_for_role(settings, "verifier", settings.verifier_model),
    )


def model_for_role(settings: Settings, role: str, model_spec: str) -> ModelLike:
    provider, model_name = _split_model_spec(model_spec)
    if provider == "groq" and settings.groq_key_pool:
        return ChatGroq(
            model=model_name,
            api_key=_key_for_role(settings.groq_key_pool, role),
        )
    if provider == "google_genai" and settings.google_key_pool:
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=_key_for_role(settings.google_key_pool, role),
        )
    return model_spec


def describe_model_routes(settings: Settings) -> dict[str, object]:
    routes = [_describe_role(settings, role) for role in _ROLE_MODEL_ATTRS]
    return {
        "provider": settings.provider,
        "google_key_count": len(settings.google_key_pool),
        "groq_key_count": len(settings.groq_key_pool),
        "roles": [route.to_dict() for route in routes],
    }


def route_summary(settings: Settings) -> str:
    visible_roles = ("orchestrator", "planner", "researcher", "analyst", "verifier", "judge")
    routes = [_describe_role(settings, role) for role in visible_roles]
    parts = []
    for route in routes:
        key = route.key_label or "no-key"
        parts.append(f"{route.role}={route.provider}:{route.model} via {key}")
    return "; ".join(parts)


def _key_for_role(keys: tuple[str, ...], role: str) -> str:
    if not keys:
        raise ValueError("API key pool cannot be empty.")
    return keys[_key_slot_for_role(keys, role)]


def _describe_role(settings: Settings, role: str) -> ModelRoute:
    attr = _ROLE_MODEL_ATTRS[role]
    model_spec = getattr(settings, attr)
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
    )


def _key_pool_for_provider(settings: Settings, provider: str) -> tuple[str, ...]:
    if provider == "groq":
        return settings.groq_key_pool
    if provider == "google_genai":
        return settings.google_key_pool
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


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    provider, separator, model_name = model_spec.partition(":")
    if not separator:
        return "", model_spec
    return provider, model_name

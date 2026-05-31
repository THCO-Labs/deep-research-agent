from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class RoutedAgentModels:
    orchestrator: ModelLike
    fast: ModelLike
    planner: ModelLike
    researcher: ModelLike
    analyst: ModelLike
    verifier: ModelLike


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


def _key_for_role(keys: tuple[str, ...], role: str) -> str:
    if not keys:
        raise ValueError("API key pool cannot be empty.")
    index = _ROLE_KEY_INDEX.get(role, 0)
    return keys[index % len(keys)]


def _split_model_spec(model_spec: str) -> tuple[str, str]:
    provider, separator, model_name = model_spec.partition(":")
    if not separator:
        return "", model_spec
    return provider, model_name

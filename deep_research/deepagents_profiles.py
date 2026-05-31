from __future__ import annotations

from threading import Lock

from collections.abc import Mapping

from deepagents.profiles import HarnessProfile, register_harness_profile

from deep_research.settings import Settings

WRITE_TODOS_TOOL = "write_todos"
GROQ_PROVIDER = "groq"
GOOGLE_PROVIDER = "google_genai"

_PROFILE_LOCK = Lock()
_CONFIGURED_PROFILE_KEYS: set[str] = set()


def configure_deepagents_profiles(settings: Settings) -> None:
    """Register provider-specific DeepAgents runtime profiles.

    Groq's free/on-demand models are useful for this project, but the
    OpenAI-compatible tool-call parser is less forgiving when a model emits a
    malformed call to DeepAgents' internal todo tool. The app already exposes
    a first-class progress stream, so Groq runs remove that internal tool from
    the visible tool set while keeping subagent dispatch and file safety.
    """
    profiles = _profiles(settings)
    if not profiles:
        return

    with _PROFILE_LOCK:
        for key, profile in profiles.items():
            if key in _CONFIGURED_PROFILE_KEYS:
                continue
            register_harness_profile(key, profile)
            _CONFIGURED_PROFILE_KEYS.add(key)


def _profiles(settings: Settings) -> Mapping[str, HarnessProfile]:
    models = {
        settings.model,
        settings.fast_model,
        settings.planner_model,
        settings.researcher_model,
        settings.analyst_model,
        settings.verifier_model,
        settings.judge_model,
    }
    groq_models = {model for model in models if model.startswith("groq:")}
    google_models = {model for model in models if model.startswith("google_genai:")}
    profiles: dict[str, HarnessProfile] = {}
    if settings.provider == "groq" or groq_models:
        groq_profile = HarnessProfile(excluded_tools=frozenset({WRITE_TODOS_TOOL}))
        for key in sorted({GROQ_PROVIDER, *groq_models}):
            profiles[key] = groq_profile
    if settings.provider == "google" or google_models:
        profiles[GOOGLE_PROVIDER] = HarnessProfile()
    return profiles

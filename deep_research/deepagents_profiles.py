from __future__ import annotations

from threading import Lock

from deepagents.profiles import HarnessProfile, register_harness_profile

from deep_research.settings import Settings

WRITE_TODOS_TOOL = "write_todos"

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
    profile_keys = _profile_keys(settings)
    if not profile_keys:
        return

    with _PROFILE_LOCK:
        for key in profile_keys:
            if key in _CONFIGURED_PROFILE_KEYS:
                continue
            register_harness_profile(
                key,
                HarnessProfile(excluded_tools=frozenset({WRITE_TODOS_TOOL})),
            )
            _CONFIGURED_PROFILE_KEYS.add(key)


def _profile_keys(settings: Settings) -> tuple[str, ...]:
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
    if settings.provider != "groq" and not groq_models:
        return ()
    return tuple(sorted({"groq", *groq_models}))

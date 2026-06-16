from __future__ import annotations

import re
import threading
from typing import Iterable


_WORDNET_DISABLED = False
_EXPANSION_CACHE: dict[tuple[tuple[str, ...], int], set[str]] = {}


def expand_terms(
    terms: Iterable[str],
    *,
    max_per_term: int = 8,
    timeout_s: float = 0.45,
) -> set[str]:
    """Expand analysis terms with WordNet when available, with a bounded fallback.

    NLTK can be installed while the WordNet corpus is missing or slow to load.
    This helper never blocks the research graph on corpus lookup: if WordNet
    cannot respond within the small budget, the process keeps a deterministic
    morphology-based expansion.
    """
    seeds = {_clean_term(term) for term in terms if _clean_term(term)}
    cache_key = (tuple(sorted(seeds)), max_per_term)
    if cache_key in _EXPANSION_CACHE:
        return set(_EXPANSION_CACHE[cache_key])
    fallback = set(seeds)
    for seed in seeds:
        fallback.update(_morphological_variants(seed))
    if not seeds or _WORDNET_DISABLED:
        _EXPANSION_CACHE[cache_key] = set(fallback)
        return fallback

    holder: list[set[str]] = []

    def _load() -> None:
        holder.append(_wordnet_expansion(seeds, max_per_term=max_per_term))

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive() or not holder:
        _disable_wordnet()
        _EXPANSION_CACHE[cache_key] = set(fallback)
        return fallback
    expanded = fallback | holder[0]
    _EXPANSION_CACHE[cache_key] = set(expanded)
    return expanded


def term_matches(text: str, seeds: Iterable[str]) -> bool:
    terms = expand_terms(seeds)
    haystack = {token.lower() for token in re.findall(r"[A-Za-z][A-Za-z-]+", text)}
    return bool(terms & haystack)


def _wordnet_expansion(seeds: set[str], *, max_per_term: int) -> set[str]:
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        _disable_wordnet()
        return set()
    expanded: set[str] = set()
    try:
        for seed in seeds:
            count = 0
            for synset in wn.synsets(seed):
                for lemma in synset.lemmas():
                    for candidate in [lemma.name(), *(related.name() for related in lemma.derivationally_related_forms())]:
                        cleaned = _clean_term(candidate.replace("_", " "))
                        if cleaned and " " not in cleaned:
                            expanded.add(cleaned)
                            count += 1
                        if count >= max_per_term:
                            break
                    if count >= max_per_term:
                        break
                if count >= max_per_term:
                    break
    except Exception:
        _disable_wordnet()
        return set()
    return expanded


def _morphological_variants(term: str) -> set[str]:
    variants = {term}
    if len(term) < 4:
        return variants
    variants.update({f"{term}s", f"{term}ed", f"{term}ing"})
    if term.endswith("e"):
        variants.update({f"{term}d", f"{term[:-1]}ing"})
    if term.endswith("y"):
        variants.update({f"{term[:-1]}ies", f"{term[:-1]}ied"})
    if term.endswith("ion"):
        variants.add(term[:-3] + "e")
    if term.endswith("ation"):
        variants.add(term[:-5] + "e")
    if term.endswith("al"):
        variants.add(term[:-2])
    return {value for value in variants if value}


def _clean_term(term: str) -> str:
    cleaned = re.sub(r"[^A-Za-z -]+", " ", str(term).lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _disable_wordnet() -> None:
    global _WORDNET_DISABLED
    _WORDNET_DISABLED = True

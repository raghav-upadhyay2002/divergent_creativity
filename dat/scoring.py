import pickle
import re
from itertools import combinations

import numpy as np
from scipy.spatial import distance

from .config import load_config

_vectors = None


def _get_vectors():
    """Load and cache the GloVe vectors from the configured path."""
    global _vectors
    if _vectors is None:
        glove_path = load_config()["paths"]["glove"]
        with open(glove_path, "rb") as f:
            _vectors = pickle.load(f)
    return _vectors


# ---------------------------------------------------------------------------
# validation code from https://github.com/jayolson/divergent-association-task
# (copied word for word from a9v2t-osfstorage-archive/calculate_stats.py,
# except vectors is now passed in rather than read from a module global)
# ---------------------------------------------------------------------------

def validate(word: str, vectors):
    """Clean up word and find best candidate to use"""
    # Strip unwanted characters
    clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
    if len(clean) <= 1:
        return None  # Word too short

    candidates = []
    if " " in clean:
        candidates.append(re.sub(r" +", "-", clean))
        candidates.append(re.sub(r" +", "", clean))
    else:
        candidates.append(clean)
        if "-" in clean:
            candidates.append(re.sub(r"-+", "", clean))

    for cand in candidates:
        if cand in vectors:
            return cand  # Return first word that is in model
    return None  # Could not find valid word


def _split(words):
    """Split a comma-separated string into words, or pass a list through."""
    if isinstance(words, str):
        return [w.strip() for w in words.split(",")]
    return list(words)


def _first_n_valid_unique(words, n, vectors):
    """Classify each raw word as used (one of the first n valid uniques) or dropped."""
    seen = set()
    used = []
    dropped = []
    for word in _split(words):
        cleaned = validate(word, vectors)
        if cleaned is not None and cleaned not in seen and len(used) < n:
            seen.add(cleaned)
            used.append(cleaned)
        else:
            dropped.append(word)
    return used, dropped


def valid_words(words):
    """Return the first 7 valid, unique words (the ones the DAT score uses)."""
    vectors = _get_vectors()
    used, _ = _first_n_valid_unique(words, 7, vectors)
    return used


def _mean_pairwise_distance_score(used, vectors):
    """Olson's DAT formula: mean cosine distance over all pairs, times 100."""
    distances = []
    for word1, word2 in combinations(used, 2):
        v1 = vectors[word1].astype(np.float32)
        v2 = vectors[word2].astype(np.float32)
        distances.append(distance.cosine(v1, v2))
    return float((sum(distances) / len(distances)) * 100)


def dat_score(words, min_words=7):
    """Return the DAT score for words, or None if fewer than min_words are valid."""
    vectors = _get_vectors()
    used, _ = _first_n_valid_unique(words, min_words, vectors)
    if len(used) < min_words:
        return None

    return _mean_pairwise_distance_score(used, vectors)


def score_details(words, min_words=7):
    """Return (score, valid_words_used, invalid_words) for the main DAT score."""
    vectors = _get_vectors()
    used, dropped = _first_n_valid_unique(words, min_words, vectors)

    score = None
    if len(used) >= min_words:
        score = _mean_pairwise_distance_score(used, vectors)

    return score, used, dropped


def score_details_all10(words, max_words=10, min_words=7):
    """Like score_details, but scores up to 10 valid words instead of exactly 7."""
    vectors = _get_vectors()
    used, dropped = _first_n_valid_unique(words, max_words, vectors)

    score = None
    if len(used) >= min_words:
        score = _mean_pairwise_distance_score(used, vectors)

    return score, used, dropped

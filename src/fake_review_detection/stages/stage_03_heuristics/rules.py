# Transparent rules: each returns suspicion in [0,1]; aggregate is a weighted mean of those signals.

from __future__ import annotations

import re
from collections import Counter

_MARKETING = frozenset(
    "amazing incredible best worst perfect love highly recommend must great awesome "
    "fantastic excellent superb outstanding disappointed terrible awful horrible "
    "guarantee deal steal bargain".split()
)


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def score_glowing_text_low_stars(rating: int, sentiment_compound: float) -> float:
    if rating > 2 or sentiment_compound <= 0.12:
        return 0.0
    return min(0.6, _clip01((sentiment_compound - 0.12) / 0.88) * 0.65)


def score_rating_sentiment_mismatch(rating: int, sentiment_compound: float) -> float:
    if rating >= 5 and sentiment_compound < -0.45:
        raw = _clip01((-sentiment_compound - 0.45) / 0.55)
        return min(0.55, raw * 0.65)
    if rating <= 2 and sentiment_compound > 0.65:
        raw = _clip01((sentiment_compound - 0.65) / 0.35)
        return min(0.55, raw * 0.65)
    return 0.0


def score_extreme_brevity_high_rating(rating: int, review_text: str) -> float:
    n_words = len(re.findall(r"\b\w+\b", review_text))
    if rating >= 5 and n_words > 0 and n_words <= 8:
        return _clip01(1.0 - n_words / 8.0)
    return 0.0


def score_title_shouting(title: str) -> float:
    if not title.strip():
        return 0.0
    letters = [c for c in title if c.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for c in letters if c.isupper())
    ratio = caps / len(letters)
    if ratio < 0.5:
        return 0.0
    return _clip01((ratio - 0.5) * 2)


def score_lexical_repetition(review_text: str) -> float:
    words = re.findall(r"\b\w+\b", review_text.lower())
    if len(words) < 10:
        return 0.0
    stop = {"the", "a", "an", "and", "or", "to", "of", "it", "is", "this", "that", "i", "my", "for", "in", "on"}
    filtered = [w for w in words if w not in stop and len(w) > 2]
    if not filtered:
        return 0.0
    top = Counter(filtered).most_common(1)[0][1]
    frac = top / len(filtered)
    if frac < 0.15:
        return 0.0
    return _clip01((frac - 0.15) / 0.35)


def score_unverified_low_engagement(verified: bool, review_length: int) -> float:
    if verified or review_length >= 80:
        return 0.0
    return _clip01(1.0 - review_length / 80.0)


def score_heavy_punctuation_high_rating(rating: int, review_text: str) -> float:
    text = review_text or ""
    n = len(text)
    if n < 45 or rating < 4:
        return 0.0
    punct = sum(1 for c in text if not c.isalnum() and not c.isspace())
    pr = punct / n
    if pr < 0.11:
        return 0.0
    return min(0.55, _clip01((pr - 0.11) / 0.22) * 0.6)


def score_marketing_language_density(review_text: str) -> float:
    # Flags survey-style praise stacks (“amazing”, “highly recommend”, …) as a spam-like pattern.
    words = re.findall(r"\b\w+\b", review_text.lower())
    if len(words) < 10:
        return 0.0
    hits = sum(1 for w in words if w in _MARKETING)
    d = hits / len(words)
    if d < 0.06:
        return 0.0
    return min(0.6, _clip01((d - 0.06) / 0.22) * 0.65)


HEURISTIC_NAMES = [
    "rating_sentiment_mismatch",
    "glowing_text_low_stars",
    "extreme_brevity_high_rating",
    "title_shouting",
    "lexical_repetition",
    "unverified_short_review",
    "heavy_punctuation_high_rating",
    "marketing_language_density",
]

DEFAULT_HEURISTIC_WEIGHTS = {
    "rating_sentiment_mismatch": 0.19,
    "glowing_text_low_stars": 0.15,
    "extreme_brevity_high_rating": 0.13,
    "title_shouting": 0.08,
    "lexical_repetition": 0.11,
    "unverified_short_review": 0.07,
    "heavy_punctuation_high_rating": 0.12,
    "marketing_language_density": 0.15,
}


def compute_heuristic_vector(
    review_text: str,
    title: str,
    *,
    rating: int,
    verified_purchase: bool,
    sentiment_compound: float,
) -> dict[str, float]:
    return {
        "rating_sentiment_mismatch": score_rating_sentiment_mismatch(rating, sentiment_compound),
        "glowing_text_low_stars": score_glowing_text_low_stars(rating, sentiment_compound),
        "extreme_brevity_high_rating": score_extreme_brevity_high_rating(rating, review_text),
        "title_shouting": score_title_shouting(title),
        "lexical_repetition": score_lexical_repetition(review_text),
        "unverified_short_review": score_unverified_low_engagement(verified_purchase, len(review_text)),
        "heavy_punctuation_high_rating": score_heavy_punctuation_high_rating(rating, review_text),
        "marketing_language_density": score_marketing_language_density(review_text),
    }


def aggregate_heuristic(
    parts: dict[str, float],
    weights: dict[str, float] | None = None,
) -> float:
    w = weights or DEFAULT_HEURISTIC_WEIGHTS
    s = sum(w.get(k, 0.0) for k in HEURISTIC_NAMES)
    if s <= 0:
        return 0.0
    return sum(parts[k] * w.get(k, 0.0) for k in HEURISTIC_NAMES) / s

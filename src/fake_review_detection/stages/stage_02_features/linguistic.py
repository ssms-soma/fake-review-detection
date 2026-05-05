# Builds one tabular row: metadata plus NLP signals that mirror the training CSV columns.

from __future__ import annotations

import re

import spacy
import textstat
from nltk.sentiment import SentimentIntensityAnalyzer

_nlp_model = None
_vader = None


def _nlp():
    global _nlp_model
    if _nlp_model is None:
        # Small English model; we only need POS tags so NER and dependency parser stay off for speed.
        _nlp_model = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _nlp_model


def _vader_analyzer():
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def count_pos_tags_from_doc(doc) -> tuple[int, int, int, int]:
    # spaCy’s coarse POS counts: how many tokens are NOUN, VERB, ADJ, ADV after tagging.
    counts = doc.count_by(spacy.attrs.POS)
    return (
        counts.get(spacy.parts_of_speech.NOUN, 0),
        counts.get(spacy.parts_of_speech.VERB, 0),
        counts.get(spacy.parts_of_speech.ADJ, 0),
        counts.get(spacy.parts_of_speech.ADV, 0),
    )


def count_pos_tags(text: str) -> tuple[int, int, int, int]:
    # Convenience wrapper: parse once inside count_pos_tags_from_doc.
    return count_pos_tags_from_doc(_nlp()(text))


def sentiment_compound(text: str) -> float:
    return float(_vader_analyzer().polarity_scores(text)["compound"])


def readability_fre(text: str) -> float:
    if not text.strip():
        return 0.0
    try:
        # Higher Flesch = easier to read; odd inputs fall back to 0 so training never crashes.
        return float(textstat.flesch_reading_ease(text))
    except Exception:
        return 0.0


def avg_word_length(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return 0.0
    return sum(len(w) for w in words) / len(words)


def coherence_encoded(rating: int, sentiment_compound: float) -> int:
    # 1 if VADER’s sign agrees with “high vs low stars” bucket; else 0 (tone vs stars clash).
    sent_pos = 1 if sentiment_compound > 0.0 else 0
    rating_pos = 1 if rating > 3 else 0
    return 1 if sent_pos == rating_pos else 0


def extract_tabular_row(
    review_text: str,
    *,
    rating: int = 3,
    verified_purchase: bool = False,
    title: str = "",
    average_rating: float | None = None,
    num_reviews: int | None = None,
    doc=None,
) -> dict[str, float | int]:
    text = review_text or ""
    title = title or ""
    ar = float(average_rating if average_rating is not None else rating)
    ar = max(1.0, min(5.0, ar))
    nr = int(num_reviews if num_reviews is not None else 1)
    nr = max(1, nr)
    rd = abs(float(rating) - ar)

    if doc is None:
        doc = _nlp()(text)  # caller can pass doc to avoid parsing the same text twice
    nouns, verbs, adj, adv = count_pos_tags_from_doc(doc)
    sent = sentiment_compound(text)
    fre = readability_fre(text)

    return {
        "VERIFIED_PURCHASE": 1 if verified_purchase else 0,
        "REVIEW_LENGTH": len(text),
        "TITLE_LENGTH": len(title),
        "SENTIMENT_SCORE": sent,
        "COHERENT_ENCODED": coherence_encoded(int(rating), sent),
        "RATING_DEVIATION": rd,
        "READABILITY_FRE": fre,
        "NUM_NOUNS": nouns,
        "NUM_VERBS": verbs,
        "NUM_ADJECTIVES": adj,
        "NUM_ADVERBS": adv,
        "AVERAGE_RATING": ar,
        "NUM_REVIEWS": nr,
    }

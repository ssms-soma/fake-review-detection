# Extra tabular fields: punctuation, caps, diversity, VADER pos/neg, POS ratios, stopword rate.

from __future__ import annotations

import re

import pandas as pd
from nltk.corpus import stopwords
from spacy.tokens import Doc
from nltk.sentiment import SentimentIntensityAnalyzer

from ...config import TABULAR_DERIVED_FEATURES

from .linguistic import _nlp, count_pos_tags_from_doc
from ..stage_01_preprocess.normalize import normalize_review_text

_vader = None
_stopwords: frozenset[str] | None = None


def _vader_analyzer():
    global _vader
    if _vader is None:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def _english_stopwords() -> frozenset[str]:
    global _stopwords
    if _stopwords is None:
        _stopwords = frozenset(stopwords.words("english"))
    return _stopwords


def _pos_stop_ratios_from_doc(doc, words_norm: list[str]) -> dict[str, float]:
    # Ratios use regex word count as denominator so they scale with review length.
    denom = max(1, len(words_norm))
    nouns, verbs, adj, adv = count_pos_tags_from_doc(doc)
    alpha_lower = [t.text.lower() for t in doc if t.is_alpha]
    if not alpha_lower:
        stop_r = 0.0
    else:
        sw = _english_stopwords()
        stop_r = sum(1 for w in alpha_lower if w in sw) / len(alpha_lower)
    return {
        "NOUN_RATIO": float(nouns / denom),
        "VERB_RATIO": float(verbs / denom),
        "ADJ_RATIO": float(adj / denom),
        "ADV_RATIO": float(adv / denom),
        "STOPWORD_RATIO": float(stop_r),
    }


def compute_derived_features(
    raw_text: str,
    normalized_text: str,
    doc: Doc | None = None,
) -> dict[str, float]:
    # raw_text keeps original casing for stylometry; norm + doc match the lowered review body.
    raw = raw_text or ""
    norm = normalized_text or ""
    n = len(raw)
    if n == 0:
        return {k: 0.0 for k in TABULAR_DERIVED_FEATURES}

    if doc is None:
        doc = _nlp()(norm)

    words_norm = re.findall(r"\b\w+\b", norm)
    ratios = _pos_stop_ratios_from_doc(doc, words_norm)

    punct = sum(1 for c in raw if not c.isalnum() and not c.isspace())
    punct_r = min(1.0, punct / n)

    letters = [c for c in raw if c.isalpha()]
    upper_r = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0

    digit_r = sum(1 for c in raw if c.isdigit()) / n

    awl = (sum(len(w) for w in words_norm) / len(words_norm)) if words_norm else 0.0

    excl_per_100 = (raw.count("!") / n) * 100.0

    n_w = len(words_norm)
    unique_ratio = (len(set(words_norm)) / n_w) if n_w else 0.0

    chunks = re.split(r"[.!?]+", norm)
    sents = [c.strip() for c in chunks if c.strip()]
    if not sents:
        mean_sent_len = 0.0
    else:
        mean_sent_len = sum(
            len(re.findall(r"\b\w+\b", s.lower())) for s in sents
        ) / len(sents)

    vad = _vader_analyzer().polarity_scores(norm if norm else raw.lower())

    out = {
        "PUNCT_RATIO": float(punct_r),
        "UPPER_RATIO": float(upper_r),
        "DIGIT_RATIO": float(digit_r),
        "AWL_BODY": float(awl),
        "EXCLAIM_PER_100": float(excl_per_100),
        "UNIQUE_WORD_RATIO": float(unique_ratio),
        "MEAN_SENT_LEN": float(mean_sent_len),
        "VADER_POS": float(vad["pos"]),
        "VADER_NEG": float(vad["neg"]),
    }
    out.update(ratios)
    return out


def augment_derived_dataframe(df: pd.DataFrame) -> None:
    # Training-only helper: nlp.pipe batches rows so big CSVs parse faster than a Python loop.
    raw_bodies = df["REVIEW_TEXT"].fillna("").astype(str).tolist()
    norms = [normalize_review_text(t) for t in raw_bodies]
    nlp = _nlp()
    docs = list(nlp.pipe(norms, batch_size=128))
    rows = [
        compute_derived_features(r, n, doc=d)
        for r, n, d in zip(raw_bodies, norms, docs)
    ]
    sty_df = pd.DataFrame(rows, index=df.index)
    for k in TABULAR_DERIVED_FEATURES:
        df[k] = sty_df[k]

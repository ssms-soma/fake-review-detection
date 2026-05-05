# Sparse TF-IDF over characters and words; classifier is LR vs calibrated SVM by validation F1.

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def _build_char_word_union() -> FeatureUnion:
    # Char ngrams catch typos and morphology; word ngrams catch phrases; both use log TF.
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        max_features=60_000,
        sublinear_tf=True,
    )
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        max_features=40_000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w\w+\b",
    )
    return FeatureUnion([("char", char), ("word", word)])


def build_tfidf_pipeline_lr() -> Pipeline:
    clf = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        solver="saga",
        random_state=42,
    )
    return Pipeline([("feats", _build_char_word_union()), ("lr", clf)])


def build_tfidf_pipeline_svc() -> Pipeline:
    base = LinearSVC(
        class_weight="balanced",
        dual="auto",
        max_iter=5000,
        random_state=42,
    )
    cal = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    return Pipeline([("feats", _build_char_word_union()), ("svm", cal)])


def fit_best_tfidf(
    X_train: list[str],
    y_train: np.ndarray,
    X_val: list[str],
    y_val: np.ndarray,
) -> tuple[Pipeline, str]:
    # Train both heads on the train fold and keep whichever wins macro-F1 on validation.
    pipe_lr = build_tfidf_pipeline_lr()
    pipe_lr.fit(X_train, y_train)
    f1_lr = float(f1_score(y_val, pipe_lr.predict(X_val), average="macro"))

    pipe_svc = build_tfidf_pipeline_svc()
    pipe_svc.fit(X_train, y_train)
    f1_svc = float(f1_score(y_val, pipe_svc.predict(X_val), average="macro"))

    if f1_svc > f1_lr:
        return pipe_svc, "linear_svc_calibrated"
    return pipe_lr, "logistic_regression"

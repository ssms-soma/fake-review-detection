# End-to-end training, joblib artifact, and predict_explain JSON for the Flask UI.

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from fake_review_detection.config import (
    ARTIFACTS_DIR,
    ENSEMBLE_PATH,
    META_PATH,
    REPORTS_DIR,
    TABULAR_FEATURES,
    TABULAR_FROM_CSV,
)
from ..stage_01_preprocess.normalize import (
    normalize_review_text,
    normalize_review_text_for_bert,
    normalize_title,
)
from ..stage_02_features import (
    augment_derived_dataframe,
    compute_derived_features,
    extract_tabular_row,
    sentiment_compound,
)
from ..stage_02_features.linguistic import _nlp
from ..stage_03_heuristics.rules import (
    DEFAULT_HEURISTIC_WEIGHTS,
    aggregate_heuristic,
    compute_heuristic_vector,
)
from ..stage_04_text_model.embedding_classifiers import (
    compare_text_models_val_f1,
    refit_text_model_full,
    text_model_predict_proba,
)
from ..stage_05_tabular_model.tabular_pipeline import build_tabular_pipeline
from ..stage_07_evaluation.metrics import metrics_from_probs
from ..stage_07_evaluation.report import StepResult, write_training_report
from ..stage_08_output.verdict import binary_class_from_probability, verdict_from_probability

from .blend import (
    EnsembleWeights,
    TrainReport,
    _apply_min_heuristic,
    _logit,
    _sigmoid,
    blend_logit,
    heuristic_to_genuine_score,
    logit_scalar,
    tune_logit_bias,
    tune_weights,
)

def _merge_title_body_for_sparse(title: str, body: str) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        return f"{title} || {body}"
    return title or body
def _tabular_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[list(TABULAR_FEATURES)].astype(np.float64).values


def train_from_csv(
    csv_path: Path,
    random_state: int = 42,
    *,
    write_reports: bool = True,
    reports_dir: Path | None = None,
) -> TrainReport:
    # Stratified 80/20: fit tabular + TF-IDF on train, tune ensemble on val, then refit on all rows.
    df = pd.read_csv(csv_path)
    need_cols = list(TABULAR_FROM_CSV) + ["LABEL_ENCODED", "REVIEW_TEXT", "RATING"]
    df = df.dropna(subset=need_cols)
    augment_derived_dataframe(df)
    y = df["LABEL_ENCODED"].astype(int).values
    raw_bodies = df["REVIEW_TEXT"].fillna("").astype(str).tolist()
    raw_titles = df.get("REVIEW_TITLE", pd.Series([""] * len(df))).fillna("").astype(str).tolist()

    texts_tfidf = [normalize_review_text(t) for t in raw_bodies]
    titles_clean = [normalize_title(t) for t in raw_titles]
    texts_sparse = [
    _merge_title_body_for_sparse(ti, bo)
    for ti, bo in zip(titles_clean, texts_tfidf, strict=True)
]

    bodies_bert = [normalize_review_text_for_bert(t) for t in raw_bodies]

    X_tab = _tabular_matrix(df)
    idx_train, idx_val = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    tab_pipe = build_tabular_pipeline()
    tab_pipe.fit(X_tab[idx_train], y[idx_train])
    p_tab_val = tab_pipe.predict_proba(X_tab[idx_val])[:, 1]

    tr_tx = [texts_sparse[i] for i in idx_train]
    va_tx = [texts_sparse[i] for i in idx_val]

    tr_titles = [titles_clean[i] for i in idx_train]
    va_titles = [titles_clean[i] for i in idx_val]

    tr_bodies_bert = [bodies_bert[i] for i in idx_train]
    va_bodies_bert = [bodies_bert[i] for i in idx_val]

    best_key, text_f1_map, text_acc_map, text_train_model, tfidf_head_tag, ft_backend = (
        compare_text_models_val_f1(
            tr_tx,
            y[idx_train],
            va_tx,
            y[idx_val],
            train_titles=tr_titles,
            train_bodies_for_bert=tr_bodies_bert,
            val_titles=va_titles,
            val_bodies_for_bert=va_bodies_bert,
        )
    )

    p_tf_val = text_model_predict_proba(
        text_train_model,
        va_tx,
        titles=va_titles,
        bodies_for_bert=va_bodies_bert,
    )

    h_val = []
    for i in idx_val:
        row = df.iloc[int(i)]
        norm_body = normalize_review_text(str(row["REVIEW_TEXT"]))
        norm_title = normalize_title(str(row.get("REVIEW_TITLE", "") or ""))
        sent = sentiment_compound(norm_body)
        parts = compute_heuristic_vector(
            norm_body,
            norm_title,
            rating=int(row["RATING"]),
            verified_purchase=bool(int(row["VERIFIED_PURCHASE"])),
            sentiment_compound=sent,
        )
        h_val.append(aggregate_heuristic(parts))
    h_val = np.array(h_val, dtype=np.float64)
    h_gen_val = 1.0 - h_val

    w = tune_weights(p_tab_val, p_tf_val, h_val, y[idx_val])
    w = _apply_min_heuristic(w)
    z_val = (
        w.w_tabular * _logit(p_tab_val)
        + w.w_tfidf * _logit(p_tf_val)
        + w.w_heuristic * _logit(h_gen_val)
    )
    bias = tune_logit_bias(z_val, y[idx_val])
    blend = blend_logit(
        p_tab_val,
        p_tf_val,
        h_gen_val,
        w.w_tabular,
        w.w_tfidf,
        w.w_heuristic,
        bias,
    )
    pred = (blend >= 0.5).astype(int)
    y_tr = y[idx_train]
    y_va = y[idx_val]

    report = TrainReport(
        val_f1_macro=float(f1_score(y_va, pred, average="macro")),
        val_accuracy=float(np.mean(pred == y_va)),
        weights={
            "tabular": w.w_tabular,
            "tfidf": w.w_tfidf,
            "heuristic": w.w_heuristic,
        },
        tfidf_val_f1=float(
            f1_score(y_va, (p_tf_val >= 0.5).astype(int), average="macro")
        ),
        tabular_val_f1=float(
            f1_score(y_va, (p_tab_val >= 0.5).astype(int), average="macro")
        ),
        logit_bias=bias,
        n_train=len(idx_train),
        n_val=len(idx_val),
        selected_text_model=best_key,
        text_model_comparison_val_f1=dict(text_f1_map),
    )

    h_train: list[float] = []
    for i in idx_train:
        row = df.iloc[int(i)]
        norm_body = normalize_review_text(str(row["REVIEW_TEXT"]))
        norm_title = normalize_title(str(row.get("REVIEW_TITLE", "") or ""))
        sent = sentiment_compound(norm_body)
        parts = compute_heuristic_vector(
            norm_body,
            norm_title,
            rating=int(row["RATING"]),
            verified_purchase=bool(int(row["VERIFIED_PURCHASE"])),
            sentiment_compound=sent,
        )
        h_train.append(aggregate_heuristic(parts))
    h_train = np.array(h_train, dtype=np.float64)
    h_gen_train = 1.0 - h_train

    p_tab_train = tab_pipe.predict_proba(X_tab[idx_train])[:, 1]
    p_tf_train = text_model_predict_proba(
    text_train_model,
    tr_tx,
    titles=tr_titles,
    bodies_for_bert=tr_bodies_bert,
)

    maj = int(np.bincount(y_tr).argmax())
    p_tr_maj = np.full(len(idx_train), 0.99 if maj == 1 else 0.01)
    p_va_maj = np.full(len(idx_val), 0.99 if maj == 1 else 0.01)

    wt = w.w_tabular + w.w_tfidf
    wtn = w.w_tabular / wt
    wfn = w.w_tfidf / wt
    z_va_2ch = wtn * _logit(p_tab_val) + wfn * _logit(p_tf_val)
    bias_2ch = tune_logit_bias(z_va_2ch, y_va)
    p_va_2ch = _sigmoid(z_va_2ch + bias_2ch)
    z_tr_2ch = wtn * _logit(p_tab_train) + wfn * _logit(p_tf_train)
    p_tr_2ch = _sigmoid(z_tr_2ch + bias_2ch)

    z_tr_full = (
        w.w_tabular * _logit(p_tab_train)
        + w.w_tfidf * _logit(p_tf_train)
        + w.w_heuristic * _logit(h_gen_train)
    )
    p_tr_full = _sigmoid(z_tr_full + bias)

    step_defs = [
        ("majority", "0. Majority baseline", p_tr_maj, p_va_maj),
        ("tabular", "1. Tabular LR only", p_tab_train, p_tab_val),
        (
            "tfidf",
            f"2. Text ({best_key}: best of TF-IDF / SBERT / FastText)",
            p_tf_train,
            p_tf_val,
        ),
        ("tab_tfidf", "3. Logit blend (tabular + TF-IDF)", p_tr_2ch, p_va_2ch),
        ("full", "4. Full ensemble (+ heuristics + bias)", p_tr_full, blend),
    ]
    steps: list[StepResult] = []
    prev_f1: float | None = None
    for key, title, ptr, pva in step_defs:
        mt = metrics_from_probs(y_tr, ptr)
        mv = metrics_from_probs(y_va, pva)
        f1v = mv["f1_macro"]
        delta = None if prev_f1 is None else f1v - prev_f1
        prev_f1 = f1v
        steps.append(StepResult(key=key, title=title, val_metrics=mv, train_metrics=mt, delta_val_f1_macro=delta))

    if write_reports:
        write_training_report(
            reports_dir or REPORTS_DIR,
            steps,
            extra={
                "ensemble_weights": asdict(w),
                "logit_bias": bias,
                "bias_two_channel": bias_2ch,
                "n_train": len(idx_train),
                "n_val": len(idx_val),
                "training_csv": str(csv_path),
                "tfidf_head": tfidf_head_tag,
                "selected_text_model": best_key,
                "text_model_comparison_val_f1": dict(text_f1_map),
                "text_model_comparison_val_accuracy": dict(text_acc_map),
                "fasttext_backend": ft_backend,
            },
        )

    # Final artifact uses every labeled row so deploy matches the coursework “full data” refit.
    tab_full = build_tabular_pipeline()
    tab_full.fit(X_tab, y)
    text_full, text_head_label = refit_text_model_full(
    best_key,
    texts_sparse,
    y,
    tfidf_head_tag,
    fasttext_backend=ft_backend,
    all_titles=titles_clean,
    all_bodies_for_bert=bodies_bert,
)

    tabular_metadata_bounds = {
        "NUM_REVIEWS": (int(df["NUM_REVIEWS"].min()), int(df["NUM_REVIEWS"].max())),
        "AVERAGE_RATING": (float(df["AVERAGE_RATING"].min()), float(df["AVERAGE_RATING"].max())),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tabular_pipeline": tab_full,
        "text_pipeline": text_full,
        "tfidf_pipeline": text_full,
        "text_model_kind": best_key,
        "fasttext_backend": ft_backend,
        "weights": asdict(w),
        "logit_bias": bias,
        "tabular_features": list(TABULAR_FEATURES),
        "tabular_metadata_bounds": tabular_metadata_bounds,
        "heuristic_subweights": dict(DEFAULT_HEURISTIC_WEIGHTS),
        "classes": [0, 1],
        "class_names": {"0": "deceptive", "1": "genuine"},
        "blend_mode": "logit",
        "bert_model_dir": str(ARTIFACTS_DIR / "bert_text_model"),
        "tfidf_head": text_head_label,
    }
    joblib.dump(payload, ENSEMBLE_PATH)

    meta = {
        "train_report": asdict(report),
        "weights": payload["weights"],
        "logit_bias": bias,
        "tabular_only_val_macro_f1": report.tabular_val_f1,
        "tfidf_only_val_macro_f1": report.tfidf_val_f1,
        "text_channel_val_macro_f1": report.tfidf_val_f1,
        "selected_text_model": best_key,
        "text_model_comparison_val_f1": dict(text_f1_map),
        "text_model_comparison_val_accuracy": dict(text_acc_map),
        "fasttext_backend": ft_backend,
        "tfidf_head": text_head_label,
        "note": "Class 0 = deceptive, 1 = genuine. Training rows: amazon_reviews_training.csv only. Tabular omits raw RATING; verified from checkbox at inference.",
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return report


def load_ensemble() -> dict[str, Any]:
    return joblib.load(ENSEMBLE_PATH)


def _tabular_explanations(
    tab_pipe: Pipeline, feature_row: np.ndarray, feature_names: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    # Linear model: contribution per feature is coef × scaled_value; sum + intercept = logit.
    scaler: MinMaxScaler = tab_pipe.named_steps["scale"]
    lr: LogisticRegression = tab_pipe.named_steps["lr"]
    x_s = scaler.transform(feature_row.reshape(1, -1))[0]
    coef = lr.coef_.ravel()
    intercept = float(lr.intercept_.ravel()[0])
    contribs = []
    for name, xi, c in zip(feature_names, x_s, coef, strict=True):
        contribs.append(
            {
                "feature": name,
                "scaled_value": float(xi),
                "coefficient": float(c),
                "logit_contribution": float(c * xi),
            }
        )
    logit = intercept + sum(t["logit_contribution"] for t in contribs)
    p_genuine = float(1.0 / (1.0 + np.exp(-logit)))
    return p_genuine, contribs


def predict_explain(
    review_text: str,
    *,
    rating: int = 3,
    verified_purchase: bool = False,
    title: str = "",
    average_rating: float | None = None,
    num_reviews: int | None = None,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Single-review inference: same feature order as training; returns probs + UI-friendly breakdown.
    if artifact is None:
        artifact = load_ensemble()

    text = normalize_review_text(review_text)
    text_bert = normalize_review_text_for_bert(review_text)
    tit = normalize_title(title)
    sparse_text = _merge_title_body_for_sparse(tit, text)
    
    feat_names: list[str] = list(artifact.get("tabular_features") or TABULAR_FEATURES)

    bounds = artifact.get("tabular_metadata_bounds") or {}
    ar_use = average_rating
    nr_use = num_reviews
    if nr_use is not None and "NUM_REVIEWS" in bounds:
        lo, hi = bounds["NUM_REVIEWS"]
        nr_use = int(max(int(lo), min(int(hi), int(nr_use))))
    if ar_use is not None and "AVERAGE_RATING" in bounds:
        lo, hi = bounds["AVERAGE_RATING"]
        ar_use = float(max(float(lo), min(float(hi), float(ar_use))))

    doc = _nlp()(text)
    row = extract_tabular_row(
        text,
        rating=rating,
        verified_purchase=verified_purchase,
        title=tit,
        average_rating=ar_use,
        num_reviews=nr_use,
        doc=doc,
    )
    row.update(compute_derived_features(review_text or "", text, doc=doc))
    X = np.array([[row[f] for f in feat_names]], dtype=np.float64)

    tab_pipe: Pipeline = artifact["tabular_pipeline"]
    tf_pipe: Any = artifact.get("text_pipeline") or artifact["tfidf_pipeline"]
    text_kind: str = str(artifact.get("text_model_kind", "tfidf"))
    w = artifact["weights"]
    h_w = artifact.get("heuristic_subweights", DEFAULT_HEURISTIC_WEIGHTS)
    bias = float(artifact.get("logit_bias", 0.0))

    p_tab_g = float(tab_pipe.predict_proba(X)[0, 1])
    p_tf_g = float(
    text_model_predict_proba(
        tf_pipe,
        [sparse_text],
        titles=[tit],
        bodies_for_bert=[text_bert],
    )[0]
)

    sent = sentiment_compound(text)
    h_parts = compute_heuristic_vector(
        text, tit, rating=rating, verified_purchase=verified_purchase, sentiment_compound=sent
    )
    h_agg = aggregate_heuristic(h_parts, h_w)
    p_heur_g = heuristic_to_genuine_score(h_agg)

    w_tab, w_tf, w_h = w["w_tabular"], w["w_tfidf"], w["w_heuristic"]
    s = w_tab + w_tf + w_h
    if s > 0:
        w_tab, w_tf, w_h = w_tab / s, w_tf / s, w_h / s

    lt = logit_scalar(p_tab_g)
    lf = logit_scalar(p_tf_g)
    lh = logit_scalar(p_heur_g)
    z = w_tab * lt + w_tf * lf + w_h * lh + bias
    p_final = float(np.clip(_sigmoid(z), 1e-7, 1.0 - 1e-7))

    _, tab_breakdown = _tabular_explanations(tab_pipe, X.ravel(), feat_names)

    predicted_genuine = binary_class_from_probability(p_final)
    v_short, v_detail = verdict_from_probability(p_final)

    return {
        "verdict_genuine": bool(predicted_genuine),
        "verdict_label": v_short,
        "verdict_detail": v_detail,
        "probability_genuine": round(p_final, 4),
        "probability_deceptive": round(1.0 - p_final, 4),
        "text_model_kind": text_kind,
        "blend_mode": artifact.get("blend_mode", "linear"),
        "logit_bias": round(bias, 4),
        "logit_z": round(float(z), 4),
        "logit_breakdown": {
            "tabular": round(w_tab * lt, 4),
            "tfidf": round(w_tf * lf, 4),
            "heuristic": round(w_h * lh, 4),
            "bias": round(bias, 4),
        },
        "components": {
            "tabular_model": {
                "weight_in_ensemble": round(w_tab, 4),
                "probability_genuine": round(p_tab_g, 4),
                "logit": round(lt, 4),
                "weighted_logit": round(w_tab * lt, 4),
                "description": "Logistic regression on MinMax-scaled tabular features (CSV-aligned NLP + derived text stats). Raw star count is not a column.",
                "per_feature_logit": sorted(
                    tab_breakdown, key=lambda z_: abs(z_["logit_contribution"]), reverse=True
                )[:8],
            },
            "char_tfidf_model": {
                "weight_in_ensemble": round(w_tf, 4),
                "probability_genuine": round(p_tf_g, 4),
                "logit": round(lf, 4),
                "weighted_logit": round(w_tf * lf, 4),
                "text_model_kind": text_kind,
                "tfidf_head": artifact.get("tfidf_head", "logistic_regression"),
                "description": {
                    "tfidf": "Char WB 3–5 + word 1–2-gram TF-IDF; LR or calibrated linear SVM (picked among TF-IDF heads on validation).",
                    "bert_finetuned": "Fine-tuned BERT sequence classifier using review title + review body as paired input; selected if it beats TF-IDF, SBERT, and FastText on validation macro-F1.",
                    "sbert": "Sentence-BERT (all-MiniLM-L6-v2) sentence embeddings + logistic regression; chosen if it beats TF-IDF and FastText on val macro-F1.",
                    "fasttext_mean": (
                        "Gensim FastText skip-gram on this corpus, mean word vectors + logistic regression (when Gensim is installed). "
                        "Otherwise: hashed char n-grams + SGD log-loss as a FastText-style subword baseline."
                    ),
                }.get(
                    text_kind,
                    "Text channel: dense or sparse representation + linear classifier.",
                ),
            },
            "heuristics": {
                "weight_in_ensemble": round(w_h, 4),
                "probability_genuine": round(p_heur_g, 4),
                "probability_genuine_mapping": round(p_heur_g, 4),
                "logit": round(lh, 4),
                "weighted_logit": round(w_h * lh, 4),
                "raw_risk_scores": {k: round(v, 4) for k, v in h_parts.items()},
                "aggregate_deceptive_risk": round(h_agg, 4),
                "description": "Weighted rule-based risk scores.",
            },
        },
        "user_metadata": {
            "rating_stars": int(rating),
            "verified_purchase": bool(verified_purchase),
            "note": "VERIFIED_PURCHASE matches the checkbox. RATING is not fed as a raw tabular feature.",
        },
        "input_features_used": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
    }

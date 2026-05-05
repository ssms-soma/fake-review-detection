# Combines tabular, TF-IDF, and heuristic probs in logit space; grid-searches nonnegative weights.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class EnsembleWeights:
    w_tabular: float
    w_tfidf: float
    w_heuristic: float

    def normalize(self) -> "EnsembleWeights":
        # Weights are stored nonnegative and renormalized to sum to 1 before blending.
        s = self.w_tabular + self.w_tfidf + self.w_heuristic
        if s <= 0:
            return EnsembleWeights(1 / 3, 1 / 3, 1 / 3)
        return EnsembleWeights(
            self.w_tabular / s, self.w_tfidf / s, self.w_heuristic / s
        )


@dataclass
class TrainReport:
    val_f1_macro: float
    val_accuracy: float
    weights: dict[str, float]
    tfidf_val_f1: float
    tabular_val_f1: float
    logit_bias: float
    n_train: int
    n_val: int
    selected_text_model: str = ""
    text_model_comparison_val_f1: dict[str, float] = field(default_factory=dict)


def _logit(p: np.ndarray) -> np.ndarray:
    # Clamp away from 0/1 so logit never hits ±inf on bad probabilities.
    p = np.clip(p.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def logit_scalar(p: float) -> float:
    p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(z: np.ndarray | float) -> Any:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def blend_logit(
    p_tab: np.ndarray,
    p_tf: np.ndarray,
    p_heur: np.ndarray,
    w_tab: float,
    w_tf: float,
    w_h: float,
    bias: float = 0.0,
) -> np.ndarray:
    z = (
        w_tab * _logit(p_tab)
        + w_tf * _logit(p_tf)
        + w_h * _logit(p_heur)
        + bias
    )
    return np.clip(_sigmoid(z), 1e-7, 1.0 - 1e-7)


def tune_logit_bias(z_val: np.ndarray, y_true: np.ndarray) -> float:
    # Small scalar shift on z before sigmoid; coarse grid search for best val macro-F1.
    best_b = 0.0
    best_f1 = -1.0
    for b in np.arange(-3.0, 3.05, 0.1):
        p = _sigmoid(z_val + b)
        pred = (p >= 0.5).astype(int)
        f1 = f1_score(y_true, pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_b = float(b)
    return best_b


MIN_HEURISTIC_WEIGHT = 0.08  # floor so transparent rules can still move the blended score


def _apply_min_heuristic(w: EnsembleWeights, min_h: float = MIN_HEURISTIC_WEIGHT) -> EnsembleWeights:
    w = w.normalize()
    if w.w_heuristic >= min_h:
        return w
    rest = 1.0 - min_h
    s = w.w_tabular + w.w_tfidf
    if s <= 0:
        return EnsembleWeights(rest / 2, rest / 2, min_h).normalize()
    return EnsembleWeights(
        rest * w.w_tabular / s,
        rest * w.w_tfidf / s,
        min_h,
    )


def tune_weights(
    p_tab: np.ndarray,
    p_tf: np.ndarray,
    h_risk: np.ndarray,
    y_true: np.ndarray,
) -> EnsembleWeights:
    # Heuristic channel uses risk; we convert to “genuine-like” prob as 1 − risk for the blend.
    best = EnsembleWeights(0.35, 0.50, 0.15)
    best_f1 = -1.0
    grid = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]
    h_gen = 1.0 - np.clip(h_risk, 0.0, 1.0)
    for a in grid:
        for b in grid:
            for c in grid:
                if a + b + c == 0:
                    continue
                w = EnsembleWeights(a, b, c).normalize()
                blend = blend_logit(
                    p_tab, p_tf, h_gen, w.w_tabular, w.w_tfidf, w.w_heuristic, 0.0
                )
                pred = (blend >= 0.5).astype(int)
                f1 = f1_score(y_true, pred, average="macro")
                if f1 > best_f1:
                    best_f1 = f1
                    best = w
    return best


def heuristic_to_genuine_score(h_risk: float) -> float:
    # Ensemble expects a pseudo-P(genuine); high risk flips to low genuine score.
    return 1.0 - float(np.clip(h_risk, 0.0, 1.0))

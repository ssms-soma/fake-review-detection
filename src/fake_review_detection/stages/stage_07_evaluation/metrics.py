# Turns probability vectors into thresholded preds and returns acc / F1 / confusion / ROC-AUC when defined.
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def metrics_from_probs(
    
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        if len(np.unique(y_true)) > 1:
            out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            out["roc_auc"] = None
    except ValueError:
        out["roc_auc"] = None
    try:
        if len(np.unique(y_true)) > 1:
            out["pr_auc"] = float(average_precision_score(y_true, y_prob))
        else:
            out["pr_auc"] = None
    except ValueError:
        out["pr_auc"] = None
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out["confusion_tn"] = int(tn)
    out["confusion_fp"] = int(fp)
    out["confusion_fn"] = int(fn)
    out["confusion_tp"] = int(tp)
    return out

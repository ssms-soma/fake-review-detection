# Scales all tabular columns to [0,1] then fits a balanced logistic regression.

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def build_tabular_pipeline() -> Pipeline:
    # clip=True keeps out-of-range user metadata inside [0,1] so logits stay stable.
    scaler = MinMaxScaler(clip=True)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    return Pipeline([("scale", scaler), ("lr", clf)])

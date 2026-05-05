# Public package: staged pipeline (preprocess through verdict) plus train/load/predict helpers.

__version__ = "2.0.0"

from fake_review_detection.config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    DATA_DIR,
    ENSEMBLE_PATH,
    META_PATH,
    REPORTS_DIR,
    TABULAR_DERIVED_FEATURES,
    TABULAR_FEATURES,
    TABULAR_FROM_CSV,
    TRAINING_CSV,
)
from fake_review_detection.stages.stage_06_ensemble.blend import EnsembleWeights, TrainReport
from fake_review_detection.stages.stage_06_ensemble.train_predict import (
    load_ensemble,
    predict_explain,
    train_from_csv,
)

__all__ = [
    "TRAINING_CSV",
    "ENSEMBLE_PATH",
    "META_PATH",
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "REPORTS_DIR",
    "TABULAR_FROM_CSV",
    "TABULAR_DERIVED_FEATURES",
    "TABULAR_FEATURES",
    "CLASS_NAMES",
    "train_from_csv",
    "predict_explain",
    "load_ensemble",
    "TrainReport",
    "EnsembleWeights",
]

# Shared paths and the ordered tabular feature list used by scaler + tabular logistic.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
REPORTS_DIR = ROOT / "reports" / "training"
TRAINING_CSV = DATA_DIR / "amazon_reviews_training.csv"
ENSEMBLE_PATH = ARTIFACTS_DIR / "ensemble.joblib"
META_PATH = ARTIFACTS_DIR / "ensemble_meta.json"

# Directory where the fine-tuned transformer model and tokenizer will be saved.
BERT_MODEL_DIR = ARTIFACTS_DIR / "bert_text_model"

# These columns are read straight from the training CSV (already engineered in the dataset).
TABULAR_FROM_CSV = [
    "VERIFIED_PURCHASE",
    "REVIEW_LENGTH",
    "TITLE_LENGTH",
    "SENTIMENT_SCORE",
    "COHERENT_ENCODED",
    "RATING_DEVIATION",
    "READABILITY_FRE",
    "NUM_NOUNS",
    "NUM_VERBS",
    "NUM_ADJECTIVES",
    "NUM_ADVERBS",
    "AVERAGE_RATING",
    "NUM_REVIEWS",
]

# We recompute these from review text at train and inference.
TABULAR_DERIVED_FEATURES = [
    "PUNCT_RATIO",
    "UPPER_RATIO",
    "DIGIT_RATIO",
    "AWL_BODY",
    "EXCLAIM_PER_100",
    "UNIQUE_WORD_RATIO",
    "MEAN_SENT_LEN",
    "VADER_POS",
    "VADER_NEG",
    # POS and stopword rates divided by word count so long and short reviews compare fairly.
    "NOUN_RATIO",
    "VERB_RATIO",
    "ADJ_RATIO",
    "ADV_RATIO",
    "STOPWORD_RATIO",
]

TABULAR_FEATURES = TABULAR_FROM_CSV + TABULAR_DERIVED_FEATURES

CLASS_NAMES = {0: "deceptive", 1: "genuine"}

# Faster transformer settings for local CPU training.
BERT_MODEL_NAME = "distilbert-base-cased"
BERT_MAX_LENGTH = 96
BERT_TRAIN_BATCH_SIZE = 8
BERT_EVAL_BATCH_SIZE = 16
BERT_NUM_EPOCHS = 1
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01
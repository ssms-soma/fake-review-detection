# Fits the full stack from the training CSV, saves ensemble.joblib, and writes reports/training/.
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import nltk

from fake_review_detection import META_PATH, REPORTS_DIR, TRAINING_CSV, train_from_csv


def main() -> None:
    nltk.download("vader_lexicon", quiet=True)  # sentiment for features and heuristics
    nltk.download("stopwords", quiet=True)  # stopword ratio in derived tabular features
    csv_path = TRAINING_CSV
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Training CSV missing: {csv_path}. Expected data/amazon_reviews_training.csv"
        )

    report = train_from_csv(csv_path, write_reports=True, reports_dir=REPORTS_DIR)
    print(json.dumps(asdict(report), indent=2))
    print("\n--- Stack steps (validation = hold-out test) ---")
    metrics_path = REPORTS_DIR / "training_metrics.json"
    if metrics_path.is_file():
        summary = json.loads(metrics_path.read_text(encoding="utf-8"))
        for s in summary.get("steps", []):
            vm = s["val_metrics"]
            d = s.get("delta_val_f1_macro")
            dstr = "" if d is None else f"  (dF1 vs prev: {d:+.4f})"
            print(
                f"  {s['title']}: acc={vm['accuracy']:.4f}  F1_macro={vm['f1_macro']:.4f}{dstr}"
            )
    print("\nArtifacts:", META_PATH.parent)
    print("Report + charts:", REPORTS_DIR)


if __name__ == "__main__":
    main()

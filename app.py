# Flask demo UI: train first with python scripts/train_pipeline.py, then run python app.py.
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import nltk
from flask import Flask, jsonify, render_template, request

from fake_review_detection import ENSEMBLE_PATH, load_ensemble, predict_explain

app = Flask(
    __name__,
    template_folder=str(ROOT / "web" / "templates"),
    static_folder=str(ROOT / "web" / "static"),
)

_artifact = None  # lazy load so import does not require artifacts on disk


def get_artifact():
    global _artifact
    if _artifact is None:
        if not ENSEMBLE_PATH.is_file():
            raise RuntimeError(
                f"Missing {ENSEMBLE_PATH}. Run: python scripts/train_pipeline.py"
            )
        _artifact = load_ensemble()
    return _artifact


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    # Grab NLTK data if missing; harmless when already installed.
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)
    data = request.get_json(silent=True) or {}
    text = (data.get("review_text") or "").strip()
    if len(text) < 5:
        return jsonify({"error": "Please enter at least a few words of review text."}), 400

    rating = int(data.get("rating") or 3)
    rating = max(1, min(5, rating))
    verified = bool(data.get("verified_purchase"))
    title = (data.get("title") or "").strip()
    avg = data.get("average_rating")
    num_rev = data.get("num_reviews")
    try:
        avg_f = float(avg) if avg is not None and avg != "" else None
    except (TypeError, ValueError):
        avg_f = None
    try:
        num_i = int(num_rev) if num_rev is not None and num_rev != "" else None
    except (TypeError, ValueError):
        num_i = None

    try:
        out = predict_explain(
            text,
            rating=rating,
            verified_purchase=verified,
            title=title,
            average_rating=avg_f,
            num_reviews=num_i,
            artifact=get_artifact(),
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(out)


def main():
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("stopwords", quiet=True)
    print("Open http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()

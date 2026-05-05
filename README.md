# Fake Review Detection System

A machine learning system for identifying whether a product review is genuine or deceptive using natural language processing, metadata, and heuristic analysis.

---

## Overview

This project implements an end-to-end pipeline that combines multiple approaches to detect fake reviews:

* Text-based modeling using TF-IDF and BERT embeddings
* Tabular modeling using linguistic and metadata features
* Rule-based heuristics for identifying suspicious patterns
* An ensemble layer to combine all signals into a final prediction

The system outputs a probability score along with interpretable signals that explain the prediction.

---

## Pipeline

1. Input review text (with optional metadata such as rating or title)
2. Preprocess and normalize text
3. Extract linguistic and behavioral features
4. Generate predictions from:

   * Text models (TF-IDF and BERT-based representations)
   * Tabular model (engineered features)
   * Heuristic rules
5. Combine outputs using an ensemble model
6. Return final probability and explanation

---

## Tech Stack

* Python
* Scikit-learn
* PyTorch / Transformers (BERT)
* spaCy
* NLTK (VADER)
* Flask

---

## Setup and Usage

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/train_pipeline.py
python app.py
```

Access the application at: http://127.0.0.1:5000

---

## Project Structure

* `app.py` – Flask application
* `scripts/train_pipeline.py` – Training pipeline
* `src/` – Core implementation (feature engineering, models, ensemble)
* `artifacts/` – Saved models
* `reports/` – Training outputs and evaluation

---

## Output

The system returns:

* Probability of a review being genuine
* Supporting signals from models and heuristics

---

## Limitations

* Predictions are probabilistic and not definitive
* Performance depends on training data quality
* Does not determine intent, only statistical patterns

---

## Future Work

* Improve model calibration and evaluation
* Optimize inference performance
* Enhance explainability of predictions

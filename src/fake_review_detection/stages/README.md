# Pipeline stages (for demos / marking)

Data flows **down this list** at inference; training uses the same stages to fit weights and write `artifacts/`.

| Folder | Role |
|--------|------|
| **`stage_01_preprocess/`** | Unicode / HTML / whitespace; body lowercased for models; title keeps case for rules. |
| **`stage_02_features/`** | Tabular row: spaCy POS, VADER, readability, metadata; **derived** stylometry + lexical diversity + VADER pos/neg. |
| **`stage_03_heuristics/`** | Rule-based deceptive-risk scores (star–text mismatch, repetition, marketing lexicon, …). |
| **`stage_04_text_model/`** | Char + word TF-IDF; LR vs calibrated LinearSVM chosen on validation. |
| **`stage_05_tabular_model/`** | MinMax + logistic regression on all tabular columns. |
| **`stage_06_ensemble/`** | Logit fusion, weight grid, bias tuning, `train_from_csv`, `predict_explain`. |
| **`stage_07_evaluation/`** | Metrics + PNG/HTML/JSON training reports. |
| **`stage_08_output/`** | Verdict labels from P(genuine). |

Shared paths and feature name lists live in **`../config/`**.

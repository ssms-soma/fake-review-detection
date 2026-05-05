# Pipeline architecture diagram

This file contains a single **Mermaid** diagram of the fake-review scoring stack: from HTTP input through preprocessing, parallel feature branches, ensemble fusion, and JSON/UI output.

```mermaid
flowchart TB
  subgraph Client
    UI[HTML/CSS form]
  end

  subgraph Flask["Flask (app.py)"]
    API["POST /api/predict"]
  end

  subgraph Pre["Preprocessing"]
    NORM["normalize_review_text / title\nNFKC · unescape HTML · whitespace"]
  end

  subgraph TabBranch["Tabular branch"]
    FEAT["extract_tabular_row\nVADER · FRE · POS · coherence · verified\n(no raw RATING column)"]
    TABPIPE["MinMaxScaler → LogisticRegression"]
    TABP["P_genuine_tab"]
    EXPLAIN["Top |coef·x_scaled| for logit"]
  end

  subgraph TxtBranch["Text branch"]
    TFIDF["FeatureUnion: char_wb 3–5 + word 1–2 ngrams\nsublinear TF · min/max df caps"]
    TXTLR["LogisticRegression or calibrated LinearSVC\n(validation pick)"]
    TXTP["P_genuine_tfidf"]
  end

  subgraph Heur["Heuristic branch"]
    HRULES["Mismatch · brevity · caps · repetition"]
    HAGG["aggregate_risk → P_genu_heur = 1 − risk"]
  end

  subgraph Fuse["Ensemble (logit blend + bias b)"]
    W["z = Σ w·logit(P) + b\nP = σ(z)"]
    OUT["P_genuine , P_deceptive , verdict"]
  end

  subgraph Art["Artifacts"]
    JOB["artifacts/ensemble.joblib"]
  end

  UI --> API
  API --> NORM
  NORM --> FEAT
  FEAT --> TABPIPE
  TABPIPE --> TABP
  TABPIPE --> EXPLAIN
  NORM --> TFIDF
  TFIDF --> TXTLR
  TXTLR --> TXTP
  NORM --> HRULES
  HRULES --> HAGG
  TABP --> W
  TXTP --> W
  HAGG --> W
  W --> OUT
  OUT --> UI
  JOB -.-> TABPIPE
  JOB -.-> TXTLR
  JOB -.-> W
```

### Reading the diagram

- **Solid arrows** are runtime data flow on each prediction.
- **Dotted arrows** show that fitted parameters (vectorizer vocabulary, scaler bounds, logistic weights, ensemble weights) are loaded from `ensemble.joblib` produced by `scripts/train_pipeline.py`.
- **Text branch** matches `tfidf_pipeline.py`: character n-grams (width 3–5, word-boundary aware) *and* word unigrams/bigrams in one `FeatureUnion`; training picks either a logistic head or a calibrated linear SVM by validation macro-F1.

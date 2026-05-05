# Fake review detection — production-style NLP pipeline

This repository is organized into **(1)** a small **Flask app** that scores a single review with an **explainable ensemble**, and **(2)** a **training script** that fits models on **`data/amazon_reviews_training.csv`**, writes **`artifacts/`**, and generates **metrics + charts** under **`reports/training/`** (see `training_report.html`). File-level documentation lives in **[CODEBASE.md](CODEBASE.md)**.

The UI shows **P(genuine)** and **P(deceptive)** plus a **breakdown**: each sub-model’s **P(genuine)**, its **weight × logit(P)** toward the combined score **z**, then **σ(z)** as the headline probability; heuristic sub-scores; and the largest **Δ logit** terms from the tabular logistic model. Nothing here is legal proof of fraud; labels come from a crowdsourced corpus and models capture **statistical cues**, not intent.

---

## Quick start

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/train_pipeline.py    # artifacts + reports/training/ (HTML + PNG + JSON)
python app.py                       # http://127.0.0.1:5000
```

The first run downloads the VADER lexicon automatically. The first training run also downloads the **Sentence-BERT** weights (~90MB for MiniLM) and **PyTorch** if missing; comparing TF-IDF, SBERT, and FastText adds CPU time (often a few minutes on a laptop for ~21k rows). After training, open **`reports/training/training_report.html`** for tables, figures, and the **text-method comparison** chart.

### What data trains the shipped model?

**Only** [`data/amazon_reviews_training.csv`](data/amazon_reviews_training.csv) — the pre-engineered Amazon MTurk-style **labeled** export (~21k rows). Extra copies (Yelp, unlabelled Amazon, etc.) were removed from this repo to keep the project minimal; add them back only if you extend the trainer.

---

## Repository layout

| Path | Role |
|------|------|
| `app.py` | Flask app: serves `web/` and `/api/predict`. |
| `src/fake_review_detection/` | Library: `config/`, numbered `stages/` (preprocess → features → heuristics → models → ensemble → evaluation → verdict). |
| `src/fake_review_detection/stages/README.md` | Short stage map for demos / marking. |
| `scripts/train_pipeline.py` | Train + write `artifacts/` and `reports/training/`. |
| `data/amazon_reviews_training.csv` | Sole training file for the default script. |
| `artifacts/` | `ensemble.joblib`, `ensemble_meta.json` (generated). |
| `reports/training/` | `training_report.html`, PNG charts, `training_metrics.json` (generated). |
| `CODEBASE.md` | Per-file codebase explanation. |
| `README_PIPELINE_DIAGRAM.md` | Mermaid inference diagram. |

---

## End-to-end pipeline (what goes in, what comes out)

### 1. Inputs

- **Required:** review body (plain text).
- **Optional but useful:** title, star rating (1–5), verified-purchase checkbox (**fed into tabular `VERIFIED_PURCHASE` as 0/1**).
- **Optional product context:** average product rating and review count. If omitted, we use a **cold-start default**: average rating = the user’s own stars, review count = 1, so `RATING_DEVIATION = 0`. That matches “I only see this one review” and is documented so users do not over-trust aggregate features.
- **Why huge review counts used to break scores:** `MinMaxScaler` was fit on the training CSV. Entering a `NUM_REVIEWS` **above the training maximum** produced scaled values **greater than 1**, which could blow up the tabular logistic logit (e.g. P(genuine) → 0). The pipeline now uses **`MinMaxScaler(clip=True)`** and **clips** optional `NUM_REVIEWS` / `AVERAGE_RATING` to the min–max seen in training (stored in the artifact), so UI fields behave like “in-distribution” inputs.

### 2. Preprocessing (`stages/stage_01_preprocess/normalize.py`)

- Unicode NFKC normalization, HTML entity unescape, whitespace cleanup; review body lowercased for models.
- Title keeps case for title-based heuristics.

### 3. Tabular NLP + derived text features (`stages/stage_02_features/`)

Aligned with the engineered columns in the training CSV:

| Feature group | Examples | Rationale |
|---------------|----------|-----------|
| Lexical / stylistic | Review length, average word length | Spam and templated reviews often differ in length and token statistics. |
| Readability | Flesch Reading Ease (`textstat`) | Scripted or low-effort campaigns sometimes cluster in readability. |
| Sentiment | VADER compound score | Mismatch with stars is a classic deception cue (handled again in heuristics). |
| Coherence | Binary: sentiment polarity sign vs. rating threshold | Encodes whether tone matches coarse star bucket. |
| Syntax | spaCy POS counts (NOUN, VERB, ADJ, ADV) | Part-of-speech ratios capture “salesy” or generic language patterns. |
| Metadata | Verified purchase (checkbox), title length | Matches the CSV column at train time; at inference uses the user’s tick state. |
| Product aggregates | Average rating, count, deviation | **Strong when known**; **neutralized** when unknown via cold-start defaults above. |

**Raw `RATING` (1–5) is not a tabular column:** the Amazon corpus correlates high stars with deceptive spam, so including star count made “same positive text + lower stars” score *better* — an artifact, not a moral signal. Stars still affect **`COHERENT_ENCODED`** (VADER sign vs. coarse bucket) and **heuristics** (e.g. glowing text with 1–2★ is flagged as inconsistent).

**CSV columns** are joined with **derived** fields computed from text: stylometry (punctuation, caps, digits, exclamation rate, avg word length), **lexical diversity** (unique-token ratio), **mean words per sentence**, and **VADER pos/neg** (in addition to compound in the main row).

**Outputs:** one vector matching `config.settings.TABULAR_FEATURES` (see `config/settings.py`).

### 4. Text sub-model: TF-IDF vs Sentence-BERT vs FastText (`stages/stage_04_text_model/`)

On the **same** stratified validation fold, training scores **three** text encoders by **macro-F1**; the winner is refit on **all** labeled rows and saved as the text channel (`text_model_kind` in `ensemble.joblib`). A bar chart compares them in **`reports/training/text_model_comparison_val_f1.png`** (also embedded in **`training_report.html`**).

| Method | File(s) | Role |
|--------|---------|------|
| **TF-IDF** | `tfidf_pipeline.py` | Sparse **char** (WB 3–5) + **word** (1–2) n-grams; head is **LR** or **calibrated LinearSVC** (whichever wins among TF-IDF heads only). |
| **Sentence-BERT** | `embedding_classifiers.py` | **`all-MiniLM-L6-v2`** encodes the **whole review** into a dense vector → scaled **logistic regression**. |
| **FastText** | `embedding_classifiers.py` | **Preferred:** **Gensim** skip-gram on this corpus + **mean** word vectors + LR (install `gensim` when a wheel exists for your Python). **Fallback:** hashed **char n-grams** + **SGD** (subword-style signal without native FastText). |

**What embeddings do here:** They are **only** used in the **text branch** of the ensemble: they map review text to a **fixed-size vector** so a **linear** classifier can emit `P(genuine)` from wording alone. They do **not** replace tabular features (POS, VADER, metadata) or heuristics—they compete to be the best “second read” of the raw text alongside those signals.

**Why one can beat another:** TF-IDF shines when **exact tokens and character patterns** separate the classes. **SBERT** can help when **meaning and paraphrase** matter more than literal n-gram overlap (it is **pretrained** on large text). **Corpus-trained FastText** can help when **domain-specific** words/subwords are informative but may trail SBERT if the training set is small relative to a strong pretrained encoder. Your run’s chart and `training_metrics.json` → `extra` show which won on **your** split.

### 5. Tabular sub-model: MinMax scale + logistic regression

- **Why MinMax:** Features live on incomparable scales (counts vs. probabilities vs. readability). Bounding to [0,1] stabilizes `lbfgs` training next to binary indicators.
- **Output:** `P(genuine)` from the metadata + linguistic vector.

**Explainability:** For a single prediction we decompose the linear part: for each feature, `coefficient × scaled_value` ranks which drove the logit toward genuine vs. deceptive (top 8 shown in the UI).

### 6. Heuristic checks (`stages/stage_03_heuristics/rules.py`)

Rule-based scores in [0,1] (higher = more suspicious), then aggregated with fixed sub-weights:

- Rating vs. sentiment mismatch (e.g. 5★ + strongly negative text).
- **Glowing text + 1–2★** (inconsistent; dampens “lower stars to game the score” on positive prose).
- Very short 5★ body.
- “Shouting” title (capital ratio).
- Lexical repetition of non-stopwords.
- Unverified + very short review (small weight — complements the verified tabular bit).
- Heavy punctuation with high rating; **marketing / evaluative lexicon density** (survey-style spam cue).

These are **transparent** and cheap; the trainer enforces a **minimum heuristic weight** in the blend so this channel can move the headline score, not only appear as diagnostics.

### 7. Ensemble weighting (`stages/stage_06_ensemble/` — validation-tuned)

On a stratified 80/20 split we **grid-search nonnegative weights** `(w_tabular, w_tfidf, w_heuristic)` that maximize **macro-F1** under a **logit-space blend**, then fit a small **logit bias** on validation. Weights renormalize to sum to 1.

**Typical shipped metrics** (see `artifacts/ensemble_meta.json` after retraining):

| Setup | Validation macro-F1 (approx.) |
|--------|-------------------------------|
| Tabular LR only | **~0.79** |
| Char TF-IDF LR only | **~0.65** |
| Blended (logit + bias) | **~0.79–0.82** (depends on heuristic floor) |
| Heuristic weight | At least **~0.08** (enforced floor so star–text rules affect the blended score; validation F1 may drop slightly vs. heuristic-free blend) |

**Final score** (per row):

\[
z = w_t \,\text{logit}(P^{\text{tab}}) + w_f \,\text{logit}(P^{\text{tfidf}}) + w_h \,\text{logit}(P^{\text{heur}}) + b,\qquad
P_{\text{genuine}} = \sigma(z)
\]

The UI lists **weight × logit(P)** per channel; these add to **z** (plus **bias**), not to **P** directly.

### 8. Training data and label semantics

- Primary training file: `data/amazon_reviews_training.csv`.
- **Class 0:** deceptive (per corpus / `__label1__` side of the original Amazon MTurk-style split).
- **Class 1:** genuine (`__label2__` side).
- Empirically, class **1** aligns with higher **verified purchase** rate and longer reviews in this export — consistent with “organic” Amazon reviews vs. the paired deceptive set.

---

## Design trade-offs (NLP engineering view)

1. **Lexical + metadata vs. deep transformers:** Transformers (BERT, etc.) can win on raw text but cost latency, GPU RAM, and explainability. Here we prioritize **fast CPU inference**, **inspectable linear factors**, and a path that matches the historical feature engineering in the coursework notebooks.
2. **Cold-start product features:** Without a product graph, aggregate features are placeholders. The README and UI state this explicitly to avoid **false precision**.
3. **Heuristics vs. learning:** Rules catch edge cases but overlap with what VADER + LR already absorb. We keep them as **explicit documentation** of business logic, not as mandatory score drivers.
4. **Calibration:** We report `predict_proba` from logistic heads. These are **not** guaranteed well-calibrated out of domain; for production you would add Platt scaling or isotonic regression on a fresh validation set.

---

## Architecture diagram

See **[README_PIPELINE_DIAGRAM.md](README_PIPELINE_DIAGRAM.md)** for a standalone Mermaid figure of the inference stack. It matches the implementation: the text branch uses **char WB 3–5** plus **word 1–2** grams in a `FeatureUnion`, and **logistic regression vs calibrated LinearSVC** is chosen on validation macro-F1 (`tfidf_pipeline.py`).

---

## Faculty Q&A (presentation cheat sheet)

### Code comments (`#` not `//`)

This project is **Python**. Single-line comments use **`#`**. The operator **`//`** is *integer floor division*, not a comment, so we do not use C/Java-style `//` comments in source files.

### Where did we get the dataset?

Training uses **only** [`data/amazon_reviews_training.csv`](data/amazon_reviews_training.csv): a **labeled** Amazon-style export (on the order of **~21k rows**) with genuine vs deceptive annotations in the usual MTurk/coursework style. **Class 0** = deceptive, **class 1** = genuine (also recorded in `artifacts/ensemble_meta.json` after you train). The web app scores text you submit; it does not download fresh Amazon pages at runtime.

### What NLP techniques are we using, and where?

| Technique | Where in code | Role |
|-----------|---------------|------|
| **Unicode + HTML cleanup** | `stage_01_preprocess/normalize.py` | NFKC, `html.unescape`, whitespace; review body lowercased for models. |
| **POS tagging (coarse counts)** | `linguistic.py` — spaCy `en_core_web_sm` | `NUM_NOUNS`, `NUM_VERBS`, `NUM_ADJECTIVES`, `NUM_ADVERBS`; ratios in `derived_text.py`. |
| **Regex word tokens** | `derived_text.py`, `rules.py`, `linguistic.avg_word_length` | Counts and lengths via `\b\w+\b` on raw or normalized text (per feature). |
| **VADER** | `linguistic.py`, `derived_text.py`, heuristics | Compound for tabular row; pos/neg facets; star–sentiment mismatch rules. |
| **Readability** | `linguistic.py` — `textstat` Flesch Reading Ease | One scalar in the tabular vector. |
| **Stopwords** | `derived_text.py` — NLTK English list | `STOPWORD_RATIO` over alphabetic tokens. |
| **Stylometry** | `derived_text.py` | Punctuation, caps, digits, exclamations, unique-token ratio, mean words per sentence. |
| **TF‑IDF** | `tfidf_pipeline.py` | `FeatureUnion`: char WB n-grams (3–5) + word n-grams (1–2), sublinear TF, df caps. |
| **Linear models** | `tabular_pipeline.py`, `tfidf_pipeline.py` | `MinMaxScaler` + balanced `LogisticRegression` (tabular); TF-IDF + LR **or** calibrated `LinearSVC` (validation pick). |
| **Hand-crafted rules** | `stage_03_heuristics/rules.py` | Risk scores → weighted aggregate → pseudo P(genuine) = 1 − risk. |
| **Ensemble** | `blend.py`, `train_predict.py` | Logit blend of three channels + tuned bias; nonnegative weights from validation grid search. |

We **do not** run transformers (BERT, etc.) here: the goal is **fast CPU inference** and **explainable** linear cues.

### Why this approach instead of a single end-to-end model?

- **Tabular LR** uses metadata and linguistics that the dataset already encodes or that we recompute identically at inference.
- **TF‑IDF + linear head** picks up **lexical and subword** patterns that a small set of summary numbers can miss.
- **Heuristics** encode **auditable** rules (mismatch, brevity, shouting, repetition, marketing lexicon).
- **Logit blending** puts channels on a **comparable scale** before the sigmoid; weights optimize **validation macro-F1**. Together this stays **lightweight** while often **matching or edging** tabular-only accuracy/F1 in our reports.

### How do we get noun/verb counts and word counts?

- **Nouns, verbs, adjectives, adverbs:** Normalized review → **spaCy** pipeline (`en_core_web_sm`, **NER and parser off**). `doc.count_by(spacy.attrs.POS)` gives token counts per coarse tag; we read `NOUN`, `VERB`, `ADJ`, `ADV` in `count_pos_tags_from_doc` (`linguistic.py`) → **`NUM_NOUNS`**, **`NUM_VERBS`**, etc.
- **POS ratios (`NOUN_RATIO`, …):** Same spaCy counts divided by **`max(1, len(words_norm))`**, where `words_norm = re.findall(r"\b\w+\b", normalized_text)` (`derived_text.py`).
- **Heuristic word counts (e.g. very short 5★ review):** `len(re.findall(r"\b\w+\b", review_text))` in `rules.py` on the string each rule uses (normalized body from the API path).
- **Average word length:** Regex words, lowercased, in `avg_word_length` (`linguistic.py`).

**Summary:** Syntactic category counts = **spaCy POS**; generic “how many words” = **regex word tokens** on the chosen string.

### The CSV already has lots of features — what are we adding or changing?

- **Loaded from CSV:** Columns in `TABULAR_FROM_CSV` (`config/settings.py`) — the pre-engineered side of the dataset (verified, lengths, sentiment score, coherence, deviation, FRE, POS **counts**, product stats).
- **Recomputed from text every time:** `TABULAR_DERIVED_FEATURES` — stylometry, diversity, sentence-length stats, VADER pos/neg, POS/stopword **ratios** — so **training rows** and **live UI** use the **same** NLP, not stale CSV-only values for those fields.
- **Raw star rating** is **not** a tabular column: it correlated with the label in a misleading way; stars still drive **coherence** and **heuristics**.
- **Inference safety:** Optional aggregates are **clipped** to training min–max; `MinMaxScaler(clip=True)` avoids logits blowing up on extreme user input.

### Full pipeline: what enters and leaves each stage

1. **Raw input:** Review text; optional title, stars, verified flag, optional product average rating and review count.
2. **Stage 1 — Preprocess:** Normalize body and title. **Out:** clean strings (body lowercased).
3. **Stage 2 — Features:** Tabular dict = CSV-aligned row + derived NLP/stylometry. **Out:** numeric vector `TABULAR_FEATURES` in fixed order.
4. **Stage 3 — Heuristics:** Rule scores + weighted risk. **Out:** aggregate risk and P_heur(genuine) = 1 − risk.
5. **Stage 4 — Text model:** TF‑IDF transform + linear classifier. **Out:** P_tfidf(genuine).
6. **Stage 5 — Tabular model:** Scale + logistic. **Out:** P_tab(genuine) + top `coef × scaled_value` terms for explanation.
7. **Stage 6 — Ensemble:** z = Σ w·logit(P) + b; σ(z) = final P(genuine). **Out:** headline probability + per-channel breakdown for JSON/UI.
8. **Stage 7 — Evaluation (train only):** Metrics and plots → `reports/training/`.
9. **Stage 8 — Verdict:** Map P(genuine) to short text bands. **Out:** user-facing label.

**Persisted bundle:** `artifacts/ensemble.joblib` (both sub-models, weights, bias, feature names, heuristic weights, clipping bounds).

### More questions she might ask

- **Validation strategy?** Stratified **80/20** split; ensemble weights and bias tuned on the validation fold; final sub-models refit on **all** rows for the shipped artifact.
- **Imbalance?** `class_weight="balanced"` on logistic (and SVM) heads.
- **Why not BERT?** Different cost/ops profile; this demo is about classical NLP, linear models, and explicit rules.
- **Are probabilities “true” frequencies?** Not necessarily out of domain; they are scores from linear heads + heuristic mapping.
- **Ethics / misuse?** Output is a statistical screening aid, not proof of dishonesty or intent.

---

## Credits

- **VADER:** Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- **spaCy:** Explosion AI.
- **GloVe** (Pennington et al., 2014) is cited in many academic write-ups of embedding baselines; this repo’s runtime model does not load GloVe.

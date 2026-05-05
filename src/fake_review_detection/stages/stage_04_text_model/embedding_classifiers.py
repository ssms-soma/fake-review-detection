# Dense text channels: Sentence-BERT + LR; FastText (Gensim) or subword hash fallback + LR.

from __future__ import annotations

from typing import Any
from .bert_classifier import FineTunedBertClassifier
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .tfidf_pipeline import fit_best_tfidf

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"


def _gensim_available() -> bool:
    try:
        from gensim.models import FastText  # noqa: F401

        return True
    except ImportError:
        return False


class SbertLogisticClassifier:
    # Sentence embedding then linear layer; encoder is lazy-loaded after unpickling.
    def __init__(self, model_name: str = SBERT_MODEL_NAME) -> None:
        self.model_name = model_name
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        self._encoder: Any = None

    def __getstate__(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["_encoder"] = None
        return d

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def fit(self, texts: list[str], y: np.ndarray) -> "SbertLogisticClassifier":
        enc = self._get_encoder()
        X = np.asarray(
            enc.encode(
                texts,
                show_progress_bar=False,
                batch_size=128,
                convert_to_numpy=True,
            ),
            dtype=np.float64,
        )
        Xs = self.scaler.fit_transform(X)
        self.lr.fit(Xs, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        enc = self._get_encoder()
        X = np.asarray(
            enc.encode(
                texts,
                show_progress_bar=False,
                batch_size=128,
                convert_to_numpy=True,
            ),
            dtype=np.float64,
        )
        Xs = self.scaler.transform(X)
        return self.lr.predict_proba(Xs)

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.predict_proba(texts)[:, 1] >= 0.5).astype(int)


class SklearnSubwordHashClassifier:
    # FastText-like subword signal without native FastText: hashed char n-grams + linear SGD.
    def __init__(
        self,
        *,
        n_features: int = 65_536,
        ngram_range: tuple[int, int] = (2, 5),
    ) -> None:
        self.n_features = n_features
        self.ngram_range = ngram_range
        self.vec = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )
        self.clf = SGDClassifier(
            loss="log_loss",
            class_weight="balanced",
            random_state=42,
            max_iter=2500,
            tol=1e-4,
        )

    def fit(self, texts: list[str], y: np.ndarray) -> "SklearnSubwordHashClassifier":
        X = self.vec.transform(texts)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict_proba(self.vec.transform(texts))

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict(self.vec.transform(texts))


class FastTextMeanClassifier:
    # Train skip-gram FastText on the corpus, average word vectors per review, then LR.
    backend_tag = "gensim"

    def __init__(
        self,
        *,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 12,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed
        self.ft: Any = None
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )

    def _tokenize(self, texts: list[str]) -> list[list[str]]:
        from gensim.utils import simple_preprocess

        return [simple_preprocess(t or "", deacc=True) for t in texts]

    def _doc_vec(self, tokens: list[str]) -> np.ndarray:
        if self.ft is None:
            return np.zeros(self.vector_size, dtype=np.float64)
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float64)
        vecs = [self.ft.wv[t] for t in tokens if t in self.ft.wv]
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float64)
        return np.mean(np.vstack(vecs), axis=0)

    def fit(self, texts: list[str], y: np.ndarray) -> "FastTextMeanClassifier":
        from gensim.models import FastText

        tokenized = self._tokenize(texts)
        self.ft = FastText(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=1,
            seed=self.seed,
            sg=1,
        )
        self.ft.build_vocab(corpus_iterable=tokenized)
        self.ft.train(
            corpus_iterable=tokenized,
            total_examples=len(tokenized),
            epochs=self.epochs,
        )
        X = np.vstack([self._doc_vec(t) for t in tokenized])
        Xs = self.scaler.fit_transform(X)
        self.lr.fit(Xs, y)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        tokenized = self._tokenize(texts)
        X = np.vstack([self._doc_vec(t) for t in tokenized])
        Xs = self.scaler.transform(X)
        return self.lr.predict_proba(Xs)

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.lr.predict(
            self.scaler.transform(
                np.vstack([self._doc_vec(t) for t in self._tokenize(texts)])
            )
        )


def _make_fasttext_branch() -> tuple[Any, str]:
    if _gensim_available():
        return FastTextMeanClassifier(), "gensim"
    return SklearnSubwordHashClassifier(), "sklearn_hash"


def compare_text_models_val_f1(
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
    *,
    train_titles: list[str] | None = None,
    train_bodies_for_bert: list[str] | None = None,
    val_titles: list[str] | None = None,
    val_bodies_for_bert: list[str] | None = None,
) -> tuple[str, dict[str, float], dict[str, float], Any, str, str]:
    tf_pipe, tf_head = fit_best_tfidf(train_texts, y_train, val_texts, y_val)
    f1_tfidf = float(
        f1_score(y_val, tf_pipe.predict(val_texts), average="macro", zero_division=0)
    )

    sbert = SbertLogisticClassifier()
    sbert.fit(train_texts, y_train)
    f1_sbert = float(
        f1_score(y_val, sbert.predict(val_texts), average="macro", zero_division=0)
    )

    ft_model, ft_backend = _make_fasttext_branch()
    ft_model.fit(train_texts, y_train)
    f1_ft = float(
        f1_score(y_val, ft_model.predict(val_texts), average="macro", zero_division=0)
    )

    models: dict[str, Any] = {
        "tfidf": tf_pipe,
        "sbert": sbert,
        "fasttext_mean": ft_model,
    }
    scores_f1 = {
        "tfidf": f1_tfidf,
        "sbert": f1_sbert,
        "fasttext_mean": f1_ft,
    }
    scores_acc = {
        "tfidf": float(accuracy_score(y_val, tf_pipe.predict(val_texts))),
        "sbert": float(accuracy_score(y_val, sbert.predict(val_texts))),
        "fasttext_mean": float(accuracy_score(y_val, ft_model.predict(val_texts))),
    }

    if (
        train_titles is not None
        and train_bodies_for_bert is not None
        and val_titles is not None
        and val_bodies_for_bert is not None
    ):
        bert = FineTunedBertClassifier()
        bert.fit(
            train_titles,
            train_bodies_for_bert,
            y_train,
            eval_titles=val_titles,
            eval_bodies=val_bodies_for_bert,
            y_eval=y_val,
        )
        bert_preds = bert.predict(val_titles, val_bodies_for_bert)
        scores_f1["bert_finetuned"] = float(
            f1_score(y_val, bert_preds, average="macro", zero_division=0)
        )
        scores_acc["bert_finetuned"] = float(accuracy_score(y_val, bert_preds))
        models["bert_finetuned"] = bert

    order = ["tfidf", "bert_finetuned", "sbert", "fasttext_mean"]
    available = [k for k in order if k in scores_f1]
    best_key = max(available, key=lambda k: (scores_f1[k], -available.index(k)))
    return best_key, scores_f1, scores_acc, models[best_key], tf_head, ft_backend

def refit_text_model_full(
    best_key: str,
    all_texts: list[str],
    y: np.ndarray,
    tfidf_head_name: str,
    *,
    fasttext_backend: str = "sklearn_hash",
    all_titles: list[str] | None = None,
    all_bodies_for_bert: list[str] | None = None,
) -> tuple[Any, str]:
    if best_key == "tfidf":
        from .tfidf_pipeline import build_tfidf_pipeline_lr, build_tfidf_pipeline_svc

        if tfidf_head_name == "linear_svc_calibrated":
            pipe = build_tfidf_pipeline_svc()
        else:
            pipe = build_tfidf_pipeline_lr()
        pipe.fit(all_texts, y)
        return pipe, tfidf_head_name

    if best_key == "sbert":
        return SbertLogisticClassifier().fit(all_texts, y), "sbert_minilm"

    if best_key == "fasttext_mean":
        if fasttext_backend == "gensim" and _gensim_available():
            return FastTextMeanClassifier().fit(all_texts, y), "fasttext_mean_pool"
        return SklearnSubwordHashClassifier().fit(all_texts, y), "fasttext_subword_hash"

    if best_key == "bert_finetuned":
        if all_titles is None or all_bodies_for_bert is None:
            raise ValueError("BERT refit requires titles and BERT bodies.")
        model = FineTunedBertClassifier().fit(all_titles, all_bodies_for_bert, y)
        return model, "bert_finetuned"

    raise ValueError(f"Unknown text model key: {best_key}")


def text_model_predict_proba(
    model: Any,
    texts: list[str],
    *,
    titles: list[str] | None = None,
    bodies_for_bert: list[str] | None = None,
) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        try:
            if titles is not None and bodies_for_bert is not None:
                proba = model.predict_proba(titles, bodies_for_bert)
            else:
                proba = model.predict_proba(texts)
        except TypeError:
            proba = model.predict_proba(texts)
    else:
        raise ValueError("Text model does not support predict_proba.")
    return np.asarray(proba[:, 1], dtype=np.float64)

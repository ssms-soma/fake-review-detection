from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from fake_review_detection.config import (
    BERT_EVAL_BATCH_SIZE,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    BERT_NUM_EPOCHS,
    BERT_TRAIN_BATCH_SIZE,
    BERT_WEIGHT_DECAY,
)


class _TextPairDataset(Dataset):
    def __init__(self, encodings: dict[str, Any], labels: np.ndarray | None = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


@dataclass
class BertTrainConfig:
    model_name: str = BERT_MODEL_NAME
    model_dir: Path = BERT_MODEL_DIR
    max_length: int = BERT_MAX_LENGTH
    train_batch_size: int = BERT_TRAIN_BATCH_SIZE
    eval_batch_size: int = BERT_EVAL_BATCH_SIZE
    num_epochs: int = BERT_NUM_EPOCHS
    learning_rate: float = BERT_LEARNING_RATE
    weight_decay: float = BERT_WEIGHT_DECAY
    seed: int = 42


class FineTunedBertClassifier:
    def __init__(self, config: BertTrainConfig | None = None) -> None:
        self.config = config or BertTrainConfig()
        self._tokenizer = None
        self._model = None

    def __getstate__(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["_tokenizer"] = None
        d["_model"] = None
        return d

    def _get_tokenizer(self):
        if self._tokenizer is None:
            load_dir = self.config.model_dir if self.config.model_dir.exists() else self.config.model_name
            self._tokenizer = AutoTokenizer.from_pretrained(load_dir)
        return self._tokenizer

    def _get_model(self):
        if self._model is None:
            load_dir = self.config.model_dir if self.config.model_dir.exists() else self.config.model_name
            self._model = AutoModelForSequenceClassification.from_pretrained(
                load_dir,
                num_labels=2,
                id2label={0: "deceptive", 1: "genuine"},
                label2id={"deceptive": 0, "genuine": 1},
            )
        return self._model

    def _encode(self, titles: list[str], bodies: list[str]) -> dict[str, Any]:
        tok = self._get_tokenizer()
        return tok(
            titles,
            bodies,
            truncation=True,
            max_length=self.config.max_length,
            padding=False,
        )

    @staticmethod
    def _compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        }

    def fit(
        self,
        titles: list[str],
        bodies: list[str],
        y: np.ndarray,
        *,
        eval_titles: list[str] | None = None,
        eval_bodies: list[str] | None = None,
        y_eval: np.ndarray | None = None,
    ) -> "FineTunedBertClassifier":
        self.config.model_dir.mkdir(parents=True, exist_ok=True)

        if eval_titles is None or eval_bodies is None or y_eval is None:
            idx = np.arange(len(y))
            idx_tr, idx_va = train_test_split(
                idx,
                test_size=0.1,
                random_state=self.config.seed,
                stratify=y,
            )
            tr_titles = [titles[i] for i in idx_tr]
            tr_bodies = [bodies[i] for i in idx_tr]
            y_tr = y[idx_tr]
            va_titles = [titles[i] for i in idx_va]
            va_bodies = [bodies[i] for i in idx_va]
            y_va = y[idx_va]
        else:
            tr_titles, tr_bodies, y_tr = titles, bodies, y
            va_titles, va_bodies, y_va = eval_titles, eval_bodies, y_eval

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
            id2label={0: "deceptive", 1: "genuine"},
            label2id={"deceptive": 0, "genuine": 1},
        )

        train_ds = _TextPairDataset(self._encode(tr_titles, tr_bodies), y_tr)
        eval_ds = _TextPairDataset(self._encode(va_titles, va_bodies), y_va)
        data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)

        args = TrainingArguments(
            output_dir=str(self.config.model_dir),
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=20,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1_macro",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            seed=self.config.seed,
            disable_tqdm=False,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=self._tokenizer,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        trainer.train()
        trainer.save_model(str(self.config.model_dir))
        self._tokenizer.save_pretrained(str(self.config.model_dir))
        self._model = trainer.model.eval()
        return self

    def predict_proba(self, titles: list[str], bodies: list[str], batch_size: int = 16) -> np.ndarray:
        tok = self._get_tokenizer()
        model = self._get_model()
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        all_probs = []

        for start in range(0, len(titles), batch_size):
            bt_titles = titles[start:start + batch_size]
            bt_bodies = bodies[start:start + batch_size]

            enc = tok(
                bt_titles,
                bt_bodies,
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
                return_tensors="pt",
            )

            enc = {k: v.to(device) for k, v in enc.items()}

            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, titles: list[str], bodies: list[str]) -> np.ndarray:
        return np.argmax(self.predict_proba(titles, bodies), axis=1)
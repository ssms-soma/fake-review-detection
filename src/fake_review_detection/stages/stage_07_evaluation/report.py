# Writes training_metrics.json, three PNG charts, and training_report.html after a train run.

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class StepResult:
    key: str
    title: str
    val_metrics: dict[str, Any]
    train_metrics: dict[str, Any]
    delta_val_f1_macro: float | None


def _strip_non_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_non_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_non_json(v) for v in obj]
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj
    if isinstance(obj, np.floating):
        x = float(obj)
        return None if np.isnan(x) else x
    if isinstance(obj, np.integer):
        return int(obj)
    return None


def _write_text_model_comparison_chart(
    reports_dir: Path,
    f1_by_method: dict[str, float],
    selected: str,
    fasttext_backend: str = "sklearn_hash",
) -> None:
    order = ["tfidf", "bert_finetuned", "sbert", "fasttext_mean"]
    ft_lbl = (
        "FastText\n(Gensim mean)"
        if fasttext_backend == "gensim"
        else "FastText-style\n(char hash)"
    )
    labels_map = {
        "tfidf": "TF-IDF\n(char + word)",
        "sbert": "Sentence-BERT\n(MiniLM)",
        "fasttext_mean": ft_lbl,
        "bert_finetuned": "Fine-tuned BERT\n(title + review)",
    }
    labels = []
    vals = []
    colors = []
    for k in order:
        if k not in f1_by_method:
            continue
        labels.append(labels_map.get(k, k))
        vals.append(float(f1_by_method[k]))
        colors.append("#2ecc71" if k == selected else "#3498db")
    if not vals:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Text channel: same split, different representations (highest → shipped)")
    ax.set_ylim(0, min(1.05, max(vals) * 1.15 + 0.02))
    for b, v in zip(bars, vals, strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            f"{v:.3f}",
            ha="center",
            fontsize=9,
        )
    ax.axhline(0, color="#666", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(reports_dir / "text_model_comparison_val_f1.png", dpi=120)
    plt.close(fig)


def write_training_report(
    reports_dir: Path,
    steps: list[StepResult],
    extra: dict[str, Any] | None = None,
) -> None:
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "steps": [asdict(s) for s in steps],
        "extra": extra or {},
    }
    (reports_dir / "training_metrics.json").write_text(
        json.dumps(_strip_non_json(payload), indent=2), encoding="utf-8"
    )

    titles = [s.title for s in steps]
    val_f1 = [s.val_metrics["f1_macro"] for s in steps]
    val_acc = [s.val_metrics["accuracy"] for s in steps]
    train_f1 = [s.train_metrics["f1_macro"] for s in steps]
    train_acc = [s.train_metrics["accuracy"] for s in steps]
    x = np.arange(len(steps))

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.35
    ax.bar(x - w / 2, val_f1, width=w, label="Validation F1 (macro)", color="#3d9cf0")
    ax.bar(x + w / 2, train_f1, width=w, label="Train F1 (macro)", color="#8b9bb4")
    ax.set_xticks(x)
    ax.set_xticklabels(titles, rotation=22, ha="right", fontsize=8)
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Model stack: train vs validation F1 at each step")
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(reports_dir / "f1_train_val_by_step.png", dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.bar(x - w / 2, val_acc, width=w, label="Validation accuracy", color="#3ecf8e")
    ax2.bar(x + w / 2, train_acc, width=w, label="Train accuracy", color="#c4d4c8")
    ax2.set_xticks(x)
    ax2.set_xticklabels(titles, rotation=22, ha="right", fontsize=8)
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy by stack step (same threshold 0.5)")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    fig2.tight_layout()
    fig2.savefig(reports_dir / "accuracy_train_val_by_step.png", dpi=120)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    deltas = [s.delta_val_f1_macro for s in steps]
    heights = [0.0 if d is None else float(d) for d in deltas]
    colors = ["#2a3545" if (d is None or d <= 0) else "#3ecf8e" for d in deltas]
    labels_d = [
        "baseline" if d is None else (f"+{d:.3f}" if d > 0 else f"{d:.3f}") for d in deltas
    ]
    bars = ax3.bar(x, heights, color=colors)
    ax3.set_xticks(x)
    ax3.set_xticklabels(titles, rotation=22, ha="right", fontsize=8)
    ax3.set_ylabel("Delta validation F1 (macro) vs previous step")
    ax3.set_title("Marginal gain from each step")
    ax3.axhline(0, color="#666", linewidth=0.8)
    for i, b in enumerate(bars):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, labels_d[i], ha="center", fontsize=7)
    fig3.tight_layout()
    fig3.savefig(reports_dir / "delta_f1_by_step.png", dpi=120)
    plt.close(fig3)

    extra = extra or {}
    cmp_f1 = extra.get("text_model_comparison_val_f1")
    if isinstance(cmp_f1, dict) and cmp_f1:
        _write_text_model_comparison_chart(
            reports_dir,
            {k: float(v) for k, v in cmp_f1.items()},
            str(extra.get("selected_text_model", "")),
            str(extra.get("fasttext_backend", "sklearn_hash")),
        )

    def _roc_cell(vm: dict) -> str:
        r = vm.get("roc_auc")
        if r is None or (isinstance(r, float) and (np.isnan(r) or np.isinf(r))):
            return "—"
        return f"{float(r):.4f}"

    rows = []
    for s in steps:
        vm = s.val_metrics
        tm = s.train_metrics
        d = s.delta_val_f1_macro
        dcell = "—" if d is None else f"{d:+.4f}"
        rows.append(
            f"<tr><td>{s.title}</td>"
            f"<td>{tm['accuracy']:.4f}</td><td>{tm['f1_macro']:.4f}</td>"
            f"<td>{vm['accuracy']:.4f}</td><td>{vm['f1_macro']:.4f}</td>"
            f"<td>{_roc_cell(vm)}</td>"
            f"<td>{dcell}</td></tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Training report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 1100px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.45rem 0.5rem; text-align: left; }}
th {{ background: #f0f0f0; }}
img {{ max-width: 100%; height: auto; margin: 1rem 0; }}
h1 {{ font-size: 1.25rem; }}
.muted {{ color: #555; font-size: 0.9rem; }}
</style></head><body>
<h1>Fake review detector — training report</h1>
<p class="muted">Validation split = hold-out test for metrics. Train columns = same models evaluated on the training fold (in-sample; expect higher).</p>
<table>
<thead><tr><th>Step</th><th>Train acc</th><th>Train F1 macro</th><th>Val acc</th><th>Val F1 macro</th><th>Val ROC-AUC</th><th>d val F1</th></tr></thead>
<tbody>{"".join(rows)}</tbody>
</table>
<h2>Charts</h2>
<p><img src="f1_train_val_by_step.png" alt="F1 by step"/></p>
<p><img src="accuracy_train_val_by_step.png" alt="Accuracy by step"/></p>
<p><img src="delta_f1_by_step.png" alt="Delta F1"/></p>
<h2>Text representation comparison (validation)</h2>
<p class="muted">Macro-F1 on the same hold-out fold for TF-IDF vs Sentence-BERT vs FastText (Gensim mean vectors if installed, otherwise a subword char-hash baseline). The green bar is the model refit on all data and bundled in <code>ensemble.joblib</code>.</p>
<p><img src="text_model_comparison_val_f1.png" alt="Text model F1 comparison"/></p>
<p class="muted">Open <code>training_metrics.json</code> for precision/recall, confusion counts, and numeric F1/accuracy per text method under <code>extra</code>.</p>
</body></html>"""
    (reports_dir / "training_report.html").write_text(html, encoding="utf-8")

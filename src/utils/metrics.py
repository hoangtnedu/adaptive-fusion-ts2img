from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return str(value)


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False, default=_json_default)


def save_confusion_matrix(y_true, y_pred, class_names, out_path: str | Path, title: str) -> None:
    out_path = Path(out_path)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_learning_curve(history_csv: str | Path, out_path: str | Path, title: str) -> None:
    history_csv = Path(history_csv)
    out_path = Path(out_path)
    hist = pd.read_csv(history_csv)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hist["epoch"], hist["train_macro_f1"], label="Train Macro-F1")
    ax.plot(hist["epoch"], hist["val_macro_f1"], label="Val Macro-F1")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro-F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_classification_report(y_true, y_pred, class_names, out_path: str | Path) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

from __future__ import annotations

from pathlib import Path

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from mvp.src.config_loader import project_paths
from mvp.src.model import batch_predict_proba, train_classifier
from mvp.src.preprocessing import normalize_batch
from mvp.src.embeddings import generate_embeddings


def load_split(name: str) -> pd.DataFrame:
    paths = project_paths()
    path: Path = paths["data"] / "generated" / f"{name}.csv"
    return pd.read_csv(path)


def evaluate_split(split: str, out_dir: Path) -> dict:
    df = load_split(split)
    texts = df["description"].astype(str).tolist()
    y_true = df["category_id"].astype(str).tolist()

    artifacts = train_classifier()
    normalized = normalize_batch(texts)
    y_pred, confidences, _ = batch_predict_proba(artifacts, normalized)

    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    fig.tight_layout()
    (out_dir / "confusion_matrix.png").parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "confusion_matrix.png")
    plt.close(fig)

    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    macro_f1 = report["macro avg"]["f1-score"]

    with (out_dir / "per_class_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({lab: report[lab] for lab in labels}, f, indent=2)

    with (out_dir / "macro_f1_score.json").open("w", encoding="utf-8") as f:
        json.dump({"macro_f1": macro_f1}, f, indent=2)

    return {"macro_f1": macro_f1, "labels": labels}


def main() -> None:
    paths = project_paths()
    eval_dir: Path = paths["evaluation"]
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_split("test", eval_dir)

    notes_path = eval_dir / "performance_notes.md"
    if not notes_path.exists():
        with notes_path.open("w", encoding="utf-8") as f:
            f.write("# Performance Notes\n\n")
            f.write("- Macro F1 (test): {macro_f1:.3f}\n".format(**metrics))
            f.write("- See confusion_matrix.png for class-wise behavior.\n")


if __name__ == "__main__":  # pragma: no cover
    main()

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from .config_loader import project_paths, load_settings
from .embeddings import generate_embeddings
from .preprocessing import normalize_batch


@dataclass
class ModelArtifacts:
    classifier: KNeighborsClassifier
    label_to_index: Dict[str, int]
    index_to_label: Dict[int, str]
    train_embeddings: np.ndarray
    train_texts: List[str]
    train_labels: List[str]


def load_training_data() -> Tuple[List[str], List[str]]:
    paths = project_paths()
    train_path: Path = paths["data"] / "generated" / "train.csv"
    df = pd.read_csv(train_path)
    texts = df["description"].astype(str).tolist()
    labels = df["category_id"].astype(str).tolist()
    return texts, labels


def train_classifier() -> ModelArtifacts:
    """Train a k-NN classifier on the synthetic dataset."""

    settings = load_settings()
    texts, labels = load_training_data()
    normalized = normalize_batch(texts)
    X = generate_embeddings(normalized)

    unique_labels = sorted(set(labels))
    label_to_index = {lab: i for i, lab in enumerate(unique_labels)}
    index_to_label = {i: lab for lab, i in label_to_index.items()}
    y = np.array([label_to_index[lab] for lab in labels], dtype="int64")

    clf = KNeighborsClassifier(n_neighbors=settings.k_neighbors, metric="cosine")
    clf.fit(X, y)

    return ModelArtifacts(
        classifier=clf,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        train_embeddings=X,
        train_texts=texts,
        train_labels=labels,
    )


def predict_proba(artifacts: ModelArtifacts, text: str) -> Tuple[str, float, np.ndarray]:
    """Predict label and return confidence + full probability distribution."""

    X = generate_embeddings([text])
    proba = artifacts.classifier.predict_proba(X)[0]
    pred_index = int(np.argmax(proba))
    pred_label = artifacts.index_to_label[pred_index]
    confidence = float(proba[pred_index])
    return pred_label, confidence, proba


def batch_predict_proba(artifacts: ModelArtifacts, texts: List[str]) -> Tuple[List[str], List[float], np.ndarray]:
    X = generate_embeddings(texts)
    proba = artifacts.classifier.predict_proba(X)
    pred_indices = np.argmax(proba, axis=1)
    labels = [artifacts.index_to_label[int(i)] for i in pred_indices]
    confidences = [float(proba[i, idx]) for i, idx in enumerate(pred_indices)]
    return labels, confidences, proba

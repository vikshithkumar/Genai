from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config_loader import Settings, Taxonomy, load_settings, load_taxonomy
from .embeddings import generate_embeddings
from .explainability import Explanation, explain_prediction
from .model import ModelArtifacts, batch_predict_proba, train_classifier
from .preprocessing import normalize_batch


@dataclass
class PredictionResult:
    description: str
    predicted_category_id: str
    predicted_category_name: str
    confidence: float
    is_low_confidence: bool
    is_unknown: bool
    explanation: Explanation


_model_artifacts: ModelArtifacts | None = None
_settings: Settings | None = None
_taxonomy: Taxonomy | None = None


def _ensure_loaded() -> None:
    global _model_artifacts, _settings, _taxonomy
    if _settings is None:
        _settings = load_settings()
    if _taxonomy is None:
        _taxonomy = load_taxonomy()
    if _model_artifacts is None:
        _model_artifacts = train_classifier()


def _map_to_taxonomy(category_id: str) -> str:
    assert _taxonomy is not None
    mapping = _taxonomy.id_to_name()
    return mapping.get(category_id, category_id)


def _apply_threshold(category_id: str, confidence: float) -> tuple[str, bool]:
    assert _settings is not None, "Settings must be loaded"
    if confidence < _settings.low_confidence_threshold:
        return _taxonomy.unknown_category_id, True  # type: ignore[union-attr]
    return category_id, False


def predict_with_confidence(description: str) -> PredictionResult:
    _ensure_loaded()
    assert _model_artifacts is not None
    assert _settings is not None
    assert _taxonomy is not None

    texts = [description]
    normalized = normalize_batch(texts)
    emb = generate_embeddings(normalized)

    labels, confidences, _ = batch_predict_proba(_model_artifacts, normalized)
    raw_label = labels[0]
    confidence = confidences[0]

    final_label, is_low = _apply_threshold(raw_label, confidence)
    name = _map_to_taxonomy(final_label)

    explanation = explain_prediction(
        _model_artifacts,
        description=description,
        predicted_category=final_label,
        confidence=confidence,
        query_emb=emb,
    )

    is_unknown = final_label == _taxonomy.unknown_category_id
    return PredictionResult(
        description=description,
        predicted_category_id=final_label,
        predicted_category_name=name,
        confidence=confidence,
        is_low_confidence=is_low,
        is_unknown=is_unknown,
        explanation=explanation,
    )


def batch_inference(descriptions: List[str]) -> List[PredictionResult]:
    _ensure_loaded()
    assert _model_artifacts is not None
    assert _settings is not None
    assert _taxonomy is not None

    normalized = normalize_batch(descriptions)
    emb = generate_embeddings(normalized)

    labels, confidences, _ = batch_predict_proba(_model_artifacts, normalized)

    results: List[PredictionResult] = []
    for i, desc in enumerate(descriptions):
        raw_label = labels[i]
        confidence = confidences[i]
        final_label, is_low = _apply_threshold(raw_label, confidence)
        name = _map_to_taxonomy(final_label)
        explanation = explain_prediction(
            _model_artifacts,
            description=desc,
            predicted_category=final_label,
            confidence=confidence,
            query_emb=emb[i : i + 1],
        )
        is_unknown = final_label == _taxonomy.unknown_category_id
        results.append(
            PredictionResult(
                description=desc,
                predicted_category_id=final_label,
                predicted_category_name=name,
                confidence=confidence,
                is_low_confidence=is_low,
                is_unknown=is_unknown,
                explanation=explanation,
            )
        )
    return results

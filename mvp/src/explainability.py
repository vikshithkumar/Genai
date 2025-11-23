from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .model import ModelArtifacts
from .preprocessing import normalize_text


@dataclass
class NeighborExplanation:
    description: str
    category_id: str
    similarity: float


@dataclass
class Explanation:
    top_neighbors: List[NeighborExplanation]
    keyword_matches: List[str]
    rationale: str


def top_k_neighbors(artifacts: ModelArtifacts, query_emb: np.ndarray, k: int = 3) -> List[NeighborExplanation]:
    sims = cosine_similarity(query_emb, artifacts.train_embeddings)[0]
    idx_sorted = np.argsort(sims)[::-1][:k]
    neighbors: List[NeighborExplanation] = []
    for idx in idx_sorted:
        neighbors.append(
            NeighborExplanation(
                description=artifacts.train_texts[int(idx)],
                category_id=artifacts.train_labels[int(idx)],
                similarity=float(sims[int(idx)]),
            )
        )
    return neighbors


def keyword_matches(description: str, category_id: str) -> List[str]:
    tokens = normalize_text(description).split()
    counts = Counter(tokens)
    matches: List[str] = []
    for token, c in counts.items():
        if c > 0:
            matches.append(f"{token} (x{c})")
    return matches


def build_rationale(predicted_category: str, confidence: float, neighbors: List[NeighborExplanation]) -> str:
    neighbor_cats = {n.category_id for n in neighbors}
    return (
        f"Prediction {predicted_category} with confidence {confidence:.2f} "
        f"based on nearest neighbors in categories {sorted(neighbor_cats)}."
    )


def explain_prediction(
    artifacts: ModelArtifacts,
    description: str,
    predicted_category: str,
    confidence: float,
    query_emb: np.ndarray,
) -> Explanation:
    neighbors = top_k_neighbors(artifacts, query_emb, k=3)
    keywords = keyword_matches(description, predicted_category)
    rationale = build_rationale(predicted_category, confidence, neighbors)
    return Explanation(top_neighbors=neighbors, keyword_matches=keywords, rationale=rationale)

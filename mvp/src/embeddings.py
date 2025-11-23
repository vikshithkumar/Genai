from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .preprocessing import normalize_batch


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers model.

    This will download the model on first use and reuse it afterwards.
    """

    return SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts: Iterable[str]) -> np.ndarray:
    """Normalize and embed a batch of texts into dense vectors."""

    model = get_model()
    normalized = normalize_batch(texts)
    emb = model.encode(list(normalized), convert_to_numpy=True, show_progress_bar=False)
    return emb.astype("float32")

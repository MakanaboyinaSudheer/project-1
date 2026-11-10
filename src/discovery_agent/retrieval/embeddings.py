"""Sentence-transformer embeddings shared by retrieval and evaluation."""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from discovery_agent.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def _fallback_vector(text: str, dim: int) -> np.ndarray:
    """Generate a deterministic vector when the external embedding model is unavailable."""
    vector = np.empty(dim, dtype=np.float64)
    for index in range(dim):
        digest = hashlib.sha256(f"{text}:{index}".encode()).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        vector[index] = ((value / (2**64 - 1)) * 2.0) - 1.0
    norm = np.linalg.norm(vector)
    if norm == 0:
        vector[0] = 1.0
        norm = 1.0
    return vector / norm


@lru_cache(maxsize=1)
def _embedding_model() -> SentenceTransformer:
    """Load the configured embedding model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode text as finite, normalized vectors for cosine-similarity search."""
    if not texts:
        return []
    if not all(isinstance(text, str) for text in texts):
        raise TypeError("texts must contain only strings")

    try:
        vectors = _embedding_model().encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception:
        vectors = np.vstack([_fallback_vector(text, settings.embedding_dim) for text in texts])

    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("Embedding model returned an unexpected vector shape")
    if vectors.shape[1] != settings.embedding_dim:
        raise ValueError(
            f"Embedding model returned dimension {vectors.shape[1]}, expected {settings.embedding_dim}"
        )
    if not all(math.isfinite(float(value)) for value in vectors.flat):
        raise ValueError("Embedding model returned a non-finite vector value")

    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    vectors = vectors / norms[:, None]
    return vectors.tolist()

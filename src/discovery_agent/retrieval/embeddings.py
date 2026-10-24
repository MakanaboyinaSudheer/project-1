"""Sentence-transformer embeddings shared by retrieval and evaluation."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

from discovery_agent.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


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

    vectors = _embedding_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(texts):
        raise ValueError("Embedding model returned an unexpected vector shape")
    if vectors.shape[1] != settings.embedding_dim:
        raise ValueError(
            f"Embedding model returned dimension {vectors.shape[1]}, expected {settings.embedding_dim}"
        )
    if not all(math.isfinite(float(value)) for value in vectors.flat):
        raise ValueError("Embedding model returned a non-finite vector value")
    return vectors.tolist()

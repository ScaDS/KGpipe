"""
Embedding computation and similarity helpers for parameter clustering.

Uses sentence-transformers to encode parameter descriptions into dense
vectors, then provides numpy-based cosine-similarity utilities.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .models import ParameterVector

logger = logging.getLogger(__name__)

# Default lightweight model; works well for short technical phrases.
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_model(model_name: str):
    """Load a SentenceTransformer model (cached after first call)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model: %s", model_name)
    return SentenceTransformer(model_name)


def embed_parameters(
    parameters: List[ParameterVector],
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 64,
    model: Optional[object] = None,
) -> np.ndarray:
    """
    Compute embeddings for a list of ParameterVectors.

    Each parameter's ``text_for_embedding()`` is encoded via the
    sentence-transformer *model_name*.  The resulting embeddings are
    stored back into each ``ParameterVector.embedding`` field **and**
    returned as a (N, D) numpy array.

    Parameters
    ----------
    parameters : list[ParameterVector]
        Parameters to embed.
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Encoding batch size.
    model : optional
        Pre-loaded SentenceTransformer instance (avoids reloading).

    Returns
    -------
    np.ndarray
        Shape ``(len(parameters), embedding_dim)``.
    """
    if not parameters:
        return np.empty((0, 0))

    if model is None:
        model = _load_model(model_name)

    texts = [p.text_for_embedding() for p in parameters]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalise to unit length so cosine similarity = dot product.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms

    for pv, emb in zip(parameters, embeddings):
        pv.embedding = emb.tolist()

    return embeddings


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute the pair-wise cosine similarity matrix.

    If the embeddings are already L2-normalised (as ``embed_parameters``
    produces), this is simply ``embeddings @ embeddings.T``.
    """
    if embeddings.size == 0:
        return np.empty((0, 0))
    # Ensure unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms
    return normed @ normed.T


"""
embeddings.py — Embedding Engine
==================================
Wraps the sentence-transformers library to produce dense vector
embeddings for persona descriptions and user posts.

Key concept — **Cosine Similarity**
------------------------------------
Cosine similarity measures the angle between two vectors in
high-dimensional space, ignoring magnitude:

    cos(A, B) = (A · B) / (||A|| × ||B||)

A value of 1.0 means identical direction (perfect semantic match),
0.0 means orthogonal (unrelated), and −1.0 means opposite.

We use it here because sentence-transformer embeddings are
normalised unit vectors, making cosine similarity equivalent to
the dot product — extremely fast to compute.
"""

import os

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from utils import get_logger

load_dotenv()
logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────
# Singleton Model Loader
# ────────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """
    Lazily load the sentence-transformer model once and cache it.
    The model name is read from the EMBEDDING_MODEL env var.
    """
    global _model
    if _model is None:
        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        logger.info("Loading embedding model: %s", model_name)
        _model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded  (dim=%d)", _model.get_sentence_embedding_dimension())
    return _model


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """
    Convert a single text string into a normalised embedding vector.

    Parameters
    ----------
    text : str
        The input sentence / paragraph.

    Returns
    -------
    np.ndarray
        1-D float32 array of shape (dim,).
    """
    model = _get_model()
    # encode returns shape (1, dim) — squeeze to (dim,)
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Batch-embed multiple texts.

    Parameters
    ----------
    texts : list[str]

    Returns
    -------
    np.ndarray of shape (len(texts), dim)
    """
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return vecs.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Because embed_text already returns L2-normalised vectors,
    this simplifies to the dot product:

        cos(a, b)  =  a · b       (when ||a|| = ||b|| = 1)

    Parameters
    ----------
    a, b : np.ndarray, 1-D

    Returns
    -------
    float in [-1, 1]
    """
    # Defensive normalisation in case raw vectors are passed
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def get_embedding_dimension() -> int:
    """Return the dimensionality of the loaded model."""
    return _get_model().get_sentence_embedding_dimension()


def unload_model() -> None:
    """
    Explicitly unload the sentence-transformer model to free RAM.
    Essential on memory-constrained systems before running Ollama.
    """
    global _model
    if _model is not None:
        del _model
        _model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Embedding model unloaded — RAM freed")


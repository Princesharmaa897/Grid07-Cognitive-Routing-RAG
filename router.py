"""
router.py — Vector-Based Semantic Router  (Phase 1)
=====================================================
Routes an incoming user post to the most relevant bot personas
using cosine similarity over sentence-transformer embeddings.

Architecture
------------
1.  On initialisation, each persona description is embedded and
    stored in a lightweight in-memory FAISS index (exact search,
    IndexFlatIP — inner-product on normalised vectors = cosine sim).
2.  ``route_post_to_bots()`` embeds the incoming post, queries the
    FAISS index, and returns personas whose similarity exceeds the
    configured threshold, sorted descending.

Why FAISS?
----------
Even though we only have 3 personas today, FAISS gives us:
  • Sub-millisecond lookup even with thousands of personas
  • A real production pattern (swap IndexFlatIP → IndexIVFFlat
    for million-scale without changing the API)
"""

import os

import faiss
import numpy as np
from dotenv import load_dotenv

from embeddings import embed_text, embed_texts, cosine_similarity
from personas import PERSONAS
from utils import get_logger

load_dotenv()
logger = get_logger(__name__)

# ────────────────────────────────────────────────────────────
# Module State
# ────────────────────────────────────────────────────────────

_index: faiss.Index | None = None
_persona_vectors: np.ndarray | None = None  # (N, dim) matrix


# ────────────────────────────────────────────────────────────
# Initialisation
# ────────────────────────────────────────────────────────────

def initialise_persona_index() -> None:
    """
    Embed every persona description and build the FAISS index.
    Must be called once before routing.
    """
    global _index, _persona_vectors

    descriptions = [p["description"] for p in PERSONAS]
    logger.info("Embedding %d persona descriptions …", len(descriptions))

    _persona_vectors = embed_texts(descriptions)            # (N, dim)
    dim = _persona_vectors.shape[1]

    # IndexFlatIP = brute-force inner-product search
    # On L2-normalised vectors, IP == cosine similarity
    _index = faiss.IndexFlatIP(dim)
    _index.add(_persona_vectors)

    logger.info(
        "FAISS index built — %d vectors, dim=%d",
        _index.ntotal, dim,
    )


# ────────────────────────────────────────────────────────────
# Routing
# ────────────────────────────────────────────────────────────

def route_post_to_bots(
    post_content: str,
    threshold: float | None = None,
) -> list[dict]:
    """
    Route a post to the most semantically relevant bots.

    Parameters
    ----------
    post_content : str
        The raw text of the user's post.
    threshold : float, optional
        Minimum cosine similarity to include a bot.
        Defaults to ROUTING_THRESHOLD from .env (or 0.30).

    Returns
    -------
    list[dict]
        Each dict has keys:
          persona_id  — slug id
          name        — display name
          similarity  — cosine score (0–1)
        Sorted descending by similarity.
    """
    if _index is None or _persona_vectors is None:
        raise RuntimeError(
            "Persona index not initialised. Call initialise_persona_index() first."
        )

    if threshold is None:
        threshold = float(os.getenv("ROUTING_THRESHOLD", "0.30"))

    # Embed the incoming post
    post_vec = embed_text(post_content).reshape(1, -1)  # (1, dim)

    # Search all personas (k = total count)
    k = _index.ntotal
    similarities, indices = _index.search(post_vec, k)

    # Filter & build result list
    results: list[dict] = []
    for sim, idx in zip(similarities[0], indices[0]):
        if sim >= threshold:
            persona = PERSONAS[idx]
            results.append({
                "persona_id": persona["id"],
                "name": persona["name"],
                "similarity": float(sim),
            })

    # Already sorted descending by FAISS, but be explicit
    results.sort(key=lambda r: r["similarity"], reverse=True)

    logger.info(
        "Post routed → %d bot(s) above threshold %.2f",
        len(results), threshold,
    )
    for r in results:
        logger.debug("  %s  sim=%.4f", r["name"], r["similarity"])

    return results

"""Intrinsic chunk quality metrics — no gold labels needed."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..data.types import Chunk


def intra_chunk_cohesion(chunks: list[Chunk], embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity within each chunk, averaged across chunks."""
    scores = []
    for ch in chunks:
        if len(ch.sentences) < 2:
            continue
        idx = [s.index for s in ch.sentences]
        chunk_embs = embeddings[idx]
        sims = cosine_similarity(chunk_embs)
        triu = sims[np.triu_indices(len(idx), k=1)]
        scores.append(float(triu.mean()))
    return float(np.mean(scores)) if scores else 0.0


def inter_chunk_separation(chunks: list[Chunk], embeddings: np.ndarray) -> float:
    """Mean cosine similarity between adjacent chunk centroids. Lower = better."""
    centroids = []
    for ch in chunks:
        if not ch.sentences:
            continue
        idx = [s.index for s in ch.sentences]
        centroids.append(embeddings[idx].mean(axis=0))

    if len(centroids) < 2:
        return 0.0

    centroids = np.array(centroids)
    sims = []
    for i in range(len(centroids) - 1):
        sims.append(float(cosine_similarity([centroids[i]], [centroids[i + 1]])[0, 0]))
    return float(np.mean(sims))


def size_coefficient_of_variation(chunks: list[Chunk]) -> float:
    """Std / mean of chunk sizes (in sentences). Lower = more uniform."""
    sizes = np.array([len(ch.sentences) for ch in chunks])
    if sizes.mean() == 0:
        return 0.0
    return float(sizes.std() / sizes.mean())


def compute_intrinsic_metrics(
    chunks: list[Chunk], embeddings: np.ndarray
) -> dict[str, float]:
    return {
        "n_chunks": len(chunks),
        "cohesion": intra_chunk_cohesion(chunks, embeddings),
        "separation": inter_chunk_separation(chunks, embeddings),
        "size_cov": size_coefficient_of_variation(chunks),
        "avg_chunk_sentences": float(
            np.mean([len(ch.sentences) for ch in chunks])
        ),
    }

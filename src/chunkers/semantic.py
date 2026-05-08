"""Semantic chunker — split where adjacent sentence similarity drops."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from ..data.types import Chunk, Sentence
from .base import BaseChunker


class SemanticChunker(BaseChunker):
    """Splits at sentence boundaries where cosine similarity drops below
    the given percentile threshold of the document's similarity distribution.

    This is the "local 1D" baseline — it only looks at adjacent pairs,
    unlike DensityChunker which uses the full pairwise similarity matrix.
    """

    def __init__(
        self,
        threshold_percentile: float = 10.0,
        min_sentences: int = 3,
    ) -> None:
        self.threshold_percentile = threshold_percentile
        self.min_sentences = min_sentences

    @property
    def name(self) -> str:
        return f"semantic-p{self.threshold_percentile:.0f}"

    def chunk_document(
        self,
        sentences: list[Sentence],
        embeddings: np.ndarray,
    ) -> list[Chunk]:
        n = len(sentences)
        if n <= self.min_sentences:
            return self._assemble_chunks(sentences, [0])

        adj_sims = np.array([
            float(cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0, 0])
            for i in range(n - 1)
        ])

        threshold = np.percentile(adj_sims, self.threshold_percentile)

        boundaries = [0]
        i = 0
        while i < n - 1:
            if adj_sims[i] < threshold:
                if i + 1 - boundaries[-1] >= self.min_sentences:
                    boundaries.append(i + 1)
            i += 1
        return self._assemble_chunks(sentences, boundaries)

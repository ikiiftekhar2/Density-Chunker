"""DensityChunker — position-weighted k-NN density in full embedding space."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.metrics.pairwise import cosine_similarity

from ..data.types import Chunk, Sentence
from .base import BaseChunker


class DensityChunker(BaseChunker):
    """Chunk by finding valleys in the position-weighted semantic density signal.

    For each sentence i, density[i] = sum_j sim(i,j) * exp(-|i-j| / sigma).
    Dense regions are topic clusters. Valleys are boundaries.
    """

    def __init__(
        self,
        sigma_position: int | str = "auto",
        smoothing_sigma: float = 2.0,
        valley_prominence: float = 0.5,
        min_sentences: int = 3,
        max_sentences: int = 40,
    ) -> None:
        self.sigma_position = sigma_position
        self.smoothing_sigma = smoothing_sigma
        self.valley_prominence = valley_prominence
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    @property
    def name(self) -> str:
        return "density"

    def chunk_document(
        self,
        sentences: list[Sentence],
        embeddings: np.ndarray,
    ) -> list[Chunk]:
        n = len(sentences)

        if n <= self.min_sentences:
            return self._assemble_chunks(sentences, [0])

        sigma = self._resolve_sigma(n)
        similarity = self.compute_similarity_matrix(embeddings)
        position_weights = self.compute_position_weights(n, sigma)
        density = self.compute_density_profile(similarity, position_weights)
        density_smooth = gaussian_filter1d(density, sigma=self.smoothing_sigma)
        valleys = self.find_valleys(density_smooth)
        boundaries = self.enforce_constraints(valleys, density_smooth, n)
        return self._assemble_chunks(sentences, boundaries)

    @staticmethod
    def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
        """(N, D) → (N, N) cosine similarity matrix."""
        return cosine_similarity(embeddings)

    @staticmethod
    def compute_position_weights(n: int, sigma: float) -> np.ndarray:
        """(N, N) matrix where W[i,j] = exp(-|i-j| / sigma)."""
        idx = np.arange(n)
        dist = np.abs(idx[:, None] - idx[None, :])
        return np.exp(-dist / sigma)

    @staticmethod
    def compute_density_profile(
        similarity: np.ndarray,
        position_weights: np.ndarray,
    ) -> np.ndarray:
        """density[i] = sum_j S[i,j] * W[i,j]."""
        return (similarity * position_weights).sum(axis=1)

    def find_valleys(self, density: np.ndarray) -> list[int]:
        if self.valley_prominence == 0:
            return [0]

        threshold = self.valley_prominence * density.std()
        valleys, _ = find_peaks(
            -density, prominence=threshold if threshold > 0 else 0.01
        )
        return valleys.tolist()

    def enforce_constraints(
        self,
        valleys: list[int],
        density: np.ndarray,
        n: int,
    ) -> list[int]:
        boundaries = [0]
        for v in valleys:
            if v > 0 and v < n:
                boundaries.append(v)

        boundaries = sorted(set(boundaries))

        merged = [boundaries[0]]
        for i in range(1, len(boundaries)):
            size = boundaries[i] - merged[-1]
            chunk_size = (
                boundaries[i + 1] - merged[-1]
                if i + 1 < len(boundaries)
                else n - merged[-1]
            )
            if chunk_size < self.min_sentences and i + 1 < len(boundaries):
                continue
            elif boundaries[i] - merged[-1] < self.min_sentences and len(merged) > 1:
                continue
            merged.append(boundaries[i])

        split_result = []
        for i, start in enumerate(merged):
            end = merged[i + 1] if i + 1 < len(merged) else n
            size = end - start
            if size > self.max_sentences:
                sub_valleys = self._find_deepest_split(density[start:end])
                if sub_valleys:
                    for sv in sorted(sub_valleys):
                        split_result.append(start + sv)
                    continue
            split_result.append(start)

        return sorted(set(split_result))

    def _resolve_sigma(self, n: int) -> float:
        if isinstance(self.sigma_position, (int, float)):
            return float(self.sigma_position)
        return max(5.0, n / 20.0)

    def _find_deepest_split(self, segment: np.ndarray) -> list[int]:
        """Split an oversized segment at its deepest valley."""
        if len(segment) < self.min_sentences * 2:
            return []
        valleys, props = find_peaks(-segment, prominence=segment.std() * 0.3)
        if len(valleys) == 0:
            return []
        best_idx = valleys[np.argmin(segment[valleys])]
        return [best_idx]

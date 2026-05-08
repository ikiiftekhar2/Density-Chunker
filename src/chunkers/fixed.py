"""Fixed-size chunker — cut every N sentences."""

from __future__ import annotations

import numpy as np

from ..data.types import Chunk, Sentence
from .base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Splits documents into chunks of a fixed number of sentences."""

    def __init__(self, chunk_size: int = 10) -> None:
        self.chunk_size = chunk_size

    @property
    def name(self) -> str:
        return f"fixed-{self.chunk_size}"

    def chunk_document(
        self,
        sentences: list[Sentence],
        embeddings: np.ndarray,
    ) -> list[Chunk]:
        n = len(sentences)
        boundaries = list(range(0, n, self.chunk_size))
        return self._assemble_chunks(sentences, boundaries)

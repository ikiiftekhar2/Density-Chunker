"""Batch embedder wrapping sentence-transformers models."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class BatchEmbedder:
    """Encodes texts into embeddings with configurable output dimension.

    Uses BGE-M3 by default, which supports Matryoshka representations —
    truncating to a lower dimension preserves semantic quality because the
    model was trained with variable-dimensional output heads.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        output_dim: int = 1024,
        device: str | None = None,
        normalize: bool = True,
        show_progress: bool = True,
        max_seq_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.output_dim = output_dim
        self.normalize = normalize
        self.show_progress = show_progress

        self._model = SentenceTransformer(
            model_name, device=device, trust_remote_code=True
        )
        self._model.max_seq_length = max_seq_length
        self._native_dim = self._model.get_embedding_dimension()

        if output_dim > self._native_dim:
            raise ValueError(
                f"Requested output_dim {output_dim} exceeds native "
                f"dimension {self._native_dim} for model {model_name}"
            )

    @property
    def native_dim(self) -> int:
        return self._native_dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        prompt_name: str | None = None,
    ) -> np.ndarray:
        """Encode a list of texts → (N, output_dim) array."""
        if not texts:
            return np.empty((0, self.output_dim), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=self.normalize,
            prompt_name=prompt_name,
            convert_to_numpy=True,
        )

        if self.output_dim < self._native_dim:
            embeddings = embeddings[:, :self.output_dim]

        if self.output_dim < self._native_dim and self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string → (1, output_dim) array.

        For BGE models, the query prompt name is 'query' to trigger
        the instruction-tuned encoding path.
        """
        return self.encode([query], prompt_name="query")

    def encode_queries(self, queries: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode multiple queries → (len(queries), output_dim) array."""
        return self.encode(queries, batch_size=batch_size, prompt_name="query")

    def __repr__(self) -> str:
        return (
            f"BatchEmbedder(model={self.model_name!r}, "
            f"output_dim={self.output_dim}, device={self._model.device})"
        )

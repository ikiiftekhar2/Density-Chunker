"""Abstract base class for chunkers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..data.types import Chunk, Sentence


class BaseChunker(ABC):
    """Every chunker takes sentences + embeddings and returns chunks."""

    @abstractmethod
    def chunk_document(
        self,
        sentences: list[Sentence],
        embeddings: np.ndarray,
    ) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            sentences: Ordered list of sentences in the document.
            embeddings: (N, D) array of sentence embeddings.

        Returns:
            List of Chunk objects with character spans and metadata.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable chunker name for experiment tracking."""
        ...

    def _assemble_chunks(
        self, sentences: list[Sentence], boundaries: list[int]
    ) -> list[Chunk]:
        """Group sentences by boundary indices into Chunk objects.

        boundaries[i] means chunk i starts at sentence boundaries[i].
        The last chunk ends at len(sentences).
        """
        chunks = []
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
            sent_group = sentences[start:end]
            if not sent_group:
                continue
            chunk_text = " ".join(s.text for s in sent_group)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    sentences=sent_group,
                    start_char=sent_group[0].start_char,
                    end_char=sent_group[-1].end_char,
                    chunk_id=f"{i}",
                    metadata={
                        "start_sent": start,
                        "end_sent": end,
                        "n_sentences": len(sent_group),
                    },
                )
            )
        return chunks

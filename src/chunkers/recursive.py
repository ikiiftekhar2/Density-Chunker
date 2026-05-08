"""Recursive chunker — LangChain's RecursiveCharacterTextSplitter."""

from __future__ import annotations

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..data.types import Chunk, Sentence
from .base import BaseChunker


class RecursiveChunker(BaseChunker):
    """Wraps LangChain's RecursiveCharacterTextSplitter.

    Operates on character-level text with standard paragraph/break separators,
    then maps resulting chunks back to sentence boundaries for evaluation.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def name(self) -> str:
        return f"recursive-{self.chunk_size}-{self.chunk_overlap}"

    def chunk_document(
        self,
        sentences: list[Sentence],
        embeddings: np.ndarray,
    ) -> list[Chunk]:
        full_text = " ".join(s.text for s in sentences)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        text_chunks = splitter.split_text(full_text)

        chunks: list[Chunk] = []
        pos = 0
        for ci, tc in enumerate(text_chunks):
            start = full_text.find(tc, pos)
            if start == -1:
                start = pos
            end = start + len(tc)
            pos = end

            chunk_sents = [
                s for s in sentences
                if s.end_char > start and s.start_char < end
            ]
            if not chunk_sents:
                continue

            chunks.append(
                Chunk(
                    text=tc.strip(),
                    sentences=chunk_sents,
                    start_char=start,
                    end_char=end,
                    chunk_id=str(ci),
                    metadata={
                        "start_sent": chunk_sents[0].index,
                        "end_sent": chunk_sents[-1].index + 1,
                        "n_sentences": len(chunk_sents),
                    },
                )
            )
        return chunks

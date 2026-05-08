"""ChromaDB indexer — collection lifecycle and batch insertion."""

from __future__ import annotations

import numpy as np

import chromadb
from chromadb.api.types import EmbeddingFunction


class ChromaIndexer:
    """Manages a ChromaDB collection for chunk storage and retrieval.

    One ChromaIndexer per experiment run. The collection name follows the
    convention ``{dataset}_{method}_{dim}_{mode}`` so different runs are
    isolated and results are reproducible from disk.
    """

    def __init__(
        self,
        collection_name: str,
        persist_dir: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir

        if persist_dir is not None:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,  # We supply embeddings directly
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self):
        return self._collection

    def add_chunks(
        self,
        chunk_ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict],
        documents: list[str] | None = None,
    ) -> None:
        """Batch-insert chunks into the collection.

        Args:
            chunk_ids: Unique IDs for each chunk.
            embeddings: (N, D) float32 array of centroid embeddings.
            metadatas: List of dicts with keys like doc_id, start_sent, end_sent.
            documents: Optional chunk text for inspection / debugging.
        """
        if len(chunk_ids) == 0:
            return

        embeddings_list = embeddings.astype(np.float32).tolist()

        # ChromaDB has a max batch size; split large inserts
        max_batch = 4000
        for start in range(0, len(chunk_ids), max_batch):
            end = start + max_batch
            self._collection.add(
                ids=chunk_ids[start:end],
                embeddings=embeddings_list[start:end],
                metadatas=metadatas[start:end],
                documents=documents[start:end] if documents else None,
            )

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self.collection_name)

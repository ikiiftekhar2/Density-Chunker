"""ChromaDB retriever — query embedding + nearest-neighbor lookup + optional rerank."""

from __future__ import annotations

from ..embedders.embedder import BatchEmbedder
from .indexer import ChromaIndexer


class ChromaRetriever:
    """Retrieves top-k chunks from a ChromaDB collection for a given query.

    Pipeline: ``embed(query) → collection.query() → optional rerank → results``.
    """

    def __init__(
        self,
        indexer: ChromaIndexer,
        embedder: BatchEmbedder,
        reranker=None,
        k: int = 10,
    ) -> None:
        self._indexer = indexer
        self._embedder = embedder
        self._reranker = reranker
        self.k = k

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        where: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Retrieve top-k chunks for a single query.

        Args:
            query: The query text.
            k: Number of results (defaults to self.k).
            where: Optional ChromaDB metadata filter dict
                   (e.g. ``{"doc_id": "cuad/contract.txt"}``).

        Returns:
            List of ``(chunk_id, score, metadata)`` tuples, sorted by
            descending similarity score.
        """
        k = k or self.k
        query_embedding = self._embedder.encode_query(query)

        results = self._indexer.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        ids = results["ids"][0]
        distances = results.get("distances", [[0.0] * len(ids)])[0]
        metadatas = results.get("metadatas", [[{}] * len(ids)])[0]

        # ChromaDB returns cosine distance; convert to similarity
        scores = [1.0 - d for d in distances]

        if self._reranker is not None:
            documents = results.get("documents", [[""] * len(ids)])[0]
            scores = self._reranker.rerank(query, documents, scores)

        scored = list(zip(ids, scores, metadatas))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def batch_retrieve(
        self,
        queries: list[str],
        k: int | None = None,
        where: dict | None = None,
    ) -> list[list[tuple[str, float, dict]]]:
        """Retrieve top-k chunks for multiple queries."""
        return [self.retrieve(q, k=k, where=where) for q in queries]

    @property
    def embedder(self) -> BatchEmbedder:
        return self._embedder

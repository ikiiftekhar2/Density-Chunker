"""Retrieval evaluation — deterministic, no API calls."""

from __future__ import annotations

import numpy as np

from ..data.types import Chunk, Query


def _span_overlaps(chunk: Chunk, query: Query) -> bool:
    """True if any gold span for this query overlaps the chunk."""
    for gold in query.gold_spans:
        if gold.file_path and chunk.metadata.get("doc_id", "") != gold.file_path:
            continue
        if chunk.start_char < gold.end_char and chunk.end_char > gold.start_char:
            return True
    return False


def _any_hit(
    doc_chunks: list[Chunk], query: Query, retrieved_ids: list[int]
) -> bool:
    """True if any retrieved chunk overlaps a gold span."""
    for rid in retrieved_ids:
        if rid < len(doc_chunks) and _span_overlaps(doc_chunks[rid], query):
            return True
    return False


def recall_at_k(
    queries: list[Query],
    chunk_map: dict[str, list[Chunk]],
    retrievals: dict[str, list[int]],
    k: int,
) -> float:
    """Fraction of queries where at least one top-k chunk overlaps a gold span."""
    hits = 0
    for q in queries:
        doc_id = q.gold_spans[0].file_path if q.gold_spans else ""
        chunks = chunk_map.get(doc_id, [])
        retrieved = retrievals.get(q.query_id, [])[:k]
        if _any_hit(chunks, q, retrieved):
            hits += 1
    return hits / len(queries) if queries else 0.0


def mrr(
    queries: list[Query],
    chunk_map: dict[str, list[Chunk]],
    retrievals: dict[str, list[int]],
) -> float:
    """Mean reciprocal rank of the first correct chunk."""
    ranks = []
    for q in queries:
        doc_id = q.gold_spans[0].file_path if q.gold_spans else ""
        chunks = chunk_map.get(doc_id, [])
        retrieved = retrievals.get(q.query_id, [])
        hit = False
        for rank, rid in enumerate(retrieved):
            if rid < len(chunks) and _span_overlaps(chunks[rid], q):
                ranks.append(1.0 / (rank + 1))
                hit = True
                break
        if not hit:
            ranks.append(0.0)
    return float(np.mean(ranks)) if ranks else 0.0


def precision_at_k(
    queries: list[Query],
    chunk_map: dict[str, list[Chunk]],
    retrievals: dict[str, list[int]],
    k: int,
) -> float:
    """Fraction of top-k chunks that overlap a gold span, averaged across queries."""
    precs = []
    for q in queries:
        doc_id = q.gold_spans[0].file_path if q.gold_spans else ""
        chunks = chunk_map.get(doc_id, [])
        retrieved = retrievals.get(q.query_id, [])[:k]
        if not retrieved:
            precs.append(0.0)
            continue
        hit_count = sum(
            1 for rid in retrieved
            if rid < len(chunks) and _span_overlaps(chunks[rid], q)
        )
        precs.append(hit_count / len(retrieved))
    return float(np.mean(precs)) if precs else 0.0


def evaluate_retrieval(
    queries: list[Query],
    chunk_map: dict[str, list[Chunk]],
    retrievals: dict[str, list[int]],
    k_values: list[int] = [1, 3, 5, 10],
) -> dict[str, float]:
    metrics = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(queries, chunk_map, retrievals, k)
        metrics[f"precision@{k}"] = precision_at_k(queries, chunk_map, retrievals, k)
    metrics["mrr"] = mrr(queries, chunk_map, retrievals)
    return metrics

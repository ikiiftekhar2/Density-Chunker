"""Quick legalbench eval from existing 1024-d cache."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunkers.base import BaseChunker
from src.chunkers.fixed import FixedSizeChunker
from src.chunkers.recursive import RecursiveChunker
from src.chunkers.semantic import SemanticChunker
from src.data.loader import load_legalbench_rag
from src.data.types import Chunk
from src.embedders.embedder import BatchEmbedder
from src.evaluation.intrinsic import compute_intrinsic_metrics
from src.evaluation.retrieval import evaluate_retrieval
from src.retrieval.indexer import ChromaIndexer
from src.retrieval.retriever import ChromaRetriever

METHODS: list[BaseChunker] = [
    FixedSizeChunker(chunk_size=5),
    FixedSizeChunker(chunk_size=10),
    RecursiveChunker(chunk_size=512, chunk_overlap=100),
    SemanticChunker(threshold_percentile=10.0),
]


def fix_cache_keys(cache: dict) -> dict:
    """Strip double-domain prefix: cuad/cuad/X -> cuad/X."""
    fixed = {}
    for key, val in cache.items():
        parts = key.split("/", 1)
        if len(parts) == 2 and parts[0] == parts[1].split("/")[0]:
            fixed[parts[1]] = val  # Strip first domain/
        else:
            fixed[key] = val
    return fixed


def run_method(chunker, doc_data, queries, embedder):
    t0 = time.time()
    all_chunks: dict[str, list[Chunk]] = {}
    intrinsic_scores = []

    for doc_id, (sentences, sent_embs) in tqdm(
        doc_data.items(), desc=f"  Chunking {chunker.name}", unit="doc", leave=False
    ):
        chunks = chunker.chunk_document(sentences, sent_embs)
        for ch in chunks:
            ch.metadata["doc_id"] = doc_id
        all_chunks[doc_id] = chunks
        intrinsic_scores.append(compute_intrinsic_metrics(chunks, sent_embs))

    intrinsic = {}
    if intrinsic_scores:
        for key in intrinsic_scores[0]:
            intrinsic[key] = float(np.mean([s[key] for s in intrinsic_scores]))

    indexer = ChromaIndexer(collection_name=f"blq_{chunker.name}")
    retriever = ChromaRetriever(indexer=indexer, embedder=embedder, k=10)

    flat_chunks = []
    for doc_id, chunks in all_chunks.items():
        for i, ch in enumerate(chunks):
            ch.chunk_id = f"{doc_id}::{i}"
            flat_chunks.append(ch)

    chunk_embs = embedder.encode([ch.text for ch in flat_chunks], batch_size=128)
    metadatas = [
        {"doc_id": ch.metadata["doc_id"], "start_char": ch.start_char, "end_char": ch.end_char}
        for ch in flat_chunks
    ]
    indexer.add_chunks(
        chunk_ids=[ch.chunk_id for ch in flat_chunks],
        embeddings=chunk_embs, metadatas=metadatas,
    )

    retrievals: dict[str, list[int]] = {}
    for q in tqdm(queries, desc=f"  Retrieving {chunker.name}", unit="query", leave=False):
        doc_id = q.gold_spans[0].file_path if q.gold_spans else ""
        results = retriever.retrieve(q.query_text, k=10, where={"doc_id": doc_id})
        chunk_indices = []
        for chunk_id, _, _ in results:
            chunk_indices.append(int(chunk_id.split("::")[-1]))
        retrievals[q.query_id] = chunk_indices

    retrieval_metrics = evaluate_retrieval(queries, all_chunks, retrievals, k_values=[1, 3, 5, 10])
    indexer.delete_collection()
    elapsed = time.time() - t0

    return {
        "method": chunker.name,
        "intrinsic": intrinsic,
        "retrieval": retrieval_metrics,
        "time_sec": round(elapsed, 1),
        "total_chunks": sum(len(c) for c in all_chunks.values()),
    }


def main():
    print("Loading cache...")
    with open(PROJECT_ROOT / "results" / "segmented_docs_cache_1024.pkl", "rb") as f:
        cache = pickle.load(f)
    print(f"  {len(cache)} docs in cache")

    # Fix double-domain keys
    doc_data = fix_cache_keys(cache)
    print(f"  Fixed keys: {len(doc_data)} docs")

    print("Loading LegalBench-RAG queries...")
    corpora = load_legalbench_rag()
    all_queries = []
    for corpus in corpora.values():
        all_queries.extend(corpus.queries)
    print(f"  {len(all_queries)} queries")

    print("Loading embedder (BGE-M3, 1024-d)...")
    embedder = BatchEmbedder(model_name="BAAI/bge-m3", output_dim=1024, max_seq_length=512)

    results = []
    for chunker in METHODS:
        print(f"\n{'='*50}\nRunning {chunker.name}...\n{'='*50}")
        result = run_method(chunker, doc_data, all_queries, embedder)
        results.append(result)
        # Print quick summary
        r = result["retrieval"]
        print(f"  R@1={r['recall@1']:.4f}  R@5={r['recall@5']:.4f}  R@10={r['recall@10']:.4f}  MRR={r['mrr']:.4f}")
        print(f"  Intrinsic: cohesion={result['intrinsic'].get('cohesion',0):.4f} separation={result['intrinsic'].get('separation',0):.4f}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS — LegalBench-RAG (1024-d baselines)")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Cohesion':>10} {'Separation':>12} {'Chunks':>8}")
    print("-" * 90)
    for r in results:
        m = r["retrieval"]
        i = r["intrinsic"]
        print(f"{r['method']:<25} {m['recall@1']:8.4f} {m['recall@5']:8.4f} {m['recall@10']:8.4f} {m['mrr']:8.4f} {i.get('cohesion',0):10.4f} {i.get('separation',0):12.4f} {r['total_chunks']:>8}")

    out_path = PROJECT_ROOT / "results" / "main" / "legalbench_baselines_1024d.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

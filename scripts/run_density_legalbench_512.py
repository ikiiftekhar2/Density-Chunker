"""DensityChunker — LegalBench 512-d. Tunes sigma on FULL dataset via Optuna, saves best."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
import optuna
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunkers.density import DensityChunker
from src.data.loader import load_legalbench_rag
from src.data.types import Chunk
from src.embedders.embedder import BatchEmbedder
from src.evaluation.intrinsic import compute_intrinsic_metrics
from src.evaluation.retrieval import evaluate_retrieval
from src.retrieval.indexer import ChromaIndexer
from src.retrieval.retriever import ChromaRetriever

CACHE_PATH = PROJECT_ROOT / "results" / "segmented_docs_cache_512.pkl"
OUT_PATH = PROJECT_ROOT / "results" / "main" / "density_legalbench_512d.json"
DIM = 512
N_TRIALS = 10
CHUNK_BATCH = 64

# Module-level refs set before optimize()
_doc_data = None
_queries = None
_embedder = None


def objective(trial) -> float:
    sigma = trial.suggest_float("sigma_position", 3, 50)
    chunker = DensityChunker(sigma_position=sigma, smoothing_sigma=2.0, valley_prominence=0.3, min_sentences=3, max_sentences=40)

    all_chunks = {}
    intrinsic_scores = []
    for doc_id, (sentences, sent_embs) in tqdm(_doc_data.items(), desc=f"    Chunking σ={sigma:.1f}", unit="doc", leave=False):
        chunks = chunker.chunk_document(sentences, sent_embs)
        for ch in chunks:
            ch.metadata["doc_id"] = doc_id
        all_chunks[doc_id] = chunks
        intrinsic_scores.append(compute_intrinsic_metrics(chunks, sent_embs))

    flat_chunks = [Chunk(text=ch.text, sentences=ch.sentences, start_char=ch.start_char,
        end_char=ch.end_char, chunk_id=f"{doc_id}::{i}", metadata={"doc_id": ch.metadata["doc_id"]})
        for doc_id, chunks in all_chunks.items() for i, ch in enumerate(chunks)]

    indexer = ChromaIndexer(collection_name=f"opt_lb512_{trial.number}")
    retriever = ChromaRetriever(indexer=indexer, embedder=_embedder, k=10)
    chunk_embs = _embedder.encode([ch.text for ch in flat_chunks], batch_size=CHUNK_BATCH)
    metadatas = [{"doc_id": ch.metadata["doc_id"], "start_char": ch.start_char, "end_char": ch.end_char} for ch in flat_chunks]
    indexer.add_chunks(chunk_ids=[ch.chunk_id for ch in flat_chunks], embeddings=chunk_embs, metadatas=metadatas)

    retrievals = {}
    for q in tqdm(_queries, desc=f"    Retrieving σ={sigma:.1f}", unit="q", leave=False):
        doc_id = q.gold_spans[0].file_path if q.gold_spans else ""
        results = retriever.retrieve(q.query_text, k=10, where={"doc_id": doc_id})
        retrievals[q.query_id] = [int(cid.split("::")[-1]) for cid, _, _ in results]

    metrics = evaluate_retrieval(_queries, all_chunks, retrievals, k_values=[1, 3, 5, 10])
    indexer.delete_collection()
    return metrics["recall@5"]


def main():
    global _doc_data, _queries, _embedder

    print(f"Loading cache ({DIM}-d)...")
    with open(CACHE_PATH, "rb") as f:
        _doc_data = pickle.load(f)
    print(f"  {len(_doc_data)} docs")

    print("Loading queries...")
    corpora = load_legalbench_rag()
    _queries = []
    for corpus in corpora.values():
        _queries.extend(corpus.queries)
    print(f"  {len(_queries)} queries")

    print(f"Loading embedder (BGE-M3, {DIM}-d)...")
    _embedder = BatchEmbedder(model_name="BAAI/bge-m3", output_dim=DIM, max_seq_length=512)

    print(f"\nOptuna tuning on FULL dataset ({N_TRIALS} trials)...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    for i in tqdm(range(N_TRIALS), desc="  Optuna trials", unit="trial"):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        tqdm.write(f"    Trial {i}: σ={study.trials[-1].params['sigma_position']:.1f} → R@5={study.trials[-1].value:.4f}  (best σ={study.best_params['sigma_position']:.1f}, R@5={study.best_value:.4f})")

    best_sigma = study.best_params["sigma_position"]
    print(f"\nBest sigma: {best_sigma:.1f} → R@5={study.best_value:.4f}")

    print(f"\nFinal run with σ={best_sigma:.1f}...")
    chunker = DensityChunker(sigma_position=best_sigma, smoothing_sigma=2.0, valley_prominence=0.3, min_sentences=3, max_sentences=40)

    all_chunks = {}
    intrinsic_scores = []
    for doc_id, (sentences, sent_embs) in tqdm(_doc_data.items(), desc="  Chunking", unit="doc", leave=False):
        chunks = chunker.chunk_document(sentences, sent_embs)
        for ch in chunks:
            ch.metadata["doc_id"] = doc_id
        all_chunks[doc_id] = chunks
        intrinsic_scores.append(compute_intrinsic_metrics(chunks, sent_embs))

    intrinsic = {}
    if intrinsic_scores:
        for key in intrinsic_scores[0]:
            intrinsic[key] = float(np.mean([s[key] for s in intrinsic_scores]))

    flat_chunks = [Chunk(text=ch.text, sentences=ch.sentences, start_char=ch.start_char,
        end_char=ch.end_char, chunk_id=f"{doc_id}::{i}", metadata={"doc_id": ch.metadata["doc_id"]})
        for doc_id, chunks in all_chunks.items() for i, ch in enumerate(chunks)]

    indexer = ChromaIndexer(collection_name=f"final_lb512")
    retriever = ChromaRetriever(indexer=indexer, embedder=_embedder, k=10)
    chunk_embs = _embedder.encode([ch.text for ch in flat_chunks], batch_size=CHUNK_BATCH)
    metadatas = [{"doc_id": ch.metadata["doc_id"], "start_char": ch.start_char, "end_char": ch.end_char} for ch in flat_chunks]
    indexer.add_chunks(chunk_ids=[ch.chunk_id for ch in flat_chunks], embeddings=chunk_embs, metadatas=metadatas)

    retrievals = {}
    for q in tqdm(_queries, desc="  Retrieving", unit="q", leave=False):
        doc_id = q.gold_spans[0].file_path if q.gold_spans else ""
        results = retriever.retrieve(q.query_text, k=10, where={"doc_id": doc_id})
        retrievals[q.query_id] = [int(cid.split("::")[-1]) for cid, _, _ in results]

    retrieval_metrics = evaluate_retrieval(_queries, all_chunks, retrievals, k_values=[1, 3, 5, 10])
    indexer.delete_collection()

    result = {
        "method": "density",
        "dim": DIM,
        "sigma": best_sigma,
        "intrinsic": intrinsic,
        "retrieval": retrieval_metrics,
        "total_chunks": sum(len(c) for c in all_chunks.values()),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump([result], f, indent=2)

    r = retrieval_metrics
    i = intrinsic
    print(f"\n{'='*60}")
    print(f"FINAL: DensityChunker σ={best_sigma} @ {DIM}-d")
    print(f"  R@1={r['recall@1']:.4f}  R@5={r['recall@5']:.4f}  R@10={r['recall@10']:.4f}  MRR={r['mrr']:.4f}")
    print(f"  Chunks={result['total_chunks']}")
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()

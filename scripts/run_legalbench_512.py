"""LegalBench 512-d baseline eval. Optimized: large batch sizes, parallel spaCy."""
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
from src.data.types import Chunk, Document, Sentence
from src.embedders.embedder import BatchEmbedder
from src.evaluation.intrinsic import compute_intrinsic_metrics
from src.evaluation.retrieval import evaluate_retrieval
from src.retrieval.indexer import ChromaIndexer
from src.retrieval.retriever import ChromaRetriever

CACHE_PATH = PROJECT_ROOT / "results" / "segmented_docs_cache_512.pkl"
SENT_CACHE_PATH = PROJECT_ROOT / "results" / "legalbench_sentences_512.pkl"
SENT_BATCH = 64
CHUNK_BATCH = 64

METHODS: list[BaseChunker] = [
    FixedSizeChunker(chunk_size=5),
    FixedSizeChunker(chunk_size=10),
    FixedSizeChunker(chunk_size=40),
    RecursiveChunker(chunk_size=512, chunk_overlap=100),
    SemanticChunker(threshold_percentile=3.0),
    SemanticChunker(threshold_percentile=5.0),
    SemanticChunker(threshold_percentile=10.0),
]


def segment_docs(documents: dict[str, Document], nlp, batch_size: int = 64) -> dict[str, list[Sentence]]:
    doc_ids, texts = [], []
    for doc_id, doc in documents.items():
        doc_ids.append(doc_id)
        texts.append(doc.text)

    result = {}
    for doc_id, spacy_doc in tqdm(
        zip(doc_ids, nlp.pipe(texts, batch_size=batch_size)),
        total=len(doc_ids), desc="Segmenting docs", unit="doc",
    ):
        sents = []
        for sent in spacy_doc.sents:
            sents.append(Sentence(
                text=sent.text.strip(), index=len(sents),
                start_char=sent.start_char, end_char=sent.end_char,
            ))
        if len(sents) >= 3:
            result[doc_id] = sents
    return result


def embed_sentences(doc_sentences, embedder):
    result = {}
    for doc_id, sents in tqdm(doc_sentences.items(), desc="Embedding sentences", unit="doc", leave=False):
        sent_embs = embedder.encode([s.text for s in sents], batch_size=SENT_BATCH)
        result[doc_id] = (sents, sent_embs)
    return result


def run_method(chunker, doc_data, queries, embedder):
    t0 = time.time()
    all_chunks = {}
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

    indexer = ChromaIndexer(collection_name=f"bl512_{chunker.name}")
    retriever = ChromaRetriever(indexer=indexer, embedder=embedder, k=10)

    flat_chunks = [Chunk(
        text=ch.text, sentences=ch.sentences, start_char=ch.start_char,
        end_char=ch.end_char, chunk_id=f"{doc_id}::{i}",
        metadata={"doc_id": ch.metadata["doc_id"]},
    ) for doc_id, chunks in all_chunks.items() for i, ch in enumerate(chunks)]

    chunk_embs = embedder.encode([ch.text for ch in flat_chunks], batch_size=CHUNK_BATCH)
    metadatas = [{"doc_id": ch.metadata["doc_id"], "start_char": ch.start_char, "end_char": ch.end_char} for ch in flat_chunks]
    indexer.add_chunks(
        chunk_ids=[ch.chunk_id for ch in flat_chunks],
        embeddings=chunk_embs, metadatas=metadatas,
    )

    retrievals = {}
    for q in tqdm(queries, desc=f"  Retrieving {chunker.name}", unit="query", leave=False):
        doc_id = q.gold_spans[0].file_path if q.gold_spans else ""
        results = retriever.retrieve(q.query_text, k=10, where={"doc_id": doc_id})
        retrievals[q.query_id] = [int(cid.split("::")[-1]) for cid, _, _ in results]

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
    import spacy
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading LegalBench-RAG...")
    corpora = load_legalbench_rag()
    print(f"Domains: {list(corpora.keys())}")

    all_docs = {}
    all_queries = []
    for domain, corpus in corpora.items():
        for doc_id, doc in corpus.documents.items():
            all_docs[doc_id] = doc
        all_queries.extend(corpus.queries)
    print(f"Total: {len(all_docs)} documents, {len(all_queries)} queries")

    if CACHE_PATH.exists():
        print(f"Loading cached docs from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            doc_data = pickle.load(f)
        print(f"  Loaded {len(doc_data)} docs")
    else:
        if SENT_CACHE_PATH.exists():
            print(f"Loading cached sentences from {SENT_CACHE_PATH}...")
            with open(SENT_CACHE_PATH, "rb") as f:
                lb_sentences = pickle.load(f)
            print(f"  Loaded {len(lb_sentences)} docs")
        else:
            lb_sentences = {}
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

        domain_docs = {}
        for domain, corpus in corpora.items():
            domain_docs[domain] = {}
            for doc_id, doc in corpus.documents.items():
                if doc_id not in lb_sentences:
                    domain_docs[domain][doc_id] = doc

        for domain, docs in domain_docs.items():
            if not docs:
                print(f"  {domain}: all {len(corpora[domain].documents)} docs already cached, skipping")
                continue
            bs = 8 if domain == "maud" else 64
            print(f"Segmenting {domain} ({len(docs)} docs, batch_size={bs})...")
            t0 = time.time()
            new_sentences = segment_docs(docs, nlp, batch_size=bs)
            print(f"  Done — {len(new_sentences)} docs in {time.time() - t0:.0f}s")
            lb_sentences.update(new_sentences)
            with open(SENT_CACHE_PATH, "wb") as f:
                pickle.dump(lb_sentences, f)
            print(f"  Sentences cache updated ({len(lb_sentences)} total)")

        print("Loading embedder (BGE-M3, 512-d)...")
        embedder = BatchEmbedder(model_name="BAAI/bge-m3", output_dim=512, max_seq_length=512)

        print("Embedding sentences at 512-d...")
        t0 = time.time()
        doc_data = embed_sentences(lb_sentences, embedder)
        print(f"  Done — {len(doc_data)} docs in {time.time() - t0:.0f}s")

        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(doc_data, f)
        print(f"  Cache saved to {CACHE_PATH}")

    print("Loading embedder (BGE-M3, 512-d)...")
    embedder = BatchEmbedder(model_name="BAAI/bge-m3", output_dim=512, max_seq_length=512)

    results = []
    for chunker in METHODS:
        print(f"\n{'='*50}\nRunning {chunker.name} (512-d)...\n{'='*50}")
        result = run_method(chunker, doc_data, all_queries, embedder)
        results.append(result)
        r = result["retrieval"]
        i = result["intrinsic"]
        print(f"  R@1={r['recall@1']:.4f}  R@5={r['recall@5']:.4f}  R@10={r['recall@10']:.4f}  MRR={r['mrr']:.4f}")
        print(f"  Cohesion={i.get('cohesion',0):.4f}  Separation={i.get('separation',0):.4f}  Chunks={result['total_chunks']}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS — LegalBench-RAG (512-d baselines)")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'R@1':>8} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Cohesion':>10} {'Separation':>12} {'Chunks':>8}")
    print("-" * 90)
    for r in results:
        m = r["retrieval"]
        i = r["intrinsic"]
        print(f"{r['method']:<25} {m['recall@1']:8.4f} {m['recall@5']:8.4f} {m['recall@10']:8.4f} {m['mrr']:8.4f} {i.get('cohesion',0):10.4f} {i.get('separation',0):12.4f} {r['total_chunks']:>8}")

    out_path = PROJECT_ROOT / "results" / "main" / "legalbench_baselines_512d.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

"""NarrativeQA 512-d baseline eval. Optimized: large batch sizes, parallel spaCy."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chunkers.base import BaseChunker
from src.chunkers.fixed import FixedSizeChunker
from src.chunkers.recursive import RecursiveChunker
from src.chunkers.semantic import SemanticChunker
from src.data.loader import load_narrativeqa
from src.data.types import Chunk, Sentence
from src.embedders.embedder import BatchEmbedder
from src.evaluation.intrinsic import compute_intrinsic_metrics
from src.retrieval.indexer import ChromaIndexer
from src.retrieval.retriever import ChromaRetriever

CACHE_PATH = PROJECT_ROOT / "results" / "nqa_docs_cache_512.pkl"
SENT_BATCH = 64
CHUNK_BATCH = 64

METHODS: list[BaseChunker] = [
    FixedSizeChunker(chunk_size=5),
    FixedSizeChunker(chunk_size=10),
    RecursiveChunker(chunk_size=512, chunk_overlap=100),
    SemanticChunker(threshold_percentile=10.0),
]

ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def segment_narrativeqa(samples, nlp):
    sample_ids = [s.sample_id for s in samples]
    contexts = [s.context for s in samples]

    result = {}
    for sid, spacy_doc in tqdm(
        zip(sample_ids, nlp.pipe(contexts, batch_size=64)),
        total=len(samples), desc="Segmenting NarrativeQA", unit="doc",
    ):
        sents = []
        for sent in spacy_doc.sents:
            sents.append(Sentence(
                text=sent.text.strip(), index=len(sents),
                start_char=sent.start_char, end_char=sent.end_char,
            ))
        if len(sents) >= 3:
            result[sid] = sents
    return result


def embed_sentences(doc_sentences, embedder):
    result = {}
    for doc_id, sents in tqdm(doc_sentences.items(), desc="Embedding sentences", unit="doc", leave=False):
        sent_embs = embedder.encode([s.text for s in sents], batch_size=SENT_BATCH)
        result[doc_id] = (sents, sent_embs)
    return result


def run_method(chunker, doc_data, samples, embedder):
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

    indexer = ChromaIndexer(collection_name=f"nqa512_{chunker.name}")
    retriever = ChromaRetriever(indexer=indexer, embedder=embedder, k=5)

    flat_chunks = [Chunk(
        text=ch.text, sentences=ch.sentences, start_char=ch.start_char,
        end_char=ch.end_char, chunk_id=f"{doc_id}::{i}",
        metadata={"doc_id": ch.metadata["doc_id"]},
    ) for doc_id, chunks in all_chunks.items() for i, ch in enumerate(chunks)]

    chunk_embs = embedder.encode([ch.text for ch in flat_chunks], batch_size=CHUNK_BATCH)
    metadatas = [{"doc_id": ch.metadata["doc_id"]} for ch in flat_chunks]
    indexer.add_chunks(
        chunk_ids=[ch.chunk_id for ch in flat_chunks],
        embeddings=chunk_embs, metadatas=metadatas,
    )

    rouge_l_scores = []
    for sample in tqdm(samples, desc=f"  Retrieving {chunker.name}", unit="query", leave=False):
        results = retriever.retrieve(sample.question, k=5, where={"doc_id": sample.sample_id})
        doc_texts = []
        for chunk_id, _, _ in results:
            doc = chunk_id.rsplit("::", 1)[0]
            idx = int(chunk_id.rsplit("::", 1)[-1])
            chunks_list = all_chunks.get(doc, [])
            if idx < len(chunks_list):
                doc_texts.append(chunks_list[idx].text[:500])

        combined = " ".join(doc_texts)
        best_rl = max(
            (ROUGE.score(ans, combined)["rougeL"].fmeasure for ans in sample.answers),
            default=0.0,
        )
        rouge_l_scores.append(best_rl)

    avg_rouge_l = float(np.mean(rouge_l_scores)) if rouge_l_scores else 0.0
    indexer.delete_collection()
    elapsed = time.time() - t0

    return {
        "method": chunker.name,
        "intrinsic": intrinsic,
        "rouge_l": avg_rouge_l,
        "time_sec": round(elapsed, 1),
        "total_chunks": sum(len(c) for c in all_chunks.values()),
    }


def main():
    import spacy
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    print("Loading NarrativeQA...")
    nqa = load_narrativeqa()
    print(f"Samples: {len(nqa.samples)}")

    if CACHE_PATH.exists():
        print(f"Loading cached docs from {CACHE_PATH}...")
        with open(CACHE_PATH, "rb") as f:
            doc_data = pickle.load(f)
        print(f"  Loaded {len(doc_data)} docs")
    else:
        print("Segmenting all NarrativeQA documents (spaCy, parallel, one-time)...")
        t0 = time.time()
        nqa_sentences = segment_narrativeqa(nqa.samples, nlp)
        print(f"  Done — {len(nqa_sentences)} docs in {time.time() - t0:.0f}s")

        print("Loading embedder (BGE-M3, 512-d)...")
        embedder = BatchEmbedder(model_name="BAAI/bge-m3", output_dim=512, max_seq_length=512)

        print("Embedding sentences at 512-d...")
        t0 = time.time()
        doc_data = embed_sentences(nqa_sentences, embedder)
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
        result = run_method(chunker, doc_data, nqa.samples, embedder)
        results.append(result)
        i = result["intrinsic"]
        print(f"  ROUGE-L={result['rouge_l']:.4f}  Cohesion={i.get('cohesion',0):.4f}  Separation={i.get('separation',0):.4f}  Chunks={result['total_chunks']}")

    print(f"\n{'='*60}")
    print("FINAL RESULTS — NarrativeQA (512-d baselines)")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'ROUGE-L':>10} {'Cohesion':>10} {'Separation':>12} {'Chunks':>8}")
    print("-" * 70)
    for r in results:
        i = r["intrinsic"]
        print(f"{r['method']:<25} {r['rouge_l']:10.4f} {i.get('cohesion',0):10.4f} {i.get('separation',0):12.4f} {r['total_chunks']:>8}")

    out_path = PROJECT_ROOT / "results" / "main" / "narrativeqa_baselines_512d.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

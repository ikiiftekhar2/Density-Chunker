"""Data loaders for LegalBench-RAG and NarrativeQA datasets."""

import json
from pathlib import Path

from .types import (
    Document,
    GoldSpan,
    LegalBenchCorpus,
    NarrativeQADataset,
    NarrativeQASample,
    Query,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"


def load_legalbench_rag(
    domain: str | None = None,
    mini: bool = False,
    data_dir: Path | None = None,
) -> LegalBenchCorpus | dict[str, LegalBenchCorpus]:
    """Load LegalBench-RAG dataset.

    Tries to load from local data directory first (Dropbox download),
    falls back to HuggingFace cache.

    Args:
        domain: Specific domain to load ('cuad', 'contractnli', 'maud', 'privacy_qa').
                If None, loads all domains and returns a dict keyed by domain.
        mini: Whether to load the mini version (776 queries).
        data_dir: Override the default datasets/legalbench_rag path.

    Returns:
        If domain is specified: a single LegalBenchCorpus.
        If domain is None: dict mapping domain name to LegalBenchCorpus.
    """
    base_dir = data_dir or DATASETS_DIR / "legalbench_rag"

    domains = [domain] if domain else ["cuad", "contractnli", "maud", "privacy_qa"]
    results = {}

    for d in domains:
        corpus = _load_single_domain(base_dir, d, mini)
        results[d] = corpus

    if domain:
        return results[domain]
    return results


def _load_single_domain(
    base_dir: Path, domain: str, mini: bool
) -> LegalBenchCorpus:
    """Load a single LegalBench-RAG domain."""
    documents = {}
    queries = []

    local_corpus_dir = base_dir / "data" / "corpus" / domain
    local_bench_file = base_dir / "data" / "benchmarks" / f"{domain}.json"

    if mini:
        local_bench_file = base_dir / "data" / "benchmarks" / f"{domain}_mini.json"

    if local_corpus_dir.exists():
        for txt_file in sorted(local_corpus_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8", errors="replace")
            rel_path = f"{domain}/{txt_file.name}"
            documents[rel_path] = Document(
                doc_id=rel_path,
                text=text,
                file_path=rel_path,
                metadata={"domain": domain},
            )

    if local_bench_file.exists():
        raw = json.loads(local_bench_file.read_text())
        bench_data = raw.get("tests", raw)
        if isinstance(bench_data, dict):
            bench_data = [bench_data]
        for i, item in enumerate(bench_data):
            query_text = item.get("query", item.get("input", ""))
            gold_spans = []

            snippets = item.get("snippets", [])
            if isinstance(snippets, str):
                snippets = json.loads(snippets)

            for snippet in snippets:
                span = snippet.get("span", [0, 0])
                gold_spans.append(
                    GoldSpan(
                        start_char=span[0] if isinstance(span, list) else 0,
                        end_char=span[1] if isinstance(span, list) else 0,
                        answer_text=snippet.get("answer", ""),
                        file_path=snippet.get("file_path", ""),
                    )
                )

            queries.append(
                Query(
                    query_id=f"{domain}_{i}",
                    query_text=query_text,
                    gold_spans=gold_spans,
                )
            )
    else:
        hf_file = base_dir / "huggingface" / f"{domain}.json"
        if mini:
            hf_file = base_dir / "huggingface" / f"{domain}_mini.json"

        if hf_file.exists():
            hf_data = json.loads(hf_file.read_text())
            for i, item in enumerate(hf_data):
                query_text = item.get("input", item.get("query", ""))
                expected = item.get("expected_output", item.get("snippets", "[]"))
                if isinstance(expected, str):
                    expected = json.loads(expected)
                if isinstance(expected, dict):
                    expected = [expected]

                gold_spans = []
                for snippet in expected:
                    span = snippet.get("span", [0, 0])
                    gold_spans.append(
                        GoldSpan(
                            start_char=span[0] if isinstance(span, list) else 0,
                            end_char=span[1] if isinstance(span, list) else 0,
                            answer_text=snippet.get("answer", ""),
                            file_path=snippet.get("file_path", ""),
                        )
                    )

                queries.append(
                    Query(
                        query_id=f"{domain}_{i}",
                        query_text=query_text,
                        gold_spans=gold_spans,
                    )
                )

    return LegalBenchCorpus(
        documents=documents,
        queries=queries,
        domain=domain,
    )


def load_narrativeqa(
    subsample: int = 0,
    data_dir: Path | None = None,
) -> NarrativeQADataset:
    """Load NarrativeQA from the LongBench preprocessed version.

    Args:
        subsample: Number of documents to keep. 0 means keep all.
        data_dir: Override the default datasets/narrativeqa path.

    Returns:
        NarrativeQADataset with samples ready for chunking and evaluation.
    """
    base_dir = data_dir or DATASETS_DIR / "narrativeqa"
    data_file = base_dir / "longbench_narrativeqa.json"

    if not data_file.exists():
        raise FileNotFoundError(
            f"NarrativeQA data not found at {data_file}. "
            "Run scripts/download_data.py --narrativeqa first."
        )

    raw = json.loads(data_file.read_text())

    samples = []
    for i, item in enumerate(raw):
        sample = NarrativeQASample(
            sample_id=item.get("id", f"narrqa_{i}"),
            question=item["question"],
            context=item["context"],
            answers=item["answers"],
            metadata={
                "length": item.get("length", 0),
                "dataset": item.get("dataset", "narrativeqa"),
            },
        )
        samples.append(sample)

    if subsample > 0 and subsample < len(samples):
        import random

        random.seed(42)
        samples = random.sample(samples, subsample)

    return NarrativeQADataset(samples=samples)


def load_legalbench_rag_mini(
    data_dir: Path | None = None,
) -> dict[str, LegalBenchCorpus]:
    """Load the mini version of LegalBench-RAG (776 queries total).

    Convenience wrapper around load_legalbench_rag(mini=True).
    """
    return load_legalbench_rag(mini=True, data_dir=data_dir)


def get_corpus_stats(corpus: LegalBenchCorpus) -> dict:
    """Compute basic statistics for a LegalBench-RAG corpus."""
    n_docs = len(corpus.documents)
    n_queries = len(corpus.queries)
    total_chars = sum(len(doc.text) for doc in corpus.documents.values())
    queries_with_spans = sum(1 for q in corpus.queries if q.gold_spans)
    total_spans = sum(len(q.gold_spans) for q in corpus.queries)

    return {
        "domain": corpus.domain,
        "n_documents": n_docs,
        "n_queries": n_queries,
        "total_chars": total_chars,
        "total_million_chars": round(total_chars / 1e6, 2),
        "queries_with_gold_spans": queries_with_spans,
        "total_gold_spans": total_spans,
    }


def get_narrativeqa_stats(dataset: NarrativeQADataset) -> dict:
    """Compute basic statistics for a NarrativeQA dataset."""
    n_samples = len(dataset.samples)
    total_chars = sum(len(s.context) for s in dataset.samples)
    avg_chars = total_chars / n_samples if n_samples else 0

    return {
        "n_samples": n_samples,
        "total_chars": total_chars,
        "total_million_chars": round(total_chars / 1e6, 2),
        "avg_context_chars": round(avg_chars),
        "avg_context_tokens_approx": round(avg_chars / 4),
    }
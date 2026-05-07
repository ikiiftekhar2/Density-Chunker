# DensityChunker

A framework for evaluating text chunking strategies in RAG pipelines. Compares fixed, recursive, semantic, and density-based chunkers on retrieval quality across legal and narrative benchmarks.

## Datasets

- **LegalBench-RAG** — 714 legal documents, 6,889 queries across 4 domains (CUAD, ContractNLI, MAUD, PrivacyQA)
- **NarrativeQA (LongBench)** — 200 stories with 200 QA pairs

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Usage

```python
from src.data.loader import load_legalbench_rag, load_narrativeqa

# Load LegalBench-RAG
corpora = load_legalbench_rag()
print({k: len(v.queries) for k, v in corpora.items()})

# Load NarrativeQA
nqa = load_narrativeqa()
print(f"{len(nqa.samples)} samples")
```

## Project Structure

```
src/
  data/           — Data types and loaders
  chunkers/       — Chunking strategies (fixed, recursive, semantic, density)
  embedders/      — Sentence transformer wrapper
  retrieval/      — ChromaDB index, retriever, reranker
  evaluation/     — Intrinsic, retrieval, end-to-end, and efficiency metrics
  visualization/  — Contour, heatmap, and comparison plots
scripts/           — Download and utility scripts
datasets/          — Local dataset storage
experiments/       — Experiment runner scripts
notebooks/         — Analysis notebooks
```

## License

MIT

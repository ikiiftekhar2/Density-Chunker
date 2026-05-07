"""Data types for DensityChunker."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Sentence:
    """A single sentence with position info."""
    text: str
    index: int
    start_char: int
    end_char: int


@dataclass
class Chunk:
    """A chunk of text produced by a chunker."""
    text: str
    sentences: list[Sentence]
    start_char: int
    end_char: int
    chunk_id: str
    metadata: dict = field(default_factory=dict)


@dataclass
class GoldSpan:
    """A gold-standard span from LegalBench-RAG annotations."""
    start_char: int
    end_char: int
    answer_text: str
    file_path: str


@dataclass
class Query:
    """A retrieval query with associated gold spans."""
    query_id: str
    query_text: str
    gold_spans: list[GoldSpan] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    """A source document."""
    doc_id: str
    text: str
    file_path: str
    metadata: dict = field(default_factory=dict)


@dataclass
class LegalBenchCorpus:
    """Full LegalBench-RAG dataset."""
    documents: dict[str, Document]  # file_path -> Document
    queries: list[Query]
    domain: str = ""


@dataclass
class NarrativeQASample:
    """A single NarrativeQA sample from LongBench."""
    sample_id: str
    question: str
    context: str  # full story text
    answers: list[str]
    metadata: dict = field(default_factory=dict)


@dataclass
class NarrativeQADataset:
    """NarrativeQA dataset from LongBench."""
    samples: list[NarrativeQASample]


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    method: str
    dataset: str
    retrieval_mode: str  # "embedding_only" or "embedding_rerank"
    metrics: dict = field(default_factory=dict)
    chunk_count: int = 0
    index_size_mb: float = 0.0
    chunking_time_sec: float = 0.0
    query_latency_ms: float = 0.0
"""Configuration for MyGRAGMemory project."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryGraphConfig:
    """Memory graph construction and retrieval configuration."""

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Graph construction
    similarity_threshold: float = 0.60
    topic_threshold: float = 0.40
    max_edges_per_node: int = 8

    # PPR
    ppr_alpha: float = 0.85
    ppr_max_iter: int = 50
    ppr_convergence_threshold: float = 1e-6

    # Retrieval fusion
    vector_weight: float = 0.6

    # Storage
    storage_path: str = "results/memories.json"

    # Default topics for experiments
    default_topics: list = field(default_factory=lambda: [
        "rag", "python", "deep-learning", "database", "nlp"
    ])

"""
Incoming: none --- {none}
Processing: package exports --- {1 job: exports}
Outgoing: retriever classes --- {6 retrievers}

Retriever Module Exports
"""

from .base import BaseRetriever, RetrieverResult
from .bm25 import BM25Retriever
from .tct_colbert import TCTColBERTRetriever
from .splade import SpladeRetriever
from .bge import BGERetriever
from .bm25_tct import BM25TCTRetriever
from .bm25_monot5 import BM25MonoT5Retriever

__all__ = [
    "BaseRetriever",
    "RetrieverResult",
    "BM25Retriever",
    "TCTColBERTRetriever",
    "SpladeRetriever",
    "BGERetriever",
    "BM25TCTRetriever",
    "BM25MonoT5Retriever",
]

RETRIEVER_REGISTRY = {
    "BM25": BM25Retriever,
    "TCT-ColBERT": TCTColBERTRetriever,
    "Splade": SpladeRetriever,
    "BGE": BGERetriever,
    "BM25_TCT": BM25TCTRetriever,
    "BM25_MonoT5": BM25MonoT5Retriever,
}


def get_retriever(name: str) -> type:
    """Get retriever class by name."""
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Unknown retriever: {name}. Available: {list(RETRIEVER_REGISTRY.keys())}")
    return RETRIEVER_REGISTRY[name]


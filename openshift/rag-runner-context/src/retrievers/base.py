"""
Incoming: query, corpus --- {str, Dict}
Processing: none (abstract) --- {0 jobs}
Outgoing: ranked results --- {RetrieverResult}

Base Retriever Interface
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional


@dataclass
class RetrieverResult:
    """Standard retriever output format."""
    qid: str
    results: List[Tuple[str, float, int]]  # (docno, score, rank)
    retriever_name: str
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_trec_lines(self, run_name: Optional[str] = None) -> List[str]:
        """Convert to TREC format lines."""
        name = run_name or self.retriever_name
        lines = []
        for docno, score, rank in self.results:
            lines.append(f"{self.qid} Q0 {docno} {rank} {score:.6f} {name}")
        return lines


class BaseRetriever(ABC):
    """Abstract base class for all retrievers."""
    
    name: str = "base"
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        qid: str,
        top_k: int = 100,
        **kwargs
    ) -> RetrieverResult:
        """Retrieve documents for a single query."""
        pass
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = 100,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """Retrieve documents for multiple queries."""
        results = {}
        for qid, query in queries.items():
            results[qid] = self.retrieve(query, qid, top_k, **kwargs)
        return results
    
    @staticmethod
    def normalize_scores(results: List[Tuple[str, float, int]]) -> List[Tuple[str, float, int]]:
        """Min-max normalize scores."""
        if not results:
            return []
        
        scores = [s for _, s, _ in results]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s > min_s else 1.0
        
        return [
            (docno, (score - min_s) / range_s, rank)
            for docno, score, rank in results
        ]


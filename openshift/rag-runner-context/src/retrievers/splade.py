"""
Incoming: query --- {str}
Processing: sparse learned retrieval --- {1 job: Pyserini SPLADE search}
Outgoing: ranked results --- {RetrieverResult}

SPLADE Sparse Learned Retriever (Pyserini)
------------------------------------------
Uses Pyserini's pre-built SPLADE index.
CPU-based (Lucene), uses multi-threading for efficiency.
Supports checkpointing for crash recovery.
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import config first - sets up environment
from src.config import config

from .base import BaseRetriever, RetrieverResult


class SpladeRetriever(BaseRetriever):
    """Sparse learned retrieval using Pyserini pre-built SPLADE index."""
    
    name = "Splade"
    
    def __init__(
        self,
        index_name: Optional[str] = None,
        encoder_name: Optional[str] = None,
        threads: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize SPLADE retriever using Pyserini pre-built index.
        
        Args:
            index_name: Pyserini pre-built index name (from config if None)
            encoder_name: Query encoder model (from config if None)
            threads: Number of threads for batch search (from config if None)
        """
        from pyserini.search.lucene import LuceneImpactSearcher
        
        # Get config values
        splade_config = config.indexes.pyserini.splade
        
        # Detect dataset from index_name if provided
        dataset = "nq"
        if index_name:
            for ds in config.datasets.supported:
                if ds in index_name.lower():
                    dataset = ds
                    break
        
        self.dataset = dataset
        self.index_name = index_name or config.get_index_name("splade", dataset)
        self.encoder_name = encoder_name or splade_config.encoder
        self.threads = threads or config.processing.threads.faiss
        
        print(f"[SPLADE] Loading impact index...")
        print(f"[SPLADE] Index: {self.index_name}")
        print(f"[SPLADE] Query encoder: {self.encoder_name}")
        print(f"[SPLADE] Threads: {self.threads}")
        
        # Use index from data folder
        project_root = config.project_root
        index_dir = project_root / "data" / dataset / "index" / "splade"
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index not found: {index_dir}")
        
        self.searcher = LuceneImpactSearcher(
            index_dir=str(index_dir),
            query_encoder=self.encoder_name
        )
        
        print(f"[SPLADE] Ready for retrieval")
    
    def retrieve(self, query: str, qid: str, top_k: int = None, **kwargs) -> RetrieverResult:
        """Retrieve documents using SPLADE."""
        top_k = top_k or config.processing.retrieval.top_k
        start = time.time()
        
        hits = self.searcher.search(query, k=top_k)
        results = [(hit.docid, float(hit.score), rank + 1) for rank, hit in enumerate(hits)]
        
        return RetrieverResult(
            qid=qid, results=results, retriever_name=self.name,
            latency_ms=(time.time() - start) * 1000,
            metadata={"index": self.index_name, "encoder": self.encoder_name}
        )
    
    def _process_mini_batch(
        self,
        queries: List[Tuple[str, str]],
        top_k: int
    ) -> List[RetrieverResult]:
        """Process a mini-batch using Pyserini batch search."""
        query_ids = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]
        
        t0 = time.time()
        batch_hits = self.searcher.batch_search(
            queries=query_texts,
            qids=query_ids,
            k=top_k,
            threads=self.threads
        )
        print(f"[SPLADE]     Searched {len(queries)} queries in {time.time()-t0:.1f}s")
        
        results = []
        for qid in query_ids:
            hits = batch_hits.get(qid, [])
            doc_results = [(hit.docid, float(hit.score), rank + 1) for rank, hit in enumerate(hits)]
            results.append(RetrieverResult(
                qid=qid, results=doc_results, retriever_name=self.name,
                latency_ms=0, metadata={"index": self.index_name}
            ))
        
        return results
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = None,
        checkpoint_path: Optional[str] = None,
        mini_batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """
        Batch retrieval with checkpointing.
        
        Args:
            queries: Dict of qid -> query text
            top_k: Docs per query
            checkpoint_path: JSONL file for crash recovery
            mini_batch_size: Queries per mini-batch (SPLADE is fast, so larger batches OK)
        """
        top_k = top_k or config.processing.retrieval.top_k
        mini_batch_size = mini_batch_size or config.processing.batch_sizes.splade_mini
        
        start = time.time()
        n_queries = len(queries)
        
        # Load checkpoint
        completed = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[SPLADE] Loading checkpoint...")
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    qid = data["qid"]
                    completed[qid] = RetrieverResult(
                        qid=qid,
                        results=[(d["docno"], d["score"], d["rank"]) for d in data["results"]],
                        retriever_name=self.name, latency_ms=0, metadata={"index": self.index_name}
                    )
            print(f"[SPLADE] Resumed: {len(completed)}/{n_queries} queries from checkpoint")
        
        # Filter pending
        pending = [(qid, text) for qid, text in queries.items() if qid not in completed]
        n_pending = len(pending)
        
        if n_pending == 0:
            print(f"[SPLADE] All {n_queries} queries completed!")
            return completed
        
        print(f"[SPLADE] Processing {n_pending} queries in batches of {mini_batch_size}")
        
        n_batches = (n_pending + mini_batch_size - 1) // mini_batch_size
        
        for i in range(n_batches):
            batch_start = i * mini_batch_size
            batch_end = min(batch_start + mini_batch_size, n_pending)
            batch = pending[batch_start:batch_end]
            
            t0 = time.time()
            print(f"[SPLADE] Batch {i+1}/{n_batches}: {len(batch)} queries...")
            
            batch_results = self._process_mini_batch(batch, top_k)
            
            # Save to checkpoint
            if checkpoint_path:
                with open(checkpoint_path, 'a') as f:
                    for r in batch_results:
                        f.write(json.dumps({
                            "qid": r.qid,
                            "results": [{"docno": d[0], "score": d[1], "rank": d[2]} for d in r.results]
                        }) + "\n")
            
            for r in batch_results:
                completed[r.qid] = r
            
            done = len(completed)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_queries - done) / rate if rate > 0 else 0
            print(f"[SPLADE]   -> {done}/{n_queries} done ({time.time()-t0:.1f}s) | ETA: {eta/60:.1f}min")
            
            # Memory cleanup
            del batch_results
            gc.collect()
        
        print(f"[SPLADE] Complete: {n_queries} queries in {time.time()-start:.1f}s")
        return completed

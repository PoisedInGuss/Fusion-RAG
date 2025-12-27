"""
Incoming: query, index, corpus_path --- {str, pt.Index, str}
Processing: two-stage retrieval --- {PyTerrier pipeline: BM25 >> TCT-ColBERT}
Outgoing: ranked results --- {RetrieverResult}

BM25 >> TCT-ColBERT Hybrid Retriever
------------------------------------
Efficient mini-batch processing:
1. Process N queries at a time (not all at once)
2. BM25 -> text load -> TCT for each mini-batch
3. Checkpoint after each batch for crash recovery
4. Clear memory between batches

Optimized for Mac M4:
- MPS acceleration for TCT-ColBERT
- Multi-threaded tokenization
"""

import gc
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

# Import config first - sets up environment
from src.config import config, get_device, ensure_pyterrier_init

from .base import BaseRetriever, RetrieverResult


class BM25TCTRetriever(BaseRetriever):
    """
    Two-stage retrieval: BM25 >> TCT-ColBERT
    
    Efficient mini-batch processing with checkpointing.
    """
    
    name = "BM25_TCT"
    
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        first_stage_k: Optional[int] = None,
        tct_batch_size: Optional[int] = None
    ):
        import pyterrier_dr as dr
        
        pt = ensure_pyterrier_init()
        
        # Get config values
        self.tct_model = config.models.tct_colbert.name
        
        self.corpus_path = corpus_path
        self.first_stage_k = first_stage_k or config.processing.retrieval.first_stage_k
        self.tct_batch_size = tct_batch_size or config.processing.batch_sizes.tct_encoding
        
        # BM25 first stage - use BEIR params for consistency
        print(f"[BM25_TCT] Loading BM25 index...")
        self.index = pt.IndexFactory.of(index_path)
        self.bm25 = pt.BatchRetrieve(
            self.index, 
            wmodel="BM25", 
            num_results=self.first_stage_k,
            metadata=["docno"],
            controls={"bm25.k_1": "0.9", "bm25.b": "0.4"}  # BEIR standard params
        )
        
        # TCT-ColBERT reranker with MPS acceleration
        self.device = get_device()
        print(f"[BM25_TCT] Loading {self.tct_model} on {self.device}...")
        self.tct_reranker = dr.TctColBert.hnp(
            batch_size=self.tct_batch_size,
            verbose=False,
            device=self.device
        ).text_scorer()
        
        # Build corpus offset index for lazy loading
        self._corpus_offsets = self._build_corpus_offsets()
        
        print(f"[BM25_TCT] Ready: BM25(k={self.first_stage_k}) >> TCT-ColBERT")
    
    def _build_corpus_offsets(self) -> Dict[str, int]:
        """Build doc_id -> byte offset map for lazy loading."""
        corpus_file = Path(self.corpus_path) / "corpus.jsonl"
        offsets = {}
        
        print(f"[BM25_TCT] Building corpus offset index...")
        with open(corpus_file, 'r', encoding='utf-8') as f:
            offset = 0
            for line in f:
                doc = json.loads(line)
                doc_id = doc.get("_id", "")
                offsets[doc_id] = offset
                offset += len(line.encode('utf-8'))
        
        print(f"[BM25_TCT] Indexed {len(offsets)} documents")
        return offsets
    
    def _load_doc_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Load document texts by ID (sorted seek for efficiency)."""
        corpus_file = Path(self.corpus_path) / "corpus.jsonl"
        texts = {}
        
        # Sort by offset for sequential disk reads
        ids_with_offset = [(did, self._corpus_offsets.get(did, -1)) for did in doc_ids]
        ids_with_offset = [(did, off) for did, off in ids_with_offset if off >= 0]
        ids_with_offset.sort(key=lambda x: x[1])
        
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for doc_id, offset in ids_with_offset:
                f.seek(offset)
                doc = json.loads(f.readline())
                text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                texts[doc_id] = text
        
        return texts
    
    def _process_mini_batch(
        self, 
        queries: List[Tuple[str, str]],  # [(qid, query_text), ...]
        top_k: int
    ) -> List[RetrieverResult]:
        """
        Process a mini-batch: BM25 -> text load -> TCT rerank
        
        Returns list of RetrieverResult for this batch.
        """
        # Build query DataFrame
        query_df = pd.DataFrame([
            {
                "qid": qid,
                "query": re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text)).strip()
            }
            for qid, text in queries
        ])
        
        # Step 1: BM25 retrieval
        t0 = time.time()
        bm25_results = self.bm25.transform(query_df)
        print(f"[BM25_TCT]     BM25: {len(bm25_results)} docs in {time.time()-t0:.1f}s")
        
        # Step 2: Load text for retrieved docs
        t0 = time.time()
        unique_docs = bm25_results["docno"].unique().tolist()
        doc_texts = self._load_doc_texts([str(d) for d in unique_docs])
        bm25_results = bm25_results.copy()
        bm25_results["text"] = bm25_results["docno"].apply(lambda x: doc_texts.get(str(x), ""))
        print(f"[BM25_TCT]     Texts: {len(doc_texts)} docs in {time.time()-t0:.1f}s")
        
        # Step 3: TCT-ColBERT reranking
        t0 = time.time()
        reranked = self.tct_reranker.transform(bm25_results)
        print(f"[BM25_TCT]     TCT: {len(reranked)} docs in {time.time()-t0:.1f}s")
        
        # Build results
        results = []
        grouped = reranked.groupby("qid")
        
        for qid, group in grouped:
            top_docs = group.nlargest(top_k, "score")
            
            doc_list = []
            for rank, (_, row) in enumerate(top_docs.iterrows(), start=1):
                doc_list.append((str(row["docno"]), float(row["score"]), rank))
            
            results.append(RetrieverResult(
                qid=qid,
                results=doc_list,
                retriever_name=self.name,
                latency_ms=0,
                metadata={"model": self.tct_model}
            ))
        
        # Cleanup intermediate dataframes
        del bm25_results, reranked, doc_texts
        gc.collect()
        
        return results
    
    def retrieve(self, query: str, qid: str, top_k: int = None, **kwargs) -> RetrieverResult:
        """Single query retrieval."""
        top_k = top_k or config.processing.retrieval.top_k
        results = self._process_mini_batch([(qid, query)], top_k)
        return results[0] if results else RetrieverResult(qid=qid, results=[], retriever_name=self.name)
    
    def retrieve_batch(
        self,
        queries: Dict[str, str],
        top_k: int = None,
        checkpoint_path: Optional[str] = None,
        mini_batch_size: Optional[int] = None,
        **kwargs
    ) -> Dict[str, RetrieverResult]:
        """
        Efficient batch retrieval with mini-batch processing and checkpointing.
        
        Args:
            queries: Dict of qid -> query text
            top_k: Docs per query
            checkpoint_path: JSONL file for crash recovery
            mini_batch_size: Queries per mini-batch (50 = ~5K docs to rerank)
        """
        top_k = top_k or config.processing.retrieval.top_k
        mini_batch_size = mini_batch_size or config.processing.batch_sizes.bm25_tct_mini
        
        start = time.time()
        n_queries = len(queries)
        
        # Load checkpoint
        completed = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[BM25_TCT] Loading checkpoint...")
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    qid = data["qid"]
                    completed[qid] = RetrieverResult(
                        qid=qid,
                        results=[(d["docno"], d["score"], d["rank"]) for d in data["results"]],
                        retriever_name=self.name,
                        latency_ms=0,
                        metadata={"model": self.tct_model}
                    )
            print(f"[BM25_TCT] Resumed: {len(completed)}/{n_queries} queries from checkpoint")
        
        # Filter pending
        pending = [(qid, text) for qid, text in queries.items() if qid not in completed]
        n_pending = len(pending)
        
        if n_pending == 0:
            print(f"[BM25_TCT] All {n_queries} queries completed!")
            return completed
        
        print(f"[BM25_TCT] Processing {n_pending} queries in batches of {mini_batch_size}")
        
        # Process mini-batches
        n_batches = (n_pending + mini_batch_size - 1) // mini_batch_size
        
        for i in range(n_batches):
            batch_start = i * mini_batch_size
            batch_end = min(batch_start + mini_batch_size, n_pending)
            batch = pending[batch_start:batch_end]
            
            t0 = time.time()
            print(f"[BM25_TCT] Batch {i+1}/{n_batches}: {len(batch)} queries...")
            
            # Process this batch
            batch_results = self._process_mini_batch(batch, top_k)
            
            # Save to checkpoint
            if checkpoint_path:
                with open(checkpoint_path, 'a') as f:
                    for r in batch_results:
                        f.write(json.dumps({
                            "qid": r.qid,
                            "results": [{"docno": d[0], "score": d[1], "rank": d[2]} for d in r.results]
                        }) + "\n")
            
            # Update completed
            for r in batch_results:
                completed[r.qid] = r
            
            # Progress
            done = len(completed)
            elapsed = time.time() - start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n_queries - done) / rate if rate > 0 else 0
            print(f"[BM25_TCT]   -> {done}/{n_queries} done ({time.time()-t0:.1f}s) | ETA: {eta/60:.1f}min")
            
            # Memory cleanup
            del batch_results
            gc.collect()
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        print(f"[BM25_TCT] Complete: {n_queries} queries in {time.time()-start:.1f}s")
        return completed

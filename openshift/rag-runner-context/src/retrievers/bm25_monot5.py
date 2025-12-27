"""
Incoming: query, index, corpus_path --- {str, pt.Index, str}
Processing: two-stage retrieval --- {PyTerrier pipeline: BM25 >> CrossEncoder}
Outgoing: ranked results --- {RetrieverResult}

BM25 >> MonoT5/CrossEncoder Hybrid Retriever
--------------------------------------------
Efficient mini-batch processing:
1. Process N queries at a time (not all at once)
2. BM25 -> text load -> CrossEncoder for each mini-batch
3. Checkpoint after each batch for crash recovery
4. Clear memory between batches

Optimized for Mac M4:
- MPS acceleration for CrossEncoder
- Multi-threaded tokenization
"""

import gc
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from sentence_transformers import CrossEncoder

# Import config first - sets up environment
from src.config import config, get_device, ensure_pyterrier_init

from .base import BaseRetriever, RetrieverResult


class CrossEncoderReranker:
    """PyTerrier Transformer wrapper for CrossEncoder reranking."""
    
    def __init__(self, model_name: str = None, batch_size: int = None, device: str = None):
        self.model_name = model_name or config.models.cross_encoder.name
        self.batch_size = batch_size or config.processing.batch_sizes.cross_encoder
        self.device = device or get_device()
        
        print(f"[CrossEncoder] Loading {self.model_name} on {self.device}...")
        self.model = CrossEncoder(self.model_name, device=self.device)
    
    def transform(self, topics_and_docs: pd.DataFrame) -> pd.DataFrame:
        """Rerank documents using CrossEncoder."""
        if len(topics_and_docs) == 0:
            return topics_and_docs
        
        pairs = [[row["query"], row.get("text", "")] for _, row in topics_and_docs.iterrows()]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        result = topics_and_docs.copy()
        result["score"] = scores
        result = result.sort_values(["qid", "score"], ascending=[True, False])
        result["rank"] = result.groupby("qid").cumcount() + 1
        
        return result


class BM25MonoT5Retriever(BaseRetriever):
    """
    Two-stage retrieval: BM25 >> CrossEncoder
    
    Efficient mini-batch processing with checkpointing.
    """
    
    name = "BM25_MonoT5"
    
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        first_stage_k: Optional[int] = None,
        ce_batch_size: Optional[int] = None
    ):
        pt = ensure_pyterrier_init()
        
        # Get config values
        self.ce_model = config.models.cross_encoder.name
        
        self.corpus_root = Path(corpus_path)
        self.first_stage_k = first_stage_k or config.processing.retrieval.first_stage_k
        self.ce_batch_size = ce_batch_size or config.processing.batch_sizes.cross_encoder
        self.corpus_format, self.corpus_file, self.tsv_path = self._resolve_corpus_source(self.corpus_root)
        
        # BM25 first stage - use BEIR params for consistency
        print(f"[BM25_MonoT5] Loading BM25 index...")
        self.index = pt.IndexFactory.of(index_path)
        self.bm25 = pt.BatchRetrieve(
            self.index,
            wmodel="BM25",
            num_results=self.first_stage_k,
            metadata=["docno"],
            controls={"bm25.k_1": "0.9", "bm25.b": "0.4"}  # BEIR standard params
        )
        
        # CrossEncoder reranker with MPS
        self.reranker = CrossEncoderReranker(
            model_name=self.ce_model,
            batch_size=self.ce_batch_size
        )
        
        # Build corpus offset index
        self._corpus_offsets = self._build_corpus_offsets()
        
        print(f"[BM25_MonoT5] Ready: BM25(k={self.first_stage_k}) >> CrossEncoder")
    
    def _resolve_corpus_source(self, root: Path) -> Tuple[str, Optional[Path], Optional[Path]]:
        """
        Determine where to load document texts from.
        
        Supports BEIR corpus.jsonl or DPR TSV (psgs_w100.tsv).
        """
        if root.is_file():
            if root.suffix == ".jsonl":
                return "jsonl", root, None
            if root.suffix == ".tsv":
                return "tsv", None, root
        
        candidates_jsonl = [
            root / "corpus.jsonl",
        ]
        for path in candidates_jsonl:
            if path.exists():
                return "jsonl", path, None
        
        candidates_tsv = [
            root / "psgs_w100.tsv",
            root / "psgs_w100" / "psgs_w100.tsv",
            root / "data" / "wikipedia_split" / "psgs_w100" / "psgs_w100.tsv",
        ]
        for path in candidates_tsv:
            if path.exists():
                return "tsv", None, path
        
        raise FileNotFoundError(
            f"[BM25_MonoT5] Could not locate corpus.jsonl or psgs_w100.tsv under {root}"
        )

    def _build_corpus_offsets(self) -> Dict[str, int]:
        """Build doc_id -> byte offset map for lazy loading."""
        offsets: Dict[str, int] = {}
        print(f"[BM25_MonoT5] Building corpus offset index ({self.corpus_format})...")
        
        if self.corpus_format == "jsonl":
            assert self.corpus_file is not None
            with self.corpus_file.open('r', encoding='utf-8') as f:
                offset = 0
                for line in f:
                    doc = json.loads(line)
                    doc_id = doc.get("_id", "")
                    offsets[doc_id] = offset
                    offset += len(line.encode('utf-8'))
        else:
            assert self.tsv_path is not None
            with self.tsv_path.open("rb") as f:
                offset = 0
                for i, line in enumerate(f):
                    parts = line.split(b"\t")
                    if len(parts) < 2:
                        offset += len(line)
                        continue
                    doc_id = parts[0].decode("utf-8", errors="ignore").strip()
                    if doc_id and doc_id.lower() != "id":
                        offsets[doc_id] = offset
                    offset += len(line)
                    if (i + 1) % 2_000_000 == 0:
                        print(f"[BM25_MonoT5] indexed {i+1:,} TSV lines...")
        
        print(f"[BM25_MonoT5] Indexed {len(offsets)} documents")
        return offsets
    
    def _load_doc_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Load document texts by ID (sorted seek for efficiency)."""
        texts: Dict[str, str] = {}
        
        ids_with_offset = [(did, self._corpus_offsets.get(did, -1)) for did in doc_ids]
        ids_with_offset = [(did, off) for did, off in ids_with_offset if off >= 0]
        ids_with_offset.sort(key=lambda x: x[1])
        
        if self.corpus_format == "jsonl":
            assert self.corpus_file is not None
            with self.corpus_file.open('r', encoding='utf-8') as f:
                for doc_id, offset in ids_with_offset:
                    f.seek(offset)
                    doc = json.loads(f.readline())
                    text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                    texts[doc_id] = text
        else:
            assert self.tsv_path is not None
            with self.tsv_path.open("rb") as f:
                for doc_id, offset in ids_with_offset:
                    f.seek(offset)
                    line = f.readline()
                    parts = line.split(b"\t")
                    if len(parts) < 2:
                        continue
                    title = ""
                    text = parts[1].decode("utf-8", errors="ignore").strip()
                    if len(parts) >= 3:
                        title = parts[2].decode("utf-8", errors="ignore").strip()
                    texts[doc_id] = f"{title} {text}".strip()
        
        return texts
    
    def _process_mini_batch(
        self,
        queries: List[Tuple[str, str]],
        top_k: int
    ) -> List[RetrieverResult]:
        """Process a mini-batch: BM25 -> text load -> CrossEncoder rerank"""
        query_df = pd.DataFrame([
            {"qid": qid, "query": re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9\s]", " ", text)).strip()}
            for qid, text in queries
        ])
        
        # Step 1: BM25
        t0 = time.time()
        bm25_results = self.bm25.transform(query_df)
        print(f"[BM25_MonoT5]     BM25: {len(bm25_results)} docs in {time.time()-t0:.1f}s")
        
        # Step 2: Load texts
        t0 = time.time()
        unique_docs = bm25_results["docno"].unique().tolist()
        doc_texts = self._load_doc_texts([str(d) for d in unique_docs])
        bm25_results = bm25_results.copy()
        bm25_results["text"] = bm25_results["docno"].apply(lambda x: doc_texts.get(str(x), ""))
        print(f"[BM25_MonoT5]     Texts: {len(doc_texts)} docs in {time.time()-t0:.1f}s")
        
        # Step 3: CrossEncoder reranking
        t0 = time.time()
        reranked = self.reranker.transform(bm25_results)
        print(f"[BM25_MonoT5]     CrossEncoder: {len(reranked)} docs in {time.time()-t0:.1f}s")
        
        # Build results
        results = []
        grouped = reranked.groupby("qid")
        
        for qid, group in grouped:
            top_docs = group.nlargest(top_k, "score")
            doc_list = [(str(row["docno"]), float(row["score"]), rank) 
                       for rank, (_, row) in enumerate(top_docs.iterrows(), start=1)]
            results.append(RetrieverResult(
                qid=qid, results=doc_list, retriever_name=self.name,
                latency_ms=0, metadata={"model": self.ce_model}
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
            mini_batch_size: Queries per mini-batch
        """
        top_k = top_k or config.processing.retrieval.top_k
        mini_batch_size = mini_batch_size or config.processing.batch_sizes.bm25_monot5_mini
        
        start = time.time()
        n_queries = len(queries)
        
        # Load checkpoint
        completed = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[BM25_MonoT5] Loading checkpoint...")
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    qid = data["qid"]
                    completed[qid] = RetrieverResult(
                        qid=qid,
                        results=[(d["docno"], d["score"], d["rank"]) for d in data["results"]],
                        retriever_name=self.name, latency_ms=0, metadata={"model": self.ce_model}
                    )
            print(f"[BM25_MonoT5] Resumed: {len(completed)}/{n_queries} queries from checkpoint")
        
        # Filter pending
        pending = [(qid, text) for qid, text in queries.items() if qid not in completed]
        n_pending = len(pending)
        
        if n_pending == 0:
            print(f"[BM25_MonoT5] All {n_queries} queries completed!")
            return completed
        
        print(f"[BM25_MonoT5] Processing {n_pending} queries in batches of {mini_batch_size}")
        
        n_batches = (n_pending + mini_batch_size - 1) // mini_batch_size
        
        for i in range(n_batches):
            batch_start = i * mini_batch_size
            batch_end = min(batch_start + mini_batch_size, n_pending)
            batch = pending[batch_start:batch_end]
            
            t0 = time.time()
            print(f"[BM25_MonoT5] Batch {i+1}/{n_batches}: {len(batch)} queries...")
            
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
            print(f"[BM25_MonoT5]   -> {done}/{n_queries} done ({time.time()-t0:.1f}s) | ETA: {eta/60:.1f}min")
            
            # Memory cleanup
            del batch_results
            gc.collect()
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        print(f"[BM25_MonoT5] Complete: {n_queries} queries in {time.time()-start:.1f}s")
        return completed

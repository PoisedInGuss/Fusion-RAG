"""
Incoming: query, corpus --- {str, Dict}
Processing: dense retrieval --- {1 job: embedding + FAISS similarity}
Outgoing: ranked results --- {RetrieverResult}

TCT-ColBERT Dense Retriever
---------------------------
Optimized for Mac M4:
- MPS acceleration
- fp16 embeddings (half memory)
- Chunked encoding with disk caching
- FAISS index for efficient search
- Checkpointing for crash recovery
"""

import gc
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from sentence_transformers import SentenceTransformer
import faiss

# Import config first - sets up environment
from src.config import config, get_device

from .base import BaseRetriever, RetrieverResult


class TCTColBERTRetriever(BaseRetriever):
    """
    Dense retrieval using TCT-ColBERT with MPS acceleration.
    
    Supports checkpointing for crash recovery during batch retrieval.
    """
    
    name = "TCT-ColBERT"
    
    def __init__(
        self,
        corpus: Dict[str, Dict[str, str]],
        cache_dir: Optional[str] = None,
        batch_size: Optional[int] = None,
        use_fp16: bool = True
    ):
        """
        Initialize TCT-ColBERT retriever.
        
        Args:
            corpus: {doc_id: {text, title}}
            cache_dir: Directory to cache embeddings
            batch_size: Batch size for encoding (from config if None)
            use_fp16: Use float16 to halve memory
        """
        # Get config values
        self.model_name = config.models.tct_colbert.name
        self.embedding_dim = config.models.tct_colbert.embedding_dim
        
        self.corpus = corpus
        self.batch_size = batch_size or config.processing.batch_sizes.tct_encoding
        self.use_fp16 = use_fp16
        self.doc_ids = list(corpus.keys())
        self.cache_dir = Path(cache_dir) if cache_dir else config.project_root / "data" / "nq" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = get_device()
        print(f"[TCT-ColBERT] Device: {self.device}, fp16: {use_fp16}")
        print(f"[TCT-ColBERT] Loading {self.model_name}...")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self._load_or_compute_index()
    
    def _get_cache_path(self) -> Path:
        """Get cache file path."""
        n_docs = len(self.doc_ids)
        dtype = "fp16" if self.use_fp16 else "fp32"
        return self.cache_dir / f"tct_colbert_{n_docs}_{dtype}.npy"
    
    def _load_or_compute_index(self):
        """Load cached embeddings or compute new ones."""
        cache_path = self._get_cache_path()
        ids_path = self.cache_dir / f"tct_colbert_{len(self.doc_ids)}_ids.npy"
        
        if cache_path.exists() and ids_path.exists():
            print(f"[TCT-ColBERT] Loading cached embeddings...")
            self.doc_embeddings = np.load(str(cache_path))
            cached_ids = np.load(str(ids_path), allow_pickle=True)
            
            if list(cached_ids) == self.doc_ids:
                print(f"[TCT-ColBERT] Loaded {len(self.doc_embeddings)} embeddings")
                self._build_faiss_index()
                return
            print("[TCT-ColBERT] Cache IDs mismatch, recomputing...")
        
        self._encode_corpus_chunked()
        np.save(str(cache_path), self.doc_embeddings)
        np.save(str(ids_path), np.array(self.doc_ids, dtype=object))
        print(f"[TCT-ColBERT] Saved embeddings to cache")
        self._build_faiss_index()
    
    def _encode_corpus_chunked(self):
        """Encode corpus in memory-efficient chunks."""
        print(f"[TCT-ColBERT] Encoding {len(self.doc_ids)} documents...")
        
        dtype = np.float16 if self.use_fp16 else np.float32
        n_docs = len(self.doc_ids)
        self.doc_embeddings = np.zeros((n_docs, self.embedding_dim), dtype=dtype)
        
        chunk_size = self.batch_size * 8
        
        for start_idx in range(0, n_docs, chunk_size):
            end_idx = min(start_idx + chunk_size, n_docs)
            chunk_ids = self.doc_ids[start_idx:end_idx]
            chunk_texts = [
                (self.corpus[d].get("title", "") + " " + self.corpus[d].get("text", "")).strip()
                for d in chunk_ids
            ]
            
            with torch.no_grad():
                embeddings = self.model.encode(
                    chunk_texts, batch_size=self.batch_size,
                    convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
                )
            
            if self.use_fp16:
                embeddings = embeddings.astype(np.float16)
            
            self.doc_embeddings[start_idx:end_idx] = embeddings
            print(f"  [{(end_idx/n_docs)*100:5.1f}%] Encoded {end_idx}/{n_docs} documents")
            
            del embeddings, chunk_texts
            gc.collect()
        
        print(f"[TCT-ColBERT] Encoding complete")
    
    def _build_faiss_index(self):
        """Build FAISS index for efficient search."""
        print("[TCT-ColBERT] Building FAISS index...")
        embeddings_f32 = self.doc_embeddings.astype(np.float32)
        
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        faiss.omp_set_num_threads(config.processing.threads.faiss)
        self.index.add(embeddings_f32)
        
        print(f"[TCT-ColBERT] FAISS index: {self.index.ntotal} vectors (threads: {faiss.omp_get_max_threads()})")
        
        del embeddings_f32
        gc.collect()
    
    def retrieve(self, query: str, qid: str, top_k: int = None, **kwargs) -> RetrieverResult:
        """Retrieve documents using dense similarity."""
        top_k = top_k or config.processing.retrieval.top_k
        start = time.time()
        
        with torch.no_grad():
            query_emb = self.model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )[0].astype(np.float32)
        
        scores, indices = self.index.search(query_emb.reshape(1, -1), top_k)
        
        results = [
            (self.doc_ids[idx], float(scores[0, i]), i + 1)
            for i, idx in enumerate(indices[0]) if idx >= 0
        ]
        
        return RetrieverResult(
            qid=qid, results=results, retriever_name=self.name,
            latency_ms=(time.time() - start) * 1000,
            metadata={"model": self.model_name, "device": self.device}
        )
    
    def _process_mini_batch(
        self,
        queries: List[Tuple[str, str]],
        top_k: int
    ) -> List[RetrieverResult]:
        """Process a mini-batch using vectorized FAISS search."""
        query_ids = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]
        
        t0 = time.time()
        with torch.no_grad():
            query_embs = self.model.encode(
                query_texts, batch_size=self.batch_size,
                convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            ).astype(np.float32)
        print(f"[TCT-ColBERT]     Encoded {len(queries)} queries in {time.time()-t0:.1f}s")
        
        t0 = time.time()
        all_scores, all_indices = self.index.search(query_embs, top_k)
        print(f"[TCT-ColBERT]     FAISS searched in {time.time()-t0:.1f}s")
        
        results = []
        for i, qid in enumerate(query_ids):
            doc_results = [
                (self.doc_ids[idx], float(all_scores[i, j]), j + 1)
                for j, idx in enumerate(all_indices[i]) if idx >= 0
            ]
            results.append(RetrieverResult(
                qid=qid, results=doc_results, retriever_name=self.name,
                latency_ms=0, metadata={"model": self.model_name}
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
            mini_batch_size: Queries per mini-batch
        """
        top_k = top_k or config.processing.retrieval.top_k
        mini_batch_size = mini_batch_size or config.processing.batch_sizes.tct_mini
        
        start = time.time()
        n_queries = len(queries)
        
        # Load checkpoint
        completed = {}
        if checkpoint_path and Path(checkpoint_path).exists():
            print(f"[TCT-ColBERT] Loading checkpoint...")
            with open(checkpoint_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    qid = data["qid"]
                    completed[qid] = RetrieverResult(
                        qid=qid,
                        results=[(d["docno"], d["score"], d["rank"]) for d in data["results"]],
                        retriever_name=self.name, latency_ms=0, metadata={"model": self.model_name}
                    )
            print(f"[TCT-ColBERT] Resumed: {len(completed)}/{n_queries} queries from checkpoint")
        
        # Filter pending
        pending = [(qid, text) for qid, text in queries.items() if qid not in completed]
        n_pending = len(pending)
        
        if n_pending == 0:
            print(f"[TCT-ColBERT] All {n_queries} queries completed!")
            return completed
        
        print(f"[TCT-ColBERT] Processing {n_pending} queries in batches of {mini_batch_size}")
        
        n_batches = (n_pending + mini_batch_size - 1) // mini_batch_size
        
        for i in range(n_batches):
            batch_start = i * mini_batch_size
            batch_end = min(batch_start + mini_batch_size, n_pending)
            batch = pending[batch_start:batch_end]
            
            t0 = time.time()
            print(f"[TCT-ColBERT] Batch {i+1}/{n_batches}: {len(batch)} queries...")
            
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
            print(f"[TCT-ColBERT]   -> {done}/{n_queries} done ({time.time()-t0:.1f}s) | ETA: {eta/60:.1f}min")
            
            # Memory cleanup
            del batch_results
            gc.collect()
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        print(f"[TCT-ColBERT] Complete: {n_queries} queries in {time.time()-start:.1f}s")
        return completed

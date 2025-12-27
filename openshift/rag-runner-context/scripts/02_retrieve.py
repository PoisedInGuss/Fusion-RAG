#!/usr/bin/env python3
"""
Incoming: index, corpus, queries --- {pt.Index, Dict, Dict}
Processing: retrieval --- {6 jobs: BM25, TCT, Splade, BGE, BM25_TCT, BM25_MonoT5}
Outgoing: TREC run files --- {.res files}

Step 2: Run Retrievers
----------------------
Runs retrievers sequentially with memory management and checkpointing.

Memory Optimizations (baked in via config):
- Java heap limited (config.processing.java)
- MPS cache cleanup between batches
- Aggressive garbage collection
- Lazy corpus loading where possible
- Checkpoint recovery for crash resilience

Usage:
    python scripts/02_retrieve.py --corpus_path /data/beir/datasets/nq
    python scripts/02_retrieve.py --corpus_path /data/beir/datasets/nq --retrievers BM25
    python scripts/02_retrieve.py --corpus_path /data/beir/datasets/hotpotqa --retrievers BM25_MonoT5

Background execution (survives terminal close):
    nohup python scripts/02_retrieve.py --corpus_path /data/beir/datasets/hotpotqa --retrievers BM25_MonoT5 > logs/retrieve.log 2>&1 &
"""

import os
import sys

# CRITICAL: Set OMP env vars and import faiss BEFORE any torch/transformers imports
# PyTorch and FAISS have OpenMP conflicts on Apple Silicon that cause segfaults
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')  # Single thread avoids OMP conflicts
import faiss  # Must be imported before torch/transformers

import gc
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config AFTER faiss - config triggers transformers which imports torch
from src.config import config, detect_dataset
from src.data_utils import load_corpus as _load_corpus, load_queries as _load_queries


def load_corpus(corpus_path: str, limit: int = None) -> Dict[str, Dict[str, str]]:
    """Load BEIR corpus (eager mode for retrieval)."""
    print(f"[02_retrieve] Loading corpus...")
    corpus = _load_corpus(corpus_path, lazy=False, limit=limit)
    print(f"[02_retrieve] Corpus: {len(corpus)} documents")
    return corpus


def load_queries(corpus_path: str, split: str = "test") -> Dict[str, str]:
    """Load BEIR queries, filtered by split."""
    queries = _load_queries(corpus_path, split=split)
    print(f"[02_retrieve] Loaded {len(queries)} {split} queries")
    return queries


def write_run(results: Dict, output_path: str, retriever_name: str, normalize: bool = False):
    """Write results to TREC run file."""
    from src.retrievers.base import BaseRetriever
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for qid in sorted(results.keys(), key=lambda x: int(x.replace("test", "")) if x.startswith("test") else x):
            result = results[qid]
            docs = result.results
            
            if normalize:
                docs = BaseRetriever.normalize_scores(docs)
            
            for docno, score, rank in docs:
                f.write(f"{qid} Q0 {docno} {rank} {score:.6f} {retriever_name}\n")
    
    print(f"[02_retrieve] Wrote {output_path}")


def clear_memory():
    """Force garbage collection and clear caches."""
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass


def run_bm25(index_path: str, queries: Dict[str, str], runs_dir: Path, top_k: int):
    """Run BM25 retriever."""
    from src.retrievers import BM25Retriever
    
    print(f"\n[02_retrieve] === BM25 ===")
    start = time.time()
    
    retriever = BM25Retriever(index_path)
    results = retriever.retrieve_batch(queries, top_k=top_k)
    
    write_run(results, str(runs_dir / "BM25.res"), "BM25", normalize=False)
    write_run(results, str(runs_dir / "BM25.norm.res"), "BM25", normalize=True)
    
    print(f"[02_retrieve] BM25 completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_tct_colbert(corpus: Dict, queries: Dict[str, str], runs_dir: Path, cache_dir: Path, top_k: int):
    """Run TCT-ColBERT retriever with checkpointing."""
    from src.retrievers import TCTColBERTRetriever
    
    print(f"\n[02_retrieve] === TCT-ColBERT ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "TCT-ColBERT.checkpoint.jsonl"
    
    retriever = TCTColBERTRetriever(
        corpus,
        cache_dir=str(cache_dir),
        batch_size=config.processing.batch_sizes.tct_encoding,
        use_fp16=True
    )
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=config.processing.batch_sizes.tct_mini
    )
    
    write_run(results, str(runs_dir / "TCT-ColBERT.res"), "TCT-ColBERT", normalize=False)
    write_run(results, str(runs_dir / "TCT-ColBERT.norm.res"), "TCT-ColBERT", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] TCT-ColBERT completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_splade(queries: Dict[str, str], runs_dir: Path, top_k: int, dataset: str = "nq"):
    """Run SPLADE retriever with checkpointing."""
    from src.retrievers import SpladeRetriever
    
    print(f"\n[02_retrieve] === SPLADE (Pyserini Pre-built) ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "Splade.checkpoint.jsonl"
    index_name = config.get_index_name("splade", dataset)
    
    retriever = SpladeRetriever(
        index_name=index_name, 
        threads=config.processing.threads.faiss
    )
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=config.processing.batch_sizes.splade_mini
    )
    
    write_run(results, str(runs_dir / "Splade.res"), "Splade", normalize=False)
    write_run(results, str(runs_dir / "Splade.norm.res"), "Splade", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] SPLADE completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_bge(queries: Dict[str, str], runs_dir: Path, top_k: int, dataset: str = "nq"):
    """Run BGE retriever with checkpointing."""
    from src.retrievers import BGERetriever
    
    print(f"\n[02_retrieve] === BGE (Pyserini Pre-built FAISS) ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "BGE.checkpoint.jsonl"
    index_name = config.get_index_name("bge", dataset)
    
    retriever = BGERetriever(index_name=index_name, use_mps=True)
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=config.processing.batch_sizes.bge_mini
    )
    
    write_run(results, str(runs_dir / "BGE.res"), "BGE", normalize=False)
    write_run(results, str(runs_dir / "BGE.norm.res"), "BGE", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] BGE completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_bm25_tct(index_path: str, corpus_path: str, queries: Dict[str, str], runs_dir: Path, top_k: int):
    """Run BM25>>TCT-ColBERT retriever with lazy corpus loading and checkpointing."""
    from src.retrievers import BM25TCTRetriever
    
    print(f"\n[02_retrieve] === BM25>>TCT-ColBERT ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "BM25_TCT.checkpoint.jsonl"
    
    retriever = BM25TCTRetriever(
        index_path, 
        corpus_path, 
        first_stage_k=config.processing.retrieval.first_stage_k,
        tct_batch_size=config.processing.batch_sizes.tct_encoding
    )
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=config.processing.batch_sizes.bm25_tct_mini
    )
    
    write_run(results, str(runs_dir / "BM25_TCT.res"), "BM25_TCT", normalize=False)
    write_run(results, str(runs_dir / "BM25_TCT.norm.res"), "BM25_TCT", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] BM25>>TCT completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def run_bm25_monot5(index_path: str, corpus_path: str, queries: Dict[str, str], runs_dir: Path, top_k: int):
    """Run BM25>>MonoT5 retriever with checkpointing."""
    from src.retrievers import BM25MonoT5Retriever
    
    print(f"\n[02_retrieve] === BM25>>MonoT5 ===")
    start = time.time()
    
    checkpoint_path = runs_dir / "BM25_MonoT5.checkpoint.jsonl"
    
    retriever = BM25MonoT5Retriever(
        index_path, 
        corpus_path, 
        first_stage_k=config.processing.retrieval.first_stage_k,
        ce_batch_size=config.processing.batch_sizes.cross_encoder
    )
    results = retriever.retrieve_batch(
        queries, 
        top_k=top_k,
        checkpoint_path=str(checkpoint_path),
        mini_batch_size=config.processing.batch_sizes.bm25_monot5_mini
    )
    
    write_run(results, str(runs_dir / "BM25_MonoT5.res"), "BM25_MonoT5", normalize=False)
    write_run(results, str(runs_dir / "BM25_MonoT5.norm.res"), "BM25_MonoT5", normalize=True)
    
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[02_retrieve] Removed checkpoint file")
    
    print(f"[02_retrieve] BM25>>MonoT5 completed in {time.time() - start:.1f}s")
    
    del retriever, results
    clear_memory()


def main():
    parser = argparse.ArgumentParser(description="Step 2: Run Retrievers")
    parser.add_argument("--corpus_path", required=True, help="Path to BEIR dataset")
    parser.add_argument("--index_path", default=None, help="PyTerrier index path")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--query_file", default=None, help="Custom queries file (BEIR format)")
    parser.add_argument("--query_split", default="test", help="Query split (test/dev/train)")
    parser.add_argument("--retrievers", default="BM25,TCT-ColBERT,BM25_TCT,BM25_MonoT5",
                        help="Comma-separated retriever names (Splade excluded by default)")
    parser.add_argument("--top_k", type=int, default=config.processing.retrieval.top_k,
                        help="Number of docs to retrieve")
    parser.add_argument("--limit", type=int, default=None, help="Limit corpus size")
    parser.add_argument("--splade_max_docs", type=int, default=None, 
                        help="Max docs for SPLADE (memory safety)")
    args = parser.parse_args()
    
    # Detect dataset from corpus path
    dataset = detect_dataset(args.corpus_path)
    
    # Setup paths
    output_dir = Path(args.output_dir) if args.output_dir else config.project_root / "data" / dataset
    runs_dir = output_dir / "runs"
    cache_dir = output_dir / "cache"
    
    # Use absolute path for PyTerrier index to avoid CWD issues
    if args.index_path:
        index_path = str(Path(args.index_path).resolve())
    else:
        index_path = str((output_dir / "index" / "pyterrier").resolve())
    
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Parse retrievers
    retrievers = [r.strip() for r in args.retrievers.split(",")]
    print(f"[02_retrieve] Dataset: {dataset}")
    print(f"[02_retrieve] Retrievers: {retrievers}")
    print(f"[02_retrieve] Cache dir: {cache_dir}")
    
    # Load queries first (small)
    if args.query_file:
        # Load from custom query file
        print(f"[02_retrieve] Loading queries from custom file: {args.query_file}")
        queries = {}
        with open(args.query_file, 'r', encoding='utf-8') as f:
            for line in f:
                q = json.loads(line)
                qid = q.get("_id", "")
                qtext = q.get("text", "")
                if qid and qtext:
                    queries[qid] = qtext
        print(f"[02_retrieve] Loaded {len(queries)} queries from {args.query_file}")
    else:
        # Load from corpus_path with split filter
        queries = load_queries(args.corpus_path, split=args.query_split)
    
    # Check which retrievers need corpus in memory
    # Note: BM25_TCT, BM25_MonoT5, and Splade use lazy/pre-built loading
    needs_corpus = any(r in retrievers for r in ["TCT-ColBERT"])
    
    corpus = None
    if needs_corpus:
        corpus = load_corpus(args.corpus_path, args.limit)
    
    # Run each retriever
    try:
        if "BM25" in retrievers:
            run_bm25(index_path, queries, runs_dir, args.top_k)
        
        if "TCT-ColBERT" in retrievers:
            run_tct_colbert(corpus, queries, runs_dir, cache_dir, args.top_k)
        
        if "Splade" in retrievers:
            run_splade(queries, runs_dir, args.top_k, dataset=dataset)
        
        if "BGE" in retrievers:
            run_bge(queries, runs_dir, args.top_k, dataset=dataset)
        
        if "BM25_TCT" in retrievers:
            run_bm25_tct(index_path, args.corpus_path, queries, runs_dir, args.top_k)
        
        if "BM25_MonoT5" in retrievers:
            run_bm25_monot5(index_path, args.corpus_path, queries, runs_dir, args.top_k)
            
    except Exception as e:
        print(f"[02_retrieve] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"\n=== Step 2 Complete ===")
    print(f"Runs directory: {runs_dir}")
    for f in sorted(runs_dir.glob("*.res")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()

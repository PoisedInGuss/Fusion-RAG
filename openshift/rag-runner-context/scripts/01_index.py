#!/usr/bin/env python3
"""
Incoming: BEIR corpus or existing FAISS index --- {corpus.jsonl, FAISS flat}
Processing: indexing --- {2 jobs: PyTerrier build, HNSW build}
Outgoing: search indexes --- {PyTerrier index, HNSW segments}

Step 1: Build Search Indexes
----------------------------
- PyTerrier: BM25 inverted index from BEIR corpus
- HNSW: Segmented HNSW from existing BGE FAISS (for fast dense search)

Assumes FAISS indexes already exist in cache (from Pyserini).
No automatic downloads - indexes must be pre-cached.

Usage:
    # Build HNSW for fast BGE retrieval
    python scripts/01_index.py --dataset hotpotqa --hnsw
    python scripts/01_index.py --dataset nq --hnsw --segments 2
    
    # Build PyTerrier BM25 index
    python scripts/01_index.py --dataset nq --corpus_path /data/beir/nq --pyterrier
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config


def build_pyterrier_index(corpus_path: str, index_path: str, limit: int = None) -> str:
    """Build PyTerrier BM25 inverted index from BEIR corpus."""
    import pyterrier as pt
    if not pt.started():
        pt.init()
    
    corpus_file = os.path.join(corpus_path, "corpus.jsonl")
    print(f"[Index] Building PyTerrier index from {corpus_file}")
    print(f"[Index] Output: {index_path}")
    os.makedirs(index_path, exist_ok=True)
    
    def stream_corpus():
        count = 0
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                doc = json.loads(line)
                yield {
                    "docno": doc.get("_id", str(count)),
                    "text": (doc.get("title", "") + " " + doc.get("text", "")).strip()
                }
                count += 1
                if count % 500000 == 0:
                    print(f"  Processed {count:,} documents...")
        print(f"[Index] Total: {count:,} documents")
    
    t0 = time.time()
    indexer = pt.IterDictIndexer(index_path, meta={"docno": 100}, verbose=True)
    index_ref = indexer.index(stream_corpus())
    
    index = pt.IndexFactory.of(index_ref)
    stats = index.getCollectionStatistics()
    print(f"[Index] Built in {time.time()-t0:.1f}s: "
          f"{stats.getNumberOfDocuments():,} docs, {stats.getNumberOfTokens():,} tokens")
    
    return str(index_ref)


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Build Search Indexes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build HNSW for BGE dense retrieval
  python scripts/01_index.py --dataset hotpotqa --hnsw
  python scripts/01_index.py --dataset nq --hnsw --segments 2
  
  # Build PyTerrier BM25 index
  python scripts/01_index.py --dataset nq --corpus_path /data/beir/nq --pyterrier
"""
    )
    
    parser.add_argument("--dataset", default="nq", choices=config.datasets.supported,
                        help="Dataset name")
    parser.add_argument("--hnsw", action="store_true",
                        help="Build HNSW segments from existing BGE FAISS")
    parser.add_argument("--segments", type=int, default=None,
                        help="Number of HNSW segments (default: 4 for hotpotqa, 2 for nq)")
    parser.add_argument("--threads", type=int, default=config.indexes.hnsw.num_threads,
                        help="Threads for HNSW build")
    parser.add_argument("--pyterrier", action="store_true",
                        help="Build PyTerrier BM25 index")
    parser.add_argument("--corpus_path", help="Path to BEIR corpus (for PyTerrier)")
    parser.add_argument("--index_path", help="Output path for PyTerrier index")
    parser.add_argument("--limit", type=int, help="Limit corpus size")
    parser.add_argument("--force", action="store_true", help="Force rebuild")
    
    args = parser.parse_args()
    
    print(f"[Index] Dataset: {args.dataset}")
    print(f"[Index] Cache: {config.cache_root}")
    
    results = {}
    
    # PyTerrier BM25
    if args.pyterrier:
        if not args.corpus_path:
            print("[Index] ERROR: --corpus_path required for --pyterrier")
            results["pyterrier"] = "ERROR"
        else:
            output_dir = config.project_root / "data" / args.dataset
            index_path = args.index_path or str(output_dir / "index" / "pyterrier")
            
            if os.path.exists(index_path) and os.listdir(index_path) and not args.force:
                print(f"[Index] PyTerrier already exists: {index_path}")
                results["pyterrier"] = "EXISTS"
            else:
                build_pyterrier_index(args.corpus_path, index_path, args.limit)
                results["pyterrier"] = "BUILT"
    
    # HNSW for BGE
    if args.hnsw:
        from src.indexing import build_hnsw_index
        
        # Check if already exists
        cache_dir = Path(os.environ.get("PYSERINI_CACHE", str(config.cache_root / "pyserini")))
        try:
            bge_hash = config.get_index_hash("bge", args.dataset)
            hnsw_meta = cache_dir / "indexes" / bge_hash / "hnsw_segments_meta.json"
            
            if hnsw_meta.exists() and not args.force:
                print(f"[Index] HNSW already exists")
                results["hnsw"] = "EXISTS"
            else:
                # Determine segments (2 for NQ ~2.7M, 4 for HotpotQA ~5.2M)
                n_segments = args.segments
                if n_segments is None:
                    n_segments = 2 if args.dataset == "nq" else 4
                
                print(f"[Index] Building HNSW: {n_segments} segments, {args.threads} threads")
                build_hnsw_index(
                    dataset=args.dataset,
                    n_segments=n_segments,
                    num_threads=args.threads
                )
                results["hnsw"] = "BUILT"
        except FileNotFoundError as e:
            print(f"[Index] ERROR: FAISS index not found. Download it first using Pyserini.")
            print(f"  {e}")
            results["hnsw"] = "FAILED"
        except Exception as e:
            print(f"[Index] HNSW ERROR: {e}")
            results["hnsw"] = "FAILED"
    
    # Summary
    if results:
        print(f"\n{'='*40}")
        print("Results:")
        for idx, status in results.items():
            symbol = "✅" if status in ["BUILT", "EXISTS"] else "❌"
            print(f"  {symbol} {idx}: {status}")
    else:
        print("\n[Index] Nothing to do. Use --hnsw or --pyterrier")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Incoming: .res run files --- {TREC format}
Processing: QPP computation --- {13 QPP methods per run}
Outgoing: .qpp files --- {per-query QPP scores}

Step 3: Compute QPP Scores
--------------------------
Computes QPP methods for each retriever run.

Usage:
    python scripts/03_qpp.py
    python scripts/03_qpp.py --runs_dir data/nq/runs --qpp_dir data/nq/qpp
"""

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config first - sets up environment
from src.config import config
from src.qpp import compute_qpp_for_res_file


def _process_qpp_file(args_tuple):
    """Process a single QPP file (module-level for pickling)."""
    res_path, qpp_output, top_k, normalize, queries_path, res_file = args_tuple
    print(f"[03_qpp] Processing {res_file}...")
    compute_qpp_for_res_file(res_path, qpp_output, top_k, normalize, queries_path)
    return res_file


def main():
    parser = argparse.ArgumentParser(description="Step 3: Compute QPP Scores")
    parser.add_argument("--runs_dir", default=None, help="Directory with .res files")
    parser.add_argument("--qpp_dir", default=None, help="Output directory for QPP files")
    parser.add_argument("--queries", default=None, help="Path to queries.jsonl (BEIR format)")
    parser.add_argument("--top_k", type=int, default=config.processing.retrieval.top_k, 
                        help="Top-k for QPP computation")
    parser.add_argument("--normalize", default=config.qpp.normalization, 
                        choices=["none", "minmax", "zscore"])
    parser.add_argument("--force", action="store_true", help="Recompute even if QPP file exists")
    args = parser.parse_args()
    
    # Setup paths - detect dataset from runs_dir
    if args.runs_dir:
        runs_dir = Path(args.runs_dir)
        # Infer dataset directory from runs path (e.g., data/hotpotqa/runs -> data/hotpotqa)
        dataset_dir = runs_dir.parent
    else:
        dataset_dir = config.project_root / "data" / "nq"
        runs_dir = dataset_dir / "runs"
    
    qpp_dir = Path(args.qpp_dir) if args.qpp_dir else dataset_dir / "qpp"
    
    # Auto-detect queries.jsonl if not specified
    queries_path = args.queries
    if not queries_path:
        # Try dataset-specific locations (BEIR-<dataset>/queries.jsonl)
        for subdir in dataset_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("BEIR"):
                candidate = subdir / "queries.jsonl"
                if candidate.exists():
                    queries_path = str(candidate)
                    break
        # Fallback to direct queries.jsonl
        if not queries_path:
            candidate = dataset_dir / "queries.jsonl"
            if candidate.exists():
                queries_path = str(candidate)
    
    if queries_path:
        print(f"[03_qpp] Using query texts from: {queries_path}")
    else:
        print(f"[03_qpp] WARNING: No queries.jsonl found - IDF-based methods will be inaccurate")
    
    os.makedirs(qpp_dir, exist_ok=True)
    
    # Find .res files (not .norm.res)
    res_files = [f for f in os.listdir(runs_dir) if f.endswith(".res") and not f.endswith(".norm.res")]
    
    if not res_files:
        print(f"[03_qpp] No .res files found in {runs_dir}")
        return
    
    import multiprocessing
    n_workers = min(8, multiprocessing.cpu_count())
    
    # Get QPP method names from config
    qpp_methods = config.qpp.methods
    
    print(f"[03_qpp] Processing {len(res_files)} run files")
    print(f"[03_qpp] QPP methods: {config.qpp.n_methods} ({', '.join(qpp_methods[:5])}...)")
    print(f"[03_qpp] Parallel processing with {n_workers} workers (of {multiprocessing.cpu_count()} cores)")
    
    # Prepare jobs
    jobs = []
    for res_file in res_files:
        res_path = runs_dir / res_file
        qpp_output = qpp_dir / res_file.replace(".res", ".res.mmnorm.qpp")
        
        if qpp_output.exists() and not args.force:
            print(f"[03_qpp] SKIP {res_file} (already exists)")
            continue
        
        jobs.append((str(res_path), str(qpp_output), args.top_k, args.normalize, queries_path, res_file))
    
    # Process files in parallel
    processed = 0
    skipped = len(res_files) - len(jobs)
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_qpp_file, job): job for job in jobs}
        
        for future in as_completed(futures):
            job = futures[future]
            try:
                res_file = future.result()
                print(f"[03_qpp] ✓ Completed {res_file}")
                processed += 1
            except Exception as e:
                print(f"[03_qpp] ✗ Failed {job[5]}: {e}")
                import traceback
                traceback.print_exc()
    
    print(f"\n[03_qpp] Processed: {processed}, Skipped: {skipped}")
    
    print(f"\n=== Step 3 Complete ===")
    print(f"QPP files: {qpp_dir}")
    print(f"Files: {list(qpp_dir.glob('*.qpp'))}")


if __name__ == "__main__":
    main()

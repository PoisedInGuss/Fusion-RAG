#!/usr/bin/env python3
"""
Incoming: fused .res files, qrels --- {TREC runs, relevance judgments}
Processing: IR evaluation --- {ir_measures: nDCG, MRR, Recall}
Outgoing: comparison results --- {JSON + table}

Step 6: Evaluate Fusion (IR Metrics)
------------------------------------
Evaluates existing fusion result files using ir_measures.
Does NOT regenerate fusion results — that is 05_fusion.py's job.

STRICT: Read-only evaluation. No fusion computation.

Usage:
    python scripts/06_eval_fusion.py
    python scripts/06_eval_fusion.py --dataset hotpotqa
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config first
from src.config import config

from src.evaluation import IREvaluator
from src.data_utils import load_qrels as _load_qrels, load_run_file as _load_run


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels from TSV file."""
    return _load_qrels(qrels_path)


def load_run_file(run_path: Path) -> Dict[str, List[Tuple[str, float]]]:
    """Load TREC-format run file, returns {qid: [(docid, score), ...]}."""
    run_data = _load_run(run_path)
    # Convert from (docid, score, rank) to (docid, score), sorted by score desc
    return {
        qid: sorted([(d, s) for d, s, r in docs], key=lambda x: -x[1])
        for qid, docs in run_data.items()
    }


def evaluate_fused_runs(
    fused_dir: Path,
    qrels_path: Path,
    output_dir: Path
):
    """Evaluate all fused run files in directory."""
    print("=" * 70)
    print("FUSION EVALUATION (ir_measures)")
    print("=" * 70)
    
    # Load qrels
    qrels = load_qrels(qrels_path)
    print(f"Qrels: {len(qrels)} queries")
    
    # Find all .res files
    res_files = sorted(fused_dir.glob("*.res"))
    if not res_files:
        print(f"No .res files found in {fused_dir}")
        return []
    
    print(f"Found {len(res_files)} fused run files")
    
    # Initialize evaluator with metrics from config
    evaluator = IREvaluator(metrics=config.evaluation.ir_metrics[:6])  # First 6 metrics
    
    results = []
    
    for res_file in res_files:
        method_name = res_file.stem  # filename without extension
        print(f"  Evaluating: {method_name}")
        
        # Load run
        run = load_run_file(res_file)
        
        # Determine method type from filename
        if method_name.startswith("learned"):
            method_type = "learned"
        elif method_name.startswith("w"):
            method_type = "qpp-weighted"
        else:
            method_type = "unweighted"
        
        # Evaluate
        metrics = evaluator.evaluate(run, qrels, per_query=False)
        metrics['n_queries'] = len([q for q in run.keys() if q in qrels])
        
        results.append({
            'method': method_name,
            'type': method_type,
            **metrics
        })
    
    # === Results Table ===
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Sort by NDCG@10
    results.sort(key=lambda x: -x.get('nDCG@10', 0))
    
    # Get baseline for improvement calculation (first unweighted method or lowest)
    baseline_ndcg = next(
        (r.get('nDCG@10', 0) for r in results if r['type'] == 'unweighted'),
        results[-1].get('nDCG@10', 0) if results else 0
    )
    
    print(f"\n{'Method':<25} {'Type':<14} {'NDCG@5':<8} {'NDCG@10':<8} {'MRR@5':<8} {'MRR@10':<8} {'R@5':<8} {'R@10':<8} {'Δ':<8}")
    print("-" * 110)
    
    for r in results:
        ndcg5 = r.get('nDCG@5', 0)
        ndcg10 = r.get('nDCG@10', 0)
        mrr5 = r.get('RR@5', 0)
        mrr10 = r.get('RR@10', 0)
        r5 = r.get('R@5', 0)
        r10 = r.get('R@10', 0)
        improvement = (ndcg10 - baseline_ndcg) / baseline_ndcg * 100 if baseline_ndcg > 0 else 0
        sign = "+" if improvement >= 0 else ""
        print(f"{r['method']:<25} {r['type']:<14} {ndcg5:.4f}   {ndcg10:.4f}   {mrr5:.4f}   {mrr10:.4f}   {r5:.4f}   {r10:.4f}   {sign}{improvement:.1f}%")
    
    print("-" * 110)
    
    # Save results
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Evaluate Fusion Results (ir_measures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluates existing fused run files. Does NOT regenerate fusion results.

Expects .res files in: <data_dir>/fused/
Expects qrels at: <corpus_path>/qrels/test.tsv

Examples:
    python scripts/06_eval_fusion.py
    python scripts/06_eval_fusion.py --dataset hotpotqa
    python scripts/06_eval_fusion.py --corpus_path /path/to/beir/nq
"""
    )
    parser.add_argument("--dataset", default="nq", choices=config.datasets.supported,
                        help="Dataset name")
    parser.add_argument("--data_dir", default=None, help="Data directory (default: data/<dataset>)")
    parser.add_argument("--corpus_path", default=None, help="Path to BEIR dataset for qrels")
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(args.data_dir) if args.data_dir else config.project_root / "data" / args.dataset
    fused_dir = data_dir / "fused"
    
    if not fused_dir.exists():
        print(f"Error: Fused directory not found: {fused_dir}")
        print("Run 05_fusion.py first to generate fusion results.")
        sys.exit(1)
    
    # Qrels path - use config
    if args.corpus_path:
        qrels_path = Path(args.corpus_path) / "qrels" / "test.tsv"
    else:
        qrels_path = config.get_qrels_path(args.dataset)
    
    if not qrels_path.exists():
        print(f"Error: Qrels not found: {qrels_path}")
        print("Specify --corpus_path to point to BEIR dataset.")
        sys.exit(1)
    
    print(f"[06_eval] Dataset: {args.dataset}")
    print(f"[06_eval] Qrels: {qrels_path}")
    
    evaluate_fused_runs(fused_dir, qrels_path, fused_dir)


if __name__ == "__main__":
    main()

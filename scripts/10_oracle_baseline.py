#!/usr/bin/env python3
"""
Incoming: ranker .res files, qrels --- {TREC runs, relevance}
Processing: oracle fusion (best ranker per query) --- {selection, evaluation}
Outgoing: oracle baseline results --- {metrics, analysis}

Step 10: Oracle Baseline Evaluation
-----------------------------------
Computes the oracle upper bound for fusion: for each query, select the 
ranker that achieves the highest effectiveness (nDCG@k, AP, etc.).

This establishes the theoretical maximum performance achievable by 
perfect ranker selection, serving as an upper bound for QPP-weighted fusion.

Research Purpose:
- Oracle = upper bound for adaptive fusion
- Gap between Oracle and best single ranker = potential improvement
- Gap between Oracle and our fusion = room for improvement

Reference: QPP-Fusion paper Section 2 - Oracle setup for evaluation.

Usage:
    python scripts/10_oracle_baseline.py --dataset nq --metric ndcg@10
    python scripts/10_oracle_baseline.py --dataset nq --metric map
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.evaluation import IREvaluator
from src.data_utils import load_qrels as _load_qrels, load_run_file as _load_run

# ir_measures for metric computation
import ir_measures
from ir_measures import nDCG, RR, R, AP


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels from TSV file."""
    return _load_qrels(qrels_path)


def load_run_file(run_path: Path) -> Dict[str, List[Tuple[str, float, int]]]:
    """Load TREC-format run file, returns {qid: [(docid, score, rank), ...]}."""
    return _load_run(run_path)


def load_all_runs(runs_dir: Path, use_normalized: bool = False) -> Dict[str, Dict[str, List[Tuple[str, float]]]]:
    """
    Load all ranker runs from directory.
    
    Args:
        runs_dir: Directory containing .res files
        use_normalized: Whether to use .norm.res files
        
    Returns:
        {ranker_name: {qid: [(docid, score), ...]}}
    """
    runs = {}
    
    suffix = ".norm.res" if use_normalized else ".res"
    res_files = sorted([f for f in runs_dir.glob(f"*{suffix}") if (use_normalized or '.norm.' not in f.name)])
    
    for res_file in res_files:
        if use_normalized:
            ranker_name = res_file.name.replace(".norm.res", "")
        else:
            ranker_name = res_file.stem
            
        run_data = load_run_file(res_file)
        
        # Convert to (docid, score) format
        runs[ranker_name] = {
            qid: [(d, s) for d, s, r in sorted(docs, key=lambda x: -x[1])]
            for qid, docs in run_data.items()
        }
    
    return runs


def parse_metric(metric_str: str):
    """Parse metric string to ir_measures metric object."""
    metric_str = metric_str.lower().strip()
    
    if 'ndcg' in metric_str:
        if '@' in metric_str:
            k = int(metric_str.split('@')[1])
            return nDCG @ k
        return nDCG @ 10
    elif metric_str in ['map', 'ap']:
        return AP
    elif 'rr' in metric_str or 'mrr' in metric_str:
        if '@' in metric_str:
            k = int(metric_str.split('@')[1])
            return RR @ k
        return RR @ 10
    elif 'recall' in metric_str or metric_str.startswith('r@'):
        if '@' in metric_str:
            k = int(metric_str.split('@')[1])
            return R @ k
        return R @ 10
    else:
        raise ValueError(f"Unknown metric: {metric_str}")


def compute_per_query_scores(
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
    qrels: Dict[str, Dict[str, int]],
    metric
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-query scores for each ranker.
    
    Args:
        runs: {ranker: {qid: [(docid, score), ...]}}
        qrels: {qid: {docid: rel}}
        metric: ir_measures metric
        
    Returns:
        {ranker: {qid: score}}
    """
    per_query_scores = {}
    
    for ranker, run in runs.items():
        # Convert to ir_measures format
        ir_run = {qid: {d: s for d, s in docs} for qid, docs in run.items()}
        ir_qrels = {qid: {d: r for d, r in rels.items()} for qid, rels in qrels.items()}
        
        # Compute per-query using iter_calc properly
        scores = {}
        for result in ir_measures.iter_calc([metric], ir_qrels, ir_run):
            # result is a tuple: (metric, qid_scores_dict)
            qid = result.query_id
            score = result.value
            scores[qid] = float(score)
        
        per_query_scores[ranker] = scores
    
    return per_query_scores


def compute_oracle_run(
    runs: Dict[str, Dict[str, List[Tuple[str, float]]]],
    qrels: Dict[str, Dict[str, int]],
    metric
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, str]]:
    """
    Build oracle adaptive run: for each query, select the ranker 
    that achieves the highest effectiveness.
    
    Args:
        runs: {ranker: {qid: [(docid, score), ...]}}
        qrels: {qid: {docid: rel}}
        metric: ir_measures metric
        
    Returns:
        oracle_run: {qid: [(docid, score), ...]}
        selections: {qid: selected_ranker}
    """
    # Compute per-query scores for all rankers
    per_query_scores = compute_per_query_scores(runs, qrels, metric)
    
    # Get all query IDs that have relevance judgments
    all_qids = set(qrels.keys())
    
    # Also include qids from runs
    for ranker_run in runs.values():
        all_qids.update(ranker_run.keys())
    
    # Filter to qids with qrels
    all_qids = all_qids & set(qrels.keys())
    
    oracle_run = {}
    selections = {}
    
    for qid in sorted(all_qids):
        best_score = -1
        best_ranker = None
        
        for ranker in runs.keys():
            if qid in per_query_scores[ranker]:
                score = per_query_scores[ranker][qid]
                if score > best_score:
                    best_score = score
                    best_ranker = ranker
        
        if best_ranker and qid in runs[best_ranker]:
            oracle_run[qid] = runs[best_ranker][qid]
            selections[qid] = best_ranker
    
    return oracle_run, selections


def evaluate_run(
    run: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    metrics: List[str]
) -> Dict[str, float]:
    """Evaluate a run on multiple metrics."""
    ir_run = {qid: {d: s for d, s in docs} for qid, docs in run.items()}
    ir_qrels = {qid: {d: r for d, r in rels.items()} for qid, rels in qrels.items()}
    
    parsed_metrics = [ir_measures.parse_measure(m) for m in metrics]
    results = ir_measures.calc_aggregate(parsed_metrics, ir_qrels, ir_run)
    
    return {str(k): float(v) for k, v in results.items()}


def analyze_oracle_selections(
    selections: Dict[str, str],
    per_query_scores: Dict[str, Dict[str, float]]
) -> Dict[str, any]:
    """
    Analyze oracle selection patterns.
    
    Args:
        selections: {qid: selected_ranker}
        per_query_scores: {ranker: {qid: score}}
        
    Returns:
        Analysis dict with statistics
    """
    # Count selections per ranker
    selection_counts = defaultdict(int)
    for ranker in selections.values():
        selection_counts[ranker] += 1
    
    total = len(selections)
    selection_pct = {r: count / total * 100 for r, count in selection_counts.items()}
    
    # Compute score improvements
    improvements = []
    for qid, best_ranker in selections.items():
        best_score = per_query_scores[best_ranker].get(qid, 0)
        other_scores = [
            per_query_scores[r].get(qid, 0) 
            for r in per_query_scores.keys() 
            if r != best_ranker
        ]
        if other_scores:
            avg_others = np.mean(other_scores)
            if avg_others > 0:
                improvements.append((best_score - avg_others) / avg_others * 100)
    
    return {
        'selection_counts': dict(selection_counts),
        'selection_percentages': selection_pct,
        'n_queries': total,
        'avg_improvement_over_others': np.mean(improvements) if improvements else 0,
        'max_improvement': max(improvements) if improvements else 0,
    }


def print_results(
    oracle_metrics: Dict[str, float],
    individual_metrics: Dict[str, Dict[str, float]],
    analysis: Dict[str, any],
    metric_name: str
):
    """Print results in research format."""
    print("\n" + "=" * 80)
    print("ORACLE BASELINE EVALUATION")
    print("=" * 80)
    
    # Oracle performance
    print("\n" + "-" * 80)
    print("Oracle Upper Bound (Best Ranker Per Query)")
    print("-" * 80)
    
    for metric, value in sorted(oracle_metrics.items()):
        print(f"  {metric:<15}: {value:.4f}")
    
    # Individual ranker performance
    print("\n" + "-" * 80)
    print("Individual Ranker Performance")
    print("-" * 80)
    
    print(f"  {'Ranker':<20} {'nDCG@10':<12} {'MRR@10':<12} {'R@10':<12}")
    print("  " + "-" * 56)
    
    sorted_rankers = sorted(
        individual_metrics.items(), 
        key=lambda x: -x[1].get('nDCG@10', 0)
    )
    
    for ranker, metrics in sorted_rankers:
        print(f"  {ranker:<20} {metrics.get('nDCG@10', 0):<12.4f} "
              f"{metrics.get('RR@10', 0):<12.4f} {metrics.get('R@10', 0):<12.4f}")
    
    # Oracle improvement over best individual
    best_individual = sorted_rankers[0]
    oracle_ndcg = oracle_metrics.get('nDCG@10', 0)
    best_ndcg = best_individual[1].get('nDCG@10', 0)
    improvement = (oracle_ndcg - best_ndcg) / best_ndcg * 100 if best_ndcg > 0 else 0
    
    print("\n" + "-" * 80)
    print("Analysis")
    print("-" * 80)
    
    print(f"\n  Best single ranker: {best_individual[0]} (nDCG@10 = {best_ndcg:.4f})")
    print(f"  Oracle upper bound: nDCG@10 = {oracle_ndcg:.4f}")
    print(f"  Oracle improvement over best: +{improvement:.2f}%")
    print(f"  → This is the maximum achievable by perfect ranker selection")
    
    # Selection analysis
    print("\n  Oracle Selection Distribution:")
    for ranker, pct in sorted(analysis['selection_percentages'].items(), key=lambda x: -x[1]):
        count = analysis['selection_counts'][ranker]
        print(f"    {ranker:<20}: {count:>5} queries ({pct:.1f}%)")
    
    print(f"\n  Avg improvement over other rankers: +{analysis['avg_improvement_over_others']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Step 10: Oracle Baseline Evaluation"
    )
    parser.add_argument("--dataset", default="nq", choices=config.datasets.supported)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--corpus_path", default=None)
    parser.add_argument("--metric", default="ndcg@10", help="Selection metric (ndcg@10, map, mrr@10)")
    parser.add_argument("--use_normalized", action="store_true", help="Use normalized .norm.res files")
    parser.add_argument("--output_json", default=None, help="Output JSON path")
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(args.data_dir) if args.data_dir else config.project_root / "data" / args.dataset
    runs_dir = data_dir / "runs"
    
    if not runs_dir.exists():
        print(f"Error: Runs directory not found: {runs_dir}")
        sys.exit(1)
    
    # Qrels
    if args.corpus_path:
        qrels_path = Path(args.corpus_path) / "qrels" / "test.tsv"
    else:
        qrels_path = config.get_qrels_path(args.dataset)
    
    if not qrels_path.exists():
        print(f"Error: Qrels not found: {qrels_path}")
        sys.exit(1)
    
    print(f"[10_oracle] Dataset: {args.dataset}")
    print(f"[10_oracle] Selection metric: {args.metric}")
    print(f"[10_oracle] Runs directory: {runs_dir}")
    
    # Load data
    qrels = load_qrels(qrels_path)
    print(f"[10_oracle] Loaded {len(qrels)} queries with qrels")
    
    runs = load_all_runs(runs_dir, use_normalized=args.use_normalized)
    print(f"[10_oracle] Loaded {len(runs)} rankers: {list(runs.keys())}")
    
    # Parse selection metric
    selection_metric = parse_metric(args.metric)
    print(f"[10_oracle] Selection metric parsed: {selection_metric}")
    
    # Compute oracle run
    print(f"\n[10_oracle] Computing oracle selections...")
    oracle_run, selections = compute_oracle_run(runs, qrels, selection_metric)
    print(f"[10_oracle] Oracle run: {len(oracle_run)} queries")
    
    # Evaluate oracle
    eval_metrics = ['nDCG@5', 'nDCG@10', 'nDCG@20', 'RR@10', 'R@10', 'R@100', 'AP']
    oracle_metrics = evaluate_run(oracle_run, qrels, eval_metrics)
    
    # Evaluate individual rankers
    individual_metrics = {}
    for ranker, run in runs.items():
        individual_metrics[ranker] = evaluate_run(run, qrels, eval_metrics)
    
    # Analyze selections
    per_query_scores = compute_per_query_scores(runs, qrels, selection_metric)
    analysis = analyze_oracle_selections(selections, per_query_scores)
    
    # Print results
    print_results(oracle_metrics, individual_metrics, analysis, args.metric)
    
    # Save JSON
    if args.output_json:
        output_data = {
            'dataset': args.dataset,
            'selection_metric': args.metric,
            'oracle_metrics': oracle_metrics,
            'individual_ranker_metrics': individual_metrics,
            'selection_analysis': analysis,
            'selections': selections,
        }
        
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n[10_oracle] Results saved to: {output_path}")
    
    print("\n=== Oracle Evaluation Complete ===")


if __name__ == "__main__":
    main()




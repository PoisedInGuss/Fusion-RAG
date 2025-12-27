#!/usr/bin/env python3
"""
Incoming: .qpp files, .norm.res files, qrels --- {QPP scores, retrieval runs, labels}
Processing: model training --- {3 models: PerRetriever, MultiOutput, MLP}
Outgoing: trained models --- {.pkl files}

Step 4: Train Fusion Weight Models
----------------------------------
Trains ML models to predict optimal per-query retriever weights
using QPP features as input.

Models:
  - per_retriever: Separate LightGBM model per retriever
  - multioutput: Single multi-output LightGBM  
  - mlp: Neural network with shared layers

STRICT: Uses ir_measures for NDCG computation. No manual implementations.

Usage:
    python scripts/04_train_fusion.py --model per_retriever
    python scripts/04_train_fusion.py --model multioutput
    python scripts/04_train_fusion.py --model mlp
    python scripts/04_train_fusion.py --model all
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config first
from src.config import config, detect_dataset
from src.models import PerRetrieverLGBM, MultiOutputLGBM, FusionMLP
from src.models.base import build_features
from src.evaluation.ir_evaluator import compute_ndcg
from src.data_utils import load_qpp_scores as _load_qpp, load_qrels as _load_qrels, load_run_as_dict


def load_qpp_scores(qpp_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Load QPP scores: {qid: {retriever: [n_qpp scores]}}"""
    qpp_data = _load_qpp(qpp_dir)
    print(f"Loaded QPP for {len(qpp_data)} queries")
    return qpp_data


def load_runs(runs_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load run files: {qid: {retriever: {docid: score}}}"""
    runs = {}
    
    for run_file in runs_dir.glob("*.norm.res"):
        retriever = run_file.stem.replace(".norm", "")
        run_data = load_run_as_dict(run_file)
        
        for qid, doc_scores in run_data.items():
            if qid not in runs:
                runs[qid] = {}
            runs[qid][retriever] = doc_scores
    
    print(f"Loaded runs for {len(runs)} queries")
    return runs


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """Load qrels: {qid: {docid: relevance}}"""
    qrels = _load_qrels(qrels_path)
    print(f"Loaded qrels for {len(qrels)} queries")
    return qrels


def compute_targets(
    qids: List[str],
    runs: Dict,
    qrels: Dict,
    retrievers: List[str]
) -> np.ndarray:
    """
    Compute target weights based on per-retriever NDCG (via ir_measures).
    
    Returns:
        Y: (n_queries, n_retrievers) target weights, normalized to sum to 1
    """
    Y = np.zeros((len(qids), len(retrievers)))
    
    for i, qid in enumerate(qids):
        if qid not in qrels:
            continue
        
        for j, retriever in enumerate(retrievers):
            if qid in runs and retriever in runs[qid]:
                docs = sorted(runs[qid][retriever].items(), key=lambda x: -x[1])
                ranked_docs = [d[0] for d in docs]
                Y[i, j] = compute_ndcg(ranked_docs, qrels[qid], k=10)
    
    # Normalize
    Y_sum = Y.sum(axis=1, keepdims=True)
    Y_sum[Y_sum == 0] = 1
    Y = Y / Y_sum
    
    return Y


def evaluate_model(
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    qids_test: List[str],
    runs: Dict,
    qrels: Dict,
    retrievers: List[str]
) -> Dict[str, float]:
    """Evaluate model on test set using ir_measures."""
    pred_weights = model.predict(X_test)
    
    # Compare strategies
    uniform_weights = np.ones(len(retrievers)) / len(retrievers)
    
    ndcg_uniform = []
    ndcg_learned = []
    ndcg_oracle = []
    
    for i, qid in enumerate(qids_test):
        if qid not in qrels or qid not in runs:
            continue
        
        # Collect all docs
        all_docs = set()
        for retriever in retrievers:
            if retriever in runs.get(qid, {}):
                all_docs.update(runs[qid][retriever].keys())
        
        for weights, scores_list in [
            (uniform_weights, ndcg_uniform),
            (pred_weights[i], ndcg_learned),
            (Y_test[i], ndcg_oracle)
        ]:
            fused = {}
            for docid in all_docs:
                score = sum(
                    weights[j] * runs[qid].get(r, {}).get(docid, 0)
                    for j, r in enumerate(retrievers)
                )
                fused[docid] = score
            
            ranked = [d for d, _ in sorted(fused.items(), key=lambda x: -x[1])]
            scores_list.append(compute_ndcg(ranked, qrels[qid], k=10))
    
    return {
        'uniform': np.mean(ndcg_uniform),
        'learned': np.mean(ndcg_learned),
        'oracle': np.mean(ndcg_oracle)
    }


def train_and_evaluate(
    model_type: str,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    qids_test: List[str],
    runs: Dict,
    qrels: Dict,
    retrievers: List[str],
    output_dir: Path
) -> Dict:
    """Train model and evaluate."""
    
    # Create model
    if model_type == "per_retriever":
        model = PerRetrieverLGBM(retrievers)
    elif model_type == "multioutput":
        model = MultiOutputLGBM(retrievers)
    elif model_type == "mlp":
        model = FusionMLP(retrievers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train - use config defaults
    if model_type == "mlp":
        model.train(X_train, Y_train, X_test, Y_test)
    else:
        model.train(X_train, Y_train, X_test, Y_test)
    
    # Evaluate
    results = evaluate_model(model, X_test, Y_test, qids_test, runs, qrels, retrievers)
    
    # Save model
    model_path = output_dir / f"fusion_{model_type}.pkl"
    model.save(str(model_path))
    
    return {
        'model_type': model_type,
        'model_path': str(model_path),
        **results
    }


def main():
    parser = argparse.ArgumentParser(description="Train Fusion Weight Models")
    parser.add_argument("--model", default="all",
                        choices=["per_retriever", "multioutput", "mlp", "all"],
                        help="Model type to train")
    parser.add_argument("--data_dir", default=None, help="Data directory")
    parser.add_argument("--dataset", default="nq", choices=config.datasets.supported,
                        help="Dataset name (for qrels path)")
    parser.add_argument("--corpus_path", default=None, help="Path to BEIR dataset")
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(args.data_dir) if args.data_dir else config.project_root / "data" / args.dataset
    runs_dir = data_dir / "runs"
    qpp_dir = data_dir / "qpp"
    output_dir = data_dir / "models"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Qrels path - use config
    if args.corpus_path:
        qrels_path = Path(args.corpus_path) / "qrels" / "test.tsv"
    else:
        qrels_path = config.get_qrels_path(args.dataset)
    
    print(f"[04_train] Dataset: {args.dataset}")
    print(f"[04_train] Qrels: {qrels_path}")
    
    # Load data
    qpp_data = load_qpp_scores(qpp_dir)
    runs = load_runs(runs_dir)
    qrels = load_qrels(qrels_path)
    
    # Detect retrievers
    retrievers = sorted(set(r for qid_runs in runs.values() for r in qid_runs.keys()))
    print(f"Retrievers: {retrievers}")
    
    # Build features and targets
    X, qids = build_features(qpp_data, retrievers)
    Y = compute_targets(qids, runs, qrels, retrievers)
    
    # Train/test split - use config
    train_ratio = config.training.train_ratio
    n_train = int(train_ratio * len(qids))
    X_train, X_test = X[:n_train], X[n_train:]
    Y_train, Y_test = Y[:n_train], Y[n_train:]
    qids_test = qids[n_train:]
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    # Train models
    models_to_train = ["per_retriever", "multioutput", "mlp"] if args.model == "all" else [args.model]
    
    results = []
    for model_type in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training: {model_type}")
        print('='*50)
        
        result = train_and_evaluate(
            model_type, X_train, Y_train, X_test, Y_test,
            qids_test, runs, qrels, retrievers, output_dir
        )
        results.append(result)
    
    # Print comparison
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Model':<20} {'Uniform':<12} {'Learned':<12} {'Oracle':<12} {'Improvement':<12}")
    print("-"*60)
    
    for r in results:
        improvement = (r['learned'] - r['uniform']) / r['uniform'] * 100
        print(f"{r['model_type']:<20} {r['uniform']:.4f}       {r['learned']:.4f}       {r['oracle']:.4f}       +{improvement:.1f}%")
    
    print("-"*60)
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    main()

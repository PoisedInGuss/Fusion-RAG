#!/usr/bin/env python3
"""
Incoming: TREC run files, QPP scores --- {.res files, .qpp files}
Processing: rank fusion --- {ranx library + QPP-weighted variants}
Outgoing: fused run --- {TREC .res file}

Fusion Methods for Multi-Retriever RAG
---------------------------------------
Implements all fusion baselines and QPP-weighted variants using ranx library.

Unweighted (via ranx):
- CombSUM: Sigma S_i(d,q)
- CombMNZ: |{i: d in R_i}| x Sigma S_i(d,q)
- RRF: Sigma 1/(k + rank_i(d,q))

QPP-Weighted (custom):
- W-CombSUM: Sigma w_i(q) x S_i(d,q)
- W-CombMNZ: |{i}| x Sigma w_i(q) x S_i(d,q)
- W-RRF: Sigma w_i(q) / (k + rank_i(d,q))

Learned:
- Learned fusion with ML model weights

STRICT: No fallbacks. Missing dependencies raise ImportError immediately.
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# STRICT: Fail immediately if ranx not available
from ranx import Run, fuse, Qrels

# Import config
from src.config import config


# =============================================================================
# Data Loading
# =============================================================================

def load_runs(res_path: str, use_normalized: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load run files into dict of DataFrames.
    
    Args:
        res_path: Directory with run files
        use_normalized: If True, load .norm.res files; else .res files
    
    Returns:
        {retriever_name: DataFrame with qid, docno, rank, score}
    """
    runs = {}
    suffix = ".norm.res" if use_normalized else ".res"
    
    files = [f for f in os.listdir(res_path) if f.endswith(suffix)]
    
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {res_path}")
    
    for f in files:
        ranker = f.replace(suffix, "")
        df = pd.read_csv(
            os.path.join(res_path, f),
            sep=r"\s+",
            names=["qid", "iter", "docno", "rank", "score", "runid"],
            dtype={"qid": str, "docno": str}
        )
        df["qid"] = df["qid"].astype(str)
        runs[ranker] = df
    
    return runs


def load_runs_as_ranx(res_path: str, use_normalized: bool = True) -> Dict[str, Run]:
    """
    Load run files as ranx Run objects.
    
    Args:
        res_path: Directory with run files
        use_normalized: If True, load .norm.res files
    
    Returns:
        {retriever_name: ranx.Run}
    """
    runs = {}
    suffix = ".norm.res" if use_normalized else ".res"
    
    files = [f for f in os.listdir(res_path) if f.endswith(suffix)]
    
    if not files:
        raise FileNotFoundError(f"No {suffix} files found in {res_path}")
    
    for f in files:
        ranker = f.replace(suffix, "")
        filepath = os.path.join(res_path, f)
        runs[ranker] = Run.from_file(filepath, kind="trec")
    
    return runs


def load_qpp_scores(qpp_path: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load QPP files: {qid: {retriever: [13 qpp_scores]}}
    
    Delegates to centralized data_utils.load_qpp_scores.
    """
    from .data_utils import load_qpp_scores as _load_qpp
    return _load_qpp(qpp_path)


def get_qpp_weight(
    qid: str,
    ranker: str,
    qpp_data: Dict[str, Dict[str, List[float]]],
    qpp_index: int = None,
    fusion_mode: bool = False
) -> float:
    """
    Get QPP weight for (query, ranker) pair.
    
    Args:
        qid: Query ID
        ranker: Retriever name
        qpp_data: QPP scores dict
        qpp_index: Which QPP method to use (0-12), or -1 for fusion. Uses config default if None.
        fusion_mode: If True, average all QPP methods
    
    Returns:
        Weight value (0-1 range after normalization)
        
    Raises:
        KeyError: If QPP data missing for qid or ranker (data integrity issue).
        IndexError: If qpp_index out of range.
    """
    # Use default from config if not specified
    if qpp_index is None:
        qpp_index = config.qpp.default_index
    
    if qid not in qpp_data:
        raise KeyError(f"QPP data missing for query '{qid}'. Run QPP computation first.")
    
    if ranker not in qpp_data[qid]:
        raise KeyError(f"QPP data missing for ranker '{ranker}' on query '{qid}'. Available: {list(qpp_data[qid].keys())}")
    
    scores = qpp_data[qid][ranker]
    
    if fusion_mode or qpp_index == -1:
        return sum(scores) / len(scores)
    
    if qpp_index >= len(scores):
        raise IndexError(f"QPP index {qpp_index} out of range. Available: 0-{len(scores)-1}")
    
    return scores[qpp_index]


def get_qpp_index(model_name: str) -> int:
    """Resolve QPP model name to index. Returns -1 for 'fusion'."""
    return config.get_qpp_index(model_name)


# =============================================================================
# Unweighted Fusion Methods (via ranx)
# =============================================================================

def combsum(runs: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple[str, float]]]:
    """
    CombSUM using ranx library.
    
    Formula: CombSUM(d,q) = Sigma S_i(d,q)
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    print("[fusion] Running CombSUM (ranx)...")
    
    ranx_runs = _df_runs_to_ranx(runs)
    fused_run = fuse(runs=list(ranx_runs.values()), method="sum")
    
    result = _ranx_to_dict(fused_run)
    print(f"[fusion] CombSUM done: {len(result)} queries")
    return result


def combmnz(runs: Dict[str, pd.DataFrame]) -> Dict[str, List[Tuple[str, float]]]:
    """
    CombMNZ using ranx library.
    
    Formula: CombMNZ(d,q) = |{i: d in R_i}| x Sigma S_i(d,q)
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    print("[fusion] Running CombMNZ (ranx)...")
    
    ranx_runs = _df_runs_to_ranx(runs)
    fused_run = fuse(runs=list(ranx_runs.values()), method="mnz")
    
    result = _ranx_to_dict(fused_run)
    print(f"[fusion] CombMNZ done: {len(result)} queries")
    return result


def rrf(runs: Dict[str, pd.DataFrame], k: int = None) -> Dict[str, List[Tuple[str, float]]]:
    """
    Reciprocal Rank Fusion using ranx library.
    
    Formula: RRF(d,q) = Sigma 1/(k + rank_i(d,q))
    
    Args:
        runs: Dict of ranker DataFrames
        k: RRF constant (from config if None)
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    k = k if k is not None else config.fusion.rrf_k
    
    print("[fusion] Running RRF (ranx)...")
    
    ranx_runs = _df_runs_to_ranx(runs)
    fused_run = fuse(runs=list(ranx_runs.values()), method="rrf", params={"k": k})
    
    result = _ranx_to_dict(fused_run)
    print(f"[fusion] RRF done: {len(result)} queries")
    return result


# =============================================================================
# QPP-Weighted Fusion Methods (custom implementation - not in ranx)
# =============================================================================

def weighted_combsum(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, List[float]]],
    qpp_index: int = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    QPP-Weighted CombSUM.
    
    Formula: W-CombSUM(d,q) = Sigma w_i(q) x S_i(d,q)
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP scores dict
        qpp_index: QPP method index (0-12) or -1 for fusion. Uses config default if None.
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    qpp_index = qpp_index if qpp_index is not None else config.qpp.default_index
    
    print(f"[fusion] Running W-CombSUM (QPP index={qpp_index})...")
    
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    fusion_mode = (qpp_index == -1)
    
    # OPTIMIZED: Pre-group all dataframes to avoid O(n^2) filtering
    grouped_runs = {ranker: df.groupby("qid") for ranker, df in runs.items()}
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        
        for ranker, grouped_df in grouped_runs.items():
            weight = get_qpp_weight(qid, ranker, qpp_data, qpp_index, fusion_mode)
            
            if qid in grouped_df.groups:
                sub = grouped_df.get_group(qid)
                for _, row in sub.iterrows():
                    doc_scores[row["docno"]] += weight * row["score"]
        
        for docid, score in doc_scores.items():
            fused[qid].append((docid, score))
    
    print(f"[fusion] W-CombSUM done: {len(fused)} queries")
    return dict(fused)


def weighted_combmnz(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, List[float]]],
    qpp_index: int = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    QPP-Weighted CombMNZ.
    
    Formula: W-CombMNZ(d,q) = |{i}| x Sigma w_i(q) x S_i(d,q)
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP scores dict
        qpp_index: QPP method index (0-12) or -1 for fusion. Uses config default if None.
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    qpp_index = qpp_index if qpp_index is not None else config.qpp.default_index
    
    print(f"[fusion] Running W-CombMNZ (QPP index={qpp_index})...")
    
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    fusion_mode = (qpp_index == -1)
    
    # OPTIMIZED: Pre-group all dataframes to avoid O(n^2) filtering
    grouped_runs = {ranker: df.groupby("qid") for ranker, df in runs.items()}
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        doc_counts = defaultdict(int)
        
        for ranker, grouped_df in grouped_runs.items():
            weight = get_qpp_weight(qid, ranker, qpp_data, qpp_index, fusion_mode)
            
            if qid in grouped_df.groups:
                sub = grouped_df.get_group(qid)
                for _, row in sub.iterrows():
                    doc_scores[row["docno"]] += weight * row["score"]
                    doc_counts[row["docno"]] += 1
        
        for docid, score in doc_scores.items():
            fused[qid].append((docid, score * doc_counts[docid]))
    
    print(f"[fusion] W-CombMNZ done: {len(fused)} queries")
    return dict(fused)


def weighted_rrf(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, List[float]]],
    qpp_index: int = None,
    k: int = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    QPP-Weighted RRF.
    
    Formula: W-RRF(d,q) = Sigma w_i(q) / (k + rank_i(d,q))
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP scores dict
        qpp_index: QPP method index (0-12) or -1 for fusion. Uses config default if None.
        k: RRF constant (from config if None)
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    qpp_index = qpp_index if qpp_index is not None else config.qpp.default_index
    k = k if k is not None else config.fusion.rrf_k
    
    print(f"[fusion] Running W-RRF (QPP index={qpp_index})...")
    
    fused = defaultdict(list)
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    fusion_mode = (qpp_index == -1)
    
    # OPTIMIZED: Pre-group all dataframes to avoid O(n^2) filtering
    grouped_runs = {ranker: df.groupby("qid") for ranker, df in runs.items()}
    
    for qid in all_qids:
        doc_scores = defaultdict(float)
        
        for ranker, grouped_df in grouped_runs.items():
            weight = get_qpp_weight(qid, ranker, qpp_data, qpp_index, fusion_mode)
            
            if qid in grouped_df.groups:
                sub = grouped_df.get_group(qid).sort_values("rank")
                for _, row in sub.iterrows():
                    doc_scores[row["docno"]] += weight / (k + row["rank"])
        
        for docid, score in doc_scores.items():
            fused[qid].append((docid, score))
    
    print(f"[fusion] W-RRF done: {len(fused)} queries")
    return dict(fused)


# =============================================================================
# Learned Fusion (ML Model Weights)
# =============================================================================

def learned_fusion(
    runs: Dict[str, pd.DataFrame],
    qpp_data: Dict[str, Dict[str, List[float]]],
    model_path: str,
    retrievers: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Fusion with learned per-query weights from ML model.
    
    Args:
        runs: Dict of ranker DataFrames
        qpp_data: QPP scores dict
        model_path: Path to trained model pickle
        retrievers: List of retriever names in model order
    
    Returns:
        {qid: [(docid, fused_score), ...]}
    """
    print(f"[fusion] Running learned fusion from {model_path}...")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data.get('model')
    model_type = model_data.get('model_type', 'PerRetrieverLGBM')
    retrievers = retrievers or model_data.get('retrievers', sorted(runs.keys()))
    n_qpp = model_data.get('n_qpp', config.qpp.n_methods)
    
    all_qids = sorted(set.union(*[set(df["qid"].unique()) for df in runs.values()]))
    n_retrievers = len(retrievers)
    
    X = np.zeros((len(all_qids), n_qpp * n_retrievers))
    for i, qid in enumerate(all_qids):
        for j, retriever in enumerate(retrievers):
            if qid in qpp_data and retriever in qpp_data[qid]:
                scores = qpp_data[qid][retriever]
                X[i, j*n_qpp:(j+1)*n_qpp] = scores[:n_qpp]
    
    pred_weights = model.predict(X)
    
    weights_dict = {}
    for i, qid in enumerate(all_qids):
        weights_dict[qid] = {r: w for r, w in zip(retrievers, pred_weights[i])}
    
    all_dfs = []
    for ranker, df in runs.items():
        df_copy = df[["qid", "docno", "score"]].copy()
        df_copy["weighted_score"] = df_copy.apply(
            lambda row: row["score"] * weights_dict.get(row["qid"], {}).get(ranker, 1.0/n_retrievers),
            axis=1
        )
        all_dfs.append(df_copy[["qid", "docno", "weighted_score"]])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    aggregated = combined.groupby(["qid", "docno"])["weighted_score"].sum().reset_index()
    
    fused = defaultdict(list)
    for _, row in aggregated.iterrows():
        fused[row["qid"]].append((row["docno"], row["weighted_score"]))
    
    print(f"[fusion] Learned fusion done: {len(fused)} queries")
    return dict(fused)


# =============================================================================
# Utility Functions
# =============================================================================

def _df_runs_to_ranx(runs: Dict[str, pd.DataFrame]) -> Dict[str, Run]:
    """Convert DataFrame runs to ranx Run objects."""
    ranx_runs = {}
    for name, df in runs.items():
        run_dict = {}
        for _, row in df.iterrows():
            qid = str(row["qid"])
            if qid not in run_dict:
                run_dict[qid] = {}
            run_dict[qid][str(row["docno"])] = float(row["score"])
        ranx_runs[name] = Run(run_dict, name=name)
    return ranx_runs


def _ranx_to_dict(run: Run) -> Dict[str, List[Tuple[str, float]]]:
    """Convert ranx Run to dict format."""
    result = {}
    run_dict = run.to_dict()
    for qid, doc_scores in run_dict.items():
        # Sort by score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])
        result[qid] = [(doc, score) for doc, score in sorted_docs]
    return result


def write_runfile(
    fused: Dict[str, List[Tuple[str, float]]],
    output_path: str,
    tag: str = "fusion"
):
    """Write fused results in TREC format."""
    with open(output_path, "w") as fout:
        for qid in sorted(fused.keys(), key=lambda x: int(x.replace("test", "")) if x.startswith("test") else x):
            ranked = sorted(fused[qid], key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                fout.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")
    
    print(f"Wrote fused run to {output_path}")


def run_fusion(
    method: str,
    runs_dir: str,
    qpp_dir: Optional[str] = None,
    qpp_model: str = None,
    model_path: Optional[str] = None,
    output_path: Optional[str] = None,
    rrf_k: int = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Run specified fusion method.
    
    Args:
        method: One of combsum, combmnz, rrf, wcombsum, wcombmnz, wrrf, learned
        runs_dir: Directory with run files
        qpp_dir: Directory with QPP files (required for weighted methods)
        qpp_model: QPP model name for weighting (from config if None)
        model_path: Path to trained model (for learned fusion)
        output_path: Output file path (optional)
        rrf_k: RRF constant (from config if None)
    
    Returns:
        Fused results dict
    """
    qpp_model = qpp_model or config.qpp.default_method
    rrf_k = rrf_k if rrf_k is not None else config.fusion.rrf_k
    
    runs = load_runs(runs_dir, use_normalized=True)
    print(f"Loaded {len(runs)} rankers: {list(runs.keys())}")
    
    method = method.lower()
    
    if method == "combsum":
        fused = combsum(runs)
        tag = "combsum"
        
    elif method == "combmnz":
        fused = combmnz(runs)
        tag = "combmnz"
        
    elif method == "rrf":
        fused = rrf(runs, k=rrf_k)
        tag = f"rrf-k{rrf_k}"
    
    elif method in ["wcombsum", "w-combsum"]:
        if not qpp_dir:
            raise ValueError("--qpp_dir required for weighted methods")
        qpp_data = load_qpp_scores(qpp_dir)
        qpp_index = get_qpp_index(qpp_model)
        fused = weighted_combsum(runs, qpp_data, qpp_index)
        tag = f"wcombsum-{qpp_model.lower()}"
        
    elif method in ["wcombmnz", "w-combmnz"]:
        if not qpp_dir:
            raise ValueError("--qpp_dir required for weighted methods")
        qpp_data = load_qpp_scores(qpp_dir)
        qpp_index = get_qpp_index(qpp_model)
        fused = weighted_combmnz(runs, qpp_data, qpp_index)
        tag = f"wcombmnz-{qpp_model.lower()}"
        
    elif method in ["wrrf", "w-rrf"]:
        if not qpp_dir:
            raise ValueError("--qpp_dir required for weighted methods")
        qpp_data = load_qpp_scores(qpp_dir)
        qpp_index = get_qpp_index(qpp_model)
        fused = weighted_rrf(runs, qpp_data, qpp_index, k=rrf_k)
        tag = f"wrrf-{qpp_model.lower()}"
    
    elif method == "learned":
        if not model_path:
            raise ValueError("--model_path required for learned fusion")
        if not qpp_dir:
            raise ValueError("--qpp_dir required for learned fusion")
        qpp_data = load_qpp_scores(qpp_dir)
        fused = learned_fusion(runs, qpp_data, model_path)
        tag = "learned"
    
    else:
        valid = config.fusion.methods
        raise ValueError(f"Unknown method '{method}'. Valid: {valid}")
    
    if output_path:
        write_runfile(fused, output_path, tag)
    
    return fused


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fusion Methods for Multi-Retriever RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
  combsum   - Sum of normalized scores (ranx)
  combmnz   - CombSUM x number of rankers returning doc (ranx)
  rrf       - Reciprocal Rank Fusion (ranx)
  wcombsum  - QPP-weighted CombSUM
  wcombmnz  - QPP-weighted CombMNZ
  wrrf      - QPP-weighted RRF
  learned   - ML model learned weights

Examples:
  python src/fusion.py --method combsum --runs_dir data/nq/runs --output fused.res
  python src/fusion.py --method wcombsum --runs_dir data/nq/runs --qpp_dir data/nq/qpp --qpp_model RSD
  python src/fusion.py --method learned --runs_dir data/nq/runs --qpp_dir data/nq/qpp --model_path models/fusion.pkl
"""
    )
    parser.add_argument("--method", required=True,
                        choices=config.fusion.methods,
                        help="Fusion method")
    parser.add_argument("--runs_dir", required=True, help="Directory with .norm.res files")
    parser.add_argument("--qpp_dir", default=None, help="Directory with .qpp files")
    parser.add_argument("--qpp_model", default=config.qpp.default_method, help="QPP model for weighting")
    parser.add_argument("--model_path", default=None, help="Path to learned model")
    parser.add_argument("--output", required=True, help="Output TREC run file")
    parser.add_argument("--rrf_k", type=int, default=config.fusion.rrf_k, help="RRF k constant")
    
    args = parser.parse_args()
    
    run_fusion(
        method=args.method,
        runs_dir=args.runs_dir,
        qpp_dir=args.qpp_dir,
        qpp_model=args.qpp_model,
        model_path=args.model_path,
        output_path=args.output,
        rrf_k=args.rrf_k
    )

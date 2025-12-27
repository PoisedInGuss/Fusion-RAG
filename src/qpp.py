#!/usr/bin/env python3
"""
Incoming: documents with scores --- {List[Dict] with score field}
Processing: QPP computation via Java bridge --- {13 QPP methods}
Outgoing: QPP scores and predictions --- {Dict with qpp_scores}

Query Performance Prediction (QPP) Operations
----------------------------------------------
Implements 13 real QPP methods via Java bridge (QPPBridge.java):

1. NQC - Normalized Query Commitment
2. SMV - Similarity Mean Variance  
3. WIG - Weighted Information Gain
4. SigmaMax - Maximum Standard Deviation
5. SigmaX - Threshold-based Std Dev
6. RSD - Retrieval Score Distribution
7. UEF - Utility Estimation Framework
8. MaxIDF - Maximum IDF
9. AvgIDF - Average IDF
10. CumNQC - Cumulative NQC
11. SNQC - Calibrated NQC
12. DenseQPP - Dense Vector QPP
13. DenseQPP-M - Matryoshka Dense QPP

STRICT: No fallbacks. Java QPP bridge is mandatory.

Usage:
    from src.qpp import QPPBridge
    qpp = QPPBridge()
    result = qpp.compute(query="what is X", scores=[0.9, 0.7, 0.5, ...])
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import config
from src.config import config


@dataclass
class QPPResult:
    """QPP computation result."""
    query: str
    retriever_name: str
    qpp_scores: Dict[str, float]
    methods_used: List[str]
    processing_time_ms: float
    predictions: Dict[str, Any]
    error: Optional[str] = None


class QPPBridge:
    """
    Python-Java bridge for QPP computation.
    
    Uses REAL QPP implementations from src/qpp/java (copied from lucene-msmarco).
    STRICT: Requires compiled Java classes. No Python fallback.
    """
    
    def __init__(self, java_dir: Optional[str] = None, persistent: bool = True):
        """
        Initialize QPP Bridge.
        
        Args:
            java_dir: Optional path to QPP Java directory. 
                      Defaults to src/qpp relative to project root.
            persistent: Use persistent Java process (much faster for batch ops)
        
        Raises:
            RuntimeError: If Java QPP bridge is not compiled/available.
        """
        self.src_dir = Path(__file__).parent
        self.project_root = config.project_root
        
        # Use src/qpp for real QPP implementations
        self.qpp_dir = Path(java_dir) if java_dir else self.src_dir / "qpp"
        self.classes_dir = self.qpp_dir / "target" / "classes"
        self.deps_dir = self.qpp_dir / "target" / "dependency"
        
        self.persistent = persistent
        self._java_process = None
        
        # Get QPP method names from config
        self.qpp_method_names = config.qpp.methods
        
        # STRICT: Check for compiled Java - raise if not available
        if not self._check_java():
            raise RuntimeError(
                f"Java QPP bridge not compiled. "
                f"Required: {self.classes_dir / 'qpp' / 'QPPBridge.class'}. "
                f"To compile: cd {self.qpp_dir} && ./build.sh"
            )
        
        print(f"[QPP] Real QPP from src/qpp ready (persistent={persistent})", file=sys.stderr)
    
    def _check_java(self) -> bool:
        """Check if Java QPPBridge is compiled and available."""
        # Check for compiled class file in lucene-msmarco
        class_file = self.classes_dir / "qpp" / "QPPBridge.class"
        return class_file.exists()
    
    def _get_classpath(self) -> str:
        """Build Java classpath from lucene-msmarco Maven build."""
        paths = [str(self.classes_dir)]
        
        # Add all dependency JARs
        if self.deps_dir.exists():
            paths.append(str(self.deps_dir / "*"))
        
        return ":".join(paths)
    
    def compute_batch(self, queries_scores: List[tuple]) -> List[QPPResult]:
        """
        Compute QPP for multiple queries in one Java call (MUCH faster).
        
        Args:
            queries_scores: List of (query_id, scores) tuples
            
        Returns:
            List of QPPResult objects
        """
        if not queries_scores:
            return []
        
        # Prepare batch input
        batch_input = {
            "queries": [
                {"qid": str(qid), "scores": scores}
                for qid, scores in queries_scores
            ]
        }
        
        classpath = self._get_classpath()
        
        try:
            result = subprocess.run(
                ["java", "-cp", classpath, "qpp.QPPBridge", "--batch"],
                input=json.dumps(batch_input),
                capture_output=True,
                text=True,
                timeout=60
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Java QPP batch computation timed out (60s limit)")
        except FileNotFoundError:
            raise RuntimeError("Java not found. Ensure Java 11+ is installed and in PATH.")
        
        if result.returncode != 0:
            raise RuntimeError(f"Java QPP failed: {result.stderr}")
        
        # Parse batch results
        try:
            batch_output = json.loads(result.stdout)
            results = []
            for item in batch_output.get("results", []):
                results.append(QPPResult(
                    query=item["qid"],
                    retriever_name="batch",
                    qpp_scores=item["qpp_scores"],
                    methods_used=self.qpp_method_names,
                    processing_time_ms=0,
                    predictions={}
                ))
            return results
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON from Java QPP: {result.stdout}")
    
    def compute(
        self,
        query: str,
        scores: List[float],
        retriever_name: str = "unknown",
        methods: Optional[List[str]] = None
    ) -> QPPResult:
        """
        Compute QPP scores for a query's retrieval scores via Java bridge.
        
        Args:
            query: Query text
            scores: List of retrieval scores (top-k documents)
            retriever_name: Name of retriever
            methods: List of QPP methods to compute (default: all from config)
            
        Returns:
            QPPResult with all QPP scores
            
        Raises:
            RuntimeError: If Java QPP computation fails.
        """
        import time
        start = time.time()
        
        methods = methods or self.qpp_method_names
        
        # Build input JSON
        input_data = {
            "query": query,
            "documents": [{"score": s} for s in scores],
            "retriever_name": retriever_name,
            "methods": methods
        }
        
        classpath = self._get_classpath()
        
        try:
            result = subprocess.run(
                ["java", "-cp", classpath, "qpp.QPPBridge"],
                input=json.dumps(input_data),
                capture_output=True,
                text=True,
                timeout=30
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Java QPP computation timed out (30s limit)")
        except FileNotFoundError:
            raise RuntimeError("Java not found. Ensure Java 11+ is installed and in PATH.")
        
        if result.returncode != 0:
            raise RuntimeError(f"Java QPP failed (exit {result.returncode}): {result.stderr}")
        
        try:
            output = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Java QPP returned invalid JSON: {e}\nStdout: {result.stdout}")
        
        return QPPResult(
            query=query,
            retriever_name=retriever_name,
            qpp_scores=output.get("qpp_scores", {}),
            methods_used=output.get("methods_used", []),
            processing_time_ms=(time.time() - start) * 1000,
            predictions=output.get("predictions", {})
        )


# ============================================================================
# Batch QPP Computation (for .res files)
# ============================================================================

def compute_qpp_for_res_file(
    res_path: str,
    output_path: Optional[str] = None,
    top_k: int = None,
    normalize: str = None,
    queries_path: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Compute QPP scores for all queries in a TREC .res file.
    
    Args:
        res_path: Path to .res file
        output_path: Path for .qpp output (optional)
        top_k: Top-k documents for QPP (from config if None)
        normalize: "minmax", "zscore", or "none" (from config if None)
        queries_path: Path to queries.jsonl (BEIR format) for actual query text.
                     Required for IDF-based QPP methods (WIG, MaxIDF, AvgIDF).
        
    Returns:
        Dict of {qid: [13 QPP scores]}
        
    Raises:
        RuntimeError: If Java QPP computation fails.
        FileNotFoundError: If res_path doesn't exist.
    """
    import numpy as np
    from collections import defaultdict
    
    # Get defaults from config
    top_k = top_k or config.processing.retrieval.top_k
    normalize = normalize or config.qpp.normalization
    qpp_method_names = config.qpp.methods
    
    if not os.path.exists(res_path):
        raise FileNotFoundError(f"Run file not found: {res_path}")
    
    # Load actual query texts if provided
    query_texts = {}
    if queries_path and os.path.exists(queries_path):
        print(f"[QPP] Loading query texts from {queries_path}")
        with open(queries_path, 'r') as f:
            for line in f:
                q = json.loads(line)
                query_texts[q["_id"]] = q["text"]
        print(f"[QPP] Loaded {len(query_texts)} query texts")
    
    # Load run file
    runs = defaultdict(list)
    with open(res_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                qid, _, docno, rank, score = parts[:5]
                runs[qid].append(float(score))
    
    if not runs:
        raise ValueError(f"No valid entries in run file: {res_path}")
    
    # Sort and truncate
    for qid in runs:
        runs[qid] = sorted(runs[qid], reverse=True)[:top_k]
    
    # Compute QPP via Java bridge (batch mode for efficiency)
    bridge = QPPBridge()
    
    # Prepare batch input - use actual query text if available
    batch_queries = [
        (query_texts.get(qid, qid), scores, qid)  # (query_text, scores, qid_for_tracking)
        for qid, scores in runs.items()
    ]
    
    # Batch compute (single Java call - no fallback)
    batch_results = bridge.compute_batch([(q[0], q[1]) for q in batch_queries])
    results = {}
    for i, qpp_result in enumerate(batch_results):
        qid = batch_queries[i][2]
        qpp_list = [qpp_result.qpp_scores.get(m, 0.0) for m in qpp_method_names]
        results[qid] = qpp_list
    
    # Normalize
    if normalize != "none" and results:
        results = _normalize_qpp(results, normalize)
    
    # Write output
    if output_path:
        with open(output_path, 'w') as f:
            for qid in sorted(results.keys(), key=_qid_sort_key):
                scores = results[qid]
                score_str = '\t'.join(f"{s:.6f}" for s in scores)
                f.write(f"{qid}\t{score_str}\n")
        print(f"Wrote QPP scores to {output_path}")
    
    return results


def _normalize_qpp(results: Dict[str, List[float]], method: str) -> Dict[str, List[float]]:
    """Normalize QPP scores across queries."""
    import numpy as np
    
    n_methods = config.qpp.n_methods
    
    # Collect per-method values
    method_values = [[] for _ in range(n_methods)]
    for scores in results.values():
        for i, score in enumerate(scores):
            method_values[i].append(score)
    
    # Compute params
    params = []
    for values in method_values:
        arr = np.array(values)
        if method == "minmax":
            vmin, vmax = arr.min(), arr.max()
            params.append((vmin, vmax - vmin if vmax > vmin else 1.0))
        else:
            params.append((arr.mean(), arr.std() if arr.std() > 0 else 1.0))
    
    # Apply normalization
    normalized = {}
    for qid, scores in results.items():
        norm_scores = []
        for i, score in enumerate(scores):
            vmin, scale = params[i]
            if method == "minmax":
                norm_scores.append((score - vmin) / scale if scale > 0 else 0.0)
            else:
                norm_scores.append((score - vmin) / scale)
        normalized[qid] = norm_scores
    
    return normalized


def _qid_sort_key(qid: str):
    """Sort numeric query IDs (including negatives) before alphanumeric IDs."""
    try:
        return (0, int(qid))
    except (ValueError, TypeError):
        try:
            return (0, int(str(qid).strip()))
        except (ValueError, TypeError):
            return (1, str(qid))


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QPP Bridge - Compute 13 QPP methods (Java required)")
    parser.add_argument("--res_file", help="TREC .res file to process")
    parser.add_argument("--output", help="Output .qpp file")
    parser.add_argument("--top_k", type=int, default=config.processing.retrieval.top_k, help="Top-k for QPP")
    parser.add_argument("--normalize", choices=["none", "minmax", "zscore"], 
                        default=config.qpp.normalization, help="Normalization")
    parser.add_argument("--test", action="store_true", help="Run test")
    args = parser.parse_args()
    
    if args.test:
        # Quick test
        bridge = QPPBridge()
        result = bridge.compute(
            query="what is machine learning",
            scores=[0.95, 0.82, 0.71, 0.65, 0.58, 0.52, 0.48, 0.41, 0.35, 0.28]
        )
        print(f"Query: {result.query}")
        print(f"QPP Scores:")
        for method, score in result.qpp_scores.items():
            print(f"  {method}: {score:.4f}")
        print(f"Predictions: {result.predictions}")
        
    elif args.res_file:
        output = args.output or args.res_file.replace(".res", ".mmnorm.qpp")
        compute_qpp_for_res_file(args.res_file, output, args.top_k, args.normalize)
    else:
        parser.print_help()

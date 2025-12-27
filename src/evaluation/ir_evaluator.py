"""
Incoming: TREC run files, qrels --- {Dict/DataFrame}
Processing: IR metric computation --- {ir_measures}
Outgoing: metric scores --- {Dict[str, float]}

IR Evaluation using ir_measures
-------------------------------
Research-grade IR evaluation with standard metrics:
- nDCG@k: Normalized Discounted Cumulative Gain
- RR (MRR): Mean Reciprocal Rank  
- R@k: Recall at k
- P@k: Precision at k
- AP (MAP): Average Precision

Uses ir_measures library (standard in IR research, used by TREC).
STRICT: No fallbacks. Missing dependencies raise ImportError immediately.
"""

from typing import Dict, List, Tuple, Optional, Union
import pandas as pd

# STRICT: Fail immediately if ir_measures not available
import ir_measures
from ir_measures import nDCG, RR, R, P, AP, Judged

# Import config
from src.config import config


class IREvaluator:
    """
    IR evaluation wrapper using ir_measures.
    
    Supports:
    - Single run evaluation
    - Multi-run comparison
    - Per-query and aggregate metrics
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metric names (e.g., ["nDCG@10", "RR@10"])
                    Uses config defaults if not specified.
        """
        self.metric_names = metrics or config.evaluation.ir_metrics
        self._metrics = [ir_measures.parse_measure(m) for m in self.metric_names]
    
    def evaluate(
        self,
        run: Union[Dict[str, List[Tuple[str, float]]], pd.DataFrame],
        qrels: Union[Dict[str, Dict[str, int]], pd.DataFrame],
        per_query: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate a single run against qrels.
        
        Args:
            run: Either:
                - Dict: {qid: [(docid, score), ...]}
                - DataFrame: columns [qid, docno, score]
            qrels: Either:
                - Dict: {qid: {docid: relevance}}
                - DataFrame: columns [qid, docno, label]
            per_query: If True, return per-query scores
            
        Returns:
            Dict with metric names as keys
        """
        ir_run = self._convert_run(run)
        ir_qrels = self._convert_qrels(qrels)
        
        if per_query:
            results = {}
            for measure, values in ir_measures.iter_calc(self._metrics, ir_qrels, ir_run):
                metric_name = str(measure)
                if metric_name not in results:
                    results[metric_name] = {}
                for qid, value in values.items():
                    results[metric_name][qid] = value
            return results
        else:
            aggregated = ir_measures.calc_aggregate(self._metrics, ir_qrels, ir_run)
            return {str(k): float(v) for k, v in aggregated.items()}
    
    def evaluate_multiple(
        self,
        runs: Dict[str, Union[Dict, pd.DataFrame]],
        qrels: Union[Dict[str, Dict[str, int]], pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Evaluate multiple runs and return comparison table.
        
        Args:
            runs: {run_name: run_data}
            qrels: Ground truth relevance judgments
            
        Returns:
            DataFrame with runs as rows, metrics as columns
        """
        results = []
        for name, run in runs.items():
            metrics = self.evaluate(run, qrels, per_query=False)
            metrics["run_name"] = name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        df = df.set_index("run_name")
        return df.sort_values(by=self.metric_names[0], ascending=False)
    
    def _convert_run(
        self, 
        run: Union[Dict[str, List[Tuple[str, float]]], pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """Convert run to ir_measures format (nested dict)."""
        if isinstance(run, pd.DataFrame):
            run_dict = {}
            for _, row in run.iterrows():
                qid = str(row.get("qid", row.get("query_id", "")))
                docid = str(row.get("docno", row.get("doc_id", "")))
                score = float(row.get("score", 0.0))
                if qid not in run_dict:
                    run_dict[qid] = {}
                run_dict[qid][docid] = score
            return run_dict
        else:
            # Input: {qid: [(docid, score), ...]}
            run_dict = {}
            for qid, docs in run.items():
                run_dict[str(qid)] = {str(d): float(s) for d, s in docs}
            return run_dict
    
    def _convert_qrels(
        self,
        qrels: Union[Dict[str, Dict[str, int]], pd.DataFrame]
    ) -> Dict[str, Dict[str, int]]:
        """Convert qrels to ir_measures format (nested dict)."""
        if isinstance(qrels, pd.DataFrame):
            qrels_dict = {}
            for _, row in qrels.iterrows():
                qid = str(row.get("qid", row.get("query_id", "")))
                docid = str(row.get("docno", row.get("doc_id", "")))
                rel = int(row.get("label", row.get("relevance", 0)))
                if qid not in qrels_dict:
                    qrels_dict[qid] = {}
                qrels_dict[qid][docid] = rel
            return qrels_dict
        else:
            return {
                str(qid): {str(d): int(r) for d, r in docs.items()}
                for qid, docs in qrels.items()
            }


def evaluate_run(
    run: Union[Dict[str, List[Tuple[str, float]]], pd.DataFrame],
    qrels: Union[Dict[str, Dict[str, int]], pd.DataFrame],
    metrics: Optional[List[str]] = None,
    per_query: bool = False
) -> Dict[str, float]:
    """
    Convenience function for single run evaluation.
    
    Args:
        run: Run results (dict or DataFrame)
        qrels: Relevance judgments
        metrics: Metric names (uses config defaults if None)
        per_query: Return per-query scores
        
    Returns:
        Dict of metric scores
    """
    evaluator = IREvaluator(metrics=metrics)
    return evaluator.evaluate(run, qrels, per_query=per_query)


def evaluate_runs(
    runs: Dict[str, Union[Dict, pd.DataFrame]],
    qrels: Union[Dict[str, Dict[str, int]], pd.DataFrame],
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Convenience function for multi-run comparison.
    
    Args:
        runs: {run_name: run_data}
        qrels: Relevance judgments
        metrics: Metric names (uses config defaults if None)
        
    Returns:
        DataFrame comparison table
    """
    evaluator = IREvaluator(metrics=metrics)
    return evaluator.evaluate_multiple(runs, qrels)


def compute_ndcg(
    ranked_docs: List[str], 
    qrels: Dict[str, int], 
    k: int = 10
) -> float:
    """
    Compute NDCG@k using ir_measures.
    
    Args:
        ranked_docs: Ordered list of doc IDs
        qrels: {docid: relevance}
        k: Cutoff
        
    Returns:
        NDCG@k score
    """
    run = {"q1": {doc: float(k - i) for i, doc in enumerate(ranked_docs[:k])}}
    qrels_fmt = {"q1": qrels}
    
    metric = ir_measures.parse_measure(f"nDCG@{k}")
    result = ir_measures.calc_aggregate([metric], qrels_fmt, run)
    
    return float(list(result.values())[0])

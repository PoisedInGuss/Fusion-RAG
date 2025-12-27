"""
Incoming: predictions, gold labels --- {FactVerificationPrediction, FactVerificationGoldLabel}
Processing: fact verification evaluation --- {accuracy, F1, evidence metrics}
Outgoing: metric scores --- {Dict[str, float]}

Fact Verification Evaluation
----------------------------
Evaluates claim verification systems (SciFact, FEVER-style):

Metrics:
- Label Accuracy: 3-way classification accuracy
- Label F1 (macro): Per-class F1 averaged
- Evidence Precision/Recall: Document-level evidence retrieval
- Sentence F1: Sentence-level evidence (if available)

Label normalization:
- SUPPORT/SUPPORTS -> SUPPORT
- CONTRADICT/REFUTE/REFUTES -> CONTRADICT  
- NOT_ENOUGH_INFO/NEI -> NOT_ENOUGH_INFO
"""

from collections import Counter
from typing import Dict, List, Optional, Union

from .base import (
    FactVerificationGoldLabel,
    FactVerificationPrediction,
    TaskEvaluator,
    TaskType,
)


class FactVerificationEvaluator(TaskEvaluator):
    """
    Evaluator for fact verification tasks (SciFact, FEVER).
    
    Computes:
    - Label accuracy (3-way classification)
    - Per-class precision, recall, F1
    - Macro F1 (average across classes)
    - Evidence retrieval metrics (if evidence available)
    """
    
    task_type = TaskType.FACT_VERIFICATION
    
    LABELS = ["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        include_evidence_metrics: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            metrics: Specific metrics to compute (None = all)
            include_evidence_metrics: Whether to compute evidence retrieval metrics
        """
        self.metric_names_requested = metrics
        self.include_evidence_metrics = include_evidence_metrics
    
    def get_metric_names(self) -> List[str]:
        """Return list of computed metrics."""
        base_metrics = [
            "accuracy",
            "macro_f1",
            "support_f1",
            "contradict_f1",
            "nei_f1",
        ]
        
        if self.include_evidence_metrics:
            base_metrics.extend([
                "evidence_precision",
                "evidence_recall",
                "evidence_f1",
            ])
        
        if self.metric_names_requested:
            return [m for m in base_metrics if m in self.metric_names_requested]
        return base_metrics
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label to standard form."""
        return FactVerificationGoldLabel.LABEL_MAP.get(
            label.upper(), 
            label.upper()
        )
    
    def evaluate(
        self,
        prediction: FactVerificationPrediction,
        gold: FactVerificationGoldLabel
    ) -> Dict[str, float]:
        """
        Evaluate single prediction.
        
        Returns:
            Dict with per-example metrics
        """
        pred_label = self._normalize_label(prediction.predicted_label)
        gold_label = gold.normalized_label
        
        results = {
            "correct": 1.0 if pred_label == gold_label else 0.0,
            "pred_label": pred_label,
            "gold_label": gold_label,
        }
        
        # Evidence metrics (document-level)
        if self.include_evidence_metrics and gold.evidence_doc_ids:
            pred_docs = set(prediction.evidence_doc_ids) if prediction.evidence_doc_ids else set()
            gold_docs = set(gold.evidence_doc_ids)
            
            if pred_docs:
                precision = len(pred_docs & gold_docs) / len(pred_docs)
            else:
                precision = 0.0
            
            if gold_docs:
                recall = len(pred_docs & gold_docs) / len(gold_docs)
            else:
                recall = 1.0 if not pred_docs else 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            results["evidence_precision"] = precision
            results["evidence_recall"] = recall
            results["evidence_f1"] = f1
        
        return results
    
    def evaluate_batch(
        self,
        predictions: List[FactVerificationPrediction],
        golds: List[FactVerificationGoldLabel],
        aggregate: bool = True
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate batch of predictions.
        
        Args:
            aggregate: If True, compute aggregated metrics
            
        Returns:
            Aggregated metrics dict or list of per-example dicts
        """
        results = []
        for pred, gold in zip(predictions, golds):
            results.append(self.evaluate(pred, gold))
        
        if not aggregate:
            return results
        
        # Compute aggregated metrics
        n_samples = len(results)
        
        # Overall accuracy
        accuracy = sum(r["correct"] for r in results) / n_samples if n_samples else 0.0
        
        # Per-class metrics
        class_metrics = self._compute_class_metrics(results)
        
        aggregated = {
            "accuracy": accuracy,
            "macro_f1": class_metrics["macro_f1"],
            "support_f1": class_metrics["SUPPORT"]["f1"],
            "contradict_f1": class_metrics["CONTRADICT"]["f1"],
            "nei_f1": class_metrics["NOT_ENOUGH_INFO"]["f1"],
            "n_samples": n_samples,
        }
        
        # Evidence metrics (average)
        if self.include_evidence_metrics:
            evidence_keys = ["evidence_precision", "evidence_recall", "evidence_f1"]
            for key in evidence_keys:
                values = [r.get(key, 0.0) for r in results if key in r]
                aggregated[key] = sum(values) / len(values) if values else 0.0
        
        return aggregated
    
    def _compute_class_metrics(
        self, 
        results: List[Dict]
    ) -> Dict[str, Dict[str, float]]:
        """Compute per-class precision, recall, F1."""
        # Count predictions and golds per class
        pred_counts = Counter(r["pred_label"] for r in results)
        gold_counts = Counter(r["gold_label"] for r in results)
        correct_counts = Counter(
            r["pred_label"] for r in results if r["correct"]
        )
        
        class_metrics = {}
        f1_scores = []
        
        for label in self.LABELS:
            tp = correct_counts.get(label, 0)
            pred_total = pred_counts.get(label, 0)
            gold_total = gold_counts.get(label, 0)
            
            precision = tp / pred_total if pred_total > 0 else 0.0
            recall = tp / gold_total if gold_total > 0 else 0.0
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            
            class_metrics[label] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": gold_total,
            }
            
            # Only include in macro F1 if class has samples
            if gold_total > 0:
                f1_scores.append(f1)
        
        class_metrics["macro_f1"] = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return class_metrics


def compute_fact_verification_metrics(
    predictions: List[Dict],
    golds: List[Dict],
    include_evidence: bool = True
) -> Dict[str, float]:
    """
    Convenience function for fact verification evaluation.
    
    Args:
        predictions: List of dicts with 'claim_id', 'predicted_label', 'evidence_doc_ids'
        golds: List of dicts with 'claim_id', 'label', 'evidence_doc_ids'
        include_evidence: Include evidence retrieval metrics
        
    Returns:
        Dict of aggregated metrics
    """
    # Convert to typed objects
    pred_objs = [
        FactVerificationPrediction(
            claim_id=p.get("claim_id", ""),
            predicted_label=p.get("predicted_label", "NOT_ENOUGH_INFO"),
            evidence_doc_ids=p.get("evidence_doc_ids", []),
            confidence=p.get("confidence"),
            rationale=p.get("rationale"),
        )
        for p in predictions
    ]
    
    gold_objs = [
        FactVerificationGoldLabel(
            claim_id=g.get("claim_id", ""),
            claim=g.get("claim", ""),
            label=g.get("label", "NOT_ENOUGH_INFO"),
            evidence_doc_ids=g.get("evidence_doc_ids", []),
            evidence_sentences=g.get("evidence_sentences", []),
        )
        for g in golds
    ]
    
    evaluator = FactVerificationEvaluator(include_evidence_metrics=include_evidence)
    return evaluator.evaluate_batch(pred_objs, gold_objs, aggregate=True)

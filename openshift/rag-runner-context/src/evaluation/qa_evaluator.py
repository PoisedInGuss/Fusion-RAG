"""
Incoming: predictions, references --- {str, List[str]}
Processing: QA metric computation --- {HuggingFace evaluate}
Outgoing: metric scores --- {Dict[str, float]}

QA Evaluation using HuggingFace evaluate
----------------------------------------
Research-grade QA evaluation with standard metrics:
- Exact Match (EM): Normalized string match
- Token F1: Token-level F1 score
- ROUGE-L: Longest common subsequence F1

Uses HuggingFace evaluate library (SQuAD metrics standard).
STRICT: No fallbacks. Missing dependencies raise ImportError immediately.
"""

from typing import Dict, List, Optional, Union

# STRICT: Fail immediately if evaluate not available
import evaluate


class QAEvaluator:
    """
    QA evaluation wrapper using HuggingFace evaluate.
    
    Supports:
    - Single answer evaluation
    - Batch evaluation
    - Multiple gold answers per question
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metric names ["em", "f1", "rouge_l", "containment"]
                    Uses ["em", "f1"] by default.
        """
        self.metric_names = metrics or ["em", "f1"]
        
        # Load HuggingFace metrics - STRICT: fail if not available
        self._squad_metric = None
        self._rouge_metric = None
        
        if "em" in self.metric_names or "f1" in self.metric_names:
            self._squad_metric = evaluate.load("squad")
        
        if "rouge_l" in self.metric_names:
            self._rouge_metric = evaluate.load("rouge")
    
    def evaluate(
        self,
        prediction: str,
        references: Union[str, List[str]],
    ) -> Dict[str, float]:
        """
        Evaluate a single prediction against reference(s).
        
        Args:
            prediction: Generated answer
            references: Gold answer(s) - single string or list
            
        Returns:
            Dict with metric scores
        """
        if isinstance(references, str):
            references = [references]
        
        results = {}
        
        if "em" in self.metric_names or "f1" in self.metric_names:
            squad_result = self._compute_squad_metrics(prediction, references)
            if "em" in self.metric_names:
                results["em"] = squad_result["exact_match"]
            if "f1" in self.metric_names:
                results["f1"] = squad_result["f1"]
        
        if "rouge_l" in self.metric_names:
            results["rouge_l"] = self._compute_rouge_l(prediction, references)
        
        if "containment" in self.metric_names:
            results["containment"] = self._compute_containment(prediction, references)
        
        return results
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[Union[str, List[str]]],
        aggregate: bool = True
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate batch of predictions.
        
        Args:
            predictions: List of generated answers
            references: List of gold answer(s)
            aggregate: If True, return averaged scores
            
        Returns:
            Aggregated metrics dict or list of per-example dicts
        """
        results = []
        for pred, refs in zip(predictions, references):
            results.append(self.evaluate(pred, refs))
        
        if not aggregate:
            return results
        
        aggregated = {}
        for metric in self.metric_names:
            values = [r.get(metric, 0.0) for r in results if metric in r]
            if values:
                aggregated[metric] = sum(values) / len(values)
        
        aggregated["n_samples"] = len(results)
        return aggregated
    
    def _compute_squad_metrics(self, prediction: str, references: List[str]) -> Dict[str, float]:
        """Compute EM and F1 using HuggingFace squad metric."""
        preds = [{"id": "0", "prediction_text": prediction}]
        refs = [{"id": "0", "answers": {"text": references, "answer_start": [0] * len(references)}}]
        result = self._squad_metric.compute(predictions=preds, references=refs)
        return {
            "exact_match": result["exact_match"] / 100.0,
            "f1": result["f1"] / 100.0
        }
    
    def _compute_rouge_l(self, prediction: str, references: List[str]) -> float:
        """Compute ROUGE-L F1 - best across references."""
        best_rouge = 0.0
        for ref in references:
            result = self._rouge_metric.compute(
                predictions=[prediction],
                references=[ref],
                use_stemmer=True
            )
            best_rouge = max(best_rouge, result["rougeL"])
        return best_rouge
    
    def _compute_containment(self, prediction: str, references: List[str]) -> float:
        """Check if any reference is contained in prediction."""
        pred_lower = prediction.lower()
        for ref in references:
            if ref.lower() in pred_lower:
                return 1.0
        return 0.0


def compute_qa_metrics(
    prediction: str,
    references: Union[str, List[str]],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Convenience function for single prediction evaluation.
    
    Args:
        prediction: Generated answer
        references: Gold answer(s)
        metrics: Metric names (default: ["em", "f1"])
        
    Returns:
        Dict of metric scores
    """
    evaluator = QAEvaluator(metrics=metrics)
    return evaluator.evaluate(prediction, references)


def compute_qa_metrics_batch(
    predictions: List[str],
    references: List[Union[str, List[str]]],
    metrics: Optional[List[str]] = None,
    aggregate: bool = True
) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Convenience function for batch evaluation.
    
    Args:
        predictions: List of generated answers
        references: List of gold answer(s)
        metrics: Metric names
        aggregate: Return averaged scores
        
    Returns:
        Aggregated metrics or list of per-example metrics
    """
    evaluator = QAEvaluator(metrics=metrics)
    return evaluator.evaluate_batch(predictions, references, aggregate=aggregate)

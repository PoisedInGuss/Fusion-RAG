"""
Incoming: runs, qrels, predictions, references --- {TREC format, gold labels}
Processing: evaluation --- {IR metrics, QA metrics, fact verification metrics}
Outgoing: metric scores --- {Dict[str, float]}

Evaluation Module
-----------------
Unified evaluation using research-grade packages:
- IR metrics: ir_measures (NDCG, MRR, Recall, MAP, etc.)
- QA metrics: HuggingFace evaluate (EM, F1, ROUGE)
- Fact Verification: 3-way classification (SUPPORT, CONTRADICT, NOT_ENOUGH_INFO)

Task Types:
- QA: NQ, HotpotQA, TriviaQA (answer extraction/generation)
- FactVerification: SciFact, FEVER (claim verification)
"""

import os

from .base import (
    TaskType,
    GoldLabel,
    QAGoldLabel,
    FactVerificationGoldLabel,
    Prediction,
    QAPrediction,
    FactVerificationPrediction,
    TaskEvaluator,
    get_task_type,
)

_SKIP_IR = os.environ.get("SRC_SKIP_IR_EVAL", "").lower() in ("1", "true", "yes")

if not _SKIP_IR:
    from .ir_evaluator import IREvaluator, evaluate_run, evaluate_runs
else:
    def _disabled(*args, **kwargs):
        raise RuntimeError("IR evaluation disabled via SRC_SKIP_IR_EVAL")

    IREvaluator = None  # type: ignore
    evaluate_run = _disabled  # type: ignore
    evaluate_runs = _disabled  # type: ignore

from .qa_evaluator import QAEvaluator, compute_qa_metrics
from .fact_verification import (
    FactVerificationEvaluator,
    compute_fact_verification_metrics,
)

__all__ = [
    # Base classes
    "TaskType",
    "GoldLabel",
    "QAGoldLabel",
    "FactVerificationGoldLabel",
    "Prediction",
    "QAPrediction",
    "FactVerificationPrediction",
    "TaskEvaluator",
    "get_task_type",
    # IR evaluation
    "IREvaluator",
    "evaluate_run",
    "evaluate_runs",
    # QA evaluation
    "QAEvaluator", 
    "compute_qa_metrics",
    # Fact verification
    "FactVerificationEvaluator",
    "compute_fact_verification_metrics",
]


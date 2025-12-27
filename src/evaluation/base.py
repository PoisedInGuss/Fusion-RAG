"""
Incoming: predictions, gold labels --- {task-specific formats}
Processing: evaluation abstraction --- {interface definition}
Outgoing: metric scores --- {Dict[str, float]}

Abstract Base Classes for Task Evaluation
-----------------------------------------
Defines interfaces for different RAG evaluation tasks:
- QA (Question Answering): NQ, HotpotQA
- FactVerification: SciFact, FEVER
- Generation: Open-ended generation

Each task type has specific:
- Gold label format
- Evaluation metrics
- Prediction format
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class TaskType(Enum):
    """Supported RAG task types."""
    QA = "qa"                           # Question answering (extractive/generative)
    FACT_VERIFICATION = "fact_verification"  # Claim verification (3-way classification)
    GENERATION = "generation"           # Open-ended generation


@dataclass
class GoldLabel:
    """Base class for gold labels."""
    pass


@dataclass
class QAGoldLabel(GoldLabel):
    """Gold label for QA tasks (NQ, HotpotQA)."""
    query: str
    answers: List[str]  # Multiple valid answers
    
    def __post_init__(self):
        if isinstance(self.answers, str):
            self.answers = [self.answers]


@dataclass  
class FactVerificationGoldLabel(GoldLabel):
    """Gold label for fact verification tasks (SciFact, FEVER)."""
    claim_id: str
    claim: str
    label: str  # SUPPORT, CONTRADICT/REFUTE, NOT_ENOUGH_INFO
    evidence_doc_ids: List[str]
    evidence_sentences: List[int]  # Sentence indices in evidence docs
    
    # Normalized label mapping
    LABEL_MAP = {
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT", 
        "CONTRADICT": "CONTRADICT",
        "REFUTE": "CONTRADICT",
        "REFUTES": "CONTRADICT",
        "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
        "NEI": "NOT_ENOUGH_INFO",
        "NOTENOUGHINFO": "NOT_ENOUGH_INFO",
    }
    
    @property
    def normalized_label(self) -> str:
        """Get normalized label."""
        return self.LABEL_MAP.get(self.label.upper(), self.label.upper())


@dataclass
class Prediction:
    """Base class for model predictions."""
    pass


@dataclass
class QAPrediction(Prediction):
    """Prediction for QA tasks."""
    query_id: str
    answer: str
    context_docs: List[str]  # Retrieved doc IDs used
    
    
@dataclass
class FactVerificationPrediction(Prediction):
    """Prediction for fact verification tasks."""
    claim_id: str
    predicted_label: str  # SUPPORT, CONTRADICT, NOT_ENOUGH_INFO
    evidence_doc_ids: List[str]  # Retrieved evidence
    confidence: Optional[float] = None
    rationale: Optional[str] = None  # Model's reasoning


class TaskEvaluator(ABC):
    """
    Abstract base class for task-specific evaluation.
    
    Subclasses implement evaluation logic for specific task types.
    """
    
    task_type: TaskType
    
    @abstractmethod
    def evaluate(
        self,
        prediction: Prediction,
        gold: GoldLabel
    ) -> Dict[str, float]:
        """
        Evaluate single prediction against gold label.
        
        Returns:
            Dict of metric_name -> score
        """
        pass
    
    @abstractmethod
    def evaluate_batch(
        self,
        predictions: List[Prediction],
        golds: List[GoldLabel],
        aggregate: bool = True
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Evaluate batch of predictions.
        
        Args:
            aggregate: If True, return averaged metrics
            
        Returns:
            Aggregated metrics or list of per-example metrics
        """
        pass
    
    @abstractmethod
    def get_metric_names(self) -> List[str]:
        """Return list of metric names this evaluator computes."""
        pass


def get_task_type(dataset: str) -> TaskType:
    """
    Determine task type from dataset name.
    
    Args:
        dataset: Dataset name (nq, hotpotqa, scifact, fever, etc.)
        
    Returns:
        TaskType enum value
    """
    FACT_VERIFICATION_DATASETS = {"scifact", "fever", "climate-fever", "healthver"}
    QA_DATASETS = {"nq", "hotpotqa", "triviaqa", "squad", "msmarco"}
    
    dataset_lower = dataset.lower()
    
    if dataset_lower in FACT_VERIFICATION_DATASETS:
        return TaskType.FACT_VERIFICATION
    elif dataset_lower in QA_DATASETS:
        return TaskType.QA
    else:
        # Default to QA for unknown datasets
        return TaskType.QA

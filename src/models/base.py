"""
Incoming: QPP features, target weights --- {numpy arrays}
Processing: model training/prediction --- {abstract}
Outgoing: predicted weights --- {numpy array}

Base class for fusion weight models.
"""

import numpy as np
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import config
from src.config import config


class BaseFusionModel(ABC):
    """
    Abstract base class for fusion weight prediction models.
    
    All models predict per-query weights for each retriever
    based on QPP features.
    """
    
    def __init__(self, retrievers: List[str], n_qpp: int = None):
        """
        Args:
            retrievers: List of retriever names
            n_qpp: Number of QPP methods (from config if None)
        """
        self.retrievers = retrievers
        self.n_retrievers = len(retrievers)
        self.n_qpp = n_qpp if n_qpp is not None else config.qpp.n_methods
        self.n_features = self.n_qpp * len(retrievers)
        self.feature_names = [f"{r}_{i}" for r in retrievers for i in range(self.n_qpp)]
        self.is_trained = False
    
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Train the model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            Y_train: Training targets (n_samples, n_retrievers)
            X_val: Validation features
            Y_val: Validation targets
        
        Returns:
            Training metrics dict
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict weights for given features.
        
        Args:
            X: Features (n_samples, n_features)
        
        Returns:
            Predicted weights (n_samples, n_retrievers), normalized to sum to 1
        """
        pass
    
    def predict_single(self, qpp_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Predict weights for a single query.
        
        Args:
            qpp_scores: {retriever: [qpp scores]}
        
        Returns:
            {retriever: weight}
        """
        # Build feature vector
        features = np.zeros(self.n_features)
        for j, retriever in enumerate(self.retrievers):
            if retriever in qpp_scores:
                scores = qpp_scores[retriever]
                features[j*self.n_qpp:(j+1)*self.n_qpp] = scores[:self.n_qpp]
        
        # Predict
        weights = self.predict(features.reshape(1, -1))[0]
        
        return {r: w for r, w in zip(self.retrievers, weights)}
    
    def save(self, path: str):
        """Save model to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self,
                'retrievers': self.retrievers,
                'n_qpp': self.n_qpp,
                'model_type': self.__class__.__name__
            }, f)
        
        print(f"Saved {self.__class__.__name__} to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'BaseFusionModel':
        """Load model from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        return data['model']
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1 per sample."""
        weights = np.clip(weights, 0, None)
        sums = weights.sum(axis=1, keepdims=True)
        sums[sums == 0] = 1
        return weights / sums


def build_features(
    qpp_data: Dict[str, Dict[str, List[float]]],
    retrievers: List[str],
    n_qpp: int = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix from QPP scores.
    
    Args:
        qpp_data: {qid: {retriever: [qpp_scores]}}
        retrievers: List of retriever names
        n_qpp: Number of QPP methods (from config if None)
    
    Returns:
        X: Feature matrix (n_queries, n_features)
        qids: List of query IDs
    """
    n_qpp = n_qpp if n_qpp is not None else config.qpp.n_methods
    
    qids = sorted(qpp_data.keys())
    X = np.zeros((len(qids), n_qpp * len(retrievers)))
    
    for i, qid in enumerate(qids):
        for j, retriever in enumerate(retrievers):
            if retriever in qpp_data.get(qid, {}):
                scores = qpp_data[qid][retriever]
                X[i, j*n_qpp:(j+1)*n_qpp] = scores[:n_qpp]
    
    return X, qids


# Re-export compute_ndcg from evaluation module
from src.evaluation.ir_evaluator import compute_ndcg

__all__ = ["BaseFusionModel", "build_features", "compute_ndcg"]

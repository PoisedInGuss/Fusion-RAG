"""
Incoming: QPP features, target weights --- {numpy arrays}
Processing: gradient boosting training/prediction --- {LightGBM models}
Outgoing: predicted weights --- {numpy array}

LightGBM-based fusion weight models.

STRICT: Requires LightGBM. No fallbacks.
"""

import numpy as np
from typing import Dict, List, Optional

# STRICT: Fail immediately if LightGBM not available
import lightgbm as lgb

# Import config
from src.config import config

from .base import BaseFusionModel


class PerRetrieverLGBM(BaseFusionModel):
    """
    Separate LightGBM model for each retriever.
    
    Trains N independent models, each predicting the weight
    for one retriever based on all QPP features.
    """
    
    def __init__(self, retrievers: List[str], n_qpp: int = None, **lgb_params):
        super().__init__(retrievers, n_qpp)
        
        self.models = {}
        
        # Get defaults from config
        lgb_config = config.training.lightgbm
        
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': lgb_config.num_leaves,
            'learning_rate': lgb_config.learning_rate,
            'feature_fraction': lgb_config.feature_fraction,
            'bagging_fraction': lgb_config.bagging_fraction,
            'bagging_freq': lgb_config.bagging_freq,
            'verbose': -1,
            **lgb_params
        }
    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        num_boost_round: int = None,
        early_stopping_rounds: int = None
    ) -> Dict:
        """Train separate model for each retriever."""
        # Get defaults from config
        lgb_config = config.training.lightgbm
        num_boost_round = num_boost_round or lgb_config.num_boost_round
        early_stopping_rounds = early_stopping_rounds or lgb_config.early_stopping_rounds
        
        print(f"\n=== Training PerRetrieverLGBM ({self.n_retrievers} models) ===")
        
        metrics = {'per_retriever': {}}
        
        for j, retriever in enumerate(self.retrievers):
            print(f"\nTraining for {retriever}...")
            
            train_data = lgb.Dataset(
                X_train, 
                label=Y_train[:, j],
                feature_name=self.feature_names
            )
            
            valid_sets = [train_data]
            if X_val is not None and Y_val is not None:
                valid_data = lgb.Dataset(X_val, label=Y_val[:, j], reference=train_data)
                valid_sets.append(valid_data)
            
            callbacks = [lgb.log_evaluation(50)]
            if early_stopping_rounds and X_val is not None:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
            
            model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks
            )
            
            self.models[retriever] = model
            
            # Feature importance
            importance = model.feature_importance(importance_type='gain')
            top_features = sorted(
                zip(self.feature_names, importance),
                key=lambda x: -x[1]
            )[:5]
            
            metrics['per_retriever'][retriever] = {
                'best_iteration': model.best_iteration,
                'top_features': top_features
            }
            print(f"  Best iter: {model.best_iteration}, Top: {top_features[:3]}")
        
        self.is_trained = True
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict weights using each retriever's model."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        weights = np.zeros((X.shape[0], self.n_retrievers))
        for j, retriever in enumerate(self.retrievers):
            weights[:, j] = self.models[retriever].predict(X)
        
        return self._normalize_weights(weights)


class MultiOutputLGBM(BaseFusionModel):
    """
    Single LightGBM model predicting all retriever weights at once.
    
    Uses multi-output regression to jointly predict weights,
    allowing the model to learn correlations between retrievers.
    """
    
    def __init__(self, retrievers: List[str], n_qpp: int = None, **lgb_params):
        super().__init__(retrievers, n_qpp)
        
        self.models = []  # One model per output (LightGBM doesn't support true multi-output)
        
        # Get defaults from config
        lgb_config = config.training.lightgbm
        
        self.lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': lgb_config.num_leaves,
            'learning_rate': lgb_config.learning_rate,
            'feature_fraction': lgb_config.feature_fraction,
            'verbose': -1,
            **lgb_params
        }
    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        num_boost_round: int = None,
        early_stopping_rounds: int = None
    ) -> Dict:
        """
        Train multi-output model.
        
        Note: LightGBM doesn't natively support multi-output,
        so we train models jointly with shared hyperparameters.
        """
        # Get defaults from config
        lgb_config = config.training.lightgbm
        num_boost_round = num_boost_round or lgb_config.num_boost_round
        early_stopping_rounds = early_stopping_rounds or lgb_config.early_stopping_rounds
        
        print(f"\n=== Training MultiOutputLGBM ({self.n_retrievers} outputs) ===")
        
        # For true multi-output, we train all at once with same stopping
        self.models = []
        metrics = {'outputs': []}
        
        # Find best iteration across all outputs
        best_iters = []
        
        for j in range(self.n_retrievers):
            train_data = lgb.Dataset(
                X_train,
                label=Y_train[:, j],
                feature_name=self.feature_names
            )
            
            valid_sets = [train_data]
            if X_val is not None:
                valid_data = lgb.Dataset(X_val, label=Y_val[:, j], reference=train_data)
                valid_sets.append(valid_data)
            
            callbacks = []
            if early_stopping_rounds and X_val is not None:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
            
            model = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks
            )
            
            self.models.append(model)
            best_iters.append(model.best_iteration)
            
            metrics['outputs'].append({
                'retriever': self.retrievers[j],
                'best_iteration': model.best_iteration
            })
        
        print(f"Best iterations: {best_iters}")
        
        # Get shared feature importance (average across outputs)
        avg_importance = np.zeros(len(self.feature_names))
        for model in self.models:
            avg_importance += model.feature_importance(importance_type='gain')
        avg_importance /= len(self.models)
        
        top_features = sorted(
            zip(self.feature_names, avg_importance),
            key=lambda x: -x[1]
        )[:10]
        
        metrics['top_features'] = top_features
        print(f"Top shared features: {top_features[:5]}")
        
        self.is_trained = True
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all weights at once."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        weights = np.column_stack([
            model.predict(X) for model in self.models
        ])
        
        return self._normalize_weights(weights)

"""
LightGBM model for demand forecasting.

Uses Huber loss for robustness to outliers and native categorical
feature support for efficient handling of item types.
"""

import lightgbm as lgb
from .base_model import BaseModel
from typing import Optional, Dict
import numpy as np


class LightGBMModel(BaseModel):
    """
    LightGBM Regressor for bakery demand forecasting.
    
    Why LightGBM for this problem?
    ------------------------------
    1. Huber loss - Robust to outlier days (festivals, catering orders)
       without needing to cap values manually
    
    2. Native categorical support - Handles 'item' encoding directly,
       no one-hot encoding needed
    
    3. Fast training - Histogram-based splitting is significantly faster
       than XGBoost/RandomForest, enabling more hyperparameter search
    
    4. Leaf-wise growth - Grows the leaf with maximum loss reduction,
       often achieving better accuracy with fewer iterations
    
    5. Feature importance - SHAP-compatible for explainability
    
    6. Handles many features well - Works efficiently with the rich
       feature set (temporal + lags + rolling + interactions)
    
    Attributes:
    -----------
    n_estimators : int
        Number of boosting rounds
    learning_rate : float
        Step size shrinkage (lower = more rounds needed but better)
    num_leaves : int
        Maximum number of leaves per tree (controls complexity)
    max_depth : int
        Maximum tree depth (-1 = no limit)
    min_child_samples : int
        Minimum data in a leaf (prevents overfitting on small groups)
    subsample : float
        Fraction of data used per iteration (adds randomness)
    colsample_bytree : float
        Fraction of features used per tree
    reg_alpha : float
        L1 regularization
    reg_lambda : float
        L2 regularization
    random_state : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize LightGBM model with time-series-tuned defaults.
        
        Parameters:
        -----------
        n_estimators : int, default=300
            Number of boosting rounds. More rounds with lower learning
            rate generally gives better results.
        
        learning_rate : float, default=0.05
            Step size shrinkage. Lower values (0.01-0.05) need more
            estimators but generalize better.
        
        num_leaves : int, default=31
            Max leaves per tree. Higher = more complex model.
            Rule of thumb: num_leaves < 2^max_depth
        
        max_depth : int, default=-1
            Max tree depth. -1 means no limit (controlled by num_leaves).
        
        min_child_samples : int, default=20
            Minimum samples in a leaf. Higher values prevent overfitting
            on rare day-of-week/item combinations.
        
        subsample : float, default=0.8
            Row sampling ratio. Adds stochasticity to reduce overfitting.
        
        colsample_bytree : float, default=0.8
            Feature sampling ratio per tree.
        
        reg_alpha : float, default=0.1
            L1 regularization for feature selection.
        
        reg_lambda : float, default=0.1
            L2 regularization for smoothing.
        
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state

        # Call parent class __init__
        super().__init__(model_name="LightGBM")

        # Add hyperparameters to metadata for tracking
        self.metadata['hyperparameters'] = {
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'min_child_samples': min_child_samples,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'huber',
            'random_state': random_state
        }

    def _build_model(self) -> lgb.LGBMRegressor:
        """
        Build and return a LGBMRegressor with Huber loss.
        
        Huber loss is robust to outliers — it behaves like MSE for
        small errors and like MAE for large errors. This is ideal
        for bakery data where special event days (2-3x normal volume)
        would otherwise dominate the loss function.
        
        Returns:
        --------
        lgb.LGBMRegressor
            Configured but untrained LightGBM model
        """
        return lgb.LGBMRegressor(
            objective='huber',
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1  # Suppress training output
        )

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Uses the 'gain' importance type by default, which measures
        the total gain brought by a feature across all splits.
        
        Returns:
        --------
        dict
            Feature names mapped to importance scores (sorted descending)
        
        Raises:
        -------
        ValueError
            If model hasn't been trained yet
        """
        if self.model is None:
            raise ValueError(
                "Cannot get feature importances from untrained model. "
                "Call .fit(X_train, y_train) first."
            )

        importances = self.model.feature_importances_

        feature_importance_dict = {
            feature_name: importance
            for feature_name, importance in zip(self.feature_names, importances)
        }

        sorted_importances = dict(
            sorted(
                feature_importance_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )
        )

        return {
            name: round(float(importance), 4)
            for name, importance in sorted_importances.items()
        }

    def __str__(self) -> str:
        """String representation for debugging."""
        trained_status = "trained" if self.model is not None else "untrained"
        return (
            f"LightGBM(n_estimators={self.n_estimators}, "
            f"lr={self.learning_rate}, leaves={self.num_leaves}, "
            f"objective=huber, {trained_status})"
        )

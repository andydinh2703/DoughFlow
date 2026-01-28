"""
Random Forest model for demand forecasting. 
"""

from sklearn.ensemble import RandomForestRegressor
from .base_model import BaseModel
from typing import Optional, Dict
import numpy as np 

# Class definition 
class RandomForestModel(BaseModel):
    """
    Random Forest Regressor for bakery demand forecasting.
    
    Why Random Forest for this problem?
    -----------------------------------
    1. No feature scaling needed - Works with mixed scales
       (day_of_week=1-7, quantity=0-145, etc.)
    
    2. Handles non-linearity naturally - Captures patterns like:
       - Friday demand spikes
       - Summer seasonality
       - Weekend vs weekday differences
    
    3. Robust to outliers - Tree splits aren't affected by extreme values
       (those 145-quantity days won't throw off the model)
    
    4. Feature importance - Tells us which features matter most
       (Is it lag_7? day_of_week? rolling averages?)
    
    5. Fast training - Good for iteration during development
    
    Attributes:
    -----------
    n_estimators : int
        Number of trees in the forest (more = better but slower)
    max_depth : int or None
        Maximum depth of each tree (None = unlimited)
    min_samples_split : int
        Minimum samples needed to split a node
    min_samples_leaf : int
        Minimum samples required in a leaf node
    random_state : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        
        random_state: int = 42
    ):
        """
        Initialize RandomForest model.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of trees in the forest.
            - More trees = better performance (up to a point)
            - Typical range: 50-500
            - Start with 100, tune if needed
        
        max_depth : int or None, default=None
            Maximum depth of each tree.
            - None = trees grow until leaves are pure
            - Setting limit prevents overfitting
            - Example: max_depth=10 means 10 levels of splits
        
        min_samples_split : int, default=2
            Minimum samples required to split a node.
            - Higher values prevent overfitting
            - Example: 10 means "need 10 samples to consider splitting"
        
        min_samples_leaf : int, default=1
            Minimum samples required in each leaf.
            - Higher values create more general predictions
            - Example: 5 means "each prediction needs ≥5 samples"
        
        random_state : int, default=42
            Random seed for reproducibility.
            - Same seed = same results every time
            - Important for debugging and comparison
        """
            
        # Store hyperparameters as instance variables
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
        # Call parent class __init__
        # This sets up: self.model, self.feature_names, self.metadata
        super().__init__(model_name="RandomForest")
        
        # Add hyperparameters to metadata for tracking
        self.metadata['hyperparameters'] = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': random_state
        }


    def _build_model(self) -> RandomForestRegressor:
        """
        Build and return a RandomForestRegressor instance.
        
        This method is called automatically by BaseModel.fit() when needed.
        You don't call this directly - it's internal.
        
        Returns:
        --------
        RandomForestRegressor
            Configured but untrained sklearn model
        """

        return RandomForestRegressor(
            n_estimators= self.n_estimators,
            max_depth= self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores for parallel training
            verbose=0   # Set to 1 if want to see training progress
        )
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Feature importance tells you which features the model
        considers most valuable for making predictions.
        
        How it works:
        -------------
        - RandomForest calculates importance based on how much
          each feature decreases impurity (variance for regression)
        - Higher score = more important
        - Scores sum to 1.0
        
        Returns:
        --------
        dict
            Feature names mapped to importance scores (sorted by importance)
            
        Example Output:
        ---------------
        {
            'lag_7': 0.3542,        # Last week's demand is most important
            'rolling_mean_7': 0.2103,
            'day_of_week': 0.1876,
            'lag_1': 0.1245,
            'month': 0.0891,
            'is_weekend': 0.0343
        }
        
        Interpretation:
        ---------------
        - lag_7 explains 35.4% of the model's decisions
        - Top 3 features explain 75.2% of decisions
        - is_weekend only contributes 3.4% (might remove it!)
        
        Raises:
        -------
        ValueError
            If model hasn't been trained yet
            
        Usage:
        ------
        >>> model = RandomForestModel()
        >>> model.fit(X_train, y_train)
        >>> importances = model.get_feature_importances()
        >>> print(f"Most important: {list(importances.keys())[0]}")
        Most important: lag_7
        """

        if self.model is None: 
            raise ValueError(
                "Cannot get feature importances from untrained model."
                "Call .fit(X_train, y_train) first."
            )
        
        # get importances from sklearn 
        importances = self.model.feature_importances_

        # dictionary mapping feature name -> importance
        feature_importance_dict = {
            feature_name: importance
            for feature_name, importance in zip(self.feature_names, importances)
        }

        # sort by importance (highest first)
        sorted_importances = dict(
            sorted(
                feature_importance_dict.items(),
                key = lambda item: item[1], # sort by value 
                reverse = True # highest first
            )
        )

        # Round for readability 
        return {
            name: round(importance, 4)
            for name, importance in sorted_importances.items()
        }
    
    def __str__(self) -> str:
        """
        String representation for debugging.

        Makes it easy to see model configuration when printing.

        Example:
        --------
        >>> model = RandomForestModel(n_estimators=100, max_depth=10)
        >>> print(model)
        RandomForest(n_estimators=100, max_depth=10, trained=False)
        """

        trained_status = "trained" if self.model is not None else "untrained"

        return (
            f"RandomForest(n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, {trained_status})"
        )
    
    


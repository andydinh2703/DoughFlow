"""
XGBoost Model
"""

from .base_model import BaseModel
import xgboost
from typing import Optional, Dict

class XGBoostModel(BaseModel):
    """
    
    
    
    """

    def __init__(
            self,
            n_estimators = 100,
            max_depth = 6,
            learning_rate = 0.1,
            subsample = 1.0,
            colsample_bytree = 1.0,
            random_state = 42
    ):
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state



        super().__init__(model_name = "XGBoost")

        # Add hyperparameters to metadata for tracking
        self.metadata['hyperparameters'] = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state
        }

    
    def _build_model(self) -> xgboost.XGBRegressor:
        """
        
        """

        return xgboost.XGBRegressor(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            learning_rate = self.learning_rate,
            subsample = self.subsample,
            colsample_bytree = self.colsample_bytree,
            random_state = self.random_state
        )



    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Feature importance tells you which features the model
        considers most valuable for making predictions.
        """

        if self.model is None: 
            raise ValueError(
                "Cannot get feature importances from untrained model."
                "Call .fit(X_train, y_train) first."
            )
        
        # get feature importances
        importances = self.model.feature_importances_

        # mapping feature importances into a dictionary 
        feature_importance_dict = {
            feature: importance 
            for feature, importance in zip(self.feature_names, importances)
        }

        # sort all the feature in descending order
        sorted_feature_importances = dict(
            sorted(
            feature_importance_dict.items(),
            key= lambda item: item[1],
            reverse= True
        ))

        return {
            name: round(importance, 4)
            for name, importance in sorted_feature_importances.items()
        }

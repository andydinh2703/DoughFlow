"""
Linear Regression model for demand forecasting.
"""

from sklearn.linear_model import LinearRegression
from .base_model import BaseModel
from typing import Optional, Dict
import numpy as np 

class LinearRegressionModel(BaseModel):
    """
    
    """

    def __init__(
            self,
            fit_intercept = True
    ):
        self.fit_intercept = fit_intercept


        super().__init__(model_name = 'LinearRegression')

    def _build_model(self):

        """
        Create and return the sklearn LinearRegression instance.

        """

        return LinearRegression(
            fit_intercept= self.fit_intercept
        )
    
    def get_coefficients(self):
        """
        Get feature coefficients after training.

        Returns
        -------
        dict
            Feature name -> coefficient value, sorted by absolute importance
        """

        if not hasattr(self, 'feature_names'):
            raise ValueError('Model must be train first.')
        
        # Create dictionary mapping feature names to coefficients
        coef_dict = dict(zip(self.feature_names, self.model.coef_))

        # sort by absolute value
        sorted_coefs = sorted(
            coef_dict.items(),
            key=lambda x: abs(x[1]),
            reverse= True
        )

        return dict(sorted_coefs)
    
    def get_intercept(self):
        """

        Get model's intercept (baseline prediction)

        """

        if not hasattr(self.model, 'intercept_'):
            raise ValueError("Model must be trained first.")
        
        return self.model.intercept_
    
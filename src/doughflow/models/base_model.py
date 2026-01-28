"""
Base model class for DoughFlow forecasting models.

This module defines the interface that all forecasting models must implement.
Following the Template Method pattern for consistency across different model types.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
from pathlib import Path
import json
from datetime import datetime


class BaseModel(ABC):
    """
    Abstract base class for all forecasting models.

    This class defines the standard interface that all models must implement.
    It ensures consistency across different model types (RandomForest, XGBoost, etc.)

    Key Design Principles:
    ----------------------
    1. Separation of Concerns: Training, prediction, and evaluation are separate
    2. Template Method Pattern: Common logic in base class, specifics in subclasses
    3. Reproducibility: All models must be saveable/loadable with metadata
    4. Transparency: Models must explain what features they expect

    Attributes:
    -----------
    model : Any
        The underlying ML model (e.g., RandomForestRegressor)
    model_name : str
        Human-readable name for this model
    feature_names : list
        Names of features the model was trained on
    metadata : dict
        Additional information about the model (training date, parameters, etc.)
    """

    def __init__(self, model_name: str = "BaseModel"):
        """
        Initialize the base model.

        Parameters:
        -----------
        model_name : str
            Name identifier for this model (e.g., "RandomForest", "XGBoost")
        """
        self.model = None  # Will be set by subclass
        self.model_name = model_name
        self.feature_names = None
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_type': model_name
        }

    @abstractmethod
    def _build_model(self) -> Any:
        """
        Build and return the underlying ML model.

        This is abstract because each model type (RandomForest, XGBoost, etc.)
        will have different initialization logic.

        Example for RandomForest:
        -------------------------
        return RandomForestRegressor(n_estimators=100, random_state=42)

        Returns:
        --------
        model : Any
            Initialized (but not trained) ML model
        """
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Train the model on the provided data.

        This method:
        1. Stores feature names (important for prediction later!)
        2. Builds the model if not already built
        3. Fits the model to the data
        4. Updates metadata with training info

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix (each row is a sample, each column is a feature)
            Example shape: (1676, 17) for 1676 samples and 17 features
        y : pd.Series
            Target values (what we're trying to predict - quantity)
            Example shape: (1676,)

        Returns:
        --------
        self : BaseModel
            Returns self to allow method chaining (e.g., model.fit(X, y).predict(X_test))
        """
        # Store feature names - CRITICAL for ensuring prediction uses same features
        self.feature_names = list(X.columns)

        # Build model if not already built
        if self.model is None:
            self.model = self._build_model()

        # Actual training happens here
        self.model.fit(X, y)

        # Update metadata for reproducibility
        self.metadata['n_samples'] = len(X)
        self.metadata['n_features'] = len(self.feature_names)
        self.metadata['features'] = self.feature_names
        self.metadata['trained_at'] = datetime.now().isoformat()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        This method validates that:
        1. Model has been trained
        2. Input features match training features (order matters!)

        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix with same columns as training data

        Returns:
        --------
        predictions : np.ndarray
            Predicted values (quantities)

        Raises:
        -------
        ValueError
            If model hasn't been trained or features don't match
        """
        if self.model is None:
            raise ValueError(
                f"{self.model_name} must be fitted before making predictions. "
                "Call .fit(X_train, y_train) first."
            )

        # Validate features match training data
        if self.feature_names is not None:
            if list(X.columns) != self.feature_names:
                raise ValueError(
                    f"Feature mismatch!\n"
                    f"Expected: {self.feature_names}\n"
                    f"Got: {list(X.columns)}"
                )

        return self.model.predict(X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on a dataset.

        Calculates multiple metrics to give a complete picture:
        - MAE: Average absolute error (in same units as target)
        - RMSE: Root mean squared error (penalizes large errors more)
        - MAPE: Mean absolute percentage error (scale-independent)
        - R²: Proportion of variance explained (0-1, higher is better)

        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            True values

        Returns:
        --------
        metrics : dict
            Dictionary with metric names and values
        """
        predictions = self.predict(X)

        # Calculate metrics
        mae = np.mean(np.abs(y - predictions))
        rmse = np.sqrt(np.mean((y - predictions) ** 2))

        # MAPE: handle zeros by adding small epsilon
        mape = np.mean(np.abs((y - predictions) / (y + 1e-10))) * 100

        # R²: proportion of variance explained
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return {
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'mape': round(mape, 2),
            'r2': round(r2, 4)
        }

    def save(self, directory: str) -> None:
        """
        Save the model and its metadata to disk.

        Saves two files:
        1. {model_name}_model.pkl - The trained model (using joblib for efficiency)
        2. {model_name}_metadata.json - Model info (features, metrics, etc.)

        Why save both?
        - .pkl: The actual trained model weights
        - .json: Human-readable info about the model (for documentation/debugging)

        Parameters:
        -----------
        directory : str
            Directory path to save the model
        """
        if self.model is None:
            raise ValueError("Cannot save untrained model. Train it first with .fit()")

        # Create directory if it doesn't exist
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save the model
        model_path = save_dir / f"{self.model_name}_model.pkl"
        joblib.dump(self.model, model_path)

        # Save metadata
        metadata_path = save_dir / f"{self.model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Metadata saved to: {metadata_path}")

    @classmethod
    def load(cls, directory: str, model_name: str) -> 'BaseModel':
        """
        Load a saved model from disk.

        This is a class method (note @classmethod) because we're creating
        a new instance of the model class.

        Parameters:
        -----------
        directory : str
            Directory where model was saved
        model_name : str
            Name of the model to load

        Returns:
        --------
        model : BaseModel
            Loaded model ready for predictions
        """
        load_dir = Path(directory)

        # Load the trained model
        model_path = load_dir / f"{model_name}_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        loaded_model = joblib.load(model_path)

        # Load metadata
        metadata_path = load_dir / f"{model_name}_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # Create new instance and populate it
        instance = cls(model_name=model_name)
        instance.model = loaded_model
        instance.metadata = metadata
        instance.feature_names = metadata.get('features', None)

        print(f"✓ Loaded model from: {model_path}")
        return instance

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get human-readable information about the model.

        Useful for debugging and documentation.

        Returns:
        --------
        info : dict
            Model information including name, features, metadata
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.model is not None,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }

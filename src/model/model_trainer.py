"""
Model training module for the MLOps project.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.preprocessing.preprocessor import Preprocessor
import joblib
import logging

class ModelTrainer:
    def __init__(self, config_path: str = "src/config.yaml"):
        """Initialize the model trainer."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.preprocessor = Preprocessor(config_path)
        self.logger = logging.getLogger(__name__)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model on the provided data.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        try:
            # Preprocess the data
            X_processed = self.preprocessor.fit_transform(X)
            
            # Train the model
            self.model.fit(X_processed, y)
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        try:
            self.model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise 
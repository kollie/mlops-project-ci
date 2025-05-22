"""
Model inference module for the MLOps project.
"""

import numpy as np
import pandas as pd
import joblib
import logging

class Predictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.logger = logging.getLogger(__name__)

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        try:
            self.model = joblib.load(path)
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def preprocess_input(self, data: pd.DataFrame) -> np.ndarray:
        """Preprocess input data."""
        try:
            # Add preprocessing steps here
            return data.values
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not loaded")
        try:
            processed_data = self.preprocess_input(data)
            return self.model.predict(processed_data)
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if self.model is None:
            raise ValueError("Model not loaded")
        try:
            processed_data = self.preprocess_input(data)
            return self.model.predict_proba(processed_data)
        except Exception as e:
            self.logger.error(f"Error getting probabilities: {str(e)}")
            raise

    def calculate_confidence(self, probabilities: np.ndarray) -> np.ndarray:
        """Calculate confidence scores."""
        return np.max(probabilities, axis=1)

    def save_predictions(self, data: pd.DataFrame, predictions: np.ndarray, 
                        probabilities: np.ndarray, path: str) -> None:
        """Save predictions to file."""
        try:
            results = pd.DataFrame({
                'prediction': predictions,
                'probability': probabilities[:, 1],
                'confidence': self.calculate_confidence(probabilities)
            })
            results.to_csv(path, index=False)
            self.logger.info(f"Predictions saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            raise 
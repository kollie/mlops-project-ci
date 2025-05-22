"""
Model inference module for the MLOps project.
"""

import numpy as np
import pandas as pd
from src.preprocessing.preprocessor import Preprocessor
import joblib
import logging
import yaml
from pathlib import Path

class Predictor:
    def __init__(self, model_path: str = "models/model.joblib", config_path: str = "src/config.yaml"):
        """Initialize the predictor."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.model = self._load_model(model_path)
        self.preprocessor = Preprocessor(config_path)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            filename=self.config['logging']['file']
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self, model_path: str):
        """Load the trained model."""
        try:
            return joblib.load(model_path)
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        # Check for required columns
        required_columns = self.config['features']['numerical_features'] + self.config['features']['categorical_features']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for invalid data types
        for col in self.config['features']['numerical_features']:
            if not pd.api.types.is_numeric_dtype(data[col]):
                raise ValueError(f"Column {col} must be numeric")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        try:
            # Validate data
            self._validate_data(data)
            
            # Preprocess data
            X_processed = self.preprocessor.transform(data)
            
            # Make predictions
            predictions = self.model.predict(X_processed)
            self.logger.info("Predictions made successfully")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        try:
            # Validate data
            self._validate_data(data)
            
            # Preprocess data
            X_processed = self.preprocessor.transform(data)
            
            # Get probabilities
            probabilities = self.model.predict_proba(X_processed)
            self.logger.info("Probabilities calculated successfully")
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error calculating probabilities: {str(e)}")
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
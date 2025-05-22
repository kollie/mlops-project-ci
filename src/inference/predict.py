import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from pathlib import Path
from typing import Union, List, Dict, Any

class Predictor:
    def __init__(self, config_path: str = "src/config.yaml", model_path: str = None):
        """
        Initialize the Predictor with configuration and model.
        
        Args:
            config_path (str): Path to the configuration file
            model_path (str, optional): Path to the saved model. If None, uses config path.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.model = None
        self.preprocessor = None
        
        if model_path is None:
            model_path = Path(self.config['data']['model_path']) / "model.joblib"
        self.load_model(model_path)
    
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
    
    def load_model(self, model_path: str):
        """
        Load the trained model and preprocessor.
        
        Args:
            model_path (str): Path to the saved model
        """
        try:
            # Load model and preprocessor
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.preprocessor = model_data['preprocessor']
            
            self.logger.info(f"Model and preprocessor loaded successfully from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Preprocessed data
        """
        try:
            if self.preprocessor is None:
                raise ValueError("Preprocessor not loaded")
            
            # Drop specified columns
            data = data.drop(columns=self.config['features']['drop_columns'], errors='ignore')
            
            # Transform the data
            processed_data = self.preprocessor.transform(data)
            
            self.logger.info("Input data preprocessed successfully")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing input data: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Predicted labels
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess the data
            processed_data = self.preprocess_input(data)
            
            # Make predictions
            predictions = self.model.predict(processed_data)
            
            self.logger.info("Predictions made successfully")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities for the input data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        try:
            if self.model is None:
                raise ValueError("Model not loaded")
            
            # Preprocess the data
            processed_data = self.preprocess_input(data)
            
            # Get prediction probabilities
            probabilities = self.model.predict_proba(processed_data)
            
            self.logger.info("Prediction probabilities calculated successfully")
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction probabilities: {str(e)}")
            raise
    
    def predict_with_confidence(self, data: pd.DataFrame, threshold: float = 0.8) -> Dict[str, Any]:
        """
        Make predictions with confidence scores.
        
        Args:
            data (pd.DataFrame): Input data
            threshold (float): Confidence threshold
            
        Returns:
            Dict[str, Any]: Dictionary containing predictions and confidence scores
        """
        try:
            # Get predictions and probabilities
            predictions = self.predict(data)
            probabilities = self.predict_proba(data)
            
            # Calculate confidence scores
            confidence_scores = np.max(probabilities, axis=1)
            
            # Create results dictionary
            results = {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidence_scores': confidence_scores,
                'high_confidence_mask': confidence_scores >= threshold
            }
            
            self.logger.info("Predictions with confidence scores calculated successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error making predictions with confidence: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    from data_loader.data_loader import DataLoader
    
    # Load test data
    loader = DataLoader()
    data = loader.load_data()
    _, _, test = loader.split_data(data)
    
    # Make predictions
    predictor = Predictor()
    predictions = predictor.predict(test.drop(columns=['readmitted']))
    results = predictor.predict_with_confidence(test.drop(columns=['readmitted']))
    
    print("Predictions:", predictions)
    print("Confidence scores:", results['confidence_scores'])

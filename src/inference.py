import pandas as pd
import numpy as np
from typing import Dict, Any, List
from .utils import setup_logging, load_config
from .data_loader import DataLoader
from .data_validation import DataValidator
from .preprocessing import DataPreprocessor
from .features import FeatureEngineer
from .model import ModelTrainer

logger = setup_logging()

class InferencePipeline:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = load_config(config_path)
        self.data_loader = DataLoader(config_path)
        self.data_validator = DataValidator(config_path)
        self.preprocessor = DataPreprocessor(config_path)
        self.feature_engineer = FeatureEngineer(config_path)
        self.model_trainer = ModelTrainer(config_path)
    
    def load_and_validate_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate input data."""
        try:
            # Load data
            data = self.data_loader.load_csv(data_path)
            
            # Validate data
            if not self.data_validator.validate_data(data):
                raise ValueError("Data validation failed")
            
            return data
        except Exception as e:
            logger.error(f"Error in load_and_validate_data: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data."""
        try:
            # Clean data
            data = self.preprocessor.clean_data(data)
            
            # Transform data
            data, _ = self.preprocessor.transform_data(data, is_training=False)
            
            return data
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            raise
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for inference."""
        try:
            # Apply feature engineering
            data, _ = self.feature_engineer.engineer_features(data)
            
            return data
        except Exception as e:
            logger.error(f"Error in engineer_features: {str(e)}")
            raise
    
    def predict(self, data_path: str) -> np.ndarray:
        """Run the complete inference pipeline."""
        try:
            # Load and validate data
            data = self.load_and_validate_data(data_path)
            
            # Preprocess data
            data = self.preprocess_data(data)
            
            # Engineer features
            data = self.engineer_features(data)
            
            # Load model
            self.model_trainer.load_model()
            
            # Make predictions
            predictions = self.model_trainer.predict(data)
            
            return predictions
        except Exception as e:
            logger.error(f"Error in predict: {str(e)}")
            raise
    
    def predict_batch(self, data_paths: List[str]) -> Dict[str, np.ndarray]:
        """Run inference on multiple data files."""
        try:
            results = {}
            for data_path in data_paths:
                predictions = self.predict(data_path)
                results[data_path] = predictions
            
            return results
        except Exception as e:
            logger.error(f"Error in predict_batch: {str(e)}")
            raise 
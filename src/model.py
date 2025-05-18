import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
from typing import Tuple, Dict, Any
from .utils import setup_logging, load_config

logger = setup_logging()

class ModelTrainer:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = load_config(config_path)
        self.model_config = self.config['model']
        self.model = None
    
    def _initialize_model(self):
        """Initialize the model based on configuration."""
        try:
            if self.model_config['name'] == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=self.model_config['parameters']['n_estimators'],
                    max_depth=self.model_config['parameters']['max_depth'],
                    min_samples_split=self.model_config['parameters']['min_samples_split'],
                    min_samples_leaf=self.model_config['parameters']['min_samples_leaf'],
                    random_state=self.model_config['random_state']
                )
            else:
                raise ValueError(f"Unknown model type: {self.model_config['name']}")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the model and return training metrics."""
        try:
            self._initialize_model()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.model_config['test_size'],
                random_state=self.model_config['random_state']
            )
            
            # Train the model
            logger.info("Training model...")
            self.model.fit(X_train, y_train)
            
            # Calculate training metrics
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            metrics = {
                'train_score': train_score,
                'test_score': test_score,
                'feature_importance': dict(zip(X.columns, self.model.feature_importances_))
            }
            
            logger.info(f"Training completed. Train score: {train_score:.4f}, Test score: {test_score:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self, model_path: str = None) -> None:
        """Save the trained model to disk."""
        try:
            if model_path is None:
                model_path = os.path.join(self.config['data']['model_path'], 'model.joblib')
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str = None) -> None:
        """Load a trained model from disk."""
        try:
            if model_path is None:
                model_path = os.path.join(self.config['data']['model_path'], 'model.joblib')
            
            # Load the model
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded")
            
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise 
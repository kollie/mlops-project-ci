import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import yaml
import logging
from pathlib import Path
import mlflow
import mlflow.sklearn

class ModelTrainer:
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Initialize the ModelTrainer with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
        self.model = None
        
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
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        try:
            mlflow.set_tracking_uri(self.config['model_registry']['tracking_uri'])
            mlflow.set_experiment(self.config['model_registry']['experiment_name'])
            self.logger.info("MLflow tracking setup complete")
        except Exception as e:
            self.logger.error(f"Error setting up MLflow: {str(e)}")
            raise
    
    def _create_model(self):
        """
        Create the model based on configuration.
        
        Returns:
            sklearn.base.BaseEstimator: Model instance
        """
        try:
            if self.config['model']['name'] == 'random_forest':
                model = RandomForestClassifier(
                    **self.config['model']['parameters']
                )
            else:
                raise ValueError(f"Unsupported model type: {self.config['model']['name']}")
            
            self.logger.info(f"Model {self.config['model']['name']} created successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the model on the provided data.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        """
        try:
            with mlflow.start_run():
                # Create and train the model
                self.model = self._create_model()
                self.model.fit(X_train, y_train)
                
                # Log model parameters
                mlflow.log_params(self.config['model']['parameters'])
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                self.logger.info("Model training completed successfully")
                
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def save_model(self, model_path: str = None):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str, optional): Path to save the model. If None, uses config path.
        """
        try:
            if model_path is None:
                model_path = Path(self.config['data']['model_path']) / "model.joblib"
            else:
                model_path = Path(model_path)
            
            # Create directory if it doesn't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save the model
            joblib.dump(self.model, model_path)
            
            self.logger.info(f"Model saved successfully to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_path: str = None):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str, optional): Path to load the model from. If None, uses config path.
        """
        try:
            if model_path is None:
                model_path = Path(self.config['data']['model_path']) / "model.joblib"
            else:
                model_path = Path(model_path)
            
            # Load the model
            self.model = joblib.load(model_path)
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    from data_loader.data_loader import DataLoader
    from preprocessing.preprocessor import Preprocessor
    
    # Load and preprocess data
    loader = DataLoader()
    data = loader.load_data()
    train, val, test = loader.split_data(data)
    
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(train.drop(columns=['readmitted']))
    y_train = train['readmitted']
    
    # Train model
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    trainer.save_model()

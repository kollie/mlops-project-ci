import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
import logging
import mlflow
import pandas as pd
from pathlib import Path

class ModelEvaluator:
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Initialize the ModelEvaluator with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._setup_mlflow()
        
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
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None) -> dict:
        """
        Calculate evaluation metrics.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_pred_proba (np.ndarray, optional): Predicted probabilities
            
        Returns:
            dict: Dictionary of metric names and values
        """
        try:
            metrics = {}
            
            # Calculate metrics based on configuration
            for metric in self.config['evaluation']['metrics']:
                if metric == 'accuracy':
                    metrics['accuracy'] = accuracy_score(y_true, y_pred)
                elif metric == 'precision':
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
                elif metric == 'recall':
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
                elif metric == 'f1':
                    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
                elif metric == 'roc_auc' and y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            
            self.logger.info("Metrics calculated successfully")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise
    
    def log_metrics(self, metrics: dict):
        """
        Log metrics to MLflow.
        
        Args:
            metrics (dict): Dictionary of metric names and values
        """
        try:
            with mlflow.start_run():
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                self.logger.info("Metrics logged to MLflow successfully")
                
        except Exception as e:
            self.logger.error(f"Error logging metrics to MLflow: {str(e)}")
            raise
    
    def save_metrics(self, metrics: dict, output_path: str = None):
        """
        Save metrics to a CSV file.
        
        Args:
            metrics (dict): Dictionary of metric names and values
            output_path (str, optional): Path to save metrics. If None, uses default path.
        """
        try:
            if output_path is None:
                output_path = Path(self.config['data']['processed_data_path']) / "metrics.csv"
            else:
                output_path = Path(output_path)
            
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save metrics to CSV
            pd.DataFrame([metrics]).to_csv(output_path, index=False)
            
            self.logger.info(f"Metrics saved successfully to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    from data_loader.data_loader import DataLoader
    from preprocessing.preprocessor import Preprocessor
    from model.train import ModelTrainer
    
    # Load and preprocess data
    loader = DataLoader()
    data = loader.load_data()
    train, val, test = loader.split_data(data)
    
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(train.drop(columns=['readmitted']))
    X_val = preprocessor.transform(val.drop(columns=['readmitted']))
    y_train = train['readmitted']
    y_val = val['readmitted']
    
    # Train model
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    
    # Evaluate model
    evaluator = ModelEvaluator()
    y_pred = trainer.model.predict(X_val)
    y_pred_proba = trainer.model.predict_proba(X_val)
    
    metrics = evaluator.calculate_metrics(y_val, y_pred, y_pred_proba)
    evaluator.log_metrics(metrics)
    evaluator.save_metrics(metrics)

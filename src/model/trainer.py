"""
Model Training Module for MLOps Project.

Handles model training and saving following the same patterns 
as other modules in the pipeline.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
import joblib
from pathlib import Path
from typing import Dict, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import mlflow
import mlflow.sklearn


class ModelTrainer:
    """
    Model training pipeline for the MLOps project.
    
    Handles model creation, training, and saving following the same
    patterns as other modules in the pipeline.
    """
    
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.report = {}
        
        # Model components
        self.model: Optional[Any] = None
        self.model_type: str = ""
        self.model_params: Dict = {}
        
        # Training tracking
        self._is_fitted: bool = False

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate required config sections
            required_sections = ['data', 'model', 'logging']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required config section: {section}")
            
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            log_config = self.config.get('logging', {})
            log_file = log_config.get('file', 'logs/training.log').replace('main.log', 'training.log')
            log_level = log_config.get('level', 'INFO')
            log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format=log_format,
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
            
            return logging.getLogger(__name__)
        except Exception:
            # Fallback logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)

    def _setup_mlflow(self):
        """Setup MLflow tracking if configured."""
        try:
            mlflow_config = self.config.get('model_registry', {})
            if mlflow_config.get('enabled', False):
                tracking_uri = mlflow_config.get('tracking_uri', 'mlruns')
                experiment_name = mlflow_config.get('experiment_name', 'diabetes_readmission')
                
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)
                
                self.logger.info(f"✅ MLflow tracking setup: {tracking_uri}")
                return True
            else:
                self.logger.info("MLflow tracking disabled in config")
                return False
        except Exception as e:
            self.logger.warning(f"MLflow setup failed: {str(e)}. Continuing without MLflow.")
            return False

    def _create_model(self) -> Any:
        """Create model instance based on configuration."""
        try:
            model_config = self.config.get('model', {})
            self.model_type = model_config.get('type', 'random_forest')
            self.model_params = model_config.get('parameters', {})
            
            self.logger.info(f"Creating model: {self.model_type}")
            self.logger.info(f"Model parameters: {self.model_params}")
            
            if self.model_type == 'random_forest':
                model = RandomForestClassifier(**self.model_params)
            elif self.model_type == 'logistic_regression':
                model = LogisticRegression(**self.model_params)
            elif self.model_type == 'decision_tree':
                model = DecisionTreeClassifier(**self.model_params)
            elif self.model_type == 'naive_bayes':
                # GaussianNB doesn't take many parameters
                nb_params = {k: v for k, v in self.model_params.items() 
                           if k in ['priors', 'var_smoothing']}
                model = GaussianNB(**nb_params)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.logger.info(f"✅ Model {self.model_type} created successfully")
            return model
            
        except Exception as e:
            self.logger.error(f"Error creating model: {str(e)}")
            raise

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the model on training data."""
        try:
            self.logger.info("🔧 Starting model training...")
            
            # Validate input
            if X_train.empty:
                raise ValueError("Cannot train on empty training data")
            if len(X_train) != len(y_train):
                raise ValueError("Training features and labels must have same length")
            
            # Initialize report
            self.report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'model_type': self.model_type,
                'model_parameters': self.model_params,
                'training_shape': X_train.shape
            }
            
            # Setup MLflow
            mlflow_enabled = self._setup_mlflow()
            
            # Create model
            self.model = self._create_model()
            
            # Convert to numpy arrays if needed
            X_train_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
            y_train_array = y_train.values if isinstance(y_train, pd.Series) else y_train
            
            if mlflow_enabled:
                with mlflow.start_run():
                    # Log parameters
                    mlflow.log_params(self.model_params)
                    mlflow.log_param("model_type", self.model_type)
                    mlflow.log_param("training_samples", len(X_train))
                    mlflow.log_param("n_features", X_train.shape[1])
                    
                    # Train model
                    self.logger.info(f"Training {self.model_type} on {len(X_train)} samples with {X_train.shape[1]} features...")
                    self.model.fit(X_train_array, y_train_array)
                    
                    # Log model
                    mlflow.sklearn.log_model(self.model, "model")
                    
            else:
                # Train without MLflow
                self.logger.info(f"Training {self.model_type} on {len(X_train)} samples with {X_train.shape[1]} features...")
                self.model.fit(X_train_array, y_train_array)
            
            # Mark as fitted
            self._is_fitted = True
            
            # Update report
            self.report.update({
                'training_completed': True,
                'n_features': X_train.shape[1],
                'n_samples': len(X_train)
            })
            
            # Write report
            self._write_report()
            
            # Log results
            self.logger.info("✅ Model training completed successfully!")
            self.logger.info(f"Model trained on {len(X_train)} samples with {X_train.shape[1]} features")
            
        except Exception as e:
            self.logger.error(f"❌ Model training failed: {str(e)}")
            self.report['training_failed'] = True
            self.report['error'] = str(e)
            self._write_report()
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        try:
            if not self._is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            predictions = self.model.predict(X_array)
            
            self.logger.info(f"Made predictions on {len(X)} samples")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Make probability predictions on new data."""
        try:
            if not self._is_fitted:
                raise ValueError("Model must be fitted before making predictions")
            
            if not hasattr(self.model, 'predict_proba'):
                raise ValueError(f"Model {self.model_type} does not support probability predictions")
            
            X_array = X.values if isinstance(X, pd.DataFrame) else X
            probabilities = self.model.predict_proba(X_array)
            
            self.logger.info(f"Made probability predictions on {len(X)} samples")
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Error making probability predictions: {str(e)}")
            raise

    def save(self, model_path: str = None) -> str:
        """Save the trained model to disk."""
        try:
            if not self._is_fitted:
                raise ValueError("Cannot save model that hasn't been fitted")
            
            if model_path is None:
                model_dir = self.config.get('data', {}).get('model_path', 'models')
                model_path = os.path.join(model_dir, f"{self.model_type}_model.joblib")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, model_path)
            
            # Save model metadata
            metadata = {
                'model_type': self.model_type,
                'model_parameters': self.model_params,
                'training_shape': self.report.get('training_shape'),
                'n_features': self.report.get('n_features'),
                'n_samples': self.report.get('n_samples'),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"✅ Model saved to: {model_path}")
            self.logger.info(f"✅ Metadata saved to: {metadata_path}")
            
            return model_path
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load(self, model_path: str) -> None:
        """Load a trained model from disk."""
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            self.model = joblib.load(model_path)
            self._is_fitted = True
            
            # Load metadata if available
            metadata_path = model_path.replace('.joblib', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.model_type = metadata.get('model_type', 'unknown')
                self.model_params = metadata.get('model_parameters', {})
            
            self.logger.info(f"✅ Model loaded from: {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance if available."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        try:
            if hasattr(self.model, 'feature_importances_'):
                # For tree-based models
                importance_df = pd.DataFrame({
                    'feature': range(len(self.model.feature_importances_)),
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                return importance_df
            elif hasattr(self.model, 'coef_'):
                # For linear models
                importance_df = pd.DataFrame({
                    'feature': range(len(self.model.coef_[0])),
                    'importance': np.abs(self.model.coef_[0])
                }).sort_values('importance', ascending=False)
                
                return importance_df
            else:
                self.logger.warning(f"Model {self.model_type} does not support feature importance")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            raise

    def _write_report(self):
        """Write training report to file."""
        try:
            report_dir = Path("logs")
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / "training_report.json"
            with open(report_file, "w") as f:
                json.dump(self.report, f, indent=2, default=str)
                
            self.logger.info(f"📋 Training report written to: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to write training report: {str(e)}")


if __name__ == "__main__":
    """Demonstrate ModelTrainer functionality."""
    print("Model training module loaded successfully.")
    print("Usage examples:")
    print("  from src.model.trainer import ModelTrainer")
    print("  trainer = ModelTrainer()")
    print("  trainer.fit(X_train, y_train)")
    print("  trainer.save()")
    print("  predictions = trainer.predict(X_test)")
"""
Model evaluation module for the MLOps project.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
import json
import logging
import yaml
from pathlib import Path

class ModelEvaluator:
    def __init__(self, config_path: str = "src/config.yaml"):
        """Initialize the model evaluator."""
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.metrics = {}

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

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Calculate ROC AUC if probabilities are available
            if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred[:, 1])
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        try:
            return confusion_matrix(y_true, y_pred)
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {str(e)}")
            raise

    def calculate_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
        """Calculate ROC curve."""
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            # Filter out infinite thresholds
            mask = np.isfinite(thresholds)
            return fpr[mask], tpr[mask], thresholds[mask]
        except Exception as e:
            self.logger.error(f"Error calculating ROC curve: {str(e)}")
            raise

    def calculate_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
        """Calculate precision-recall curve."""
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            # Add threshold for precision=1, recall=0
            thresholds = np.append(thresholds, 1.0)
            return precision, recall, thresholds
        except Exception as e:
            self.logger.error(f"Error calculating precision-recall curve: {str(e)}")
            raise

    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate classification report."""
        try:
            return classification_report(y_true, y_pred)
        except Exception as e:
            self.logger.error(f"Error generating classification report: {str(e)}")
            raise

    def save_metrics(self, metrics: dict, path: str) -> None:
        """Save evaluation metrics to file."""
        try:
            with open(path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Metrics saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise 
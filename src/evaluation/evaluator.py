"""
Model evaluation module for the MLOps project.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import json
import logging

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        self.logger = logging.getLogger(__name__)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
        """Calculate evaluation metrics."""
        try:
            self.metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, pos_label='YES'),
                'recall': recall_score(y_true, y_pred, pos_label='YES'),
                'f1_score': f1_score(y_true, y_pred, pos_label='YES'),
                'roc_auc': roc_auc_score(y_true == 'YES', y_prob[:, 1])
            }
            return self.metrics
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            raise

    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix."""
        return confusion_matrix(y_true, y_pred)

    def calculate_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
        """Calculate ROC curve."""
        return roc_curve(y_true == 'YES', y_prob)

    def calculate_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
        """Calculate precision-recall curve."""
        return precision_recall_curve(y_true == 'YES', y_prob)

    def save_metrics(self, metrics: dict, path: str) -> None:
        """Save evaluation metrics to file."""
        try:
            with open(path, 'w') as f:
                json.dump(metrics, f, indent=4)
            self.logger.info(f"Metrics saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise 
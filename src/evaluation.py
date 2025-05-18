import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Any, List
from .utils import setup_logging, load_config, plot_confusion_matrix, save_plot

logger = setup_logging()

class ModelEvaluator:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = load_config(config_path)
        self.evaluation_config = self.config['evaluation']
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute various classification metrics."""
        try:
            metrics = {}
            
            # Compute basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            
            # Log metrics
            for metric_name, value in metrics.items():
                logger.info(f"{metric_name}: {value:.4f}")
            
            return metrics
        except Exception as e:
            logger.error(f"Error computing metrics: {str(e)}")
            raise
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate a detailed classification report."""
        try:
            report = classification_report(y_true, y_pred)
            logger.info("\nClassification Report:\n" + report)
            return report
        except Exception as e:
            logger.error(f"Error generating classification report: {str(e)}")
            raise
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None) -> None:
        """Plot and save confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            plt = plot_confusion_matrix(cm, labels)
            save_plot(plt, 'confusion_matrix.png')
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str] = None) -> Dict[str, Any]:
        """Run complete model evaluation."""
        try:
            # Compute metrics
            metrics = self.compute_metrics(y_true, y_pred)
            
            # Generate classification report
            report = self.generate_classification_report(y_true, y_pred)
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y_true, y_pred, labels)
            
            return {
                'metrics': metrics,
                'classification_report': report
            }
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise 
"""
Model evaluation class for the MLOps project.
Supports dynamic metric selection, logging, and artifact saving.
"""

import logging
import json
import yaml
from pathlib import Path

from src.evaluation import metrics


class ModelEvaluator:
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Initialize the ModelEvaluator with logging and metrics setup.
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self.metric_names = self.config.get("evaluation", {}).get("metrics", ["accuracy", "precision", "recall", "f1"])
        self.save_path = self.config.get("artifacts", {}).get("metrics", "models/metrics.json")

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        logging.basicConfig(
            level=self.config.get("logging", {}).get("level", "INFO"),
            format=self.config.get("logging", {}).get("format", "%(asctime)s - %(levelname)s - %(message)s"),
            filename=self.config.get("logging", {}).get("file")
        )

    def evaluate(self, y_true, y_pred, y_prob=None) -> dict:
        """
        Evaluate model predictions using configured metrics.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted class labels.
            y_prob (np.ndarray): Predicted class probabilities (optional, required for AUC/curves).

        Returns:
            dict: Dictionary of computed metrics.
        """
        results = {}

        for name in self.metric_names:
            try:
                if name == "accuracy":
                    results["Accuracy"] = metrics.compute_accuracy(y_true, y_pred)
                elif name == "precision":
                    results["Precision (PPV)"] = metrics.compute_precision(y_true, y_pred)
                elif name == "recall":
                    results["Recall (Sensitivity)"] = metrics.compute_recall(y_true, y_pred)
                elif name == "f1":
                    results["F1 Score"] = metrics.compute_f1(y_true, y_pred)
                elif name == "roc_auc":
                    if y_prob is not None:
                        results["ROC AUC"] = metrics.compute_roc_auc(y_true, y_prob)
                elif name == "specificity":
                    cm = metrics.compute_confusion_matrix(y_true, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    results["Specificity"] = tn / (tn + fp)
                elif name == "npv":
                    cm = metrics.compute_confusion_matrix(y_true, y_pred)
                    tn, fp, fn, tp = cm.ravel()
                    results["NPV"] = tn / (tn + fn)
            except Exception as e:
                self.logger.error(f"Error calculating {name}: {str(e)}")
                results[name] = None  # Fail-safe

        return results

    def save_metrics(self, metrics_dict: dict, split: str = "test"):
        """
        Save evaluation metrics to JSON.

        Args:
            metrics_dict (dict): Metrics dictionary.
            split (str): Name of the dataset split (e.g., 'test' or 'validation').
        """
        try:
            path = Path(self.save_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                with open(path, "r") as f:
                    full_metrics = json.load(f)
            else:
                full_metrics = {}

            full_metrics[split] = metrics_dict

            with open(path, "w") as f:
                json.dump(full_metrics, f, indent=4)

            self.logger.info(f"Metrics saved to {path}")

        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise

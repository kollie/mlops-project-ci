"""
Model Evaluation Module for MLOps Project.

Handles model evaluation, metrics calculation, and reporting following the same patterns
as other modules in the pipeline.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)
import mlflow
import mlflow.sklearn


class ModelEvaluator:
    """
    Model evaluation pipeline for the MLOps project.

    Handles model evaluation, metrics calculation, and reporting following the same
    patterns as other modules in the pipeline.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.report = {}

        # Evaluation components
        self.metrics: Dict[str, float] = {}
        self.confusion_matrix: Optional[np.ndarray] = None
        self.classification_report: str = ""

        # Curve data
        self.roc_curve_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.pr_curve_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate required config sections
            required_sections = ["data", "model", "logging"]
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
            log_config = self.config.get("logging", {})
            log_file = log_config.get("file", "logs/evaluation.log").replace(
                "main.log", "evaluation.log"
            )
            log_level = log_config.get("level", "INFO")
            log_format = log_config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Configure logging
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format=log_format,
                handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            )

            return logging.getLogger(__name__)
        except Exception:
            # Fallback logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)

    def _setup_mlflow(self):
        """Setup MLflow tracking if configured."""
        try:
            mlflow_config = self.config.get("model_registry", {})
            if mlflow_config.get("enabled", False):
                tracking_uri = mlflow_config.get("tracking_uri", "mlruns")
                experiment_name = mlflow_config.get(
                    "experiment_name", "diabetes_readmission"
                )

                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(experiment_name)

                self.logger.info(f"✅ MLflow tracking setup: {tracking_uri}")
                return True
            else:
                self.logger.info("MLflow tracking disabled in config")
                return False
        except Exception as e:
            self.logger.warning(
                f"MLflow setup failed: {str(e)}. Continuing without MLflow."
            )
            return False

    def _calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        try:
            metrics = {}

            # Basic metrics
            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["recall"] = recall_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
            metrics["f1_score"] = f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )

            # ROC AUC if probabilities are available
            if y_pred_proba is not None:
                try:
                    if len(np.unique(y_true)) == 2:  # Binary classification
                        if len(y_pred_proba.shape) == 2:
                            metrics["roc_auc"] = roc_auc_score(
                                y_true, y_pred_proba[:, 1]
                            )
                        else:
                            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
                    else:  # Multiclass
                        metrics["roc_auc"] = roc_auc_score(
                            y_true, y_pred_proba, multi_class="ovr", average="weighted"
                        )
                except Exception as e:
                    self.logger.warning(f"Could not calculate ROC AUC: {e}")

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {str(e)}")
            raise

    def _calculate_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> np.ndarray:
        """Calculate confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            self.logger.info(f"Confusion matrix calculated: shape {cm.shape}")
            return cm
        except Exception as e:
            self.logger.error(f"Error calculating confusion matrix: {str(e)}")
            raise

    def _generate_classification_report(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> str:
        """Generate detailed classification report."""
        try:
            report = classification_report(y_true, y_pred)
            self.logger.info("Classification report generated")
            return report
        except Exception as e:
            self.logger.error(f"Error generating classification report: {str(e)}")
            raise

    def _calculate_roc_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate ROC curve data."""
        try:
            if len(np.unique(y_true)) != 2:
                self.logger.warning(
                    "ROC curve calculation only supported for binary classification"
                )
                return None, None, None

            # Handle probability format
            if len(y_pred_proba.shape) == 2:
                y_prob = y_pred_proba[:, 1]  # Positive class probabilities
            else:
                y_prob = y_pred_proba

            fpr, tpr, thresholds = roc_curve(y_true, y_prob)

            # Filter out infinite thresholds
            mask = np.isfinite(thresholds)
            fpr_clean = fpr[mask]
            tpr_clean = tpr[mask]
            thresholds_clean = thresholds[mask]

            self.logger.info(
                f"ROC curve calculated with {len(thresholds_clean)} points"
            )
            return fpr_clean, tpr_clean, thresholds_clean

        except Exception as e:
            self.logger.error(f"Error calculating ROC curve: {str(e)}")
            raise

    def _calculate_precision_recall_curve(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate precision-recall curve data."""
        try:
            if len(np.unique(y_true)) != 2:
                self.logger.warning(
                    "PR curve calculation only supported for binary classification"
                )
                return None, None, None

            # Handle probability format
            if len(y_pred_proba.shape) == 2:
                y_prob = y_pred_proba[:, 1]  # Positive class probabilities
            else:
                y_prob = y_pred_proba

            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

            # Add threshold for precision=1, recall=0 point
            thresholds = np.append(thresholds, 1.0)

            self.logger.info(f"PR curve calculated with {len(thresholds)} points")
            return precision, recall, thresholds

        except Exception as e:
            self.logger.error(f"Error calculating precision-recall curve: {str(e)}")
            raise

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None,
        dataset_name: str = "test",
    ) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        try:
            self.logger.info(f"🔍 Starting model evaluation on {dataset_name} data...")

            # Validate inputs
            if len(y_true) != len(y_pred):
                raise ValueError("y_true and y_pred must have the same length")
            if len(y_true) == 0:
                raise ValueError("Cannot evaluate on empty data")

            # Initialize report with proper type conversion
            unique_labels, label_counts = np.unique(y_true, return_counts=True)
            self.report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "dataset_name": dataset_name,
                "n_samples": int(len(y_true)),
                "n_classes": int(len(unique_labels)),
                "class_distribution": {
                    str(int(k)): int(v) for k, v in zip(unique_labels, label_counts)
                },
            }

            # Rest of the method remains the same...
            # Setup MLflow
            mlflow_enabled = self._setup_mlflow()

            if mlflow_enabled:
                with mlflow.start_run():
                    # Calculate metrics
                    self.metrics = self._calculate_basic_metrics(
                        y_true, y_pred, y_pred_proba
                    )
                    self.confusion_matrix = self._calculate_confusion_matrix(
                        y_true, y_pred
                    )
                    self.classification_report = self._generate_classification_report(
                        y_true, y_pred
                    )

                    # Calculate curves if probabilities available
                    if y_pred_proba is not None:
                        self.roc_curve_data = self._calculate_roc_curve(
                            y_true, y_pred_proba
                        )
                        self.pr_curve_data = self._calculate_precision_recall_curve(
                            y_true, y_pred_proba
                        )

                    # Log metrics to MLflow
                    mlflow.log_metrics(self.metrics)
                    mlflow.log_param("dataset", dataset_name)
                    mlflow.log_param("n_samples", len(y_true))

            else:
                # Evaluate without MLflow
                self.metrics = self._calculate_basic_metrics(
                    y_true, y_pred, y_pred_proba
                )
                self.confusion_matrix = self._calculate_confusion_matrix(y_true, y_pred)
                self.classification_report = self._generate_classification_report(
                    y_true, y_pred
                )

                # Calculate curves if probabilities available
                if y_pred_proba is not None:
                    self.roc_curve_data = self._calculate_roc_curve(
                        y_true, y_pred_proba
                    )
                    self.pr_curve_data = self._calculate_precision_recall_curve(
                        y_true, y_pred_proba
                    )

            # Update report
            self.report.update(
                {
                    "metrics": self.metrics,
                    "confusion_matrix": self.confusion_matrix.tolist()
                    if self.confusion_matrix is not None
                    else None,
                    "classification_report": self.classification_report,
                    "evaluation_completed": True,
                }
            )

            # Write report
            self._write_report()

            # Log results
            self.logger.info("✅ Model evaluation completed successfully!")
            self.logger.info(f"Evaluation results for {dataset_name}:")
            for metric, value in self.metrics.items():
                self.logger.info(f"  {metric}: {value:.4f}")

            return {
                "metrics": self.metrics,
                "confusion_matrix": self.confusion_matrix,
                "classification_report": self.classification_report,
                "roc_curve_data": self.roc_curve_data,
                "pr_curve_data": self.pr_curve_data,
            }

        except Exception as e:
            self.logger.error(f"❌ Model evaluation failed: {str(e)}")
            self.report["evaluation_failed"] = True
            self.report["error"] = str(e)
            self._write_report()
            raise

    def get_metrics(self) -> Dict[str, float]:
        """Get calculated metrics."""
        if not self.metrics:
            raise ValueError("No metrics available. Run evaluate() first.")
        return self.metrics.copy()

    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if self.confusion_matrix is None:
            raise ValueError("No confusion matrix available. Run evaluate() first.")
        return self.confusion_matrix.copy()

    def get_classification_report(self) -> str:
        """Get classification report."""
        if not self.classification_report:
            raise ValueError(
                "No classification report available. Run evaluate() first."
            )
        return self.classification_report

    def get_roc_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data."""
        if self.roc_curve_data is None or self.roc_curve_data[0] is None:
            raise ValueError(
                "No ROC curve data available. Ensure probabilities were provided and it's binary classification."
            )
        return self.roc_curve_data

    def get_pr_curve_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get precision-recall curve data."""
        if self.pr_curve_data is None or self.pr_curve_data[0] is None:
            raise ValueError(
                "No PR curve data available. Ensure probabilities were provided and it's binary classification."
            )
        return self.pr_curve_data

    def save_metrics(self, file_path: str = None) -> str:
        """Save metrics to JSON file."""
        try:
            if not self.metrics:
                raise ValueError("No metrics to save. Run evaluate() first.")

            if file_path is None:
                metrics_dir = self.config.get("data", {}).get(
                    "processed_data_path", "data/processed"
                )
                file_path = os.path.join(metrics_dir, "evaluation_metrics.json")

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Prepare data for saving
            save_data = {
                "metrics": self.metrics,
                "confusion_matrix": self.confusion_matrix.tolist()
                if self.confusion_matrix is not None
                else None,
                "classification_report": self.classification_report,
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            # Save to file
            with open(file_path, "w") as f:
                json.dump(save_data, f, indent=2, default=str)

            self.logger.info(f"✅ Metrics saved to: {file_path}")
            return file_path

        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")
            raise

    def compare_models(
        self, other_metrics: Dict[str, float], primary_metric: str = "f1_score"
    ) -> Dict[str, Any]:
        """Compare this model's metrics with another model."""
        try:
            if not self.metrics:
                raise ValueError(
                    "No metrics available for comparison. Run evaluate() first."
                )

            if primary_metric not in self.metrics:
                raise ValueError(
                    f"Primary metric '{primary_metric}' not found in current metrics"
                )
            if primary_metric not in other_metrics:
                raise ValueError(
                    f"Primary metric '{primary_metric}' not found in comparison metrics"
                )

            # Ensure we're working with Python float types
            current_value = float(self.metrics[primary_metric])
            other_value = float(other_metrics[primary_metric])

            comparison = {
                "current_model": self.metrics,
                "other_model": other_metrics,
                "differences": {},
                "better_metrics": [],
                "worse_metrics": [],
                "primary_metric": primary_metric,
                "is_better": bool(current_value > other_value),  # Ensure Python bool
            }

            # Calculate differences
            for metric in self.metrics:
                if metric in other_metrics:
                    diff = float(self.metrics[metric]) - float(other_metrics[metric])
                    comparison["differences"][metric] = diff

                    if diff > 0:
                        comparison["better_metrics"].append(metric)
                    elif diff < 0:
                        comparison["worse_metrics"].append(metric)

            self.logger.info(
                f"Model comparison completed. Current model is {'better' if comparison['is_better'] else 'worse'} on {primary_metric}"
            )

            return comparison

        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            raise

    def _write_report(self):
        """Write evaluation report to file."""
        try:
            report_dir = Path("logs")
            report_dir.mkdir(exist_ok=True)

            report_file = report_dir / "evaluation_report.json"

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to native Python types."""
                if isinstance(obj, dict):
                    return {
                        str(key): convert_numpy_types(value)
                        for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, "item"):  # Handle numpy scalars
                    return obj.item()
                else:
                    return obj

            # Convert the report to handle numpy types
            report_converted = convert_numpy_types(self.report)

            with open(report_file, "w") as f:
                json.dump(report_converted, f, indent=2, default=str)

            self.logger.info(f"📋 Evaluation report written to: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to write evaluation report: {str(e)}")


if __name__ == "__main__":
    """Demonstrate ModelEvaluator functionality."""
    print("Model evaluation module loaded successfully.")
    print("Usage examples:")
    print("  from src.evaluation.evaluator import ModelEvaluator")
    print("  evaluator = ModelEvaluator()")
    print("  results = evaluator.evaluate(y_true, y_pred, y_pred_proba)")
    print("  metrics = evaluator.get_metrics()")
    print("  evaluator.save_metrics()")

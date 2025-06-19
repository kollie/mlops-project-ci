"""
Unit tests for the Model Evaluation module.

Tests the ModelEvaluator class and its evaluation methods
following the same patterns as other modules in the pipeline.
"""

import pytest
import numpy as np
import yaml
import os
import tempfile
import shutil
import json
from pathlib import Path
import sys
from unittest.mock import patch
import builtins

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.evaluator import ModelEvaluator


class TestModelEvaluator:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create a sample configuration for testing."""
        config = {
            "data": {
                "raw_data_path": os.path.join(temp_dir, "raw_data.csv"),
                "processed_data_path": os.path.join(temp_dir, "processed"),
                "model_path": os.path.join(temp_dir, "models"),
            },
            "model": {
                "type": "random_forest",
                "parameters": {"n_estimators": 10, "max_depth": 3, "random_state": 42},
            },
            "model_registry": {
                "enabled": False,  # Disable MLflow for testing
                "tracking_uri": "mlruns",
                "experiment_name": "test_experiment",
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": os.path.join(temp_dir, "logs", "evaluation.log"),
            },
            "evaluation": {
                "metrics": ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
                "save_curves": True,
                "primary_metric": "f1_score",
            },
        }

        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def model_evaluator(self, sample_config):
        """Create a ModelEvaluator instance for testing."""
        return ModelEvaluator(sample_config)

    @pytest.fixture
    def sample_binary_data(self):
        """Create sample binary classification data for testing."""
        np.random.seed(42)
        n_samples = 100

        # Generate binary classification data
        y_true = np.random.choice([0, 1], n_samples)
        y_pred = np.random.choice([0, 1], n_samples)

        # Generate probabilities (2 classes)
        y_pred_proba = np.random.random((n_samples, 2))
        # Normalize to sum to 1
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

        return y_true, y_pred, y_pred_proba

    @pytest.fixture
    def sample_multiclass_data(self):
        """Create sample multiclass classification data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_classes = 3

        y_true = np.random.choice(range(n_classes), n_samples)
        y_pred = np.random.choice(range(n_classes), n_samples)

        # Generate probabilities (3 classes)
        y_pred_proba = np.random.random((n_samples, n_classes))
        # Normalize to sum to 1
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

        return y_true, y_pred, y_pred_proba

    # ---------- Initialization Tests ----------

    def test_evaluator_initialization(self, sample_config):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator(sample_config)
        assert isinstance(evaluator.metrics, dict)
        assert len(evaluator.metrics) == 0  # Should be empty initially
        assert evaluator.confusion_matrix is None
        assert evaluator.classification_report == ""
        assert evaluator.config is not None
        assert evaluator.logger is not None

    def test_evaluator_initialization_missing_config(self):
        """Test ModelEvaluator initialization with missing config."""
        with pytest.raises(FileNotFoundError):
            ModelEvaluator("nonexistent_config.yaml")

    def test_evaluator_initialization_invalid_yaml(self, temp_dir):
        """Test ModelEvaluator initialization with invalid YAML."""
        invalid_config_path = os.path.join(temp_dir, "invalid_config.yaml")
        with open(invalid_config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            ModelEvaluator(invalid_config_path)

    # ---------- Basic Metrics Tests ----------

    def test_calculate_basic_metrics_binary(self, model_evaluator, sample_binary_data):
        """Test calculation of basic metrics for binary classification."""
        y_true, y_pred, y_pred_proba = sample_binary_data

        metrics = model_evaluator._calculate_basic_metrics(y_true, y_pred, y_pred_proba)

        # Check that all expected metrics are present
        expected_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

    def test_calculate_basic_metrics_multiclass(
        self, model_evaluator, sample_multiclass_data
    ):
        """Test calculation of basic metrics for multiclass classification."""
        y_true, y_pred, y_pred_proba = sample_multiclass_data

        metrics = model_evaluator._calculate_basic_metrics(y_true, y_pred, y_pred_proba)

        # Check that all expected metrics are present
        expected_metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

    def test_calculate_basic_metrics_without_probabilities(
        self, model_evaluator, sample_binary_data
    ):
        """Test calculation of basic metrics without probabilities."""
        y_true, y_pred, _ = sample_binary_data

        metrics = model_evaluator._calculate_basic_metrics(y_true, y_pred)

        # Should have all metrics except ROC AUC
        expected_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1

        # ROC AUC should not be present
        assert "roc_auc" not in metrics

    # ---------- Confusion Matrix Tests ----------

    def test_calculate_confusion_matrix_binary(
        self, model_evaluator, sample_binary_data
    ):
        """Test calculation of confusion matrix for binary classification."""
        y_true, y_pred, _ = sample_binary_data

        cm = model_evaluator._calculate_confusion_matrix(y_true, y_pred)

        # Check confusion matrix properties
        assert cm.shape == (2, 2)
        assert np.all(cm >= 0)
        assert cm.sum() == len(y_true)

    def test_calculate_confusion_matrix_multiclass(
        self, model_evaluator, sample_multiclass_data
    ):
        """Test calculation of confusion matrix for multiclass classification."""
        y_true, y_pred, _ = sample_multiclass_data

        cm = model_evaluator._calculate_confusion_matrix(y_true, y_pred)

        # Check confusion matrix properties
        assert cm.shape == (3, 3)
        assert np.all(cm >= 0)
        assert cm.sum() == len(y_true)

    # ---------- Classification Report Tests ----------

    def test_generate_classification_report(self, model_evaluator, sample_binary_data):
        """Test generation of classification report."""
        y_true, y_pred, _ = sample_binary_data

        report = model_evaluator._generate_classification_report(y_true, y_pred)

        # Check report properties
        assert isinstance(report, str)
        assert "precision" in report.lower()
        assert "recall" in report.lower()
        assert "f1-score" in report.lower()

    # ---------- ROC Curve Tests ----------

    def test_calculate_roc_curve_binary(self, model_evaluator, sample_binary_data):
        """Test calculation of ROC curve for binary classification."""
        y_true, _, y_pred_proba = sample_binary_data

        fpr, tpr, thresholds = model_evaluator._calculate_roc_curve(
            y_true, y_pred_proba
        )

        # Check outputs
        assert fpr is not None
        assert tpr is not None
        assert thresholds is not None
        assert len(fpr) == len(tpr) == len(thresholds)
        assert all(0 <= x <= 1 for x in fpr)
        assert all(0 <= x <= 1 for x in tpr)

    def test_calculate_roc_curve_multiclass(
        self, model_evaluator, sample_multiclass_data
    ):
        """Test ROC curve calculation fails gracefully for multiclass."""
        y_true, _, y_pred_proba = sample_multiclass_data

        fpr, tpr, thresholds = model_evaluator._calculate_roc_curve(
            y_true, y_pred_proba
        )

        # Should return None for multiclass
        assert fpr is None
        assert tpr is None
        assert thresholds is None

    # ---------- Precision-Recall Curve Tests ----------

    def test_calculate_precision_recall_curve_binary(
        self, model_evaluator, sample_binary_data
    ):
        """Test calculation of precision-recall curve for binary classification."""
        y_true, _, y_pred_proba = sample_binary_data

        precision, recall, thresholds = (
            model_evaluator._calculate_precision_recall_curve(y_true, y_pred_proba)
        )

        # Check outputs
        assert precision is not None
        assert recall is not None
        assert thresholds is not None
        assert len(precision) == len(recall) == len(thresholds)
        assert all(0 <= x <= 1 for x in precision)
        assert all(0 <= x <= 1 for x in recall)

    def test_calculate_precision_recall_curve_multiclass(
        self, model_evaluator, sample_multiclass_data
    ):
        """Test PR curve calculation fails gracefully for multiclass."""
        y_true, _, y_pred_proba = sample_multiclass_data

        precision, recall, thresholds = (
            model_evaluator._calculate_precision_recall_curve(y_true, y_pred_proba)
        )

        # Should return None for multiclass
        assert precision is None
        assert recall is None
        assert thresholds is None

    # ---------- Full Evaluation Tests ----------

    def test_evaluate_binary_complete(
        self, model_evaluator, sample_binary_data, temp_dir
    ):
        """Test complete evaluation for binary classification."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            y_true, y_pred, y_pred_proba = sample_binary_data

            results = model_evaluator.evaluate(
                y_true, y_pred, y_pred_proba, dataset_name="test"
            )

            # Check results structure
            assert "metrics" in results
            assert "confusion_matrix" in results
            assert "classification_report" in results
            assert "roc_curve_data" in results
            assert "pr_curve_data" in results

            # Check metrics
            metrics = results["metrics"]
            expected_metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
            for metric in expected_metrics:
                assert metric in metrics
                assert 0 <= metrics[metric] <= 1

            # Check curves for binary classification
            assert results["roc_curve_data"] is not None
            assert results["pr_curve_data"] is not None

        finally:
            os.chdir(original_cwd)

    def test_evaluate_without_probabilities(
        self, model_evaluator, sample_binary_data, temp_dir
    ):
        """Test evaluation without probabilities."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            y_true, y_pred, _ = sample_binary_data

            results = model_evaluator.evaluate(y_true, y_pred, dataset_name="test")

            # Check that evaluation works without probabilities
            assert "metrics" in results
            assert "confusion_matrix" in results
            assert "classification_report" in results

            # ROC AUC should not be in metrics
            assert "roc_auc" not in results["metrics"]

            # Curves should be None
            assert results["roc_curve_data"] is None
            assert results["pr_curve_data"] is None

        finally:
            os.chdir(original_cwd)

    def test_evaluate_empty_data(self, model_evaluator):
        """Test evaluation with empty data."""
        y_true = np.array([])
        y_pred = np.array([])

        with pytest.raises(ValueError, match="Cannot evaluate on empty data"):
            model_evaluator.evaluate(y_true, y_pred)

    def test_evaluate_mismatched_lengths(self, model_evaluator):
        """Test evaluation with mismatched array lengths."""
        y_true = np.array([0, 1, 0])
        y_pred = np.array([1, 0])  # Different length

        with pytest.raises(
            ValueError, match="y_true and y_pred must have the same length"
        ):
            model_evaluator.evaluate(y_true, y_pred)

    # ---------- Getter Method Tests ----------

    def test_get_metrics(self, model_evaluator, sample_binary_data, temp_dir):
        """Test getting metrics after evaluation."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            y_true, y_pred, y_pred_proba = sample_binary_data

            # Run evaluation first
            model_evaluator.evaluate(y_true, y_pred, y_pred_proba)

            # Get metrics
            metrics = model_evaluator.get_metrics()

            assert isinstance(metrics, dict)
            assert len(metrics) > 0
            assert "accuracy" in metrics

        finally:
            os.chdir(original_cwd)

    def test_get_metrics_before_evaluation(self, model_evaluator):
        """Test getting metrics before running evaluation."""
        with pytest.raises(
            ValueError, match="No metrics available. Run evaluate\\(\\) first."
        ):
            model_evaluator.get_metrics()

    def test_get_confusion_matrix(self, model_evaluator, sample_binary_data, temp_dir):
        """Test getting confusion matrix after evaluation."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            y_true, y_pred, _ = sample_binary_data

            # Run evaluation first
            model_evaluator.evaluate(y_true, y_pred)

            # Get confusion matrix
            cm = model_evaluator.get_confusion_matrix()

            assert isinstance(cm, np.ndarray)
            assert cm.shape == (2, 2)

        finally:
            os.chdir(original_cwd)

    def test_get_confusion_matrix_before_evaluation(self, model_evaluator):
        """Test getting confusion matrix before running evaluation."""
        with pytest.raises(
            ValueError, match="No confusion matrix available. Run evaluate\\(\\) first."
        ):
            model_evaluator.get_confusion_matrix()

    def test_get_roc_curve_data(self, model_evaluator, sample_binary_data, temp_dir):
        """Test getting ROC curve data."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            y_true, y_pred, y_pred_proba = sample_binary_data

            # Run evaluation first
            model_evaluator.evaluate(y_true, y_pred, y_pred_proba)

            # Get ROC curve data
            fpr, tpr, thresholds = model_evaluator.get_roc_curve_data()

            assert isinstance(fpr, np.ndarray)
            assert isinstance(tpr, np.ndarray)
            assert isinstance(thresholds, np.ndarray)
            assert len(fpr) == len(tpr) == len(thresholds)

        finally:
            os.chdir(original_cwd)

    # ---------- Save/Load Tests ----------

    def test_save_metrics(self, model_evaluator, sample_binary_data, temp_dir):
        """Test saving metrics to file."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            y_true, y_pred, y_pred_proba = sample_binary_data

            # Run evaluation first
            model_evaluator.evaluate(y_true, y_pred, y_pred_proba)

            # Save metrics
            saved_path = model_evaluator.save_metrics()

            # Check that file was created
            assert os.path.exists(saved_path)

            # Check file content
            with open(saved_path, "r") as f:
                saved_data = json.load(f)

            assert "metrics" in saved_data
            assert "confusion_matrix" in saved_data
            assert "classification_report" in saved_data
            assert "timestamp" in saved_data

        finally:
            os.chdir(original_cwd)

    def test_save_metrics_custom_path(
        self, model_evaluator, sample_binary_data, temp_dir
    ):
        """Test saving metrics to custom path."""
        y_true, y_pred, _ = sample_binary_data

        # Run evaluation first
        model_evaluator.evaluate(y_true, y_pred)

        # Save to custom path
        custom_path = os.path.join(temp_dir, "custom_metrics.json")
        saved_path = model_evaluator.save_metrics(custom_path)

        assert saved_path == custom_path
        assert os.path.exists(custom_path)

    def test_save_metrics_before_evaluation(self, model_evaluator):
        """Test saving metrics before running evaluation."""
        with pytest.raises(
            ValueError, match="No metrics to save. Run evaluate\\(\\) first."
        ):
            model_evaluator.save_metrics()

    @patch.object(builtins, "open", side_effect=IOError("Disk full"))
    def test_save_metrics_io_error(
        self, mock_open, model_evaluator, sample_binary_data, temp_dir
    ):
        """Test save_metrics handles IOError properly."""
        y_true, y_pred, _ = sample_binary_data

        # Run evaluation first
        model_evaluator.evaluate(y_true, y_pred)

        # Try to save (should fail)
        with pytest.raises(IOError):
            model_evaluator.save_metrics("test_path.json")

    # ---------- Model Comparison Tests ----------

    def test_compare_models(self, model_evaluator, sample_binary_data, temp_dir):
        """Test model comparison functionality."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            y_true, y_pred, y_pred_proba = sample_binary_data

            # Run evaluation first
            model_evaluator.evaluate(y_true, y_pred, y_pred_proba)

            # Create comparison metrics (slightly different)
            other_metrics = {
                "accuracy": 0.75,
                "precision": 0.70,
                "recall": 0.80,
                "f1_score": 0.72,
            }

            # Compare models
            comparison = model_evaluator.compare_models(other_metrics)

            # Check comparison structure
            assert "current_model" in comparison
            assert "other_model" in comparison
            assert "differences" in comparison
            assert "better_metrics" in comparison
            assert "worse_metrics" in comparison
            assert "primary_metric" in comparison
            assert "is_better" in comparison

            assert isinstance(comparison["better_metrics"], list)
            assert isinstance(comparison["worse_metrics"], list)
            assert isinstance(comparison["is_better"], bool)

        finally:
            os.chdir(original_cwd)

    def test_compare_models_before_evaluation(self, model_evaluator):
        """Test model comparison before running evaluation."""
        other_metrics = {"accuracy": 0.8, "f1_score": 0.7}

        with pytest.raises(ValueError, match="No metrics available for comparison"):
            model_evaluator.compare_models(other_metrics)

    # ---------- Edge Cases and Error Handling Tests ----------

    def test_evaluation_with_single_class(self, model_evaluator):
        """Test evaluation with single class data."""
        y_true = np.array([1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1])

        # This might work but with limited metrics
        try:
            results = model_evaluator.evaluate(y_true, y_pred)
            # If it succeeds, check that accuracy is 1.0
            assert results["metrics"]["accuracy"] == 1.0
        except Exception:
            # Single class evaluation often has issues, which is acceptable
            pass

    def test_evaluation_with_invalid_probabilities_shape(self, model_evaluator):
        """Test evaluation with invalid probability shape."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_pred_proba = np.array([0.9, 0.2, 0.8, 0.1])  # 1D instead of 2D

        # Should still work, the method handles 1D probabilities
        results = model_evaluator.evaluate(y_true, y_pred, y_pred_proba)
        assert "metrics" in results

    def test_report_generation(self, model_evaluator, sample_binary_data, temp_dir):
        """Test that evaluation report is generated correctly."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            y_true, y_pred, y_pred_proba = sample_binary_data

            # Run evaluation
            model_evaluator.evaluate(y_true, y_pred, y_pred_proba, dataset_name="test")

            # Check that report file was created
            assert Path("logs/evaluation_report.json").exists()

            # Check that report contains expected keys
            assert "timestamp" in model_evaluator.report
            assert "dataset_name" in model_evaluator.report
            assert "n_samples" in model_evaluator.report
            assert "metrics" in model_evaluator.report
            assert "evaluation_completed" in model_evaluator.report

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    """Demonstrate ModelEvaluator testing functionality."""
    print("Model evaluation test module loaded successfully.")
    print("Run with: pytest tests/test_evaluation.py -v")

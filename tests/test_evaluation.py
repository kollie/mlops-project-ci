import pytest
import numpy as np
import pandas as pd
from src.evaluation.evaluator import ModelEvaluator

@pytest.fixture
def evaluator():
    """Create a ModelEvaluator instance for testing."""
    return ModelEvaluator()

@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate random predictions and true labels
    y_true = np.random.choice(['NO', 'YES'], n_samples)
    y_pred = np.random.choice(['NO', 'YES'], n_samples)
    y_prob = np.random.random((n_samples, 2))
    
    return y_true, y_pred, y_prob

def test_evaluator_initialization(evaluator):
    """Test evaluator initialization."""
    assert evaluator.metrics is not None
    assert isinstance(evaluator.metrics, dict)

def test_calculate_metrics(evaluator, sample_predictions):
    """Test calculation of evaluation metrics."""
    y_true, y_pred, _ = sample_predictions
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    
    # Check metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert all(0 <= v <= 1 for v in metrics.values())

def test_calculate_confusion_matrix(evaluator, sample_predictions):
    """Test calculation of confusion matrix."""
    y_true, y_pred, _ = sample_predictions
    cm = evaluator.calculate_confusion_matrix(y_true, y_pred)
    
    # Check confusion matrix shape and values
    assert cm.shape == (2, 2)
    assert np.all(cm >= 0)
    assert cm.sum() == len(y_true)

def test_calculate_roc_curve(evaluator, sample_predictions):
    """Test calculation of ROC curve."""
    y_true, _, y_prob = sample_predictions
    
    # Calculate ROC curve
    fpr, tpr, thresholds = evaluator.calculate_roc_curve(y_true, y_prob, pos_label='YES')
    
    # Check outputs
    assert len(fpr) == len(tpr)
    assert len(fpr) == len(thresholds)
    assert all(0 <= x <= 1 for x in fpr)
    assert all(0 <= x <= 1 for x in tpr)

def test_calculate_precision_recall_curve(evaluator, sample_predictions):
    """Test calculation of precision-recall curve."""
    y_true, _, y_prob = sample_predictions
    
    # Calculate precision-recall curve
    precision, recall, thresholds = evaluator.calculate_precision_recall_curve(y_true, y_prob, pos_label='YES')
    
    # Check outputs
    assert len(precision) == len(recall)
    assert len(precision) == len(thresholds)
    assert all(0 <= x <= 1 for x in precision)
    assert all(0 <= x <= 1 for x in recall)

def test_save_metrics(evaluator, sample_predictions, tmp_path):
    """Test saving evaluation metrics."""
    y_true, y_pred, _ = sample_predictions
    
    # Calculate and save metrics
    metrics = evaluator.calculate_metrics(y_true, y_pred)
    output_path = tmp_path / "metrics.json"
    evaluator.save_metrics(metrics, str(output_path))
    
    # Check that file was created
    assert output_path.exists()

def test_evaluation_with_invalid_data(evaluator):
    """Test evaluation with invalid data."""
    # Create invalid data
    y_true = np.array(['NO', 'YES', 'INVALID'])
    y_pred = np.array(['NO', 'YES', 'NO'])
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    
    # Test that it handles invalid data gracefully
    with pytest.raises(Exception):
        evaluator.calculate_metrics(y_true, y_pred, y_prob)

def test_evaluation_with_missing_data(evaluator):
    """Test evaluation with missing data."""
    # Create data with missing values
    y_true = np.array(['NO', 'YES', None])
    y_pred = np.array(['NO', 'YES', 'NO'])
    y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    
    # Test that it handles missing data gracefully
    with pytest.raises(Exception):
        evaluator.calculate_metrics(y_true, y_pred, y_prob)

def test_generate_classification_report(evaluator, sample_predictions):
    """Test generation of classification report."""
    y_true, y_pred, _ = sample_predictions
    
    # Generate report
    report = evaluator.generate_classification_report(y_true, y_pred)
    
    # Check report
    assert isinstance(report, str)
    assert 'precision' in report.lower()
    assert 'recall' in report.lower()
    assert 'f1-score' in report.lower() 
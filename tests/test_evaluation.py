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
    return {
        'y_true': np.array(['NO', 'YES', 'NO', 'YES', 'NO']),
        'y_pred': np.array(['NO', 'YES', 'NO', 'NO', 'YES']),
        'y_prob': np.array([
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.6, 0.4],
            [0.4, 0.6]
        ])
    }

def test_evaluator_initialization(evaluator):
    """Test evaluator initialization."""
    assert evaluator.metrics is not None
    assert isinstance(evaluator.metrics, dict)

def test_calculate_metrics(evaluator, sample_predictions):
    """Test calculation of evaluation metrics."""
    metrics = evaluator.calculate_metrics(
        sample_predictions['y_true'],
        sample_predictions['y_pred'],
        sample_predictions['y_prob']
    )
    
    # Check that all required metrics are present
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics
    
    # Check metric values are valid
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1

def test_calculate_confusion_matrix(evaluator, sample_predictions):
    """Test calculation of confusion matrix."""
    cm = evaluator.calculate_confusion_matrix(
        sample_predictions['y_true'],
        sample_predictions['y_pred']
    )
    
    # Check confusion matrix shape and values
    assert cm.shape == (2, 2)
    assert np.all(cm >= 0)
    assert cm.sum() == len(sample_predictions['y_true'])

def test_calculate_roc_curve(evaluator, sample_predictions):
    """Test calculation of ROC curve."""
    fpr, tpr, thresholds = evaluator.calculate_roc_curve(
        sample_predictions['y_true'],
        sample_predictions['y_prob'][:, 1]
    )
    
    # Check ROC curve values
    assert len(fpr) == len(tpr) == len(thresholds)
    assert np.all(fpr >= 0) and np.all(fpr <= 1)
    assert np.all(tpr >= 0) and np.all(tpr <= 1)
    assert np.all(thresholds >= 0) and np.all(thresholds <= 1)

def test_calculate_precision_recall_curve(evaluator, sample_predictions):
    """Test calculation of precision-recall curve."""
    precision, recall, thresholds = evaluator.calculate_precision_recall_curve(
        sample_predictions['y_true'],
        sample_predictions['y_prob'][:, 1]
    )
    
    # Check precision-recall curve values
    assert len(precision) == len(recall) == len(thresholds)
    assert np.all(precision >= 0) and np.all(precision <= 1)
    assert np.all(recall >= 0) and np.all(recall <= 1)
    assert np.all(thresholds >= 0) and np.all(thresholds <= 1)

def test_save_metrics(evaluator, sample_predictions, tmp_path):
    """Test saving evaluation metrics."""
    # Calculate metrics
    metrics = evaluator.calculate_metrics(
        sample_predictions['y_true'],
        sample_predictions['y_pred'],
        sample_predictions['y_prob']
    )
    
    # Save metrics
    metrics_path = tmp_path / "metrics.json"
    evaluator.save_metrics(metrics, str(metrics_path))
    
    # Check that file was created
    assert metrics_path.exists()

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
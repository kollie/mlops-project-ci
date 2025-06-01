import pytest
import numpy as np
from src.evaluation.metrics import (
    calculate_metrics,
    calculate_confusion_matrix,
    calculate_roc_curve,
    calculate_precision_recall_curve,
    generate_classification_report,
)

@pytest.fixture
def sample_predictions():
    np.random.seed(42)
    n = 100
    y_true = np.random.choice(['NO', 'YES'], n)
    y_pred = np.random.choice(['NO', 'YES'], n)
    y_prob = np.random.rand(n, 2)
    return y_true, y_pred, y_prob

def test_calculate_metrics(sample_predictions):
    y_true, y_pred, _ = sample_predictions
    metrics = calculate_metrics(y_true, y_pred)
    assert isinstance(metrics, dict)
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        assert key in metrics

def test_confusion_matrix(sample_predictions):
    y_true, y_pred, _ = sample_predictions
    cm = calculate_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm.sum() == len(y_true)

def test_roc_curve(sample_predictions):
    y_true, _, y_prob = sample_predictions
    fpr, tpr, thresholds = calculate_roc_curve(y_true, y_prob, pos_label='YES')
    assert len(fpr) == len(tpr) == len(thresholds)
    assert all(0 <= x <= 1 for x in fpr)
    assert all(0 <= x <= 1 for x in tpr)

def test_precision_recall_curve(sample_predictions):
    y_true, _, y_prob = sample_predictions
    p, r, thresholds = calculate_precision_recall_curve(y_true, y_prob)
    assert len(p) == len(r) == len(thresholds)
    assert all(0 <= x <= 1 for x in p)
    assert all(0 <= x <= 1 for x in r)

def test_classification_report(sample_predictions):
    y_true, y_pred, _ = sample_predictions
    report = generate_classification_report(y_true, y_pred)
    assert isinstance(report, str)
    assert 'precision' in report.lower()


def test_calculate_metrics_with_invalid_labels():
    y_true = np.array(['NO', 'YES', 'MAYBE'])  # Invalid label
    y_pred = np.array(['YES', 'NO', 'NO'])
    with pytest.raises(Exception):
        _ = calculate_metrics(y_true, y_pred)

def test_calculate_metrics_with_missing_values():
    y_true = np.array(['NO', 'YES', None])  # Missing value
    y_pred = np.array(['YES', 'NO', 'NO'])
    with pytest.raises(Exception):
        _ = calculate_metrics(y_true, y_pred)

def test_classification_report_contains_expected_terms(sample_predictions):
    y_true, y_pred, _ = sample_predictions
    report = generate_classification_report(y_true, y_pred).lower()
    for term in ['precision', 'recall', 'f1-score']:
        assert term in report

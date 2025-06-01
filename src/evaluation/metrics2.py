import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

def compute_accuracy(y_true, y_pred):
    """Compute accuracy score."""
    return accuracy_score(y_true, y_pred)

def compute_precision(y_true, y_pred):
    """Compute precision score (weighted)."""
    return precision_score(y_true, y_pred, average='weighted')

def compute_recall(y_true, y_pred):
    """Compute recall score (weighted)."""
    return recall_score(y_true, y_pred, average='weighted')

def compute_f1(y_true, y_pred):
    """Compute F1 score (weighted)."""
    return f1_score(y_true, y_pred, average='weighted')

def compute_roc_auc(y_true, y_prob):
    """Compute ROC AUC score."""
    if y_prob is None or len(y_prob.shape) != 2:
        raise ValueError("y_prob must be a 2D array of predicted probabilities")
    return roc_auc_score(y_true, y_prob[:, 1])

def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)

def compute_classification_report(y_true, y_pred):
    """Generate classification report as a string."""
    return classification_report(y_true, y_pred)

def compute_roc_curve_data(y_true, y_prob):
    """Return ROC curve data (fpr, tpr, thresholds)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
    mask = np.isfinite(thresholds)
    return fpr[mask], tpr[mask], thresholds[mask]

def compute_precision_recall_curve_data(y_true, y_prob):
    """Return precision-recall curve data (precision, recall, thresholds)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob[:, 1])
    thresholds = np.append(thresholds, 1.0)  # Extend threshold to 1.0
    return precision, recall, thresholds

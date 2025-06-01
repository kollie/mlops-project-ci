"""
Tests for the analyze_class_imbalance method.
"""

import pytest
import pandas as pd
import numpy as np

def test_analyze_class_imbalance_basic(eda_instance, sample_data):
    """Test basic functionality of analyze_class_imbalance."""
    metrics = eda_instance.analyze_class_imbalance(sample_data)
    
    # Check if all required keys are present
    assert 'class_counts' in metrics
    assert 'class_ratios' in metrics
    assert 'imbalance_ratio' in metrics
    
    # Check if ratios sum to 1
    assert abs(sum(metrics['class_ratios'].values()) - 1.0) < 1e-10
    
    # Check if imbalance ratio is positive
    assert metrics['imbalance_ratio'] > 0

def test_analyze_class_imbalance_balanced(eda_instance):
    """Test analyze_class_imbalance with balanced classes."""
    # Create balanced data
    data = pd.DataFrame({
        'target': ['A', 'B'] * 50  # 50 samples of each class
    })
    
    metrics = eda_instance.analyze_class_imbalance(data, target_col='target')
    
    # Check if imbalance ratio is 1.0 (perfectly balanced)
    assert abs(metrics['imbalance_ratio'] - 1.0) < 1e-10
    
    # Check if class ratios are equal
    ratios = list(metrics['class_ratios'].values())
    assert abs(ratios[0] - ratios[1]) < 1e-10

def test_analyze_class_imbalance_highly_imbalanced(eda_instance):
    """Test analyze_class_imbalance with highly imbalanced classes."""
    # Create highly imbalanced data
    data = pd.DataFrame({
        'target': ['A'] * 90 + ['B'] * 10  # 90% class A, 10% class B
    })
    
    metrics = eda_instance.analyze_class_imbalance(data, target_col='target')
    
    # Check if imbalance ratio is 9.0 (90:10 ratio)
    assert abs(metrics['imbalance_ratio'] - 9.0) < 1e-10

def test_analyze_class_imbalance_empty(eda_instance):
    """Test analyze_class_imbalance with empty DataFrame."""
    empty_df = pd.DataFrame({'target': []})
    
    with pytest.raises(ValueError):
        eda_instance.analyze_class_imbalance(empty_df, target_col='target')

def test_analyze_class_imbalance_missing_target(eda_instance, sample_data):
    """Test analyze_class_imbalance with missing target column."""
    with pytest.raises(KeyError):
        eda_instance.analyze_class_imbalance(sample_data, target_col='nonexistent_column')

def test_analyze_class_imbalance_single_class(eda_instance):
    """Test analyze_class_imbalance with single class."""
    data = pd.DataFrame({
        'target': ['A'] * 100  # All samples belong to class A
    })
    
    metrics = eda_instance.analyze_class_imbalance(data, target_col='target')
    
    # Check if class ratio is 1.0 (100% class A)
    assert abs(metrics['class_ratios']['A'] - 1.0) < 1e-10 
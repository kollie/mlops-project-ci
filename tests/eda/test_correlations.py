"""
Tests for the analyze_correlations method.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_analyze_correlations_basic(eda_instance, sample_data, plots_dir):
    """Test basic functionality of analyze_correlations."""
    eda_instance.analyze_correlations(sample_data)
    
    # Check if correlation plots were created
    assert (plots_dir / 'correlation_heatmap.png').exists()
    assert (plots_dir / 'top_correlations.png').exists()

def test_analyze_correlations_custom_target(eda_instance, sample_data, plots_dir):
    """Test analyze_correlations with custom target column."""
    # Create a custom target column
    sample_data['custom_target'] = np.random.randn(len(sample_data))
    eda_instance.analyze_correlations(sample_data, target_col='custom_target')
    
    # Check if correlation plots were created
    assert (plots_dir / 'correlation_heatmap.png').exists()
    assert (plots_dir / 'top_correlations.png').exists()

def test_analyze_correlations_no_numerical(eda_instance, plots_dir):
    """Test analyze_correlations with no numerical columns."""
    # Create DataFrame with only categorical columns
    data = pd.DataFrame({
        'cat1': ['A', 'B', 'C'] * 10,
        'cat2': ['X', 'Y', 'Z'] * 10
    })
    
    with pytest.raises(ValueError):
        eda_instance.analyze_correlations(data)

def test_analyze_correlations_empty(eda_instance, plots_dir):
    """Test analyze_correlations with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        eda_instance.analyze_correlations(empty_df)

def test_analyze_correlations_with_missing(eda_instance, plots_dir):
    """Test analyze_correlations with missing values."""
    # Create DataFrame with missing values
    data = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [1.1, 2.2, 3.3, np.nan, 5.5],
        'target': [0, 1, 0, 1, 0]
    })
    
    eda_instance.analyze_correlations(data, target_col='target')
    
    # Check if correlation plots were created
    assert (plots_dir / 'correlation_heatmap.png').exists()
    assert (plots_dir / 'top_correlations.png').exists()

def test_analyze_correlations_perfect_correlation(eda_instance, plots_dir):
    """Test analyze_correlations with perfectly correlated features."""
    # Create DataFrame with perfectly correlated features
    data = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [2, 4, 6, 8, 10],  # Perfect correlation with col1
        'target': [0, 1, 0, 1, 0]
    })
    
    eda_instance.analyze_correlations(data, target_col='target')
    
    # Check if correlation plots were created
    assert (plots_dir / 'correlation_heatmap.png').exists()
    assert (plots_dir / 'top_correlations.png').exists() 
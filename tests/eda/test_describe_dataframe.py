"""
Tests for the describe_dataframe method.
"""

import pytest
import pandas as pd
import numpy as np

def test_describe_dataframe_basic(eda_instance, sample_data):
    """Test basic functionality of describe_dataframe."""
    stats = eda_instance.describe_dataframe(sample_data)
    
    # Check if all required keys are present
    assert 'shape' in stats
    assert 'dtypes' in stats
    assert 'summary' in stats
    assert 'memory_usage' in stats
    assert 'unique_values' in stats
    assert 'skewness' in stats
    assert 'kurtosis' in stats
    
    # Check shape
    assert stats['shape'] == sample_data.shape
    
    # Check dtypes
    assert len(stats['dtypes']) == len(sample_data.columns)
    
    # Check unique values
    assert all(col in stats['unique_values'] for col in sample_data.columns)
    
    # Check numerical statistics
    numerical_cols = sample_data.select_dtypes(include=['int64', 'float64']).columns
    assert all(col in stats['skewness'] for col in numerical_cols)
    assert all(col in stats['kurtosis'] for col in numerical_cols)

def test_describe_dataframe_empty(eda_instance):
    """Test describe_dataframe with empty DataFrame."""
    empty_df = pd.DataFrame()
    stats = eda_instance.describe_dataframe(empty_df)
    
    assert stats['shape'] == (0, 0)
    assert len(stats['dtypes']) == 0
    assert len(stats['summary']) == 0
    assert len(stats['memory_usage']) == 0
    assert len(stats['unique_values']) == 0
    assert len(stats['skewness']) == 0
    assert len(stats['kurtosis']) == 0

def test_describe_dataframe_single_column(eda_instance):
    """Test describe_dataframe with single column DataFrame."""
    df = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
    stats = eda_instance.describe_dataframe(df)
    
    assert stats['shape'] == (5, 1)
    assert len(stats['dtypes']) == 1
    assert 'col1' in stats['summary']
    assert 'col1' in stats['memory_usage']
    assert 'col1' in stats['unique_values']
    assert 'col1' in stats['skewness']
    assert 'col1' in stats['kurtosis'] 
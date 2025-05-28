"""
Tests for the run_analysis method.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_run_analysis_basic(eda_instance, sample_data, plots_dir):
    """Test basic functionality of run_analysis."""
    processed_df = eda_instance.run_analysis(sample_data)
    
    # Check if processed DataFrame is returned
    assert isinstance(processed_df, pd.DataFrame)
    
    # Check if all expected plots were created
    assert (plots_dir / 'target_distribution.png').exists()
    assert (plots_dir / 'target_distribution_pie.png').exists()
    assert (plots_dir / 'correlation_heatmap.png').exists()
    assert (plots_dir / 'top_correlations.png').exists()
    
    # Check if distributions directory exists
    dist_dir = plots_dir / 'distributions'
    assert dist_dir.exists()
    
    # Check if numerical columns have distribution plots
    numerical_cols = sample_data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        assert (dist_dir / f'{col}_distribution.png').exists()

def test_run_analysis_empty(eda_instance, plots_dir):
    """Test run_analysis with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        eda_instance.run_analysis(empty_df)

def test_run_analysis_no_numerical(eda_instance, plots_dir):
    """Test run_analysis with no numerical columns."""
    # Create DataFrame with only categorical columns
    data = pd.DataFrame({
        'cat1': ['A', 'B', 'C'] * 10,
        'cat2': ['X', 'Y', 'Z'] * 10,
        'readmitted': ['NO', 'YES'] * 15
    })
    
    processed_df = eda_instance.run_analysis(data)
    
    # Check if processed DataFrame is returned
    assert isinstance(processed_df, pd.DataFrame)
    
    # Check if target distribution plots were created
    assert (plots_dir / 'target_distribution.png').exists()
    assert (plots_dir / 'target_distribution_pie.png').exists()

def test_run_analysis_with_missing(eda_instance, plots_dir):
    """Test run_analysis with missing values."""
    # Create DataFrame with missing values
    data = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [1.1, 2.2, 3.3, np.nan, 5.5],
        'readmitted': ['NO', 'YES', 'NO', 'YES', 'NO']
    })
    
    processed_df = eda_instance.run_analysis(data)
    
    # Check if processed DataFrame is returned
    assert isinstance(processed_df, pd.DataFrame)
    
    # Check if missing values were handled
    assert not processed_df.isna().any().any()

def test_run_analysis_custom_target(eda_instance, plots_dir):
    """Test run_analysis with custom target column."""
    # Create DataFrame with custom target
    data = pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randn(100),
        'custom_target': np.random.choice(['A', 'B'], 100)
    })
    
    processed_df = eda_instance.run_analysis(data, target_col='custom_target')
    
    # Check if processed DataFrame is returned
    assert isinstance(processed_df, pd.DataFrame)
    
    # Check if target distribution plots were created
    assert (plots_dir / 'target_distribution.png').exists()
    assert (plots_dir / 'target_distribution_pie.png').exists()

def test_run_analysis_preserves_data_types(eda_instance, sample_data):
    """Test if run_analysis preserves data types."""
    processed_df = eda_instance.run_analysis(sample_data)
    
    # Check if data types are preserved
    for col in sample_data.columns:
        if col in processed_df.columns:
            assert processed_df[col].dtype == sample_data[col].dtype 
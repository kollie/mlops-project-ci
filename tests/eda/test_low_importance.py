"""
Tests for the remove_low_importance_columns method.
"""

import pytest
import pandas as pd
import numpy as np

def test_remove_low_importance_columns_basic(eda_instance, sample_data):
    """Test basic functionality of remove_low_importance_columns."""
    # Create data with known correlations
    data = pd.DataFrame({
        'high_corr': np.random.randn(100),
        'low_corr': np.random.randn(100),
        'readmitted': np.random.choice(['NO', 'YES'], 100)
    })
    
    # Make high_corr correlated with target
    data['high_corr'] = data['readmitted'].map({'NO': -1, 'YES': 1}) + 0.1 * np.random.randn(100)
    
    df_cleaned, removed_cols = eda_instance.remove_low_importance_columns(
        data, target_col='readmitted', threshold=0.1
    )
    
    # Check if low correlation column was removed
    assert 'low_corr' in removed_cols
    assert 'low_corr' not in df_cleaned.columns
    assert 'high_corr' in df_cleaned.columns
    assert 'readmitted' in df_cleaned.columns

def test_remove_low_importance_columns_no_removal(eda_instance, sample_data):
    """Test remove_low_importance_columns with no columns to remove."""
    # Create data with high correlations
    data = pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randn(100),
        'readmitted': np.random.choice(['NO', 'YES'], 100)
    })
    
    # Make both columns correlated with target
    data['col1'] = data['readmitted'].map({'NO': -1, 'YES': 1}) + 0.1 * np.random.randn(100)
    data['col2'] = data['readmitted'].map({'NO': -1, 'YES': 1}) + 0.1 * np.random.randn(100)
    
    df_cleaned, removed_cols = eda_instance.remove_low_importance_columns(
        data, target_col='readmitted', threshold=0.1
    )
    
    # Check if no columns were removed
    assert len(removed_cols) == 0
    assert len(df_cleaned.columns) == len(data.columns)

def test_remove_low_importance_columns_all_removed(eda_instance):
    """Test remove_low_importance_columns with all columns below threshold."""
    # Create data with low correlations
    data = pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randn(100),
        'readmitted': np.random.choice(['NO', 'YES'], 100)
    })
    
    df_cleaned, removed_cols = eda_instance.remove_low_importance_columns(
        data, target_col='readmitted', threshold=0.9
    )
    
    # Check if all columns except target were removed
    assert len(removed_cols) == 2
    assert len(df_cleaned.columns) == 1
    assert 'readmitted' in df_cleaned.columns

def test_remove_low_importance_columns_empty(eda_instance):
    """Test remove_low_importance_columns with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError):
        eda_instance.remove_low_importance_columns(empty_df)

def test_remove_low_importance_columns_missing_target(eda_instance, sample_data):
    """Test remove_low_importance_columns with missing target column."""
    with pytest.raises(KeyError):
        eda_instance.remove_low_importance_columns(
            sample_data, target_col='nonexistent_column'
        )

def test_remove_low_importance_columns_custom_threshold(eda_instance):
    """Test remove_low_importance_columns with custom threshold."""
    # Create data with known correlations
    data = pd.DataFrame({
        'col1': np.random.randn(100),
        'col2': np.random.randn(100),
        'readmitted': np.random.choice(['NO', 'YES'], 100)
    })
    
    # Make col1 correlated with target
    data['col1'] = data['readmitted'].map({'NO': -1, 'YES': 1}) + 0.1 * np.random.randn(100)
    
    # Test with different thresholds
    df_cleaned1, removed_cols1 = eda_instance.remove_low_importance_columns(
        data, target_col='readmitted', threshold=0.1
    )
    df_cleaned2, removed_cols2 = eda_instance.remove_low_importance_columns(
        data, target_col='readmitted', threshold=0.5
    )
    
    # Check if more columns are removed with higher threshold
    assert len(removed_cols1) <= len(removed_cols2) 
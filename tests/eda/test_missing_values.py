"""
Tests for the find_missing_values and handle_missing_values methods.
"""

import pytest
import pandas as pd
import numpy as np

def test_find_missing_values_basic(eda_instance, sample_data):
    """Test basic functionality of find_missing_values."""
    missing_values = eda_instance.find_missing_values(sample_data)
    
    # Check if all required keys are present
    assert 'nan_values' in missing_values
    assert 'question_mark_values' in missing_values
    
    # Check if missing values were found
    assert 'age' in missing_values['nan_values']
    assert 'race' in missing_values['question_mark_values']

def test_find_missing_values_no_missing(eda_instance):
    """Test find_missing_values with no missing values."""
    # Create data without missing values
    data = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['A', 'B', 'C', 'D', 'E']
    })
    
    missing_values = eda_instance.find_missing_values(data)
    
    # Check if no missing values were found
    assert len(missing_values['nan_values']) == 0
    assert len(missing_values['question_mark_values']) == 0

def test_find_missing_values_empty(eda_instance):
    """Test find_missing_values with empty DataFrame."""
    empty_df = pd.DataFrame()
    missing_values = eda_instance.find_missing_values(empty_df)
    
    assert len(missing_values['nan_values']) == 0
    assert len(missing_values['question_mark_values']) == 0

def test_handle_missing_values_basic(eda_instance, sample_data):
    """Test basic functionality of handle_missing_values."""
    # Get original missing value counts
    original_missing = eda_instance.find_missing_values(sample_data)
    
    # Handle missing values
    df_cleaned = eda_instance.handle_missing_values(sample_data)
    
    # Get new missing value counts
    new_missing = eda_instance.find_missing_values(df_cleaned)
    
    # Check if all missing values were handled
    assert len(new_missing['nan_values']) == 0
    assert len(new_missing['question_mark_values']) == 0
    
    # Check if data types are preserved
    assert df_cleaned.dtypes.equals(sample_data.dtypes)

def test_handle_missing_values_no_missing(eda_instance):
    """Test handle_missing_values with no missing values."""
    # Create data without missing values
    data = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['A', 'B', 'C', 'D', 'E']
    })
    
    df_cleaned = eda_instance.handle_missing_values(data)
    
    # Check if data remains unchanged
    pd.testing.assert_frame_equal(data, df_cleaned)

def test_handle_missing_values_empty(eda_instance):
    """Test handle_missing_values with empty DataFrame."""
    empty_df = pd.DataFrame()
    df_cleaned = eda_instance.handle_missing_values(empty_df)
    
    # Check if empty DataFrame is returned
    assert df_cleaned.empty
    assert len(df_cleaned.columns) == 0

def test_handle_missing_values_all_missing(eda_instance):
    """Test handle_missing_values with all values missing in a column."""
    # Create data with all missing values in one column
    data = pd.DataFrame({
        'col1': [np.nan] * 5,
        'col2': ['A', 'B', 'C', 'D', 'E']
    })
    
    df_cleaned = eda_instance.handle_missing_values(data)
    
    # Check if missing values were filled
    assert not df_cleaned['col1'].isna().any()
    assert df_cleaned['col1'].dtype == 'float64' 
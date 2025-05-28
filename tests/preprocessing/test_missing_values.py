import pytest
import pandas as pd
import numpy as np

def test_handle_missing_values_basic(preprocessor, sample_data_with_missing):
    """Test basic missing value handling."""
    processed_data = preprocessor._handle_missing_values(sample_data_with_missing)
    
    # Check that '?' values are replaced with NaN
    assert not (processed_data == '?').any().any()
    
    # Check that original NaN values are preserved
    assert processed_data['time_in_hospital'].isna().sum() == 1
    assert processed_data['num_lab_procedures'].isna().sum() == 1
    assert processed_data['num_procedures'].isna().sum() == 1
    assert processed_data['num_medications'].isna().sum() == 1

def test_handle_missing_values_no_missing(preprocessor, sample_data):
    """Test handling of data with no missing values."""
    processed_data = preprocessor._handle_missing_values(sample_data)
    
    # Check that the data remains unchanged
    pd.testing.assert_frame_equal(processed_data, sample_data)

def test_handle_missing_values_empty(preprocessor):
    """Test handling of empty DataFrame."""
    empty_df = pd.DataFrame()
    processed_data = preprocessor._handle_missing_values(empty_df)
    
    # Check that empty DataFrame is returned unchanged
    assert processed_data.empty

def test_handle_missing_values_all_missing(preprocessor):
    """Test handling of DataFrame with all missing values."""
    data = {
        'col1': ['?', '?', '?'],
        'col2': [np.nan, np.nan, np.nan]
    }
    df = pd.DataFrame(data)
    
    processed_data = preprocessor._handle_missing_values(df)
    
    # Check that '?' values are replaced with NaN
    assert not (processed_data == '?').any().any()
    
    # Check that all values are NaN
    assert processed_data.isna().all().all() 
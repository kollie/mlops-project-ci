import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_run_preprocessing_basic(preprocessor, sample_data):
    """Test basic run_preprocessing functionality."""
    X_processed, y_processed = preprocessor.run_preprocessing(sample_data)
    
    # Check output types
    assert isinstance(X_processed, pd.DataFrame)
    assert isinstance(y_processed, pd.Series)
    
    # Check that target is encoded
    assert set(y_processed.unique()).issubset({0, 1})
    
    # Check that features are processed
    assert X_processed.shape[1] <= len(preprocessor.config['features']['numerical_features']) + \
        len(preprocessor.config['features']['categorical_features'])
    
    # Check that dropped columns are not present
    for col in preprocessor.config['features']['drop_columns']:
        assert col not in X_processed.columns

def test_run_preprocessing_with_missing(preprocessor, sample_data_with_missing):
    """Test run_preprocessing with missing values."""
    X_processed, y_processed = preprocessor.run_preprocessing(sample_data_with_missing)
    
    # Check that no missing values in processed data
    assert not X_processed.isna().any().any()
    
    # Check that target is encoded
    assert set(y_processed.unique()).issubset({0, 1})

def test_run_preprocessing_empty(preprocessor):
    """Test run_preprocessing with empty DataFrame."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        preprocessor.run_preprocessing(empty_df)

def test_run_preprocessing_custom_target(preprocessor, sample_data):
    """Test run_preprocessing with custom target column."""
    # Rename target column
    sample_data = sample_data.rename(columns={'readmitted': 'target'})
    
    X_processed, y_processed = preprocessor.run_preprocessing(sample_data, target_col='target')
    
    # Check that target is encoded
    assert set(y_processed.unique()).issubset({0, 1})
    assert y_processed.name == 'target'

def test_run_preprocessing_logging(preprocessor, sample_data):
    """Test that run_preprocessing logs correctly."""
    # Clear log file
    log_file = Path("logs/preprocessing.log")
    if log_file.exists():
        log_file.unlink()
    
    # Run preprocessing
    preprocessor.run_preprocessing(sample_data)
    
    # Check that log file exists and contains expected messages
    assert log_file.exists()
    log_content = log_file.read_text()
    assert "Starting preprocessing pipeline" in log_content
    assert "Handling missing values" in log_content
    assert "Dropping specified columns" in log_content
    assert "Encoding target variable" in log_content
    assert "Creating and fitting preprocessing pipeline" in log_content
    assert "Selecting features" in log_content
    assert "Preprocessing completed" in log_content 
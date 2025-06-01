import pytest
import pandas as pd
import numpy as np

def test_create_preprocessing_pipeline(preprocessor):
    """Test creation of preprocessing pipeline."""
    pipeline = preprocessor._create_preprocessing_pipeline()
    
    # Check that pipeline is created
    assert pipeline is not None
    
    # Check that transformers are created
    assert 'num' in pipeline.named_transformers_
    assert 'cat' in pipeline.named_transformers_

def test_fit_transform_basic(preprocessor, sample_data):
    """Test basic fit_transform functionality."""
    X_processed, y_processed = preprocessor.fit_transform(sample_data)
    
    # Check that output is correct type
    assert isinstance(X_processed, pd.DataFrame)
    assert isinstance(y_processed, pd.Series)
    
    # Check that target is encoded
    assert set(y_processed.unique()).issubset({0, 1})
    
    # Check that features are processed
    assert X_processed.shape[1] <= len(preprocessor.config['features']['numerical_features']) + \
        len(preprocessor.config['features']['categorical_features'])

def test_transform_without_fit(preprocessor, sample_data):
    """Test that transform raises error if not fitted."""
    with pytest.raises(ValueError):
        preprocessor.transform(sample_data)

def test_fit_transform_with_missing(preprocessor, sample_data_with_missing):
    """Test fit_transform with missing values."""
    X_processed, y_processed = preprocessor.fit_transform(sample_data_with_missing)
    
    # Check that output is correct type
    assert isinstance(X_processed, pd.DataFrame)
    assert isinstance(y_processed, pd.Series)
    
    # Check that no missing values in processed data
    assert not X_processed.isna().any().any()
    
    # Check that target is encoded
    assert set(y_processed.unique()).issubset({0, 1})

def test_feature_selection(preprocessor, sample_data):
    """Test feature selection functionality."""
    X_processed, y_processed = preprocessor.fit_transform(sample_data)
    
    # Check that number of features is correct
    expected_n_features = preprocessor.config['feature_selection']['n_features']
    assert X_processed.shape[1] == expected_n_features

def test_pipeline_persistence(preprocessor, sample_data, tmp_path):
    """Test saving and loading the pipeline."""
    # Fit the pipeline
    preprocessor.fit(sample_data)
    
    # Save the pipeline
    save_path = tmp_path / "preprocessor.joblib"
    preprocessor.save(str(save_path))
    
    # Create new preprocessor and load
    new_preprocessor = preprocessor.__class__(config_path=preprocessor.config_path)
    new_preprocessor.load(str(save_path))
    
    # Transform data with both preprocessors
    X1, y1 = preprocessor.transform(sample_data)
    X2, y2 = new_preprocessor.transform(sample_data)
    
    # Check that results are identical
    pd.testing.assert_frame_equal(X1, X2)
    pd.testing.assert_series_equal(y1, y2) 
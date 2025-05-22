import pytest
import pandas as pd
import numpy as np
from src.preprocessing.preprocessor import Preprocessor

@pytest.fixture
def preprocessor():
    """Create a Preprocessor instance for testing."""
    return Preprocessor()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'age': np.random.randint(0, 100, n_samples),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'num_lab_procedures': np.random.randint(0, 100, n_samples),
        'num_procedures': np.random.randint(0, 10, n_samples),
        'num_medications': np.random.randint(0, 20, n_samples),
        'number_outpatient': np.random.randint(0, 10, n_samples),
        'number_emergency': np.random.randint(0, 10, n_samples),
        'number_inpatient': np.random.randint(0, 10, n_samples),
        'number_diagnoses': np.random.randint(0, 10, n_samples),
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'readmitted': np.random.choice(['NO', 'YES'], n_samples)
    })
    
    return data

def test_preprocessing_pipeline_creation(preprocessor):
    """Test creation of preprocessing pipeline."""
    # The pipeline should be created in the constructor
    assert preprocessor.preprocessor is None  # Initially None until fit is called

def test_fit_transform(preprocessor, sample_data):
    """Test fitting and transforming data."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Check output
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[0] == len(X)
    assert preprocessor.preprocessor is not None

def test_transform_without_fit(preprocessor, sample_data):
    """Test transforming data without fitting first."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    
    # Try to transform without fitting
    with pytest.raises(ValueError):
        preprocessor.transform(X)

def test_handle_numerical_features(preprocessor, sample_data):
    """Test handling of numerical features."""
    # Get numerical features
    numerical_features = sample_data.select_dtypes(include=['int64', 'float64']).columns
    
    # Transform numerical features
    transformed_data = preprocessor.transform(sample_data)
    
    # Check that numerical features were transformed
    assert isinstance(transformed_data, np.ndarray)
    assert transformed_data.shape[0] == len(sample_data)
    assert transformed_data.shape[1] >= len(numerical_features)

def test_handle_categorical_features(preprocessor, sample_data):
    """Test handling of categorical features."""
    # Get categorical features
    categorical_features = sample_data.select_dtypes(include=['object']).columns
    
    # Transform categorical features
    transformed_data = preprocessor.transform(sample_data)
    
    # Check that categorical features were transformed
    assert isinstance(transformed_data, np.ndarray)
    assert transformed_data.shape[0] == len(sample_data)
    assert transformed_data.shape[1] >= len(categorical_features)

def test_preprocessing_pipeline(preprocessor, sample_data):
    """Test complete preprocessing pipeline."""
    # Transform data
    transformed_data = preprocessor.transform(sample_data)
    
    # Check transformed data
    assert isinstance(transformed_data, np.ndarray)
    assert transformed_data.shape[0] == len(sample_data)
    assert not np.any(np.isnan(transformed_data))
    assert not np.any(np.isinf(transformed_data))

def test_preprocessing_with_missing_values(preprocessor):
    """Test preprocessing with missing values."""
    # Create data with missing values
    data = pd.DataFrame({
        'age': [25, np.nan, 35],
        'time_in_hospital': [5, 7, np.nan],
        'num_medications': [10, 15, 20],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'],
        'gender': ['Male', 'Female', 'Male']
    })
    
    # Transform data
    transformed_data = preprocessor.transform(data)
    
    # Check that missing values were handled
    assert isinstance(transformed_data, np.ndarray)
    assert not np.any(np.isnan(transformed_data))
    assert not np.any(np.isinf(transformed_data))

def test_preprocessing_with_invalid_data(preprocessor):
    """Test preprocessing with invalid data."""
    # Create invalid data
    data = pd.DataFrame({
        'age': ['invalid', 'invalid', 'invalid'],
        'time_in_hospital': ['invalid', 'invalid', 'invalid'],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'],
        'gender': ['Female', 'Male', 'Female']
    })
    
    # Test that it handles invalid data gracefully
    with pytest.raises(Exception):
        preprocessor.fit_transform(data) 
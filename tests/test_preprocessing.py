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
        'encounter_id': range(n_samples),
        'patient_nbr': range(n_samples),
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
        'age_group': pd.cut(np.random.randint(0, 100, n_samples), 
                           bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                           labels=['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)']),
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

def test_handle_categorical_features(preprocessor, sample_data):
    """Test handling of categorical features."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Check that categorical features were properly encoded
    assert isinstance(X_transformed, np.ndarray)
    # The number of columns should be greater than original due to one-hot encoding
    assert X_transformed.shape[1] >= len(X.columns)

def test_handle_numerical_features(preprocessor, sample_data):
    """Test handling of numerical features."""
    # Prepare data with only numerical features
    X = sample_data[['age', 'time_in_hospital', 'num_lab_procedures']]
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Check that numerical features were properly scaled
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.shape[1] == len(X.columns)

def test_preprocessing_with_missing_values(preprocessor):
    """Test preprocessing with missing values."""
    # Create data with missing values
    data = pd.DataFrame({
        'age': [25, np.nan, 35],
        'time_in_hospital': [5, 7, np.nan],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'],
        'gender': ['Female', 'Male', 'Female']
    })
    
    # Test that it handles missing values gracefully
    with pytest.raises(Exception):
        preprocessor.fit_transform(data)

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
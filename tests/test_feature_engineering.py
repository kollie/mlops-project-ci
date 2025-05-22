import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.feature_engineering import FeatureEngineer

@pytest.fixture
def feature_engineer():
    """Create a FeatureEngineer instance for testing."""
    return FeatureEngineer()

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

def test_create_features(feature_engineer, sample_data):
    """Test feature creation."""
    # Create features
    features = feature_engineer.create_features(sample_data)
    
    # Check that features were created
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(sample_data)
    assert all(features.dtypes != 'object')  # All features should be numeric

def test_select_features(feature_engineer, sample_data):
    """Test feature selection."""
    # Create features first
    features = feature_engineer.create_features(sample_data)
    
    # Select features
    selected_features = feature_engineer.select_features(features, sample_data['readmitted'])
    
    # Check that features were selected
    assert isinstance(selected_features, pd.DataFrame)
    assert len(selected_features) == len(features)
    assert len(selected_features.columns) <= len(features.columns)

def test_engineer_features(feature_engineer, sample_data):
    """Test complete feature engineering pipeline."""
    # Engineer features
    engineered_data = feature_engineer.engineer_features(sample_data)
    
    # Check that features were engineered
    assert isinstance(engineered_data, pd.DataFrame)
    assert len(engineered_data) == len(sample_data)
    assert 'readmitted' in engineered_data.columns
    assert all(engineered_data.dtypes != 'object')  # All features should be numeric

def test_feature_engineering_with_missing_values(feature_engineer):
    """Test feature engineering with missing values."""
    # Create data with missing values
    data = pd.DataFrame({
        'age': [25, np.nan, 35],
        'time_in_hospital': [5, 7, np.nan],
        'num_medications': [10, 15, 20],
        'readmitted': ['NO', 'YES', 'NO']
    })
    
    # Test that it handles missing values gracefully
    with pytest.raises(ValueError):
        feature_engineer.engineer_features(data) 
import pytest
import pandas as pd
import numpy as np
from src.features.feature_engineering import FeatureEngineer

@pytest.fixture
def feature_engineer():
    """Create a FeatureEngineer instance for testing."""
    return FeatureEngineer()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'encounter_id': range(100),
        'patient_nbr': range(100),
        'age': np.random.randint(0, 100, 100),
        'time_in_hospital': np.random.randint(1, 15, 100),
        'medication_1': ['Yes'] * 100,
        'medication_2': ['No'] * 100,
        'diagnosis_1': ['250.00'] * 100,
        'diagnosis_2': ['250.01'] * 100,
        'readmitted': ['NO'] * 100
    })

def test_create_features(feature_engineer, sample_data):
    """Test feature creation."""
    # Create features
    data_with_features = feature_engineer.create_features(sample_data)
    
    # Check that new features were created
    assert 'age_group' in data_with_features.columns
    assert 'length_of_stay_group' in data_with_features.columns
    assert 'num_medications' in data_with_features.columns
    assert 'num_diagnoses' in data_with_features.columns
    
    # Check feature values
    assert data_with_features['num_medications'].dtype in [np.int64, np.int32]
    assert data_with_features['num_diagnoses'].dtype in [np.int64, np.int32]
    assert data_with_features['age_group'].dtype == 'category'
    assert data_with_features['length_of_stay_group'].dtype == 'category'

def test_select_features(feature_engineer, sample_data):
    """Test feature selection."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Select features
    X_selected, selected_features = feature_engineer.select_features(X, y)
    
    # Check output types
    assert isinstance(X_selected, pd.DataFrame)
    assert isinstance(selected_features, list)
    
    # Check dimensions
    assert len(selected_features) <= len(X.columns)
    assert X_selected.shape[1] == len(selected_features)

def test_engineer_features(feature_engineer, sample_data):
    """Test complete feature engineering pipeline."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Engineer features
    X_engineered, selected_features = feature_engineer.engineer_features(X, y)
    
    # Check output types
    assert isinstance(X_engineered, pd.DataFrame)
    assert isinstance(selected_features, list)
    
    # Check that the output contains engineered features
    assert X_engineered.shape[1] == len(selected_features)
    
    # Check that the feature selector was created
    assert feature_engineer.feature_selector is not None

def test_feature_engineering_with_missing_values(feature_engineer):
    """Test feature engineering with missing values."""
    # Create data with missing values
    data = pd.DataFrame({
        'age': [25, np.nan, 35],
        'time_in_hospital': [5, 7, np.nan],
        'medication_1': ['Yes', 'No', 'Yes'],
        'diagnosis_1': ['250.00', '250.01', '250.02'],
        'readmitted': ['NO', 'YES', 'NO']
    })
    
    # Test that it handles missing values gracefully
    with pytest.raises(Exception):
        feature_engineer.create_features(data)

def test_feature_engineering_with_invalid_data(feature_engineer):
    """Test feature engineering with invalid data."""
    # Create invalid data
    data = pd.DataFrame({
        'age': ['invalid', 'invalid', 'invalid'],
        'time_in_hospital': ['invalid', 'invalid', 'invalid'],
        'medication_1': ['Yes', 'No', 'Yes'],
        'diagnosis_1': ['250.00', '250.01', '250.02'],
        'readmitted': ['NO', 'YES', 'NO']
    })
    
    # Test that it handles invalid data gracefully
    with pytest.raises(Exception):
        feature_engineer.create_features(data) 
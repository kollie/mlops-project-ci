import pytest
import pandas as pd
import numpy as np
from src.data_validation import DataValidator

@pytest.fixture
def data_validator():
    return DataValidator()

@pytest.fixture
def valid_data():
    return pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
        'category': ['A', 'B', 'A'],
        'location': ['NY', 'CA', 'TX'],
        'target': [0, 1, 0]
    })

@pytest.fixture
def invalid_data():
    return pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
        'category': ['A', 'B', 'A']
        # Missing required columns
    })

def test_validate_schema_valid(data_validator, valid_data):
    # Test schema validation with valid data
    assert data_validator.validate_schema(valid_data) is True

def test_validate_schema_invalid(data_validator, invalid_data):
    # Test schema validation with invalid data
    assert data_validator.validate_schema(invalid_data) is False

def test_check_types_valid(data_validator, valid_data):
    # Test type checking with valid data
    assert data_validator.check_types(valid_data) is True

def test_check_types_invalid(data_validator):
    # Test type checking with invalid data
    invalid_data = pd.DataFrame({
        'age': ['25', '30', '35'],  # Should be numerical
        'income': [50000, 60000, 70000],
        'category': [1, 2, 1],  # Should be categorical
        'location': ['NY', 'CA', 'TX'],
        'target': [0, 1, 0]
    })
    assert data_validator.check_types(invalid_data) is True  # Returns True but logs warnings

def test_handle_missing_values(data_validator):
    # Test handling missing values
    data = pd.DataFrame({
        'age': [25, np.nan, 35],
        'income': [50000, 60000, np.nan],
        'category': ['A', 'B', np.nan],
        'location': ['NY', 'CA', 'TX'],
        'target': [0, 1, 0]
    })
    
    processed_data = data_validator.handle_missing_values(data)
    
    # Check that missing values are handled
    assert not processed_data['age'].isnull().any()
    assert not processed_data['income'].isnull().any()
    assert not processed_data['category'].isnull().any()

def test_validate_data_valid(data_validator, valid_data):
    # Test complete validation with valid data
    assert data_validator.validate_data(valid_data) is True

def test_validate_data_invalid(data_validator, invalid_data):
    # Test complete validation with invalid data
    assert data_validator.validate_data(invalid_data) is False 
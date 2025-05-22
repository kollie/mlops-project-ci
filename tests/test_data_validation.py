import pytest
import pandas as pd
import numpy as np
from src.validation import DataValidator

@pytest.fixture
def validator():
    return DataValidator()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'encounter_id': range(100),
        'patient_nbr': range(100),
        'race': ['Caucasian'] * 100,
        'gender': ['Female'] * 100,
        'age': range(100),
        'time_in_hospital': range(100),
        'num_lab_procedures': range(100),
        'num_procedures': range(100),
        'num_medications': range(100),
        'number_outpatient': range(100),
        'number_emergency': range(100),
        'number_inpatient': range(100),
        'readmitted': ['NO'] * 50 + ['YES'] * 50
    })

def test_validate_data(validator, sample_data):
    assert validator.validate_data(sample_data) is True

def test_validate_data_missing_columns(validator):
    data = pd.DataFrame({'encounter_id': range(100)})
    with pytest.raises(ValueError, match="Missing required columns"):
        validator.validate_data(data)

def test_validate_data_missing_values(validator, sample_data):
    sample_data.loc[0, 'age'] = np.nan
    with pytest.raises(ValueError, match="Data contains missing values"):
        validator.validate_data(sample_data)

def test_validate_data_invalid_target(validator, sample_data):
    sample_data.loc[0, 'readmitted'] = 'MAYBE'
    with pytest.raises(ValueError, match="Target variable must contain only 'NO' or 'YES'"):
        validator.validate_data(sample_data)

def test_validate_schema(validator, sample_data):
    schema = validator.validate_schema(sample_data)
    assert schema['num_rows'] == 100
    assert schema['num_columns'] == 13
    assert 'column_types' in schema
    assert 'missing_values' in schema
    assert 'unique_values' in schema

def test_validate_schema_valid(data_validator, valid_data):
    # Test schema validation with valid data
    assert data_validator.validate_schema(valid_data) is True

def test_validate_schema_invalid(data_validator, invalid_data):
    # Test schema validation with invalid data
    assert data_validator.validate_schema(invalid_data) is False

def test_check_types_valid(data_validator, valid_data):
    # Test type checking with valid data
    assert data_validator.check_types(valid_data) is True

def test_check_types_invalid(validator):
    # Test type checking with invalid data
    invalid_data = pd.DataFrame({
        'encounter_id': ['1', '2', '3'],  # Should be numerical
        'patient_nbr': ['1', '2', '3'],  # Should be numerical
        'race': ['Caucasian'] * 3,
        'gender': ['Female'] * 3,
        'age': ['25', '30', '35'],  # Should be numerical
        'time_in_hospital': range(3),
        'num_lab_procedures': range(3),
        'num_procedures': range(3),
        'num_medications': range(3),
        'number_outpatient': range(3),
        'number_emergency': range(3),
        'number_inpatient': range(3),
        'readmitted': ['NO', 'YES', 'NO']
    })
    with pytest.raises(ValueError, match="encounter_id must be numeric"):
        validator.validate_data(invalid_data)

def test_handle_missing_values(validator):
    # Test handling missing values
    data = pd.DataFrame({
        'encounter_id': range(3),
        'patient_nbr': range(3),
        'race': ['Caucasian'] * 3,
        'gender': ['Female'] * 3,
        'age': [25, np.nan, 35],
        'time_in_hospital': range(3),
        'num_lab_procedures': range(3),
        'num_procedures': range(3),
        'num_medications': range(3),
        'number_outpatient': range(3),
        'number_emergency': range(3),
        'number_inpatient': range(3),
        'readmitted': ['NO', 'YES', 'NO']
    })
    
    with pytest.raises(ValueError, match="Data contains missing values"):
        validator.validate_data(data)

def test_validate_data_valid(data_validator, valid_data):
    # Test complete validation with valid data
    assert data_validator.validate_data(valid_data) is True

def test_validate_data_invalid(data_validator, invalid_data):
    # Test complete validation with invalid data
    assert data_validator.validate_data(invalid_data) is False 
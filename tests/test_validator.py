import pytest
import pandas as pd
import numpy as np
from src.validation.validator import LightDataValidator


# ---------- Fixtures ----------

@pytest.fixture
def validator():
    return LightDataValidator()

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'race': ['Caucasian'] * 101,
        'gender': ['Female'] * 101,
        'age': ['[30-40)'] * 101,
        'time_in_hospital': np.random.randint(1, 15, 101),
        'num_lab_procedures': np.random.randint(1, 100, 101),
        'num_procedures': np.random.randint(0, 10, 101),
        'num_medications': np.random.randint(1, 50, 101),
        'number_outpatient': np.random.randint(0, 20, 101),
        'number_emergency': np.random.randint(0, 20, 101),
        'number_inpatient': np.random.randint(0, 20, 101),
        'number_diagnoses': np.random.randint(1, 20, 101),
        'readmitted': ['NO'] * 101
    })
    # Add fallback: pretend 'age' is 'age_group'
    df['age_group'] = df['age']
    return df


# ---------- Tests ----------

def test_validate_schema_passes(validator, sample_data):
    assert validator.validate_schema(sample_data) is True

def test_validate_schema_fails(validator, sample_data):
    sample_data.drop(columns=['race'], inplace=True)
    assert validator.validate_schema(sample_data) is False

def test_validate_data_types_passes(validator, sample_data):
    assert validator.validate_data_types(sample_data) is True

def test_validate_data_types_fails(validator, sample_data):
    sample_data['num_medications'] = sample_data['num_medications'].astype(str)
    assert validator.validate_data_types(sample_data) is False

def test_validate_missing_values_passes(validator, sample_data):
    assert validator.validate_missing_values(sample_data) is True

def test_validate_missing_values_fails(validator, sample_data):
    sample_data['age'] = np.nan
    assert validator.validate_missing_values(sample_data, threshold=0.1) is False

def test_validate_target_distribution_passes(validator, sample_data):
    assert validator.validate_target_distribution(sample_data) is True

def test_validate_target_distribution_fails(validator, sample_data):
    sample_data['readmitted'] = ['NO'] * 100 + ['RARE']
    assert validator.validate_target_distribution(sample_data) is False

def test_validate_all_combined_pass(validator, sample_data):
    assert all(validator.validate_all(sample_data).values())

def test_validate_all_combined_fails(validator, sample_data):
    sample_data.drop(columns=['race'], inplace=True)
    results = validator.validate_all(sample_data)
    assert results['schema'] is False

def test_empty_dataframe_fails_schema(validator):
    empty_df = pd.DataFrame()
    assert validator.validate_schema(empty_df) is False

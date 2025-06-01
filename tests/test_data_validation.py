import pytest
import pandas as pd
import numpy as np
from src.validation.data_validator import DataValidator


# ---------- Fixtures ----------

@pytest.fixture
def data_validator():
    return DataValidator()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
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
        'readmitted': ['NO'] * 100 + ['RARE']  # make 'RARE' less than 1%
    })


# ---------- Tests ----------

def test_cleaning_with_missing_values(data_validator, sample_data):
    sample_data.loc[0, 'age'] = np.nan
    cleaned = data_validator.validate_and_clean(sample_data)
    assert len(cleaned) == len(sample_data) - 1

def test_type_mismatch_raises_error(data_validator, sample_data):
    sample_data['num_medications'] = sample_data['num_medications'].astype(str)
    with pytest.raises(TypeError):
        data_validator.validate_and_clean(sample_data)

def test_missing_column_fails(data_validator, sample_data):
    sample_data.drop(columns=['race'], inplace=True)
    with pytest.raises(ValueError):
        data_validator.validate_and_clean(sample_data)

def test_target_distribution_fails(data_validator, sample_data):
    with pytest.raises(ValueError):
        data_validator.validate_and_clean(sample_data)

def test_full_validation_passes(data_validator):
    df = pd.DataFrame({
        'race': ['Caucasian'] * 100,
        'gender': ['Female'] * 100,
        'age': ['[30-40)'] * 100,
        'time_in_hospital': np.random.randint(1, 15, 100),
        'num_lab_procedures': np.random.randint(1, 100, 100),
        'num_procedures': np.random.randint(0, 10, 100),
        'num_medications': np.random.randint(1, 50, 100),
        'number_outpatient': np.random.randint(0, 20, 100),
        'number_emergency': np.random.randint(0, 20, 100),
        'number_inpatient': np.random.randint(0, 20, 100),
        'number_diagnoses': np.random.randint(1, 20, 100),
        'readmitted': ['NO'] * 50 + ['YES'] * 50
    })
    cleaned = data_validator.validate_and_clean(df)
    assert isinstance(cleaned, pd.DataFrame)
    assert 'readmitted' in cleaned.columns

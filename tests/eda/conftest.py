"""
Common fixtures for EDA tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.eda import EDA

@pytest.fixture
def sample_data():
    """Create a sample dataset for testing."""
    data = {
        'age': ['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)'],
        'gender': ['F', 'M', 'F', 'M', 'F'],
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'],
        'time_in_hospital': [5, 3, 7, 2, 4],
        'num_lab_procedures': [45, 32, 58, 23, 41],
        'num_procedures': [1, 0, 2, 1, 0],
        'num_medications': [15, 8, 20, 12, 10],
        'number_outpatient': [0, 2, 1, 0, 3],
        'number_emergency': [1, 0, 2, 1, 0],
        'number_inpatient': [0, 1, 0, 2, 1],
        'diag_1': ['250.01', '401.9', '272.4', '250.00', '401.1'],
        'diag_2': ['250.00', '272.4', '401.9', '250.01', '401.1'],
        'diag_3': ['272.4', '250.00', '401.1', '250.01', '401.9'],
        'readmitted': ['NO', 'YES', 'NO', 'YES', 'NO']
    }
    df = pd.DataFrame(data)
    
    # Convert numerical columns to appropriate types
    numerical_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient'
    ]
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

@pytest.fixture
def sample_data_with_missing():
    """Create a sample dataset with missing values."""
    data = {
        'age': ['[0-10)', '?', '[20-30)', '[30-40)', '[40-50)'],
        'gender': ['F', 'M', '?', 'M', 'F'],
        'race': ['Caucasian', '?', 'Hispanic', 'Asian', 'Other'],
        'time_in_hospital': [5, np.nan, 7, 2, 4],
        'num_lab_procedures': [45, 32, np.nan, 23, 41],
        'num_procedures': [1, 0, 2, np.nan, 0],
        'num_medications': [15, 8, 20, 12, np.nan],
        'number_outpatient': [0, 2, 1, 0, 3],
        'number_emergency': [1, 0, 2, 1, 0],
        'number_inpatient': [0, 1, 0, 2, 1],
        'diag_1': ['250.01', '?', '272.4', '250.00', '401.1'],
        'diag_2': ['250.00', '272.4', '?', '250.01', '401.1'],
        'diag_3': ['272.4', '250.00', '401.1', '?', '401.9'],
        'readmitted': ['NO', 'YES', 'NO', 'YES', 'NO']
    }
    df = pd.DataFrame(data)
    
    # Convert numerical columns to appropriate types
    numerical_cols = [
        'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency',
        'number_inpatient'
    ]
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

@pytest.fixture
def sample_data_perfect_correlation():
    """Create a sample dataset with perfect correlation."""
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [2, 4, 6, 8, 10],  # Perfect correlation with feature1
        'feature3': [1, 3, 5, 7, 9],   # High correlation with feature1
        'readmitted': ['NO', 'YES', 'NO', 'YES', 'NO']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_data_with_inf():
    """Create a sample dataset with infinite values."""
    data = {
        'feature1': [1, 2, np.inf, 4, 5],
        'feature2': [2, 4, 6, 8, 10],
        'feature3': [1, 3, 5, 7, 9],
        'readmitted': ['NO', 'YES', 'NO', 'YES', 'NO']
    }
    return pd.DataFrame(data)

@pytest.fixture
def empty_dataframe():
    """Create an empty DataFrame with the correct columns."""
    return pd.DataFrame(columns=[
        'age', 'gender', 'race', 'time_in_hospital', 'num_lab_procedures',
        'num_procedures', 'num_medications', 'number_outpatient',
        'number_emergency', 'number_inpatient', 'diag_1', 'diag_2', 'diag_3',
        'readmitted'
    ])

@pytest.fixture
def eda_instance(tmp_path):
    """Create an EDA instance with temporary log file."""
    log_file = tmp_path / "eda.log"
    return EDA(log_file=str(log_file))

@pytest.fixture
def plots_dir(tmp_path):
    """Create a temporary plots directory."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    return plots_dir 
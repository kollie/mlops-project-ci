import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import tempfile

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
    return pd.DataFrame(data)

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
    return pd.DataFrame(data)

@pytest.fixture
def config_file():
    """Create a temporary config file for testing."""
    config = {
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'features': {
            'categorical_features': ['age', 'gender', 'race', 'diag_1', 'diag_2', 'diag_3'],
            'numerical_features': [
                'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient'
            ],
            'drop_columns': ['diag_1', 'diag_2', 'diag_3']
        },
        'feature_selection': {
            'n_features': 10
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f)
        return f.name

@pytest.fixture
def preprocessor(config_file):
    """Create a Preprocessor instance with test config."""
    from src.preprocessing.preprocessor import Preprocessor
    return Preprocessor(config_path=config_file) 
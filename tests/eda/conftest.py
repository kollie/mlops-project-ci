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
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(0, 100, n_samples),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'num_lab_procedures': np.random.randint(1, 50, n_samples),
        'num_procedures': np.random.randint(0, 10, n_samples),
        'num_medications': np.random.randint(1, 30, n_samples),
        'number_outpatient': np.random.randint(0, 20, n_samples),
        'number_emergency': np.random.randint(0, 10, n_samples),
        'number_inpatient': np.random.randint(0, 15, n_samples),
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'readmitted': np.random.choice(['NO', 'YES', '>30'], n_samples)
    }
    
    # Add some missing values
    data['age'][5:10] = np.nan
    data['race'][15:20] = '?'
    
    return pd.DataFrame(data)

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
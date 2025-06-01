import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader.data_loader import DataLoader


# ---------- Fixtures ----------

@pytest.fixture
def data_loader(tmp_path):
    """Create a DataLoader with overridden config paths for testing."""
    loader = DataLoader()
    loader.config['data']['processed_data_path'] = str(tmp_path / "processed")
    loader.config['data']['test_data_path'] = str(tmp_path / "test")
    loader.config['data']['train_data_path'] = str(tmp_path / "processed/train.csv")
    loader.config['data']['validation_data_path'] = str(tmp_path / "processed/validation.csv")
    return loader

@pytest.fixture
def sample_data():
    """Create sample data for splitting/saving tests."""
    return pd.DataFrame({
        'encounter_id': range(100),
        'patient_nbr': range(100),
        'race': ['Caucasian'] * 100,
        'gender': ['Female'] * 100,
        'age': np.random.randint(0, 100, 100),
        'age_group': ['[30-40)'] * 100,
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


# ---------- Tests ----------

def test_load_data_with_fallback(data_loader, tmp_path):
    """Try loading from Google Drive; fallback to local mock data."""
    url = "https://drive.google.com/uc?id=1WCn9cjVQH2LP8E5ZC1-cNKVKBE8OrFGy"
    fallback_csv = tmp_path / "fallback_data.csv"

    # Create fallback data
    fallback_data = pd.DataFrame({
        'encounter_id': [1, 2, 3],
        'patient_nbr': [1, 2, 3],
        'race': ['Caucasian'] * 3,
        'gender': ['Female'] * 3,
        'age': [45, 50, 60],
        'age_group': ['[40-50)'] * 3,
        'time_in_hospital': [3, 5, 2],
        'num_lab_procedures': [10, 20, 15],
        'num_procedures': [1, 2, 1],
        'num_medications': [5, 10, 7],
        'number_outpatient': [0, 1, 0],
        'number_emergency': [0, 0, 1],
        'number_inpatient': [0, 1, 0],
        'number_diagnoses': [1, 2, 3],
        'readmitted': ['NO', 'YES', 'NO']
    })
    fallback_data.to_csv(fallback_csv, index=False)

    try:
        data = data_loader.load_data(url)
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'readmitted' in data.columns
    except Exception as e:
        print(f"Falling back due to error: {e}")
        data = data_loader.load_data(str(fallback_csv))
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 3
        assert 'readmitted' in data.columns


def test_split_data(data_loader, sample_data):
    """Ensure split outputs 3 mutually exclusive sets."""
    train, val, test = data_loader.split_data(sample_data)

    assert all(isinstance(split, pd.DataFrame) for split in [train, val, test])
    assert all(len(split) > 0 for split in [train, val, test])

    all_ids = set(sample_data['encounter_id'])
    split_ids = set(train['encounter_id']) | set(val['encounter_id']) | set(test['encounter_id'])

    assert all_ids == split_ids
    assert not set(train['encounter_id']) & set(val['encounter_id'])
    assert not set(train['encounter_id']) & set(test['encounter_id'])
    assert not set(val['encounter_id']) & set(test['encounter_id'])


def test_save_split_data(data_loader, sample_data, tmp_path):
    """Test writing split data to disk."""
    train, val, test = data_loader.split_data(sample_data)
    data_loader.save_split_data(train, val, test)

    assert Path(data_loader.config['data']['train_data_path']).exists()
    assert Path(data_loader.config['data']['validation_data_path']).exists()
    assert Path(data_loader.config['data']['test_data_path']).joinpath("test.csv").exists()
    
def test_load_data_missing_path(data_loader):
    data_loader.config['data']['raw_data_path'] = None
    with pytest.raises(ValueError):
        data_loader.load_data()

def test_load_data_invalid_format(data_loader):
    data_loader.config['data']['raw_data_format'] = 'xml'
    with pytest.raises(ValueError):
        data_loader.load_data("fake.csv")

def test_load_data_file_not_found(data_loader):
    with pytest.raises(FileNotFoundError):
        data_loader.load_data("data/raw/nonexistent_file.csv")
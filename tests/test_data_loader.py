import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader.data_loader import DataLoader

@pytest.fixture
def data_loader():
    """Create a DataLoader instance for testing."""
    return DataLoader()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'encounter_id': range(100),
        'patient_nbr': range(100),
        'race': ['Caucasian'] * 100,
        'gender': ['Female'] * 100,
        'age': np.random.randint(0, 100, 100),
        'weight': ['?'] * 100,
        'admission_type_id': np.random.randint(1, 9, 100),
        'discharge_disposition_id': np.random.randint(1, 30, 100),
        'admission_source_id': np.random.randint(1, 25, 100),
        'time_in_hospital': np.random.randint(1, 15, 100),
        'num_lab_procedures': np.random.randint(1, 100, 100),
        'num_procedures': np.random.randint(0, 10, 100),
        'num_medications': np.random.randint(1, 50, 100),
        'number_outpatient': np.random.randint(0, 20, 100),
        'number_emergency': np.random.randint(0, 20, 100),
        'number_inpatient': np.random.randint(0, 20, 100),
        'diag_1': ['250.00'] * 100,
        'diag_2': ['250.00'] * 100,
        'diag_3': ['250.00'] * 100,
        'number_diagnoses': np.random.randint(1, 20, 100),
        'max_glu_serum': ['None'] * 100,
        'A1Cresult': ['None'] * 100,
        'metformin': ['No'] * 100,
        'repaglinide': ['No'] * 100,
        'nateglinide': ['No'] * 100,
        'chlorpropamide': ['No'] * 100,
        'glimepiride': ['No'] * 100,
        'acetohexamide': ['No'] * 100,
        'glipizide': ['No'] * 100,
        'glyburide': ['No'] * 100,
        'tolbutamide': ['No'] * 100,
        'pioglitazone': ['No'] * 100,
        'rosiglitazone': ['No'] * 100,
        'acarbose': ['No'] * 100,
        'miglitol': ['No'] * 100,
        'troglitazone': ['No'] * 100,
        'tolazamide': ['No'] * 100,
        'examide': ['No'] * 100,
        'citoglipton': ['No'] * 100,
        'insulin': ['No'] * 100,
        'glyburide-metformin': ['No'] * 100,
        'glipizide-metformin': ['No'] * 100,
        'glimepiride-pioglitazone': ['No'] * 100,
        'metformin-rosiglitazone': ['No'] * 100,
        'metformin-pioglitazone': ['No'] * 100,
        'change': ['No'] * 100,
        'diabetesMed': ['Yes'] * 100,
        'readmitted': ['NO'] * 100
    })

def test_load_data(data_loader, tmp_path):
    """Test loading data from CSV file."""
    # Create a temporary CSV file
    sample_data = pd.DataFrame({
        'encounter_id': [1, 2, 3],
        'patient_nbr': [1, 2, 3],
        'readmitted': ['NO', 'YES', 'NO']
    })
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    
    # Test loading data
    data = data_loader.load_data(str(csv_path))
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert 'readmitted' in data.columns

def test_split_data(data_loader, sample_data):
    """Test splitting data into train, validation, and test sets."""
    train, val, test = data_loader.split_data(sample_data)
    
    # Check that we have three datasets
    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    
    # Check that the splits are non-empty
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    
    # Check that the splits are mutually exclusive
    train_ids = set(train['encounter_id'])
    val_ids = set(val['encounter_id'])
    test_ids = set(test['encounter_id'])
    
    assert not train_ids.intersection(val_ids)
    assert not train_ids.intersection(test_ids)
    assert not val_ids.intersection(test_ids)

def test_save_split_data(data_loader, sample_data, tmp_path):
    """Test saving split datasets."""
    # Create temporary directories
    processed_dir = tmp_path / "processed"
    test_dir = tmp_path / "test"
    processed_dir.mkdir()
    test_dir.mkdir()
    
    # Split the data
    train, val, test = data_loader.split_data(sample_data)
    
    # Save the splits
    data_loader.save_split_data(train, val, test)
    
    # Check that files were created
    assert Path("data/processed/train.csv").exists()
    assert Path("data/processed/validation.csv").exists()
    assert Path("data/test/test.csv").exists() 
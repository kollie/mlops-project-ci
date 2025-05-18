import pytest
import pandas as pd
import os
from src.data_loader import DataLoader

@pytest.fixture
def data_loader():
    return DataLoader()

@pytest.fixture
def sample_csv_data(tmp_path):
    # Create a temporary CSV file
    data = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000],
        'category': ['A', 'B', 'A']
    })
    file_path = tmp_path / "test_data.csv"
    data.to_csv(file_path, index=False)
    return str(file_path)

def test_load_csv(data_loader, sample_csv_data):
    # Test loading CSV file
    data = data_loader.load_csv(sample_csv_data)
    assert isinstance(data, pd.DataFrame)
    assert len(data) == 3
    assert list(data.columns) == ['age', 'income', 'category']

def test_save_data(data_loader, tmp_path):
    # Test saving data to CSV
    data = pd.DataFrame({
        'age': [25, 30, 35],
        'income': [50000, 60000, 70000]
    })
    file_path = str(tmp_path / "output.csv")
    data_loader.save_data(data, file_path)
    assert os.path.exists(file_path)
    
    # Verify saved data
    loaded_data = pd.read_csv(file_path)
    assert loaded_data.equals(data)

def test_load_csv_file_not_found(data_loader):
    # Test loading non-existent file
    with pytest.raises(Exception):
        data_loader.load_csv("non_existent_file.csv")

def test_save_data_invalid_path(data_loader):
    # Test saving to invalid path
    data = pd.DataFrame({'col': [1, 2, 3]})
    with pytest.raises(Exception):
        data_loader.save_data(data, "/invalid/path/data.csv") 
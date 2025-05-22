import pytest
import numpy as np
import pandas as pd
from src.model.model_trainer import ModelTrainer

@pytest.fixture
def model_trainer():
    """Create a ModelTrainer instance for testing."""
    return ModelTrainer()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'encounter_id': range(100),
        'patient_nbr': range(100),
        'age': np.random.randint(0, 100, 100),
        'time_in_hospital': np.random.randint(1, 15, 100),
        'num_lab_procedures': np.random.randint(1, 100, 100),
        'num_procedures': np.random.randint(0, 10, 100),
        'num_medications': np.random.randint(1, 50, 100),
        'number_outpatient': np.random.randint(0, 20, 100),
        'number_emergency': np.random.randint(0, 20, 100),
        'number_inpatient': np.random.randint(0, 20, 100),
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'] * 33 + ['Caucasian'],
        'gender': ['Female', 'Male'] * 50,
        'age_group': ['[0-10)', '[10-20)', '[20-30)'] * 33 + ['[0-10)'],
        'readmitted': ['NO', 'YES', 'NO'] * 33 + ['NO']
    })

def test_model_initialization(model_trainer):
    """Test model initialization."""
    assert model_trainer.model is None
    assert model_trainer.model_params is not None

def test_train_model(model_trainer, sample_data):
    """Test model training."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Train model
    model_trainer.train(X, y)
    
    # Check that model was trained
    assert model_trainer.model is not None
    assert hasattr(model_trainer.model, 'predict')

def test_predict(model_trainer, sample_data):
    """Test model predictions."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Train model
    model_trainer.train(X, y)
    
    # Make predictions
    predictions = model_trainer.predict(X)
    
    # Check predictions
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert set(np.unique(predictions)).issubset({'NO', 'YES'})

def test_predict_proba(model_trainer, sample_data):
    """Test prediction probabilities."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Train model
    model_trainer.train(X, y)
    
    # Get prediction probabilities
    probabilities = model_trainer.predict_proba(X)
    
    # Check probabilities
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (len(X), 2)  # 2 classes
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

def test_save_load_model(model_trainer, sample_data, tmp_path):
    """Test saving and loading model."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Train model
    model_trainer.train(X, y)
    
    # Save model
    model_path = tmp_path / "model.joblib"
    model_trainer.save_model(str(model_path))
    
    # Create new trainer and load model
    new_trainer = ModelTrainer()
    new_trainer.load_model(str(model_path))
    
    # Check that loaded model makes same predictions
    original_preds = model_trainer.predict(X)
    loaded_preds = new_trainer.predict(X)
    np.testing.assert_array_equal(original_preds, loaded_preds)

def test_model_with_invalid_data(model_trainer):
    """Test model with invalid data."""
    # Create invalid data
    X = pd.DataFrame({
        'age': ['invalid'] * 10,
        'time_in_hospital': ['invalid'] * 10
    })
    y = pd.Series(['NO'] * 10)
    
    # Test that it handles invalid data gracefully
    with pytest.raises(Exception):
        model_trainer.train(X, y)

def test_model_with_missing_data(model_trainer):
    """Test model with missing data."""
    # Create data with missing values
    X = pd.DataFrame({
        'age': [25, np.nan, 35],
        'time_in_hospital': [5, 7, np.nan]
    })
    y = pd.Series(['NO', 'YES', 'NO'])
    
    # Test that it handles missing data gracefully
    with pytest.raises(Exception):
        model_trainer.train(X, y) 
import pytest
import pandas as pd
import numpy as np
from src.model.model_trainer import ModelTrainer

@pytest.fixture
def model_trainer():
    """Create a ModelTrainer instance for testing."""
    return ModelTrainer()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'age': np.random.randint(0, 100, n_samples),
        'time_in_hospital': np.random.randint(1, 15, n_samples),
        'num_lab_procedures': np.random.randint(0, 100, n_samples),
        'num_procedures': np.random.randint(0, 10, n_samples),
        'num_medications': np.random.randint(0, 20, n_samples),
        'number_outpatient': np.random.randint(0, 10, n_samples),
        'number_emergency': np.random.randint(0, 10, n_samples),
        'number_inpatient': np.random.randint(0, 10, n_samples),
        'number_diagnoses': np.random.randint(0, 10, n_samples),
        'race': np.random.choice(['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'readmitted': np.random.choice(['NO', 'YES'], n_samples)
    })
    
    return data

def test_model_initialization(model_trainer):
    """Test model initialization."""
    assert model_trainer.model is None
    assert model_trainer.preprocessor is not None

def test_train_model(model_trainer, sample_data):
    """Test model training."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Train model
    model_trainer.train(X, y)
    
    # Check that model was trained
    assert model_trainer.model is not None

def test_predict(model_trainer, sample_data):
    """Test model prediction."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Train model first
    model_trainer.train(X, y)
    
    # Make predictions
    predictions = model_trainer.predict(X)
    
    # Check predictions
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)
    assert all(pred in ['NO', 'YES'] for pred in predictions)

def test_predict_proba(model_trainer, sample_data):
    """Test model probability prediction."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Train model first
    model_trainer.train(X, y)
    
    # Get probabilities
    probabilities = model_trainer.predict_proba(X)
    
    # Check probabilities
    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)

def test_save_load_model(model_trainer, sample_data, tmp_path):
    """Test model saving and loading."""
    # Prepare data
    X = sample_data.drop(columns=['readmitted'])
    y = sample_data['readmitted']
    
    # Train model first
    model_trainer.train(X, y)
    
    # Save model
    model_path = tmp_path / "model.joblib"
    model_trainer.save_model(str(model_path))
    
    # Check that file was created
    assert model_path.exists()
    
    # Create new trainer and load model
    new_trainer = ModelTrainer()
    new_trainer.load_model(str(model_path))
    
    # Check that model was loaded
    assert new_trainer.model is not None
    
    # Check that predictions match
    original_preds = model_trainer.predict(X)
    loaded_preds = new_trainer.predict(X)
    assert np.array_equal(original_preds, loaded_preds)

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
import pytest
import pandas as pd
import numpy as np
from src.inference.predictor import Predictor

@pytest.fixture
def predictor():
    """Create a Predictor instance for testing."""
    return Predictor(load_model=False)

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
        'age_group': pd.cut(np.random.randint(0, 100, n_samples), 
                           bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                           labels=['[0-10)', '[10-20)', '[20-30)', '[30-40)', '[40-50)', 
                                 '[50-60)', '[60-70)', '[70-80)', '[80-90)', '[90-100)'])
    })
    
    return data

def test_predictor_initialization(predictor):
    """Test predictor initialization."""
    assert predictor.model is None
    assert predictor.preprocessor is not None

def test_predict_without_model(predictor, sample_data):
    """Test prediction without model."""
    with pytest.raises(ValueError, match="Model not loaded"):
        predictor.predict(sample_data)

def test_predict_proba_without_model(predictor, sample_data):
    """Test probability prediction without model."""
    with pytest.raises(ValueError, match="Model not loaded"):
        predictor.predict_proba(sample_data)

def test_inference_with_invalid_data(predictor):
    """Test inference with invalid data."""
    # Create invalid data
    data = pd.DataFrame({
        'age': ['invalid'] * 10,
        'time_in_hospital': ['invalid'] * 10
    })
    
    # Test that it handles invalid data gracefully
    with pytest.raises(ValueError):
        predictor.predict(data)

def test_inference_with_missing_data(predictor):
    """Test inference with missing data."""
    # Create data with missing values
    data = pd.DataFrame({
        'age': [25, np.nan, 35],
        'time_in_hospital': [5, 7, np.nan]
    })
    
    # Test that it handles missing data gracefully
    with pytest.raises(ValueError):
        predictor.predict(data)

def test_calculate_confidence(predictor):
    """Test confidence calculation."""
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    confidence = predictor.calculate_confidence(probabilities)
    assert np.allclose(confidence, np.array([0.8, 0.7, 0.9]))

def test_save_predictions(predictor, tmp_path):
    """Test saving predictions."""
    # Create sample predictions
    predictions = np.array(['NO', 'YES', 'NO'])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    
    # Save predictions
    output_path = tmp_path / "predictions.csv"
    predictor.save_predictions(pd.DataFrame(), predictions, probabilities, str(output_path))
    
    # Check that file was created
    assert output_path.exists() 
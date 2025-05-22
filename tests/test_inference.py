import pytest
import numpy as np
import pandas as pd
from src.inference.predictor import Predictor

@pytest.fixture
def predictor():
    """Create a Predictor instance for testing."""
    return Predictor()

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'encounter_id': range(10),
        'patient_nbr': range(10),
        'age': np.random.randint(0, 100, 10),
        'time_in_hospital': np.random.randint(1, 15, 10),
        'num_lab_procedures': np.random.randint(1, 100, 10),
        'num_procedures': np.random.randint(0, 10, 10),
        'num_medications': np.random.randint(1, 50, 10),
        'number_outpatient': np.random.randint(0, 20, 10),
        'number_emergency': np.random.randint(0, 20, 10),
        'number_inpatient': np.random.randint(0, 20, 10),
        'race': ['Caucasian', 'AfricanAmerican', 'Hispanic'] * 3 + ['Caucasian'],
        'gender': ['Female', 'Male'] * 5,
        'age_group': ['[0-10)', '[10-20)', '[20-30)'] * 3 + ['[0-10)']
    })

def test_predictor_initialization(predictor):
    """Test predictor initialization."""
    assert predictor.model is None
    assert predictor.preprocessor is None

def test_load_model(predictor, tmp_path):
    """Test loading a trained model."""
    # Create a dummy model file
    model_path = tmp_path / "model.joblib"
    with open(model_path, 'w') as f:
        f.write("dummy model")
    
    # Test loading model
    with pytest.raises(Exception):  # Should raise exception for invalid model file
        predictor.load_model(str(model_path))

def test_preprocess_input(predictor, sample_data):
    """Test preprocessing input data."""
    # Preprocess data
    processed_data = predictor.preprocess_input(sample_data)
    
    # Check output
    assert isinstance(processed_data, np.ndarray)
    assert processed_data.shape[0] == len(sample_data)

def test_predict(predictor, sample_data):
    """Test making predictions."""
    # Try to predict without loading model
    with pytest.raises(ValueError):
        predictor.predict(sample_data)

def test_predict_proba(predictor, sample_data):
    """Test getting prediction probabilities."""
    # Try to get probabilities without loading model
    with pytest.raises(ValueError):
        predictor.predict_proba(sample_data)

def test_calculate_confidence(predictor, sample_data):
    """Test calculating confidence scores."""
    # Create dummy probabilities
    probabilities = np.array([
        [0.8, 0.2],
        [0.3, 0.7],
        [0.9, 0.1]
    ])
    
    # Calculate confidence
    confidence = predictor.calculate_confidence(probabilities)
    
    # Check confidence scores
    assert isinstance(confidence, np.ndarray)
    assert len(confidence) == len(probabilities)
    assert np.all(confidence >= 0) and np.all(confidence <= 1)

def test_save_predictions(predictor, sample_data, tmp_path):
    """Test saving predictions."""
    # Create dummy predictions
    predictions = np.array(['NO', 'YES', 'NO'])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    
    # Save predictions
    output_path = tmp_path / "predictions.csv"
    predictor.save_predictions(sample_data, predictions, probabilities, str(output_path))
    
    # Check that file was created
    assert output_path.exists()

def test_inference_with_invalid_data(predictor):
    """Test inference with invalid data."""
    # Create invalid data
    data = pd.DataFrame({
        'age': ['invalid'] * 10,
        'time_in_hospital': ['invalid'] * 10
    })
    
    # Test that it handles invalid data gracefully
    with pytest.raises(Exception):
        predictor.preprocess_input(data)

def test_inference_with_missing_data(predictor):
    """Test inference with missing data."""
    # Create data with missing values
    data = pd.DataFrame({
        'age': [25, np.nan, 35],
        'time_in_hospital': [5, 7, np.nan]
    })
    
    # Test that it handles missing data gracefully
    with pytest.raises(Exception):
        predictor.preprocess_input(data) 
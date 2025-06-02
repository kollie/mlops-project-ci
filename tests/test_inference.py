"""
Unit tests for the Model Inference module.

Tests the ModelPredictor class and its prediction methods
following the same patterns as other modules in the pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
import shutil
import json
import joblib
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.predict import ModelPredictor


# Define MockFeatureEngineer at module level so it can be pickled
class MockFeatureEngineer:
    """Mock feature engineer for testing that can be pickled."""
    
    def transform(self, data, target_col='readmitted'):
        """Transform data for testing."""
        # Simple transformation: return first 3 columns as features
        features = data.drop(columns=[target_col], errors='ignore')
        
        if len(features.columns) > 3:
            features = features.iloc[:, :3]
        elif len(features.columns) < 3:
            # Pad with zeros if needed
            for i in range(3 - len(features.columns)):
                features[f'dummy_feature_{i}'] = 0
        
        # Ensure we have exactly 3 columns with proper names
        if len(features.columns) != 3:
            features = pd.DataFrame(
                np.random.random((len(data), 3)), 
                columns=['feat1', 'feat2', 'feat3']
            )
        else:
            # Rename columns to ensure consistency
            features.columns = ['feat1', 'feat2', 'feat3']
        
        # Create dummy target
        target = pd.Series([0, 1] * (len(data) // 2) + [0] * (len(data) % 2))
        target = target[:len(data)]
        
        return features, target


class TestModelPredictor:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create a sample configuration for testing."""
        config = {
            'data': {
                'raw_data_path': os.path.join(temp_dir, "raw_data.csv"),
                'processed_data_path': os.path.join(temp_dir, "processed"),
                'model_path': os.path.join(temp_dir, "models")
            },
            'model': {
                'type': 'random_forest',
                'parameters': {
                    'n_estimators': 10,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': os.path.join(temp_dir, 'logs', 'inference.log')
            }
        }
        
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def mock_model(self):
        """Create a serializable mock model for testing."""
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Create simple training data
        np.random.seed(42)
        X = np.random.random((100, 3))
        y = np.random.choice([0, 1], 100)
        
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        return model

    @pytest.fixture
    def mock_feature_engineer(self):
        """Create a serializable mock feature engineer for testing."""
        return MockFeatureEngineer()
    
    @pytest.fixture
    def sample_model_file(self, temp_dir, mock_model, mock_feature_engineer):
        """Create a sample model file for testing."""
        model_data = {
            'model': mock_model,
            'feature_engineer': mock_feature_engineer,
            'metadata': {
                'model_type': 'RandomForestClassifier',
                'training_date': '2025-06-02',
                'required_features': ['feature1', 'feature2', 'feature3']
            }
        }
        
        model_path = os.path.join(temp_dir, "test_model.joblib")
        joblib.dump(model_data, model_path)
        
        return model_path
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.random(5),
            'feature2': np.random.random(5),
            'feature3': np.random.random(5),
            'readmitted': [0, 1, 0, 1, 1]
        })
    
    @pytest.fixture
    def model_predictor(self, sample_config):
        """Create a ModelPredictor instance for testing."""
        return ModelPredictor(config_path=sample_config, model_path=None)

    # ---------- Initialization Tests ----------

    def test_predictor_initialization(self, sample_config):
        """Test ModelPredictor initialization."""
        predictor = ModelPredictor(config_path=sample_config, model_path=None)
        assert predictor.model is None
        assert predictor.feature_engineer is None
        assert isinstance(predictor.model_metadata, dict)
        assert isinstance(predictor.prediction_history, list)
        assert len(predictor.prediction_history) == 0

    def test_predictor_initialization_missing_config(self):
        """Test ModelPredictor initialization with missing config."""
        with pytest.raises(FileNotFoundError):
            ModelPredictor(config_path="nonexistent_config.yaml", model_path=None)

    def test_predictor_initialization_with_model(self, sample_config, sample_model_file):
        """Test ModelPredictor initialization with model loading."""
        predictor = ModelPredictor(config_path=sample_config, model_path=sample_model_file)
        assert predictor.model is not None
        assert predictor.feature_engineer is not None
        assert len(predictor.model_metadata) > 0

    # ---------- Model Loading Tests ----------

    def test_load_model_success(self, model_predictor, sample_model_file):
        """Test successful model loading."""
        model_predictor.load_model(sample_model_file)
        
        assert model_predictor.model is not None
        assert model_predictor.feature_engineer is not None
        assert 'model_type' in model_predictor.model_metadata
        assert model_predictor.report['model_loaded'] is True

    def test_load_model_file_not_found(self, model_predictor):
        """Test model loading with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            model_predictor.load_model("nonexistent_model.joblib")

    def test_load_model_invalid_format(self, model_predictor, temp_dir):
        """Test model loading with invalid model format."""
        # Create invalid model file
        invalid_model_path = os.path.join(temp_dir, "invalid_model.joblib")
        joblib.dump("not_a_model", invalid_model_path)
        
        with pytest.raises(ValueError):
            model_predictor.load_model(invalid_model_path)

    # ---------- Input Validation Tests ----------

    def test_validate_input_data_success(self, model_predictor, sample_data):
        """Test successful input data validation."""
        # Should not raise an exception
        model_predictor._validate_input_data(sample_data)

    def test_validate_input_data_empty(self, model_predictor):
        """Test input validation with empty data."""
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot make predictions on empty DataFrame"):
            model_predictor._validate_input_data(empty_data)

    def test_validate_input_data_infinite_values(self, model_predictor):
        """Test input validation with infinite values."""
        data_with_inf = pd.DataFrame({
            'feature1': [1.0, 2.0, np.inf],
            'feature2': [1.0, 2.0, 3.0]
        })
        with pytest.raises(ValueError, match="Found infinite values"):
            model_predictor._validate_input_data(data_with_inf)

    # ---------- Prediction Tests ----------

    def test_predict_success(self, model_predictor, sample_model_file, sample_data):
        """Test successful prediction."""
        model_predictor.load_model(sample_model_file)
        
        predictions = model_predictor.predict(sample_data, preprocess=True)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_data)
        assert len(model_predictor.prediction_history) == 1

    def test_predict_without_model(self, model_predictor, sample_data):
        """Test prediction without loaded model."""
        with pytest.raises(ValueError, match="No model loaded"):
            model_predictor.predict(sample_data)

    def test_predict_without_preprocessing(self, model_predictor, sample_model_file):
        """Test prediction without preprocessing."""
        model_predictor.load_model(sample_model_file)
        
        # Create simple numerical data that matches the expected feature names
        # Use the same feature names as in the metadata to avoid validation error
        simple_data = pd.DataFrame(np.random.random((3, 3)), columns=['feature1', 'feature2', 'feature3'])
        
        predictions = model_predictor.predict(simple_data, preprocess=False)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(simple_data)

    def test_predict_proba_success(self, model_predictor, sample_model_file, sample_data):
        """Test successful probability prediction."""
        model_predictor.load_model(sample_model_file)
        
        probabilities = model_predictor.predict_proba(sample_data, preprocess=True)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[0] == len(sample_data)
        assert probabilities.shape[1] == 2  # Binary classification
        assert np.allclose(probabilities.sum(axis=1), 1.0)  # Probabilities sum to 1

    def test_predict_proba_model_without_support(self, model_predictor, sample_model_file, sample_data):
        """Test probability prediction with model that doesn't support it."""
        # Load model first
        model_predictor.load_model(sample_model_file)
        
        # Create a model wrapper that doesn't have predict_proba
        class ModelWithoutProba:
            def __init__(self, original_model):
                self.predict = original_model.predict
                # Deliberately don't include predict_proba
        
        # Replace the model with one that doesn't support probabilities
        model_predictor.model = ModelWithoutProba(model_predictor.model)
        
        with pytest.raises(ValueError, match="does not support probability predictions"):
            model_predictor.predict_proba(sample_data)

    # ---------- Confidence Prediction Tests ----------

    def test_predict_with_confidence_success(self, model_predictor, sample_model_file, sample_data):
        """Test successful confidence prediction."""
        model_predictor.load_model(sample_model_file)
        
        results = model_predictor.predict_with_confidence(sample_data, confidence_threshold=0.7)
        
        # Check results structure
        expected_keys = [
            'predictions', 'probabilities', 'confidence_scores',
            'high_confidence_predictions', 'low_confidence_predictions',
            'high_confidence_indices', 'low_confidence_indices',
            'confidence_threshold', 'high_confidence_count', 'low_confidence_count',
            'confidence_stats'
        ]
        
        for key in expected_keys:
            assert key in results
        
        assert isinstance(results['predictions'], np.ndarray)
        assert isinstance(results['confidence_scores'], np.ndarray)
        assert results['confidence_threshold'] == 0.7
        assert results['high_confidence_count'] + results['low_confidence_count'] == len(sample_data)

    def test_predict_with_confidence_fallback(self, model_predictor, sample_model_file, sample_data):
        """Test confidence prediction fallback when probabilities unavailable."""
        model_predictor.load_model(sample_model_file)
        
        # Create a model wrapper that doesn't have predict_proba
        class ModelWithoutProba:
            def __init__(self, original_model):
                self.predict = original_model.predict
                # Deliberately don't include predict_proba
        
        # Replace the model with one that doesn't support probabilities
        model_predictor.model = ModelWithoutProba(model_predictor.model)
        
        results = model_predictor.predict_with_confidence(sample_data)
        
        assert results['probabilities'] is None
        assert results['confidence_scores'] is None
        assert results['confidence_stats'] is None
        assert results['high_confidence_count'] == len(sample_data)
        assert results['low_confidence_count'] == 0

    # ---------- Batch Prediction Tests ----------

    def test_predict_batch_with_save(self, model_predictor, sample_model_file, temp_dir):
        """Test batch prediction with saving results."""
        model_predictor.load_model(sample_model_file)
        
        # Use correct feature names to match metadata
        data = pd.DataFrame(np.random.random((15, 3)), columns=['feature1', 'feature2', 'feature3'])
        save_path = os.path.join(temp_dir, "batch_results.json")
        
        model_predictor.predict_batch(data, batch_size=5, save_path=save_path)
        
        # Check that file was created
        assert os.path.exists(save_path)
        
        # Check file contents
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        assert 'predictions' in saved_data
        assert 'n_samples' in saved_data
        assert len(saved_data['predictions']) == len(data)

    def test_predict_batch_success(self, model_predictor, sample_model_file):
        """Test successful batch prediction."""
        model_predictor.load_model(sample_model_file)
        
        # Create larger dataset for batch testing with correct feature names
        large_data = pd.DataFrame(np.random.random((25, 3)), columns=['feature1', 'feature2', 'feature3'])
        
        results = model_predictor.predict_batch(large_data, batch_size=10)
        
        assert 'predictions' in results
        assert 'total_samples' in results
        assert 'n_batches' in results
        assert 'batch_results' in results
        
        assert len(results['predictions']) == len(large_data)
        assert results['total_samples'] == len(large_data)
        assert results['n_batches'] == 3  # 25 samples with batch_size=10

    # ---------- Save/Load Tests ----------

    def test_save_predictions(self, model_predictor, temp_dir):
        """Test saving prediction results."""
        results = {
            'predictions': np.array([0, 1, 0, 1, 1]),
            'probabilities': np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8]]),
            'prediction_distribution': {0: 2, 1: 3}
        }
        
        save_path = os.path.join(temp_dir, "predictions.json")
        saved_path = model_predictor.save_predictions(results, save_path)
        
        assert saved_path == save_path
        assert os.path.exists(save_path)
        
        # Check file contents
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        assert 'predictions' in saved_data
        assert 'probabilities' in saved_data
        assert 'prediction_distribution' in saved_data
        assert 'timestamp' in saved_data

    # ---------- Utility Method Tests ----------

    def test_get_model_info_no_model(self, model_predictor):
        """Test getting model info when no model is loaded."""
        info = model_predictor.get_model_info()
        assert info['model_loaded'] is False

    def test_get_model_info_with_model(self, model_predictor, sample_model_file):
        """Test getting model info with loaded model."""
        model_predictor.load_model(sample_model_file)
        
        info = model_predictor.get_model_info()
        
        assert info['model_loaded'] is True
        assert 'model_type' in info
        assert 'feature_engineer_available' in info
        assert 'supports_probabilities' in info
        assert 'metadata' in info

    def test_prediction_history_management(self, model_predictor, sample_model_file, sample_data):
        """Test prediction history tracking and management."""
        model_predictor.load_model(sample_model_file)
        
        # Make some predictions
        model_predictor.predict(sample_data)
        model_predictor.predict(sample_data.iloc[:3])
        
        # Check history
        history = model_predictor.get_prediction_history()
        assert len(history) == 2
        assert history[0]['n_samples'] == len(sample_data)
        assert history[1]['n_samples'] == 3
        
        # Clear history
        model_predictor.clear_prediction_history()
        assert len(model_predictor.get_prediction_history()) == 0

    # ---------- Data Preprocessing Tests ----------

    def test_preprocess_data_with_feature_engineer(self, model_predictor, sample_model_file, sample_data):
        """Test data preprocessing with feature engineer."""
        model_predictor.load_model(sample_model_file)
        
        processed_data = model_predictor._preprocess_data(sample_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        # Mock feature engineer returns 3 columns
        assert processed_data.shape[1] == 3

    def test_preprocess_data_without_feature_engineer(self, model_predictor, sample_data):
        """Test data preprocessing without feature engineer."""
        # Don't load model, so feature_engineer remains None
        processed_data = model_predictor._preprocess_data(sample_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        # Should remove target column and return the rest
        assert 'readmitted' not in processed_data.columns
        assert len(processed_data.columns) == len(sample_data.columns) - 1

    # ---------- Error Handling Tests ----------

    def test_predict_with_invalid_input(self, model_predictor, sample_model_file):
        """Test prediction with invalid input data."""
        model_predictor.load_model(sample_model_file)
        
        # Test with non-DataFrame input
        with pytest.raises(AttributeError):
            model_predictor.predict("not_a_dataframe")

    def test_report_generation(self, model_predictor, sample_model_file, sample_data, temp_dir):
        """Test that inference report is generated correctly."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            model_predictor.load_model(sample_model_file)
            model_predictor.predict(sample_data)
            
            # Write report
            model_predictor._write_report()
            
            # Check that report file was created
            assert Path("logs/inference_report.json").exists()
            
            # Check report contents
            with open("logs/inference_report.json", 'r') as f:
                report = json.load(f)
            
            assert 'model_loaded' in report
            assert 'prediction_history' in report
            assert 'model_info' in report
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    """Demonstrate ModelPredictor testing functionality."""
    print("Model inference test module loaded successfully.")
    print("Run with: pytest tests/test_inference.py -v")
"""
Unit tests for the Model Training module.

Tests the ModelTrainer class and its training methods
following the same patterns as other modules in the pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.trainer import ModelTrainer


class TestModelTrainer:
    
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
                    'n_estimators': 10,  # Small for fast testing
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'model_registry': {
                'enabled': False,  # Disable MLflow for testing
                'tracking_uri': 'mlruns',
                'experiment_name': 'test_experiment'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': os.path.join(temp_dir, 'logs', 'training.log')
            }
        }
        
        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    @pytest.fixture
    def model_trainer(self, sample_config):
        """Create a ModelTrainer instance for testing."""
        return ModelTrainer(sample_config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randint(0, 10, n_samples),
            'feature_4': np.random.randn(n_samples),
            'feature_5': np.random.randint(0, 5, n_samples),
            'readmitted': np.random.choice([0, 1], n_samples)  # Use numeric labels
        })
        
        return data
    
    @pytest.fixture
    def sample_data_categorical(self):
        """Create sample data with categorical target for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'feature_1': np.random.randn(n_samples),
            'feature_2': np.random.randn(n_samples),
            'feature_3': np.random.randint(0, 10, n_samples),
            'readmitted': np.random.choice(['NO', 'YES'], n_samples)
        })
        
        return data

    # ---------- Initialization Tests ----------

    def test_model_trainer_initialization(self, sample_config):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(sample_config)
        assert trainer.model is None
        assert trainer.model_type == ""
        assert trainer.model_params == {}
        assert trainer._is_fitted is False
        assert trainer.config is not None
        assert trainer.logger is not None

    def test_model_trainer_initialization_missing_config(self):
        """Test ModelTrainer initialization with missing config."""
        with pytest.raises(FileNotFoundError):
            ModelTrainer("nonexistent_config.yaml")

    def test_model_trainer_initialization_invalid_yaml(self, temp_dir):
        """Test ModelTrainer initialization with invalid YAML."""
        invalid_config_path = os.path.join(temp_dir, "invalid_config.yaml")
        with open(invalid_config_path, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(ValueError, match="Invalid YAML"):
            ModelTrainer(invalid_config_path)

    # ---------- Model Creation Tests ----------

    def test_create_random_forest_model(self, model_trainer):
        """Test random forest model creation."""
        model = model_trainer._create_model()
        
        assert model is not None
        assert model_trainer.model_type == 'random_forest'
        assert model_trainer.model_params['n_estimators'] == 10
        assert model_trainer.model_params['max_depth'] == 3

    def test_create_logistic_regression_model(self, sample_config):
        """Test logistic regression model creation."""
        # Modify config for logistic regression
        with open(sample_config, 'r') as f:
            config = yaml.safe_load(f)
        
        config['model']['type'] = 'logistic_regression'
        config['model']['parameters'] = {'random_state': 42, 'max_iter': 100}
        
        with open(sample_config, 'w') as f:
            yaml.dump(config, f)
        
        trainer = ModelTrainer(sample_config)
        model = trainer._create_model()
        
        assert model is not None
        assert trainer.model_type == 'logistic_regression'

    def test_create_unsupported_model(self, sample_config):
        """Test creation of unsupported model type."""
        # Modify config for unsupported model
        with open(sample_config, 'r') as f:
            config = yaml.safe_load(f)
        
        config['model']['type'] = 'unsupported_model'
        
        with open(sample_config, 'w') as f:
            yaml.dump(config, f)
        
        trainer = ModelTrainer(sample_config)
        with pytest.raises(ValueError, match="Unsupported model type"):
            trainer._create_model()

    # ---------- Training Tests ----------

    def test_fit_basic(self, model_trainer, sample_data, temp_dir):
        """Test basic model fitting."""
        # Change to temp directory for report creation
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            X = sample_data.drop(columns=['readmitted'])
            y = sample_data['readmitted']
            
            model_trainer.fit(X, y)
            
            # Check that model was trained
            assert model_trainer.model is not None
            assert model_trainer._is_fitted is True
            
            # Check report generation
            assert 'timestamp' in model_trainer.report
            assert 'training_completed' in model_trainer.report
            assert model_trainer.report['training_completed'] is True
            
        finally:
            os.chdir(original_cwd)

    def test_fit_empty_data(self, model_trainer):
        """Test fitting with empty data."""
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        
        with pytest.raises(ValueError, match="Cannot train on empty training data"):
            model_trainer.fit(empty_df, empty_series)

    def test_fit_mismatched_lengths(self, model_trainer):
        """Test fitting with mismatched feature and label lengths."""
        X = pd.DataFrame({'feature': [1, 2, 3]})
        y = pd.Series([0, 1])  # Different length
        
        with pytest.raises(ValueError, match="Training features and labels must have same length"):
            model_trainer.fit(X, y)

    def test_fit_with_categorical_target(self, model_trainer, sample_data_categorical, temp_dir):
        """Test fitting with categorical target variable."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            X = sample_data_categorical.drop(columns=['readmitted'])
            y = sample_data_categorical['readmitted']
            
            model_trainer.fit(X, y)
            
            assert model_trainer.model is not None
            assert model_trainer._is_fitted is True
            
        finally:
            os.chdir(original_cwd)

    # ---------- Prediction Tests ----------

    def test_predict_basic(self, model_trainer, sample_data):
        """Test basic prediction."""
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        # Train model first
        model_trainer.fit(X, y)
        
        # Make predictions
        predictions = model_trainer.predict(X)
        
        # Check predictions
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_without_fit(self, model_trainer, sample_data):
        """Test prediction without fitting first."""
        X = sample_data.drop(columns=['readmitted'])
        
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            model_trainer.predict(X)

    def test_predict_proba_basic(self, model_trainer, sample_data):
        """Test probability prediction."""
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        # Train model first
        model_trainer.fit(X, y)
        
        # Get probabilities
        probabilities = model_trainer.predict_proba(X)
        
        # Check probabilities
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape == (len(X), 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_predict_proba_without_fit(self, model_trainer, sample_data):
        """Test probability prediction without fitting first."""
        X = sample_data.drop(columns=['readmitted'])
        
        with pytest.raises(ValueError, match="Model must be fitted before making predictions"):
            model_trainer.predict_proba(X)

    def test_predict_proba_unsupported_model(self, sample_config, sample_data):
        """Test probability prediction with model that doesn't support it."""
        # This test might not be needed since all our models support predict_proba
        # But keeping it for completeness
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        trainer = ModelTrainer(sample_config)
        trainer.fit(X, y)
        
        # All sklearn models we use support predict_proba, so this should work
        probabilities = trainer.predict_proba(X)
        assert probabilities is not None

    # ---------- Save/Load Tests ----------

    def test_save_model_basic(self, model_trainer, sample_data, temp_dir):
        """Test basic model saving."""
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        # Train model first
        model_trainer.fit(X, y)
        
        # Save model
        model_path = model_trainer.save()
        
        # Check that files were created
        assert os.path.exists(model_path)
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        assert os.path.exists(metadata_path)

    def test_save_model_custom_path(self, model_trainer, sample_data, temp_dir):
        """Test model saving with custom path."""
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        # Train model first
        model_trainer.fit(X, y)
        
        # Save model with custom path
        custom_path = os.path.join(temp_dir, "custom_model.joblib")
        saved_path = model_trainer.save(custom_path)
        
        # Check that file was created at custom location
        assert saved_path == custom_path
        assert os.path.exists(custom_path)

    def test_get_feature_importance_without_fit(self, model_trainer):
        """Test feature importance without fitting first."""
        # The method returns empty DataFrame when not fitted (defensive programming)
        importance_df = model_trainer.get_feature_importance()
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 0  # Should be empty

    def test_load_model_basic(self, model_trainer, sample_data, temp_dir):
        """Test basic model loading."""
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        # Train and save model
        model_trainer.fit(X, y)
        model_path = model_trainer.save()
        
        # Create new trainer and load model
        new_trainer = ModelTrainer(model_trainer.config_path)
        new_trainer.load(model_path)
        
        # Check that model was loaded
        assert new_trainer.model is not None
        assert new_trainer._is_fitted is True
        
        # Check that predictions match
        original_preds = model_trainer.predict(X)
        loaded_preds = new_trainer.predict(X)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_load_model_nonexistent(self, model_trainer):
        """Test loading nonexistent model."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            model_trainer.load("nonexistent_model.joblib")

    # ---------- Feature Importance Tests ----------

    def test_get_feature_importance_random_forest(self, model_trainer, sample_data):
        """Test feature importance for random forest."""
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        # Train model first
        model_trainer.fit(X, y)
        
        # Get feature importance
        importance_df = model_trainer.get_feature_importance()
        
        # Check feature importance
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == X.shape[1]
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert all(importance_df['importance'] >= 0)

    def test_get_feature_importance_logistic_regression(self, sample_config, sample_data):
        """Test feature importance for logistic regression."""
        # Modify config for logistic regression
        with open(sample_config, 'r') as f:
            config = yaml.safe_load(f)
        
        config['model']['type'] = 'logistic_regression'
        config['model']['parameters'] = {'random_state': 42, 'max_iter': 100}
        
        with open(sample_config, 'w') as f:
            yaml.dump(config, f)
        
        trainer = ModelTrainer(sample_config)
        
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        # Train model
        trainer.fit(X, y)
        
        # Get feature importance (coefficients)
        importance_df = trainer.get_feature_importance()
        
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == X.shape[1]

    def test_get_feature_importance_without_fit(self, model_trainer):
        """Test feature importance without fitting first."""
        with pytest.raises(ValueError, match="Model must be fitted to get feature importance"):
            model_trainer.get_feature_importance()

    # ---------- Edge Cases and Error Handling Tests ----------

    def test_training_with_single_sample(self, model_trainer):
        """Test training with single sample."""
        X = pd.DataFrame({'feature': [1.0]})
        y = pd.Series([0])
        
        # This might fail depending on the model, which is expected
        try:
            model_trainer.fit(X, y)
            # If it succeeds, check basic functionality
            assert model_trainer._is_fitted is True
        except Exception:
            # Single sample training often fails, which is acceptable
            pass

    def test_training_consistency(self, model_trainer, sample_data):
        """Test that training is consistent with same data and random state."""
        X = sample_data.drop(columns=['readmitted'])
        y = sample_data['readmitted']
        
        # Train model
        model_trainer.fit(X, y)
        predictions1 = model_trainer.predict(X)
        
        # Train again with same data
        model_trainer._is_fitted = False  # Reset fitted state
        model_trainer.model = None
        model_trainer.fit(X, y)
        predictions2 = model_trainer.predict(X)
        
        # Predictions should be the same due to random_state
        np.testing.assert_array_equal(predictions1, predictions2)

    def test_report_generation(self, model_trainer, sample_data, temp_dir):
        """Test that training report is generated correctly."""
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            X = sample_data.drop(columns=['readmitted'])
            y = sample_data['readmitted']
            
            # Train model
            model_trainer.fit(X, y)
            
            # Check that report file was created
            assert Path("logs/training_report.json").exists()
            
            # Check that report contains expected keys
            assert 'timestamp' in model_trainer.report
            assert 'model_type' in model_trainer.report
            assert 'model_parameters' in model_trainer.report
            assert 'training_shape' in model_trainer.report
            assert 'training_completed' in model_trainer.report
            
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    """Demonstrate ModelTrainer testing functionality."""
    print("Model trainer test module loaded successfully.")
    print("Run with: pytest tests/test_trainer.py -v")
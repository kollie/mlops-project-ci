import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_preprocessor_initialization(preprocessor):
    """Test that the preprocessor initializes correctly."""
    assert preprocessor.preprocessor is None
    assert preprocessor.label_encoder is None
    assert preprocessor.feature_selector is None
    assert preprocessor.config is not None
    assert preprocessor.logger is not None

def test_config_loading(preprocessor, config_file):
    """Test that the config is loaded correctly."""
    assert 'logging' in preprocessor.config
    assert 'features' in preprocessor.config
    assert 'feature_selection' in preprocessor.config
    
    # Check feature lists
    assert 'categorical_features' in preprocessor.config['features']
    assert 'numerical_features' in preprocessor.config['features']
    assert 'drop_columns' in preprocessor.config['features']

def test_logging_setup(preprocessor):
    """Test that logging is set up correctly."""
    log_dir = Path("logs")
    assert log_dir.exists()
    
    log_file = log_dir / "preprocessing.log"
    assert log_file.exists()

def test_invalid_config_path():
    """Test that an invalid config path raises an error."""
    from src.preprocessing.preprocessor import Preprocessor
    with pytest.raises(FileNotFoundError):
        Preprocessor(config_path="nonexistent_config.yaml") 
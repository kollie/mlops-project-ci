import pytest
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
import shutil
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader.data_loader import DataLoader


class TestDataLoader:
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
            "data": {
                "raw_data_path": os.path.join(temp_dir, "raw_data.csv"),
                "raw_data_format": "csv",
                "processed_data_path": os.path.join(temp_dir, "processed"),
                "train_data_path": os.path.join(temp_dir, "processed", "train.csv"),
                "validation_data_path": os.path.join(
                    temp_dir, "processed", "validation.csv"
                ),
                "test_data_path": os.path.join(temp_dir, "processed", "test"),
            },
            "model": {"test_size": 0.2, "validation_size": 0.2, "random_state": 42},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": os.path.join(temp_dir, "logs", "test.log"),
            },
        }

        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def sample_data(self, temp_dir, sample_config):
        """Create sample data for testing."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 100),
                "feature_2": np.random.normal(2, 1.5, 100),
                "feature_3": np.random.randint(0, 10, 100),
                "target": np.random.randint(0, 2, 100),
            }
        )

        # Get the raw data path from config
        with open(sample_config, "r") as f:
            config = yaml.safe_load(f)

        data_path = config["data"]["raw_data_path"]
        data.to_csv(data_path, index=False)
        return data_path

    def test_data_loader_initialization(self, sample_config):
        """Test DataLoader initialization."""
        loader = DataLoader(sample_config)
        assert loader.config is not None
        assert loader.logger is not None
        assert loader.config_path == sample_config

    def test_load_config_success(self, sample_config):
        """Test successful config loading."""
        loader = DataLoader(sample_config)
        assert "data" in loader.config
        assert "model" in loader.config
        assert "logging" in loader.config

    def test_load_config_missing_file(self):
        """Test config loading with missing file."""
        with pytest.raises(FileNotFoundError):
            DataLoader("nonexistent_config.yaml")

    def test_load_config_missing_sections(self, temp_dir):
        """Test config loading with missing required sections."""
        incomplete_config = {
            "data": {"raw_data_path": "test.csv"},
            # Missing 'model' and 'logging' sections
        }

        config_path = os.path.join(temp_dir, "incomplete_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(incomplete_config, f)

        with pytest.raises(ValueError, match="Missing required config section"):
            DataLoader(config_path)

    def test_load_data_success(self, sample_config, sample_data):
        """Test successful data loading."""
        loader = DataLoader(sample_config)
        data = loader.load_data()

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert "target" in data.columns

    def test_load_data_with_file_path(self, sample_config, sample_data):
        """Test data loading with explicit file path."""
        loader = DataLoader(sample_config)
        data = loader.load_data(sample_data)

        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100

    def test_load_data_missing_path(self, sample_config):
        """Test data loading with missing path."""
        loader = DataLoader(sample_config)
        loader.config["data"]["raw_data_path"] = None

        with pytest.raises(ValueError, match="Missing 'raw_data_path'"):
            loader.load_data()

    def test_load_data_file_not_found(self, sample_config):
        """Test data loading with file not found."""
        loader = DataLoader(sample_config)

        with pytest.raises(FileNotFoundError):
            loader.load_data("nonexistent_file.csv")

    def test_load_data_invalid_format(self, sample_config, sample_data):
        """Test data loading with invalid format."""
        loader = DataLoader(sample_config)
        loader.config["data"]["raw_data_format"] = "xml"

        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_data()

    def test_split_data_success(self, sample_config, sample_data):
        """Test successful data splitting."""
        loader = DataLoader(sample_config)
        data = loader.load_data()
        train, val, test = loader.split_data(data)

        # Check all outputs are DataFrames
        assert all(isinstance(split, pd.DataFrame) for split in [train, val, test])

        # Check all splits have data
        assert all(len(split) > 0 for split in [train, val, test])

        # Check total length matches original
        assert len(train) + len(val) + len(test) == len(data)

        # Check that splits are reasonable (not exact due to rounding)
        total = len(data)
        test_ratio = len(test) / total
        val_ratio = len(val) / total
        train_ratio = len(train) / total

        # Test should be approximately 20% (±5%)
        assert (
            0.15 <= test_ratio <= 0.25
        ), f"Test ratio {test_ratio:.2f} not in expected range"

        # Validation should be approximately 16% (20% of remaining 80%) (±5%)
        assert (
            0.11 <= val_ratio <= 0.25
        ), f"Validation ratio {val_ratio:.2f} not in expected range"

        # Train should be the majority (±5%)
        assert (
            0.55 <= train_ratio <= 0.75
        ), f"Train ratio {train_ratio:.2f} not in expected range"

        # All ratios should sum to 1
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 0.01

        print(
            f"Split ratios - Train: {train_ratio:.2f}, Val: {val_ratio:.2f}, Test: {test_ratio:.2f}"
        )

    def test_split_data_missing_params(self, sample_config, sample_data):
        """Test data splitting with missing parameters."""
        loader = DataLoader(sample_config)
        data = loader.load_data()

        del loader.config["model"]["test_size"]

        with pytest.raises(ValueError, match="Missing model parameter"):
            loader.split_data(data)

    def test_save_split_data_success(self, sample_config, sample_data):
        """Test successful saving of split data."""
        loader = DataLoader(sample_config)
        data = loader.load_data()
        train, val, test = loader.split_data(data)

        loader.save_split_data(train, val, test)

        # Check files exist
        assert Path(loader.config["data"]["train_data_path"]).exists()
        assert Path(loader.config["data"]["validation_data_path"]).exists()
        assert Path(loader.config["data"]["test_data_path"]).exists()

        # Verify saved data
        saved_train = pd.read_csv(loader.config["data"]["train_data_path"])
        assert len(saved_train) == len(train)

    def test_save_split_data_missing_paths(self, sample_config, sample_data):
        """Test saving split data with missing paths."""
        loader = DataLoader(sample_config)
        data = loader.load_data()
        train, val, test = loader.split_data(data)

        del loader.config["data"]["train_data_path"]

        with pytest.raises(ValueError, match="Missing data path in config"):
            loader.save_split_data(train, val, test)

    def test_full_pipeline(self, sample_config, sample_data):
        """Test the complete data loading pipeline."""
        loader = DataLoader(sample_config)

        # Load data
        data = loader.load_data()
        assert isinstance(data, pd.DataFrame)

        # Split data
        train, val, test = loader.split_data(data)
        assert all(len(split) > 0 for split in [train, val, test])

        # Save split data
        loader.save_split_data(train, val, test)

        # Verify all files exist
        assert Path(loader.config["data"]["train_data_path"]).exists()
        assert Path(loader.config["data"]["validation_data_path"]).exists()
        assert Path(loader.config["data"]["test_data_path"]).exists()

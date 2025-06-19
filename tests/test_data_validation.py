import pytest
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.validation.data_validator import DataValidator


class TestDataValidator:
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
                "test_data_path": os.path.join(temp_dir, "processed", "test.csv"),
            },
            "model": {"test_size": 0.2, "validation_size": 0.2, "random_state": 42},
            "features": {
                "numerical_features": [
                    "time_in_hospital",
                    "num_lab_procedures",
                    "num_procedures",
                    "num_medications",
                    "number_outpatient",
                    "number_emergency",
                    "number_inpatient",
                    "number_diagnoses",
                ],
                "categorical_features": ["race", "gender", "age_group"],
                "target_column": "readmitted",
            },
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
    def validator(self, sample_config):
        """Create a DataValidator with test config."""
        return DataValidator(sample_config)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "race": ["Caucasian"] * 101,
                "gender": ["Female"] * 101,
                "age_group": ["[30-40)"] * 101,
                "time_in_hospital": np.random.randint(1, 15, 101),
                "num_lab_procedures": np.random.randint(1, 100, 101),
                "num_procedures": np.random.randint(0, 10, 101),
                "num_medications": np.random.randint(1, 50, 101),
                "number_outpatient": np.random.randint(0, 20, 101),
                "number_emergency": np.random.randint(0, 20, 101),
                "number_inpatient": np.random.randint(0, 20, 101),
                "number_diagnoses": np.random.randint(1, 20, 101),
                "readmitted": ["NO"] * 50 + ["YES"] * 51,  # Balanced target
            }
        )

    @pytest.fixture
    def sample_data_with_age_fallback(self):
        """Create sample data with 'age' instead of 'age_group' to test fallback."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "race": ["Caucasian"] * 101,
                "gender": ["Female"] * 101,
                "age": ["[30-40)"] * 101,  # Using 'age' instead of 'age_group'
                "time_in_hospital": np.random.randint(1, 15, 101),
                "num_lab_procedures": np.random.randint(1, 100, 101),
                "num_procedures": np.random.randint(0, 10, 101),
                "num_medications": np.random.randint(1, 50, 101),
                "number_outpatient": np.random.randint(0, 20, 101),
                "number_emergency": np.random.randint(0, 20, 101),
                "number_inpatient": np.random.randint(0, 20, 101),
                "number_diagnoses": np.random.randint(1, 20, 101),
                "readmitted": ["NO"] * 50 + ["YES"] * 51,
            }
        )

    # ---------- Unit Tests ----------

    def test_validator_initialization(self, sample_config):
        """Test DataValidator initialization."""
        validator = DataValidator(sample_config)
        assert validator.config is not None
        assert validator.logger is not None
        assert isinstance(validator.required_columns, list)
        assert len(validator.required_columns) > 0

    def test_validator_initialization_missing_config(self):
        """Test DataValidator initialization with missing config."""
        with pytest.raises(FileNotFoundError):
            DataValidator("nonexistent_config.yaml")

    def test_validate_schema_passes(self, validator, sample_data):
        """Test schema validation passes with valid data."""
        result = validator.validate_schema(sample_data)
        assert result is True
        assert validator.report.get("schema_valid") is True

    def test_validate_schema_fails_missing_column(self, validator, sample_data):
        """Test schema validation fails with missing required column."""
        sample_data_missing = sample_data.drop(columns=["race"])
        result = validator.validate_schema(sample_data_missing)
        assert result is False
        assert "missing_columns" in validator.report
        assert "race" in validator.report["missing_columns"]

    def test_validate_schema_age_fallback(
        self, validator, sample_data_with_age_fallback
    ):
        """Test schema validation handles age/age_group fallback."""
        result = validator.validate_schema(sample_data_with_age_fallback)
        assert result is True  # Should pass due to age->age_group fallback

    def test_validate_data_types_passes(self, validator, sample_data):
        """Test data type validation passes with correct types."""
        result = validator.validate_data_types(sample_data)
        assert result is True
        assert validator.report.get("data_types_valid") is True

    def test_validate_data_types_fails(self, validator, sample_data):
        """Test data type validation fails with incorrect types."""
        # Convert a numerical column to string
        sample_data["num_medications"] = sample_data["num_medications"].astype(str)
        result = validator.validate_data_types(sample_data)
        assert result is False
        assert "type_issues" in validator.report
        assert "num_medications" in validator.report["type_issues"]

    def test_validate_missing_values_passes(self, validator, sample_data):
        """Test missing values validation passes with no missing data."""
        result = validator.validate_missing_values(sample_data)
        assert result is True
        assert validator.report.get("missing_values_valid") is True

    def test_validate_missing_values_fails(self, validator, sample_data):
        """Test missing values validation fails with high missing ratio."""
        # Add many missing values to a column
        sample_data.loc[:80, "race"] = np.nan  # 80% missing
        result = validator.validate_missing_values(sample_data, threshold=0.1)
        assert result is False
        assert "high_missing_columns" in validator.report
        assert "race" in validator.report["high_missing_columns"]

    def test_validate_target_distribution_passes(self, validator, sample_data):
        """Test target distribution validation passes with balanced data."""
        result = validator.validate_target_distribution(sample_data)
        assert result is True
        assert validator.report.get("target_distribution_valid") is True

    def test_validate_target_distribution_fails(self, validator, sample_data):
        """Test target distribution validation fails with imbalanced data."""
        # Create severe imbalance
        sample_data["readmitted"] = ["NO"] * 100 + ["RARE"]
        result = validator.validate_target_distribution(
            sample_data, min_class_ratio=0.05
        )
        assert result is False
        assert validator.report.get("target_imbalance") is True

    def test_validate_all_combined_pass(self, validator, sample_data):
        """Test all validations pass together."""
        results = validator.validate_all(sample_data)
        assert all(results.values())
        assert validator.report.get("all_validations_passed") is True

    def test_validate_all_combined_fails(self, validator, sample_data):
        """Test validation fails when one check fails."""
        sample_data.drop(columns=["race"], inplace=True)
        results = validator.validate_all(sample_data)
        assert results["schema"] is False
        assert validator.report.get("all_validations_passed") is False

    def test_validate_and_clean_success(self, validator, sample_data):
        """Test validation and cleaning succeeds with valid data."""
        clean_data = validator.validate_and_clean(sample_data)
        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) > 0
        assert validator.report.get("cleaning_completed") is True

    def test_validate_and_clean_with_missing_values(self, validator, sample_data):
        """Test validation and cleaning handles missing values."""
        # Add some missing values
        sample_data.loc[:5, "race"] = np.nan
        clean_data = validator.validate_and_clean(sample_data, missing_strategy="drop")

        assert isinstance(clean_data, pd.DataFrame)
        assert len(clean_data) < len(sample_data)  # Some rows should be dropped
        assert clean_data["race"].isnull().sum() == 0  # No missing values in result

    def test_validate_and_clean_schema_failure(self, validator, sample_data):
        """Test validation and cleaning fails when schema validation fails."""
        sample_data.drop(columns=["race"], inplace=True)

        with pytest.raises(ValueError, match="Schema validation failed"):
            validator.validate_and_clean(sample_data)

    def test_empty_dataframe_fails_schema(self, validator):
        """Test empty dataframe fails schema validation."""
        empty_df = pd.DataFrame()
        result = validator.validate_schema(empty_df)
        assert result is False

    def test_validator_without_features_config(self, temp_dir):
        """Test validator works when features config is missing."""
        # Create config without features section
        config = {
            "data": {"raw_data_path": "test.csv"},
            "model": {"test_size": 0.2},
            "logging": {"level": "INFO", "file": os.path.join(temp_dir, "test.log")},
        }

        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        validator = DataValidator(config_path)
        assert validator.required_columns == []

        # Should pass schema validation when no requirements are defined
        sample_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        assert validator.validate_schema(sample_data) is True

    def test_report_generation(self, validator, sample_data):
        """Test that validation report is generated correctly."""
        validator.validate_all(sample_data)

        # Check report contains expected keys
        assert "initial_rows" in validator.report
        assert "initial_columns" in validator.report
        assert "timestamp" in validator.report
        assert "validation_results" in validator.report

        # Check that report values make sense
        assert validator.report["initial_rows"] == len(sample_data)
        assert validator.report["initial_columns"] == len(sample_data.columns)

    def test_missing_value_strategies(self, validator, sample_data):
        """Test different missing value handling strategies."""
        # Add missing values
        sample_data.loc[:10, "race"] = np.nan

        # Test drop strategy
        clean_data_drop = validator.validate_and_clean(
            sample_data.copy(), missing_strategy="drop"
        )
        assert (
            len(clean_data_drop) == len(sample_data) - 11
        )  # 11 rows with missing values

        # Test unknown strategy (should fallback to drop)
        clean_data_unknown = validator.validate_and_clean(
            sample_data.copy(), missing_strategy="unknown"
        )
        assert len(clean_data_unknown) == len(sample_data) - 11

    def test_validation_with_custom_thresholds(self, validator, sample_data):
        """Test validation with custom threshold parameters."""
        # Add some missing values
        sample_data.loc[:20, "race"] = np.nan  # 20% missing

        # Test with strict threshold
        results_strict = validator.validate_all(sample_data, missing_threshold=0.1)
        assert results_strict["missing_values"] is False

        # Test with lenient threshold
        results_lenient = validator.validate_all(sample_data, missing_threshold=0.3)
        assert results_lenient["missing_values"] is True

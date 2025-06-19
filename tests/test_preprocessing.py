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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.preprocessor import Preprocessor


class TestPreprocessor:
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
            "features": {
                "categorical_features": ["age", "gender", "race"],
                "numerical_features": [
                    "time_in_hospital",
                    "num_lab_procedures",
                    "num_procedures",
                    "num_medications",
                    "number_outpatient",
                    "number_emergency",
                    "number_inpatient",
                ],
                "drop_columns": ["diag_1", "diag_2", "diag_3"],
                "target_column": "readmitted",
            },
            "feature_selection": {"n_features": 10},
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": os.path.join(temp_dir, "logs", "preprocessing.log"),
            },
        }

        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def preprocessor(self, sample_config):
        """Create a Preprocessor with test config."""
        return Preprocessor(sample_config)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)

        # Create base patterns
        age_pattern = ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"]
        gender_pattern = ["F", "M"]
        race_pattern = ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"]
        diag_pattern = ["250.01", "401.9", "272.4", "250.00", "401.1"]
        readmitted_pattern = ["NO", "YES"]

        # Generate exactly 100 values for each column
        data = {
            "age": (age_pattern * 20)[:100],
            "gender": (gender_pattern * 50)[:100],
            "race": (race_pattern * 20)[:100],
            "time_in_hospital": list(np.random.randint(1, 15, 100)),
            "num_lab_procedures": list(np.random.randint(10, 100, 100)),
            "num_procedures": list(np.random.randint(0, 10, 100)),
            "num_medications": list(np.random.randint(1, 50, 100)),
            "number_outpatient": list(np.random.randint(0, 20, 100)),
            "number_emergency": list(np.random.randint(0, 20, 100)),
            "number_inpatient": list(np.random.randint(0, 20, 100)),
            "diag_1": (diag_pattern * 20)[:100],
            "diag_2": (diag_pattern * 20)[:100],
            "diag_3": (diag_pattern * 20)[:100],
            "readmitted": (readmitted_pattern * 50)[:100],
        }

        # Verify all columns have exactly 100 values
        for key, values in data.items():
            assert (
                len(values) == 100
            ), f"Column {key} has {len(values)} values, expected 100"

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values for testing."""
        np.random.seed(42)

        # Create base patterns that will be repeated
        age_pattern = ["[0-10)", "?", "[20-30)", "[30-40)", "[40-50)"]
        gender_pattern = ["F", "M", "?"]
        race_pattern = ["Caucasian", "?", "Hispanic", "Asian", "Other"]
        diag1_pattern = ["250.01", "?", "272.4", "250.00", "401.1"]
        diag2_pattern = ["250.00", "272.4", "?", "250.01", "401.1"]
        diag3_pattern = ["272.4", "250.00", "401.1", "?", "401.9"]
        readmitted_pattern = ["NO", "YES"]

        # Generate exactly 100 values for each column
        data = {
            "age": (age_pattern * 20)[:100],  # Repeat and truncate to exactly 100
            "gender": (gender_pattern * 34)[:100],  # Repeat and truncate to exactly 100
            "race": (race_pattern * 20)[:100],  # Repeat and truncate to exactly 100
            "diag_1": (diag1_pattern * 20)[:100],  # Repeat and truncate to exactly 100
            "diag_2": (diag2_pattern * 20)[:100],  # Repeat and truncate to exactly 100
            "diag_3": (diag3_pattern * 20)[:100],  # Repeat and truncate to exactly 100
            "readmitted": (readmitted_pattern * 50)[
                :100
            ],  # Repeat and truncate to exactly 100
        }

        # Generate numerical columns with some missing values
        # Start with random values
        time_in_hospital = list(np.random.randint(1, 15, 100))
        num_lab_procedures = list(np.random.randint(10, 100, 100))
        num_procedures = list(np.random.randint(0, 10, 100))
        num_medications = list(np.random.randint(1, 50, 100))

        # Introduce some NaN values at specific positions
        time_in_hospital[1] = np.nan
        time_in_hospital[15] = np.nan
        num_lab_procedures[2] = np.nan
        num_lab_procedures[25] = np.nan
        num_procedures[3] = np.nan
        num_procedures[35] = np.nan
        num_medications[4] = np.nan
        num_medications[45] = np.nan

        # Add numerical columns to data
        data.update(
            {
                "time_in_hospital": time_in_hospital,
                "num_lab_procedures": num_lab_procedures,
                "num_procedures": num_procedures,
                "num_medications": num_medications,
                "number_outpatient": list(np.random.randint(0, 20, 100)),
                "number_emergency": list(np.random.randint(0, 20, 100)),
                "number_inpatient": list(np.random.randint(0, 20, 100)),
            }
        )

        # Verify all columns have exactly 100 values
        for key, values in data.items():
            assert (
                len(values) == 100
            ), f"Column {key} has {len(values)} values, expected 100"

        return pd.DataFrame(data)

    # ---------- Initialization Tests ----------

    def test_preprocessor_initialization(self, sample_config):
        """Test Preprocessor initialization."""
        preprocessor = Preprocessor(sample_config)
        assert preprocessor.preprocessor is None
        assert preprocessor.label_encoder is None
        assert preprocessor.feature_selector is None
        assert preprocessor.config is not None
        assert preprocessor.logger is not None

    def test_preprocessor_initialization_missing_config(self):
        """Test Preprocessor initialization with missing config."""
        with pytest.raises(FileNotFoundError):
            Preprocessor("nonexistent_config.yaml")

    def test_config_loading(self, preprocessor):
        """Test config loading functionality."""
        assert "logging" in preprocessor.config
        assert "features" in preprocessor.config
        assert "categorical_features" in preprocessor.config["features"]
        assert "numerical_features" in preprocessor.config["features"]
        assert "drop_columns" in preprocessor.config["features"]

    def test_invalid_config_content(self, temp_dir):
        """Test handling of invalid config content."""
        # Create invalid YAML file
        invalid_config = os.path.join(temp_dir, "invalid_config.yaml")
        with open(invalid_config, "w") as f:
            f.write("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            Preprocessor(invalid_config)

    def test_missing_config_sections(self, temp_dir):
        """Test handling of missing required config sections."""
        # Create config missing required sections
        incomplete_config = os.path.join(temp_dir, "incomplete_config.yaml")
        config = {"logging": {"level": "INFO"}}  # Missing other required sections

        with open(incomplete_config, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="Missing required config section"):
            Preprocessor(incomplete_config)

    # ---------- Missing Values Handling Tests ----------

    def test_handle_missing_values_basic(self, preprocessor, sample_data_with_missing):
        """Test basic missing value handling."""
        processed_data = preprocessor._handle_missing_values(sample_data_with_missing)

        # Check that '?' values are replaced with NaN
        assert not (processed_data == "?").any().any()

        # Check that original NaN values are preserved
        assert processed_data["time_in_hospital"].isna().sum() >= 1
        assert processed_data["num_lab_procedures"].isna().sum() >= 1

    def test_handle_missing_values_no_missing(self, preprocessor, sample_data):
        """Test handling of data with no missing values."""
        processed_data = preprocessor._handle_missing_values(sample_data)

        # Check that the data shape is preserved
        assert processed_data.shape == sample_data.shape
        assert not processed_data.isna().any().any()

    def test_handle_missing_values_empty(self, preprocessor):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        processed_data = preprocessor._handle_missing_values(empty_df)

        # Check that empty DataFrame is returned unchanged
        assert processed_data.empty

    def test_handle_missing_values_all_missing(self, preprocessor):
        """Test handling of DataFrame with all missing values."""
        data = {"col1": ["?", "?", "?"], "col2": [np.nan, np.nan, np.nan]}
        df = pd.DataFrame(data)

        processed_data = preprocessor._handle_missing_values(df)

        # Check that '?' values are replaced with NaN
        assert not (processed_data == "?").any().any()
        assert processed_data["col1"].isna().all()
        assert processed_data["col2"].isna().all()

    # ---------- Feature Column Identification Tests ----------

    def test_get_feature_columns(self, preprocessor, sample_data):
        """Test feature column identification."""
        numerical_features, categorical_features = preprocessor._get_feature_columns(
            sample_data
        )

        # Check that features are correctly identified
        expected_numerical = [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
        ]
        expected_categorical = ["age", "gender", "race"]

        assert set(numerical_features) == set(expected_numerical)
        assert set(categorical_features) == set(expected_categorical)

        # Check that dropped columns are not included
        for col in preprocessor.config["features"]["drop_columns"]:
            assert col not in numerical_features + categorical_features

    def test_get_feature_columns_missing_columns(self, preprocessor):
        """Test feature column identification with missing columns."""
        # Create data without some expected columns
        data = pd.DataFrame(
            {
                "time_in_hospital": [1, 2, 3],
                "age": ["[20-30)", "[30-40)", "[40-50)"],
                "readmitted": ["NO", "YES", "NO"],
            }
        )

        numerical_features, categorical_features = preprocessor._get_feature_columns(
            data
        )

        # Should only include columns that actually exist
        assert "time_in_hospital" in numerical_features
        assert "age" in categorical_features
        assert len(numerical_features) == 1
        assert len(categorical_features) == 1

    # ---------- Pipeline Creation Tests ----------

    def test_create_preprocessing_pipeline(self, preprocessor, sample_data):
        """Test creation of preprocessing pipeline."""
        numerical_features, categorical_features = preprocessor._get_feature_columns(
            sample_data
        )
        pipeline = preprocessor._create_preprocessing_pipeline(
            numerical_features, categorical_features
        )

        # Check that pipeline is created
        assert pipeline is not None

        # Check that transformers are created
        transformer_names = [name for name, _, _ in pipeline.transformers]
        if numerical_features:
            assert "num" in transformer_names
        if categorical_features:
            assert "cat" in transformer_names

    def test_create_preprocessing_pipeline_no_features(self, preprocessor):
        """Test pipeline creation with no features."""
        with pytest.raises(ValueError, match="No valid features available"):
            preprocessor._create_preprocessing_pipeline([], [])

    # ---------- Target Encoding Tests ----------

    def test_encode_target(self, preprocessor, sample_data):
        """Test target variable encoding."""
        target_series = sample_data["readmitted"]
        encoded_series = preprocessor._encode_target(target_series)

        # Check that target is encoded to 0 and 1
        assert set(encoded_series.unique()).issubset({0, 1})
        assert encoded_series.name == "readmitted"

        # Check that label encoder is fitted
        assert preprocessor.label_encoder is not None
        assert len(preprocessor.label_encoder.classes_) == 2

    def test_encode_target_multiclass(self, preprocessor):
        """Test target encoding with multiple classes."""
        target_series = pd.Series(["A", "B", "C", "A", "B", "C"], name="target")
        encoded_series = preprocessor._encode_target(target_series)

        # Check that all classes are encoded
        assert set(encoded_series.unique()) == {0, 1, 2}
        assert len(preprocessor.label_encoder.classes_) == 3

    # ---------- Feature Selection Tests ----------

    def test_select_features_basic(self, preprocessor, sample_data):
        """Test basic feature selection."""
        # First fit the preprocessor to get transformed data
        preprocessor.fit(sample_data)

        # Create some sample transformed data
        X = np.random.randn(100, 15)  # 15 features
        y = np.random.randint(0, 2, 100)  # Binary target

        # Set feature names
        preprocessor._feature_names = [f"feature_{i}" for i in range(15)]

        # Test feature selection
        X_selected = preprocessor._select_features(X, y)

        # Should select top k features
        expected_k = preprocessor.config["feature_selection"]["n_features"]
        assert X_selected.shape[1] == min(expected_k, X.shape[1])

    def test_select_features_no_selection(self, preprocessor, sample_data):
        """Test when no feature selection is configured."""
        # Remove feature selection from config
        del preprocessor.config["feature_selection"]

        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        preprocessor._feature_names = [f"feature_{i}" for i in range(5)]

        X_selected = preprocessor._select_features(X, y)

        # Should return original features
        assert X_selected.shape == X.shape

    def test_select_features_zero_variance(self, preprocessor):
        """Test feature selection with zero variance features."""
        # Create data with zero variance features (columns 1 and 3 have zero variance)
        X = np.array(
            [
                [1, 5, 2, 5],  # feature 1 has variance, feature 3 has variance
                [2, 5, 3, 5],  # feature 1 has variance, feature 3 has variance
                [3, 5, 4, 5],  # feature 1 has variance, feature 3 has variance
                [4, 5, 5, 5],  # feature 1 has variance, feature 3 has variance
            ],
            dtype=float,
        )
        y = np.array([0, 1, 0, 1])

        # Make sure we have zero variance features (columns 1 and 3)
        assert np.var(X[:, 1]) == 0  # Column 1 should have zero variance
        assert np.var(X[:, 3]) == 0  # Column 3 should have zero variance
        assert np.var(X[:, 0]) > 0  # Column 0 should have variance
        assert np.var(X[:, 2]) > 0  # Column 2 should have variance

        preprocessor._feature_names = ["f0_var", "f1_zero_var", "f2_var", "f3_zero_var"]

        # Set up feature selection config to select fewer features than available
        preprocessor.config["feature_selection"] = {"n_features": 3}

        X_selected = preprocessor._select_features(X, y)

        # Should remove zero variance features, leaving only 2 features (f0_var and f2_var)
        assert X_selected.shape[1] == 2
        assert X_selected.shape[1] < X.shape[1]

    # ---------- Main Processing Tests ----------

    def test_fit_basic(self, preprocessor, sample_data):
        """Test basic fit functionality."""
        preprocessor.fit(sample_data)

        # Check that components are fitted
        assert preprocessor.preprocessor is not None
        assert preprocessor.label_encoder is not None

        # Check that feature names are set
        assert len(preprocessor._feature_names) > 0

    def test_fit_empty_data(self, preprocessor):
        """Test fit with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot fit on empty DataFrame"):
            preprocessor.fit(empty_df)

    def test_fit_missing_target(self, preprocessor, sample_data):
        """Test fit with missing target column."""
        data_no_target = sample_data.drop(columns=["readmitted"])
        with pytest.raises(KeyError, match="Target column 'readmitted' not found"):
            preprocessor.fit(data_no_target)

    def test_transform_without_fit(self, preprocessor, sample_data):
        """Test that transform raises error if not fitted."""
        with pytest.raises(ValueError, match="Preprocessing pipeline not fitted yet"):
            preprocessor.transform(sample_data)

    def test_fit_transform_basic(self, preprocessor, sample_data):
        """Test basic fit_transform functionality."""
        X_processed, y_processed = preprocessor.fit_transform(sample_data)

        # Check output types
        assert isinstance(X_processed, pd.DataFrame)
        assert isinstance(y_processed, pd.Series)

        # Check that target is encoded
        assert set(y_processed.unique()).issubset({0, 1})

        # Check that no missing values in processed data
        assert not X_processed.isna().any().any()

        # Check that dropped columns are not present
        for col in preprocessor.config["features"]["drop_columns"]:
            assert col not in X_processed.columns

    def test_fit_transform_with_missing(self, preprocessor, sample_data_with_missing):
        """Test fit_transform with missing values."""
        X_processed, y_processed = preprocessor.fit_transform(sample_data_with_missing)

        # Check that no missing values in processed data
        assert not X_processed.isna().any().any()

        # Check that target is encoded
        assert set(y_processed.unique()).issubset({0, 1})

    def test_fit_transform_custom_target(self, preprocessor, sample_data):
        """Test fit_transform with custom target column."""
        # Rename target column
        sample_data = sample_data.rename(columns={"readmitted": "target"})

        X_processed, y_processed = preprocessor.fit_transform(
            sample_data, target_col="target"
        )

        # Check that target is encoded
        assert set(y_processed.unique()).issubset({0, 1})
        assert y_processed.name == "target"

    def test_transform_after_fit(self, preprocessor, sample_data):
        """Test transform after fitting."""
        # Fit first
        preprocessor.fit(sample_data)

        # Then transform
        X_processed, y_processed = preprocessor.transform(sample_data)

        # Check that output is correct type
        assert isinstance(X_processed, pd.DataFrame)
        assert isinstance(y_processed, pd.Series)

        # Check that no missing values in processed data
        assert not X_processed.isna().any().any()

    def test_transform_empty_data(self, preprocessor, sample_data):
        """Test transform with empty DataFrame."""
        # Fit first
        preprocessor.fit(sample_data)

        # Try to transform empty data
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError, match="Cannot transform empty DataFrame"):
            preprocessor.transform(empty_df)

    # ---------- Feature Names Tests ----------

    def test_get_feature_names_before_fit(self, preprocessor):
        """Test getting feature names before fitting."""
        with pytest.raises(ValueError, match="Pipeline not fitted yet"):
            preprocessor.get_feature_names()

    def test_get_feature_names_after_fit(self, preprocessor, sample_data):
        """Test getting feature names after fitting."""
        preprocessor.fit(sample_data)
        feature_names = preprocessor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

    # ---------- Pipeline Persistence Tests ----------

    def test_save_pipeline(self, preprocessor, sample_data, temp_dir):
        """Test saving the preprocessing pipeline."""
        # Fit the pipeline
        preprocessor.fit(sample_data)

        # Save the pipeline
        save_path = os.path.join(temp_dir, "preprocessor.joblib")
        preprocessor.save_pipeline(save_path)

        # Check that file exists
        assert os.path.exists(save_path)

    def test_load_pipeline(self, preprocessor, sample_data, temp_dir):
        """Test loading the preprocessing pipeline."""
        # Fit and save the pipeline
        preprocessor.fit(sample_data)
        save_path = os.path.join(temp_dir, "preprocessor.joblib")
        preprocessor.save_pipeline(save_path)

        # Create new preprocessor and load
        new_preprocessor = Preprocessor(preprocessor.config_path)
        new_preprocessor.load_pipeline(save_path)

        # Transform data with both preprocessors
        X1, y1 = preprocessor.transform(sample_data)
        X2, y2 = new_preprocessor.transform(sample_data)

        # Check that results are identical
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_save_pipeline_before_fit(self, preprocessor, temp_dir):
        """Test saving pipeline before fitting."""
        save_path = os.path.join(temp_dir, "preprocessor.joblib")

        # Should save even if not fitted (components will be None)
        preprocessor.save_pipeline(save_path)
        assert os.path.exists(save_path)

    def test_load_nonexistent_pipeline(self, preprocessor):
        """Test loading non-existent pipeline."""
        with pytest.raises(FileNotFoundError):
            preprocessor.load_pipeline("nonexistent_pipeline.joblib")

    # ---------- Report Generation Tests ----------

    def test_report_generation(self, preprocessor, sample_data, temp_dir):
        """Test that preprocessing report is generated correctly."""
        # Change to temp directory for report file creation
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Run preprocessing
            preprocessor.fit_transform(sample_data)

            # Check that report file was created
            assert Path("logs/preprocessing_report.json").exists()

            # Check that report contains expected keys
            assert "timestamp" in preprocessor.report
            assert "original_shape" in preprocessor.report
            assert "target_column" in preprocessor.report
            assert "fitting_completed" in preprocessor.report

        finally:
            os.chdir(original_cwd)

    def test_logging_output(self, preprocessor, sample_data, temp_dir):
        """Test that preprocessing logs correctly."""
        import logging
        import io

        # Change to temp directory for log file creation
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Create a simple log capture
            log_capture = io.StringIO()
            handler = logging.StreamHandler(log_capture)
            handler.setLevel(logging.INFO)

            # Add handler to the preprocessor's logger
            preprocessor.logger.addHandler(handler)
            preprocessor.logger.setLevel(logging.INFO)

            # Test that logging works at all
            test_message = "Test logging message for preprocessing"
            preprocessor.logger.info(test_message)
            handler.flush()

            captured = log_capture.getvalue()
            if test_message not in captured:
                pytest.skip("Logger not working in test environment")

            # Clear the capture for actual test
            log_capture.truncate(0)
            log_capture.seek(0)

            # Run preprocessing
            preprocessor.fit_transform(sample_data)
            handler.flush()

            # Check that some logging occurred
            final_logs = log_capture.getvalue()
            assert (
                len(final_logs.strip()) > 0
            ), "No logs were captured during preprocessing"

            # Just check that it contains some key words related to preprocessing
            log_lower = final_logs.lower()
            preprocessing_keywords = [
                "preprocessing",
                "fitting",
                "encoding",
                "pipeline",
                "features",
            ]

            found_keywords = [
                keyword for keyword in preprocessing_keywords if keyword in log_lower
            ]
            assert (
                len(found_keywords) >= 2
            ), f"Expected at least 2 preprocessing-related keywords, found: {found_keywords}"

        finally:
            os.chdir(original_cwd)

    # ---------- Edge Cases and Error Handling Tests ----------

    def test_no_numerical_features(self, sample_config):
        """Test preprocessing with no numerical features."""
        # Create data with only categorical features
        data = pd.DataFrame(
            {
                "cat1": ["A", "B", "C", "D", "E"],
                "cat2": ["X", "Y", "Z", "X", "Y"],
                "readmitted": ["YES", "NO", "YES", "NO", "YES"],
            }
        )

        # Create a fresh preprocessor with modified config
        preprocessor = Preprocessor(sample_config)

        # Update config to reflect this data
        preprocessor.config["features"]["numerical_features"] = []
        preprocessor.config["features"]["categorical_features"] = ["cat1", "cat2"]
        preprocessor.config["features"]["drop_columns"] = []  # No columns to drop

        X_processed, y_processed = preprocessor.fit_transform(data)

        # Should still work
        assert isinstance(X_processed, pd.DataFrame)
        assert isinstance(y_processed, pd.Series)
        assert X_processed.shape[0] == len(data)
        assert len(y_processed) == len(data)

    def test_no_categorical_features(self, sample_config):
        """Test preprocessing with no categorical features."""
        # Create data with only numerical features
        data = pd.DataFrame(
            {
                "num1": [1, 2, 3, 4, 5],
                "num2": [1.1, 2.2, 3.3, 4.4, 5.5],
                "readmitted": ["YES", "NO", "YES", "NO", "YES"],
            }
        )

        # Create a fresh preprocessor with modified config
        preprocessor = Preprocessor(sample_config)

        # Update config to reflect this data
        preprocessor.config["features"]["numerical_features"] = ["num1", "num2"]
        preprocessor.config["features"]["categorical_features"] = []
        preprocessor.config["features"]["drop_columns"] = []  # No columns to drop

        X_processed, y_processed = preprocessor.fit_transform(data)

        # Should still work
        assert isinstance(X_processed, pd.DataFrame)
        assert isinstance(y_processed, pd.Series)
        assert X_processed.shape[0] == len(data)
        assert len(y_processed) == len(data)

    def test_consistent_preprocessing(self, preprocessor, sample_data):
        """Test that preprocessing is consistent across multiple runs."""
        # Run preprocessing twice
        X1, y1 = preprocessor.fit_transform(sample_data.copy())

        # Reset preprocessor
        preprocessor.preprocessor = None
        preprocessor.label_encoder = None
        preprocessor.feature_selector = None

        X2, y2 = preprocessor.fit_transform(sample_data.copy())

        # Results should be identical (same random seed)
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)


if __name__ == "__main__":
    """Demonstrate Preprocessor testing functionality."""
    print("Preprocessor test module loaded successfully.")
    print("Run with: pytest tests/test_preprocessing.py -v")

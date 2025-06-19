"""
Unit tests for the Feature Engineering module.

Tests the FeatureEngineer class and its feature creation methods
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.features.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
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
                "processed_data_path": os.path.join(temp_dir, "processed"),
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
                    "number_diagnoses",
                ],
                "drop_columns": [],
                "target_column": "readmitted",
            },
            "feature_engineering": {
                "apply_selection": True,
                "n_features_to_select": 10,
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": os.path.join(temp_dir, "logs", "feature_engineering.log"),
            },
        }

        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def feature_engineer(self, sample_config):
        """Create a FeatureEngineer instance with test config."""
        return FeatureEngineer(sample_config)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        data = {
            "age": ["[20-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)"],
            "time_in_hospital": [3, 7, 5, 12, 8],
            "num_medications": [5, 15, 10, 25, 20],
            "number_outpatient": [0, 2, 1, 3, 2],
            "number_emergency": [1, 0, 2, 1, 3],
            "number_inpatient": [0, 1, 0, 2, 1],
            "number_diagnoses": [5, 8, 6, 12, 9],
            "diag_1": ["401.9", "250.02", "272.4", "585.6", "414.01"],
            "diag_2": ["250.00", "401.9", "585.9", "250.02", "272.4"],
            "diag_3": ["272.4", "585.6", "414.01", "401.9", "250.00"],
            "change": ["No", "Ch", "No", "Ch", "No"],
            "diabetesMed": ["No", "Yes", "No", "Yes", "No"],
            "readmitted": ["NO", "YES", "NO", "YES", "NO"],
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_data_missing_cols(self):
        """Create sample data with some missing columns for testing edge cases."""
        data = {
            "time_in_hospital": [3, 7, 5],
            "num_medications": [5, 15, 10],
            "readmitted": ["NO", "YES", "NO"],
        }
        return pd.DataFrame(data)

    # ---------- Initialization Tests ----------

    def test_feature_engineer_initialization(self, sample_config):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer(sample_config)
        assert engineer.feature_selector is None
        assert engineer.selected_features == []
        assert engineer.original_features == []
        assert engineer.engineered_features == []
        assert engineer.config is not None
        assert engineer.logger is not None

    def test_feature_engineer_initialization_missing_config(self):
        """Test FeatureEngineer initialization with missing config."""
        with pytest.raises(FileNotFoundError):
            FeatureEngineer("nonexistent_config.yaml")

    # ---------- Age Group Feature Tests ----------

    def test_create_age_groups_basic(self, feature_engineer, sample_data):
        """Test basic age group creation."""
        result = feature_engineer._create_age_groups(sample_data)

        assert "age_group" in result.columns
        assert result["age_group"].dtype.name == "category"
        assert not result["age_group"].isna().any()

        # Check that all values are valid categories
        valid_categories = ["Young", "Adult", "Senior", "Elderly"]
        assert all(
            val in valid_categories for val in result["age_group"].cat.categories
        )

    def test_create_age_groups_missing_column(
        self, feature_engineer, sample_data_missing_cols
    ):
        """Test age group creation with missing age column."""
        result = feature_engineer._create_age_groups(sample_data_missing_cols)

        # Should return original data unchanged
        assert "age_group" not in result.columns
        assert result.shape == sample_data_missing_cols.shape

    def test_create_age_groups_invalid_values(self, feature_engineer):
        """Test age group creation with invalid age values."""
        data = pd.DataFrame(
            {"age": ["invalid", None, "[30-40)"], "readmitted": ["NO", "YES", "NO"]}
        )

        result = feature_engineer._create_age_groups(data)

        # Should handle invalid values gracefully
        assert "age_group" in result.columns
        # At least one valid mapping should work
        assert not result["age_group"].isna().all()

    # ---------- Length of Stay Group Tests ----------

    def test_create_length_of_stay_groups_basic(self, feature_engineer, sample_data):
        """Test basic length of stay group creation."""
        result = feature_engineer._create_length_of_stay_groups(sample_data)

        assert "los_group" in result.columns
        assert result["los_group"].dtype.name == "category"
        assert not result["los_group"].isna().any()

        # Check categories
        valid_categories = ["Short", "Medium", "Long", "Extended"]
        assert all(
            val in valid_categories for val in result["los_group"].cat.categories
        )

    def test_create_length_of_stay_groups_missing_column(
        self, feature_engineer, sample_data_missing_cols
    ):
        """Test LOS group creation with missing time_in_hospital column."""
        data = sample_data_missing_cols.drop(columns=["time_in_hospital"])
        result = feature_engineer._create_length_of_stay_groups(data)

        # Should return original data unchanged
        assert "los_group" not in result.columns

    def test_create_length_of_stay_groups_edge_values(self, feature_engineer):
        """Test LOS group creation with edge values."""
        data = pd.DataFrame(
            {
                "time_in_hospital": [0, 1, 3, 7, 14, 30],
                "readmitted": ["NO", "YES", "NO", "YES", "NO", "YES"],
            }
        )

        result = feature_engineer._create_length_of_stay_groups(data)

        assert "los_group" in result.columns
        assert not result["los_group"].isna().any()
        # Check that 30 days maps to 'Extended'
        assert (
            result[result["time_in_hospital"] == 30]["los_group"].iloc[0] == "Extended"
        )

    # ---------- Total Visits Feature Tests ----------

    def test_create_total_visits_basic(self, feature_engineer, sample_data):
        """Test basic total visits creation."""
        result = feature_engineer._create_total_visits(sample_data)

        assert "total_visits" in result.columns
        assert result["total_visits"].dtype in ["int64", "float64"]

        # Check calculation is correct
        expected = (
            sample_data["number_outpatient"]
            + sample_data["number_emergency"]
            + sample_data["number_inpatient"]
        )
        pd.testing.assert_series_equal(
            result["total_visits"], expected, check_names=False
        )

    def test_create_total_visits_missing_columns(self, feature_engineer):
        """Test total visits creation with missing visit columns."""
        data = pd.DataFrame(
            {
                "number_outpatient": [1, 2, 3],
                "readmitted": ["NO", "YES", "NO"],
                # Missing number_emergency and number_inpatient
            }
        )

        result = feature_engineer._create_total_visits(data)

        # Should return original data unchanged
        assert "total_visits" not in result.columns

    def test_create_total_visits_zero_values(self, feature_engineer):
        """Test total visits creation with zero values."""
        data = pd.DataFrame(
            {
                "number_outpatient": [0, 0, 0],
                "number_emergency": [0, 0, 0],
                "number_inpatient": [0, 0, 0],
                "readmitted": ["NO", "YES", "NO"],
            }
        )

        result = feature_engineer._create_total_visits(data)

        assert "total_visits" in result.columns
        assert all(result["total_visits"] == 0)

    # ---------- Medication Intensity Feature Tests ----------

    def test_create_medication_intensity_basic(self, feature_engineer, sample_data):
        """Test basic medication intensity creation."""
        result = feature_engineer._create_medication_intensity(sample_data)

        assert "medication_intensity" in result.columns
        assert result["medication_intensity"].dtype in ["float64"]

        # Check calculation is correct (avoiding division by zero)
        expected = sample_data["num_medications"] / sample_data[
            "time_in_hospital"
        ].replace(0, 1)
        pd.testing.assert_series_equal(
            result["medication_intensity"], expected, check_names=False
        )

    def test_create_medication_intensity_zero_hospital_time(self, feature_engineer):
        """Test medication intensity with zero hospital time."""
        data = pd.DataFrame(
            {
                "num_medications": [10, 5, 15],
                "time_in_hospital": [0, 1, 2],
                "readmitted": ["NO", "YES", "NO"],
            }
        )

        result = feature_engineer._create_medication_intensity(data)

        assert "medication_intensity" in result.columns
        # First row should use 1 instead of 0 for division
        assert result["medication_intensity"].iloc[0] == 10.0  # 10/1
        assert result["medication_intensity"].iloc[1] == 5.0  # 5/1
        assert result["medication_intensity"].iloc[2] == 7.5  # 15/2

    def test_create_medication_intensity_missing_columns(self, feature_engineer):
        """Test medication intensity with missing columns."""
        data = pd.DataFrame(
            {
                "num_medications": [10, 5, 15],
                "readmitted": ["NO", "YES", "NO"],
                # Missing time_in_hospital
            }
        )

        result = feature_engineer._create_medication_intensity(data)

        # Should return original data unchanged
        assert "medication_intensity" not in result.columns

    # ---------- Binary Flags Tests ----------

    def test_create_binary_flags_basic(self, feature_engineer, sample_data):
        """Test basic binary flags creation."""
        result = feature_engineer._create_binary_flags(sample_data)

        expected_flags = [
            "has_emergency_visits",
            "was_medicated",
            "many_diagnoses",
            "medication_changed",
            "uses_diabetes_med",
        ]

        for flag in expected_flags:
            if flag in result.columns:  # Only check flags that were created
                assert result[flag].dtype in ["int64", "uint8"]
                assert set(result[flag].unique()).issubset({0, 1})

    def test_create_binary_flags_edge_cases(self, feature_engineer):
        """Test binary flags with edge case values."""
        data = pd.DataFrame(
            {
                "number_emergency": [0, 1, 5],
                "num_medications": [0, 1, 10],
                "number_diagnoses": [8, 9, 15],
                "change": ["No", "Ch", "No"],
                "diabetesMed": ["No", "Yes", "No"],
                "readmitted": ["NO", "YES", "NO"],
            }
        )

        result = feature_engineer._create_binary_flags(data)

        # Check specific logic
        if "has_emergency_visits" in result.columns:
            assert result["has_emergency_visits"].iloc[0] == 0  # 0 emergency visits
            assert result["has_emergency_visits"].iloc[1] == 1  # 1 emergency visit
            assert result["has_emergency_visits"].iloc[2] == 1  # 5 emergency visits

        if "many_diagnoses" in result.columns:
            assert result["many_diagnoses"].iloc[0] == 0  # 8 diagnoses
            assert result["many_diagnoses"].iloc[1] == 1  # 9 diagnoses
            assert result["many_diagnoses"].iloc[2] == 1  # 15 diagnoses

    # ---------- Diagnosis Features Tests ----------

    def test_create_diagnosis_features_basic(self, feature_engineer, sample_data):
        """Test basic diagnosis features creation."""
        result = feature_engineer._create_diagnosis_features(sample_data)

        # Check that diagnosis features are created
        possible_features = ["has_diabetes_diagnosis", "has_circulatory_diagnosis"]
        created_features = [f for f in possible_features if f in result.columns]

        assert len(created_features) > 0  # At least one feature should be created

        for feature in created_features:
            assert result[feature].dtype in ["int64", "uint8"]
            assert set(result[feature].unique()).issubset({0, 1})

    def test_create_diagnosis_features_diabetes_detection(self, feature_engineer):
        """Test diabetes diagnosis detection."""
        data = pd.DataFrame(
            {
                "diag_1": ["250.02", "401.9", "272.4"],
                "diag_2": ["401.9", "250.00", "585.6"],
                "diag_3": ["272.4", "585.6", "250.99"],
                "readmitted": ["NO", "YES", "NO"],
            }
        )

        result = feature_engineer._create_diagnosis_features(data)

        if "has_diabetes_diagnosis" in result.columns:
            # All rows should have diabetes diagnosis (250.xx codes)
            assert all(result["has_diabetes_diagnosis"] == 1)

    def test_create_diagnosis_features_circulatory_detection(self, feature_engineer):
        """Test circulatory diagnosis detection."""
        data = pd.DataFrame(
            {
                "diag_1": ["401.9", "250.02", "272.4"],  # 401.9 is circulatory
                "diag_2": ["250.02", "401.9", "585.6"],  # 401.9 is circulatory
                "diag_3": ["272.4", "585.6", "414.01"],  # 414.01 is circulatory
                "readmitted": ["NO", "YES", "NO"],
            }
        )

        result = feature_engineer._create_diagnosis_features(data)

        if "has_circulatory_diagnosis" in result.columns:
            # All rows should have circulatory diagnosis (390-459 range)
            assert all(result["has_circulatory_diagnosis"] == 1)

    def test_create_diagnosis_features_missing_columns(self, feature_engineer):
        """Test diagnosis features with missing diagnosis columns."""
        data = pd.DataFrame(
            {"some_other_column": [1, 2, 3], "readmitted": ["NO", "YES", "NO"]}
        )

        result = feature_engineer._create_diagnosis_features(data)

        # Should return original data unchanged
        diagnosis_features = ["has_diabetes_diagnosis", "has_circulatory_diagnosis"]
        for feature in diagnosis_features:
            assert feature not in result.columns

    # ---------- Feature Selection Tests ----------

    def test_apply_feature_selection_basic(self, feature_engineer, sample_data):
        """Test basic feature selection."""
        # Create some features first - use ONLY numerical features for this test
        X = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_2": [2.0, 3.0, 4.0, 5.0, 6.0],
                "feature_3": [3.0, 4.0, 5.0, 6.0, 7.0],
                "feature_4": [4.0, 5.0, 6.0, 7.0, 8.0],
                "feature_5": [5.0, 6.0, 7.0, 8.0, 9.0],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        # Add more numerical features to have more than 10
        for i in range(6, 16):
            X[f"feature_{i}"] = np.random.randn(len(X))

        result = feature_engineer._apply_feature_selection(X, y)

        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 10  # Should select 10 features
        assert len(feature_engineer.selected_features) == 10

    def test_apply_feature_selection_fewer_than_k(self, feature_engineer):
        """Test feature selection when fewer features than k exist."""
        X = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [2, 3, 4, 5, 6],
                "feature3": [3, 4, 5, 6, 7],
            }
        )
        y = pd.Series([0, 1, 0, 1, 0])

        result = feature_engineer._apply_feature_selection(X, y)

        # Should return all features since 3 < 10
        assert result.shape[1] == 3
        assert list(result.columns) == ["feature1", "feature2", "feature3"]

    def test_apply_feature_selection_disabled(self, feature_engineer):
        """Test feature selection when disabled in config."""
        # Modify config to disable feature selection
        feature_engineer.config["feature_engineering"]["n_features_to_select"] = None

        X = pd.DataFrame(
            np.random.randn(10, 15), columns=[f"feat_{i}" for i in range(15)]
        )
        y = pd.Series([0, 1] * 5)

        result = feature_engineer._apply_feature_selection(X, y)

        # Should return all features
        assert result.shape[1] == 15
        assert len(feature_engineer.selected_features) == 15

    # ---------- Main Pipeline Tests ----------

    def test_fit_transform_basic(self, feature_engineer, sample_data, temp_dir):
        """Test basic fit_transform functionality."""
        # Change to temp directory for report creation
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            X_result, y_result = feature_engineer.fit_transform(sample_data)

            # Check output types
            assert isinstance(X_result, pd.DataFrame)
            assert isinstance(y_result, pd.Series)

            # Check that features were created
            assert len(feature_engineer.engineered_features) > 0
            assert len(feature_engineer.selected_features) > 0

            # Check that target is unchanged
            pd.testing.assert_series_equal(
                y_result, sample_data["readmitted"], check_names=False
            )

            # Check report generation
            assert "timestamp" in feature_engineer.report
            assert "original_shape" in feature_engineer.report
            assert "engineering_completed" in feature_engineer.report

        finally:
            os.chdir(original_cwd)

    def test_fit_transform_empty_data(self, feature_engineer):
        """Test fit_transform with empty DataFrame."""
        empty_df = pd.DataFrame()
        with pytest.raises(
            ValueError, match="Cannot perform feature engineering on empty DataFrame"
        ):
            feature_engineer.fit_transform(empty_df)

    def test_fit_transform_missing_target(self, feature_engineer, sample_data):
        """Test fit_transform with missing target column."""
        data_no_target = sample_data.drop(columns=["readmitted"])
        with pytest.raises(KeyError, match="Target column 'readmitted' not found"):
            feature_engineer.fit_transform(data_no_target)

    def test_transform_after_fit(self, feature_engineer, sample_data):
        """Test transform after fitting."""
        # Fit first
        feature_engineer.fit_transform(sample_data)

        # Create new data for transform
        new_data = sample_data.copy()
        new_data.iloc[0, 0] = "[70-80)"  # Change first age value

        # Transform
        X_result, y_result = feature_engineer.transform(new_data)

        # Check output types
        assert isinstance(X_result, pd.DataFrame)
        assert isinstance(y_result, pd.Series)

        # Check that same features are present
        assert X_result.shape[1] == len(feature_engineer.selected_features)

    def test_transform_without_fit(self, feature_engineer, sample_data):
        """Test transform without fitting first."""
        # This should work since transform doesn't require prior fitting
        # (it applies the same transformations)
        X_result, y_result = feature_engineer.transform(sample_data)
        assert isinstance(X_result, pd.DataFrame)
        assert isinstance(y_result, pd.Series)

    # ---------- Utility Methods Tests ----------

    def test_get_feature_names_after_fit(self, feature_engineer, sample_data):
        """Test getting feature names after fitting."""
        feature_engineer.fit_transform(sample_data)
        feature_names = feature_engineer.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert feature_names == feature_engineer.selected_features

    def test_get_feature_names_before_fit(self, feature_engineer):
        """Test getting feature names before fitting."""
        with pytest.raises(ValueError, match="Pipeline not fitted yet"):
            feature_engineer.get_feature_names()

    def test_get_engineered_features(self, feature_engineer, sample_data):
        """Test getting engineered feature names."""
        feature_engineer.fit_transform(sample_data)
        engineered_features = feature_engineer.get_engineered_features()

        assert isinstance(engineered_features, list)
        assert len(engineered_features) > 0
        assert engineered_features == feature_engineer.engineered_features

    # ---------- Edge Cases and Error Handling Tests ----------

    def test_feature_engineering_with_minimal_data(self, feature_engineer):
        """Test feature engineering with minimal required data."""
        minimal_data = pd.DataFrame({"readmitted": ["NO", "YES"]})

        X_result, y_result = feature_engineer.fit_transform(minimal_data)

        # Should handle gracefully even with minimal data
        assert isinstance(X_result, pd.DataFrame)
        assert isinstance(y_result, pd.Series)

    def test_feature_engineering_consistency(self, feature_engineer, sample_data):
        """Test that feature engineering is consistent across multiple runs."""
        # Run twice with same data
        X1, y1 = feature_engineer.fit_transform(sample_data.copy())

        # Reset the feature engineer
        feature_engineer.feature_selector = None
        feature_engineer.selected_features = []
        feature_engineer.original_features = []
        feature_engineer.engineered_features = []

        X2, y2 = feature_engineer.fit_transform(sample_data.copy())

        # Results should be identical
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_report_generation(self, feature_engineer, sample_data, temp_dir):
        """Test that feature engineering report is generated correctly."""
        # Change to temp directory for report file creation
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            # Run feature engineering
            feature_engineer.fit_transform(sample_data)

            # Check that report file was created
            assert Path("logs/feature_engineering_report.json").exists()

            # Check that report contains expected keys
            assert "timestamp" in feature_engineer.report
            assert "original_shape" in feature_engineer.report
            assert "target_column" in feature_engineer.report
            assert "engineering_completed" in feature_engineer.report
            assert "engineered_features" in feature_engineer.report
            assert "selected_features" in feature_engineer.report

        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    """Demonstrate FeatureEngineer testing functionality."""
    print("Feature engineering test module loaded successfully.")
    print("Run with: pytest tests/test_feature_engineering.py -v")

import pytest
import pandas as pd
import numpy as np
import yaml
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Configure matplotlib before any other imports
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Set non-interactive backend

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.eda.eda import EDAAnalyzer

plt.ioff()  # Turn off interactive mode


class TestEDAAnalyzer:
    @pytest.fixture(autouse=True)
    def setup_matplotlib(self):
        """Setup matplotlib for headless testing."""
        matplotlib.use("Agg")
        plt.ioff()
        yield
        # Cleanup any open figures
        plt.close("all")

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()

        # Change to temp directory to avoid polluting project directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        yield temp_dir

        # Cleanup
        os.chdir(original_cwd)
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
                "file": os.path.join(temp_dir, "logs", "eda.log"),
            },
        }

        config_path = os.path.join(temp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def eda_analyzer(self, sample_config):
        """Create an EDAAnalyzer with test config."""
        return EDAAnalyzer(sample_config)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "race": ["Caucasian", "AfricanAmerican", "Hispanic"] * 34,
                "gender": ["Female", "Male"] * 51,
                "age_group": ["[30-40)", "[40-50)", "[50-60)"] * 34,
                "time_in_hospital": np.random.randint(1, 15, 102),
                "num_lab_procedures": np.random.randint(1, 100, 102),
                "num_procedures": np.random.randint(0, 10, 102),
                "num_medications": np.random.randint(1, 50, 102),
                "number_outpatient": np.random.randint(0, 20, 102),
                "number_emergency": np.random.randint(0, 20, 102),
                "number_inpatient": np.random.randint(0, 20, 102),
                "number_diagnoses": np.random.randint(1, 20, 102),
                "readmitted": ["NO", "YES"] * 51,
            }
        )

    @pytest.fixture
    def plots_dir(self, eda_analyzer):
        """Get the plots directory from EDA analyzer."""
        return eda_analyzer.plots_dir

    # ---------- Initialization Tests ----------

    def test_eda_analyzer_initialization(self, sample_config):
        """Test EDAAnalyzer initialization."""
        analyzer = EDAAnalyzer(sample_config)
        assert analyzer.config is not None
        assert analyzer.logger is not None
        assert analyzer.plots_dir.exists()

    def test_eda_analyzer_initialization_missing_config(self):
        """Test EDAAnalyzer initialization with missing config."""
        with pytest.raises(FileNotFoundError):
            EDAAnalyzer("nonexistent_config.yaml")

    def test_matplotlib_backend(self):
        """Test that matplotlib is using the correct backend."""
        import matplotlib

        assert matplotlib.get_backend() == "agg"

    # ---------- Dataset Description Tests ----------

    def test_describe_dataset_basic(self, eda_analyzer, sample_data):
        """Test basic functionality of describe_dataset."""
        description = eda_analyzer.describe_dataset(sample_data)

        # Check if all required keys are present
        assert "basic_info" in description
        assert "column_types" in description
        assert "missing_values" in description

        # Check basic info
        assert description["basic_info"]["shape"] == sample_data.shape
        assert description["basic_info"]["total_rows"] == len(sample_data)
        assert description["basic_info"]["total_columns"] == len(sample_data.columns)

        # Check column types
        assert "numeric_columns" in description["column_types"]
        assert "categorical_columns" in description["column_types"]

    def test_describe_dataset_empty(self, eda_analyzer):
        """Test describe_dataset with empty DataFrame."""
        empty_df = pd.DataFrame()
        description = eda_analyzer.describe_dataset(empty_df)

        assert "error" in description
        assert description["error"] == "Empty dataset"

    # ---------- Target Analysis Tests ----------

    def test_analyze_target_variable_basic(self, eda_analyzer, sample_data, plots_dir):
        """Test basic functionality of analyze_target_variable."""
        analysis = eda_analyzer.analyze_target_variable(sample_data)

        # Check if all required keys are present
        assert "target_column" in analysis
        assert "unique_values" in analysis
        assert "value_counts" in analysis
        assert "percentages" in analysis
        assert "class_balance" in analysis

        # Wait a bit for file system to sync, then check if plots were created
        import time

        time.sleep(0.1)

        target_plots_dir = plots_dir / "target_analysis"
        assert (
            target_plots_dir.exists()
        ), f"Target plots directory not found: {target_plots_dir}"

        dist_plot = target_plots_dir / "readmitted_distribution.png"
        pie_plot = target_plots_dir / "readmitted_pie.png"

        assert dist_plot.exists(), f"Distribution plot not found: {dist_plot}"
        assert pie_plot.exists(), f"Pie plot not found: {pie_plot}"

    def test_analyze_target_variable_custom_target(
        self, eda_analyzer, sample_data, plots_dir
    ):
        """Test analyze_target_variable with custom target column."""
        # Create a custom target column
        sample_data["custom_target"] = np.random.choice(
            ["A", "B", "C"], len(sample_data)
        )
        analysis = eda_analyzer.analyze_target_variable(
            sample_data, target_col="custom_target"
        )

        assert analysis["target_column"] == "custom_target"

        # Check if plots were created
        target_plots_dir = plots_dir / "target_analysis"
        assert (target_plots_dir / "custom_target_distribution.png").exists()
        assert (target_plots_dir / "custom_target_pie.png").exists()

    def test_analyze_target_variable_missing_target(self, eda_analyzer, sample_data):
        """Test analyze_target_variable with missing target column."""
        analysis = eda_analyzer.analyze_target_variable(
            sample_data, target_col="nonexistent_column"
        )
        assert "error" in analysis

    # ---------- Feature Distribution Tests ----------
    def test_analyze_feature_distributions_basic(
        self, eda_analyzer, sample_data, plots_dir
    ):
        """Test basic functionality of analyze_feature_distributions."""
        distributions = eda_analyzer.analyze_feature_distributions(sample_data)

        # Get numerical columns
        numerical_cols = eda_analyzer._get_numeric_columns(sample_data)

        # Check if statistics are returned for numerical columns
        for col in numerical_cols:
            assert col in distributions
            assert "mean" in distributions[col]
            assert "std" in distributions[col]
            assert "skewness" in distributions[col]

        # Wait a bit for file system to sync
        import time

        time.sleep(0.1)

        # Check if plots were created for each numerical column
        dist_dir = plots_dir / "distributions"
        assert dist_dir.exists(), f"Distributions directory not found: {dist_dir}"

        for col in numerical_cols:
            plot_file = dist_dir / f"{col}_distribution.png"
            assert (
                plot_file.exists()
            ), f"Distribution plot not found for {col}: {plot_file}"

    def test_analyze_feature_distributions_no_numerical(self, eda_analyzer):
        """Test analyze_feature_distributions with no numerical columns."""
        # Create DataFrame with only categorical columns
        data = pd.DataFrame(
            {"cat1": ["A", "B", "C"] * 10, "cat2": ["X", "Y", "Z"] * 10}
        )

        distributions = eda_analyzer.analyze_feature_distributions(data)
        assert "error" in distributions

    def test_analyze_feature_distributions_with_missing(self, eda_analyzer, plots_dir):
        """Test analyze_feature_distributions with missing values."""
        # Create DataFrame with missing values
        data = pd.DataFrame(
            {"col1": [1, 2, np.nan, 4, 5], "col2": [1.1, 2.2, 3.3, np.nan, 5.5]}
        )

        distributions = eda_analyzer.analyze_feature_distributions(data)

        # Should handle missing values gracefully
        assert "col1" in distributions
        assert "col2" in distributions

    # ---------- Correlation Analysis Tests ----------

    def test_analyze_correlations_basic(self, eda_analyzer, sample_data, plots_dir):
        """Test basic functionality of analyze_correlations."""
        correlations = eda_analyzer.analyze_correlations(sample_data)

        # Check if all required keys are present
        assert "correlation_matrix" in correlations
        assert "target_correlations" in correlations
        assert "high_correlations" in correlations

        # Check if correlation heatmap was created
        corr_dir = plots_dir / "correlations"
        assert (corr_dir / "correlation_heatmap.png").exists()

    def test_analyze_correlations_insufficient_columns(self, eda_analyzer):
        """Test analyze_correlations with insufficient numeric columns."""
        # Create DataFrame with only one numeric column
        data = pd.DataFrame(
            {"col1": [1, 2, 3, 4, 5], "cat1": ["A", "B", "C", "D", "E"]}
        )

        correlations = eda_analyzer.analyze_correlations(data)
        assert "error" in correlations

    # ---------- Missing Values Analysis Tests ----------

    def test_analyze_missing_values_basic(self, eda_analyzer, sample_data):
        """Test basic functionality of analyze_missing_values."""
        analysis = eda_analyzer.analyze_missing_values(sample_data)

        # Check if all required keys are present
        assert "total_missing_values" in analysis
        assert "columns_with_missing" in analysis
        assert "rows_with_missing" in analysis
        assert "complete_rows" in analysis

    def test_analyze_missing_values_with_missing(self, eda_analyzer, plots_dir):
        """Test analyze_missing_values with actual missing values."""
        # Create DataFrame with missing values
        data = pd.DataFrame(
            {
                "col1": [1, 2, np.nan, 4, 5],
                "col2": [1.1, np.nan, 3.3, np.nan, 5.5],
                "col3": [1, 2, 3, 4, 5],  # No missing values
            }
        )

        analysis = eda_analyzer.analyze_missing_values(data)

        assert analysis["total_missing_values"] == 3
        assert analysis["columns_with_missing"] == 2
        assert "col1" in analysis["missing_by_column"]
        assert "col2" in analysis["missing_by_column"]
        assert "col3" not in analysis["missing_by_column"]

        # Check if missing values heatmap was created
        assert (plots_dir / "missing_values_heatmap.png").exists()

    def test_analyze_missing_values_no_missing(self, eda_analyzer, sample_data):
        """Test analyze_missing_values with no missing values."""
        analysis = eda_analyzer.analyze_missing_values(sample_data)

        assert analysis["total_missing_values"] == 0
        assert analysis["columns_with_missing"] == 0
        assert len(analysis["missing_by_column"]) == 0

    # ---------- Full Analysis Tests ----------

    def test_run_full_analysis_basic(self, eda_analyzer, sample_data, plots_dir):
        """Test basic functionality of run_full_analysis."""
        report = eda_analyzer.run_full_analysis(sample_data)

        # Check if all required sections are present
        assert "dataset_description" in report
        assert "target_analysis" in report
        assert "feature_distributions" in report
        assert "correlation_analysis" in report
        assert "missing_values_analysis" in report
        assert "summary" in report

        # Check summary
        assert "analysis_completed" in report["summary"]
        assert report["summary"]["analysis_completed"] is True

        # Check if plots were created
        assert (plots_dir / "target_analysis" / "readmitted_distribution.png").exists()
        assert (plots_dir / "correlations" / "correlation_heatmap.png").exists()

    def test_run_full_analysis_custom_target(self, eda_analyzer, sample_data):
        """Test run_full_analysis with custom target column."""
        sample_data["custom_target"] = np.random.choice(["A", "B"], len(sample_data))

        report = eda_analyzer.run_full_analysis(sample_data, target_col="custom_target")

        assert report["target_analysis"]["target_column"] == "custom_target"

    def test_run_full_analysis_empty(self, eda_analyzer):
        """Test run_full_analysis with empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(Exception):  # Should raise an exception
            eda_analyzer.run_full_analysis(empty_df)

    # ---------- Helper Method Tests ----------

    def test_get_numeric_columns(self, eda_analyzer, sample_data):
        """Test _get_numeric_columns helper method."""
        numeric_cols = eda_analyzer._get_numeric_columns(sample_data)

        expected_numeric = [
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_outpatient",
            "number_emergency",
            "number_inpatient",
            "number_diagnoses",
        ]

        assert set(numeric_cols) == set(expected_numeric)

    def test_get_categorical_columns(self, eda_analyzer, sample_data):
        """Test _get_categorical_columns helper method."""
        categorical_cols = eda_analyzer._get_categorical_columns(sample_data)

        expected_categorical = ["race", "gender", "age_group", "readmitted"]

        assert set(categorical_cols) == set(expected_categorical)

    def test_find_high_correlations(self, eda_analyzer):
        """Test _find_high_correlations helper method."""
        # Create DataFrame with known high correlation
        data = pd.DataFrame(
            {
                "col1": [1, 2, 3, 4, 5],
                "col2": [2, 4, 6, 8, 10],  # Perfect correlation with col1
                "col3": [5, 4, 3, 2, 1],  # Perfect negative correlation
            }
        )

        corr_matrix = data.corr()
        high_corr = eda_analyzer._find_high_correlations(corr_matrix, threshold=0.8)

        # Should find high correlations
        assert len(high_corr) > 0

    # ---------- Error Handling Tests ----------

    def test_plots_directory_creation(self, sample_config, temp_dir):
        """Test that plots directory is created properly."""
        # Change to temp directory to avoid polluting real plots directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            analyzer = EDAAnalyzer(sample_config)

            # Check that plots directory and subdirectories exist
            assert analyzer.plots_dir.exists()
            assert (analyzer.plots_dir / "distributions").exists()
            assert (analyzer.plots_dir / "correlations").exists()
            assert (analyzer.plots_dir / "target_analysis").exists()

        finally:
            os.chdir(original_cwd)

    def test_report_generation(self, eda_analyzer, sample_data, temp_dir):
        """Test that EDA report is generated correctly."""
        # Change to temp directory for report file creation
        original_cwd = os.getcwd()
        os.chdir(temp_dir)

        try:
            report = eda_analyzer.run_full_analysis(sample_data)

            # Check that report file was created
            assert Path("logs/eda_report.json").exists()

            # Check that report contains expected keys
            assert "timestamp" in report
            assert "dataset_shape" in report

        finally:
            os.chdir(original_cwd)

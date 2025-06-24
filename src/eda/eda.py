import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
import os
import json
from pathlib import Path
from typing import Dict, List


class EDAAnalyzer:
    """
    Exploratory Data Analysis (EDA) analyzer for the MLOps pipeline.

    Performs comprehensive data analysis including distributions, correlations,
    missing values analysis, and target analysis.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.plots_dir = self._setup_plots_directory()
        self.report = {}

        # Set plotting style and ensure non-interactive backend
        matplotlib.use("Agg")
        plt.style.use(
            "default"
        )  # Use default instead of seaborn-v0_8 for compatibility
        plt.ioff()  # Turn off interactive mode
        sns.set_palette("husl")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate required config sections
            required_sections = ["data", "model", "logging"]
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required config section: {section}")

            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")

    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            log_config = self.config.get("logging", {})
            log_file = log_config.get("file", "logs/eda.log").replace(
                "main.log", "eda.log"
            )
            log_level = log_config.get("level", "INFO")
            log_format = log_config.get(
                "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Create logs directory if it doesn't exist
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Configure logging
            logging.basicConfig(
                level=getattr(logging, log_level.upper()),
                format=log_format,
                handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
            )

            return logging.getLogger(__name__)
        except Exception:
            # Fallback logging
            logging.basicConfig(level=logging.INFO)
            return logging.getLogger(__name__)

    def _setup_plots_directory(self) -> Path:
        """Setup plots directory structure."""
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (plots_dir / "distributions").mkdir(exist_ok=True)
        (plots_dir / "correlations").mkdir(exist_ok=True)
        (plots_dir / "target_analysis").mkdir(exist_ok=True)

        return plots_dir

    def _get_numeric_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of numeric columns."""
        return data.select_dtypes(include=[np.number]).columns.tolist()

    def _get_categorical_columns(self, data: pd.DataFrame) -> List[str]:
        """Get list of categorical columns."""
        return data.select_dtypes(include=["object", "category"]).columns.tolist()

    def describe_dataset(self, data: pd.DataFrame) -> Dict:
        """Generate comprehensive dataset description."""
        try:
            if data.empty:
                self.logger.warning("Empty dataset provided for description")
                return {"error": "Empty dataset"}

            self.logger.info("Generating dataset description...")

            numeric_cols = self._get_numeric_columns(data)
            categorical_cols = self._get_categorical_columns(data)

            description = {
                "basic_info": {
                    "shape": data.shape,
                    "total_rows": len(data),
                    "total_columns": len(data.columns),
                    "memory_usage_mb": round(
                        data.memory_usage(deep=True).sum() / 1024**2, 2
                    ),
                },
                "column_types": {
                    "numeric_columns": len(numeric_cols),
                    "categorical_columns": len(categorical_cols),
                    "numeric_column_names": numeric_cols,
                    "categorical_column_names": categorical_cols,
                },
                "missing_values": {
                    "total_missing": int(data.isnull().sum().sum()),
                    "missing_by_column": data.isnull().sum().to_dict(),
                    "missing_percentage": (data.isnull().sum() / len(data) * 100)
                    .round(2)
                    .to_dict(),
                },
            }

            # Add numeric statistics if numeric columns exist
            if numeric_cols:
                description["numeric_summary"] = data[numeric_cols].describe().to_dict()

            self.logger.info("Dataset description completed")
            return description

        except Exception as e:
            self.logger.error(f"Error in dataset description: {str(e)}")
            return {"error": str(e)}

    def analyze_target_variable(
        self, data: pd.DataFrame, target_col: str = None
    ) -> Dict:
        """Analyze target variable distribution and characteristics."""
        try:
            # Get target column from config if not provided
            if target_col is None:
                target_col = self.config.get("features", {}).get(
                    "target_column", "readmitted"
                )

            if target_col not in data.columns:
                raise KeyError(f"Target column '{target_col}' not found in data")

            self.logger.info(f"Analyzing target variable: {target_col}")

            # Basic target analysis
            value_counts = data[target_col].value_counts()
            percentages = (value_counts / len(data) * 100).round(2)

            analysis = {
                "target_column": target_col,
                "unique_values": list(data[target_col].unique()),
                "value_counts": value_counts.to_dict(),
                "percentages": percentages.to_dict(),
                "missing_values": int(data[target_col].isnull().sum()),
                "class_balance": {
                    "is_balanced": percentages.min() > 30,  # Simple balance check
                    "imbalance_ratio": round(percentages.max() / percentages.min(), 2),
                },
            }

            # Create target distribution plots
            self._plot_target_distribution(data, target_col)

            self.logger.info("Target variable analysis completed")
            return analysis

        except Exception as e:
            self.logger.error(f"Error in target analysis: {str(e)}")
            return {"error": str(e)}

    def _plot_target_distribution(self, data: pd.DataFrame, target_col: str):
        """Create target distribution plots."""
        try:
            # Bar plot
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts = data[target_col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            ax.set_title(f"Distribution of {target_col}")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Count")
            plt.setp(ax.get_xticklabels(), rotation=45)
            plt.tight_layout()
            plt.savefig(
                self.plots_dir / "target_analysis" / f"{target_col}_distribution.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Pie chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                value_counts.values,
                labels=value_counts.index,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.set_title(f"{target_col} Distribution")
            plt.tight_layout()
            plt.savefig(
                self.plots_dir / "target_analysis" / f"{target_col}_pie.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            self.logger.info(f"Target distribution plots saved for {target_col}")

        except Exception as e:
            self.logger.error(f"Error creating target plots: {str(e)}")

    def analyze_feature_distributions(self, data: pd.DataFrame) -> Dict:
        """Analyze distributions of numeric features."""
        try:
            numeric_cols = self._get_numeric_columns(data)

            if not numeric_cols:
                self.logger.warning(
                    "No numeric columns found for distribution analysis"
                )
                return {"error": "No numeric columns found"}

            self.logger.info(
                f"Analyzing distributions for {len(numeric_cols)} numeric features"
            )

            distribution_stats = {}

            for col in numeric_cols:
                # Calculate statistics
                series = data[col].dropna()
                if len(series) == 0:
                    continue

                stats = {
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "skewness": float(series.skew()),
                    "kurtosis": float(series.kurtosis()),
                }
                distribution_stats[col] = stats

                # Create distribution plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(series, kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(
                    self.plots_dir / "distributions" / f"{col}_distribution.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

            self.logger.info("Feature distribution analysis completed")
            return distribution_stats

        except Exception as e:
            self.logger.error(f"Error in feature distribution analysis: {str(e)}")
            return {"error": str(e)}

    def analyze_correlations(self, data: pd.DataFrame, target_col: str = None) -> Dict:
        """Analyze correlations between features and with target variable."""
        try:
            # Get target column from config if not provided
            if target_col is None:
                target_col = self.config.get("features", {}).get(
                    "target_column", "readmitted"
                )

            numeric_cols = self._get_numeric_columns(data)

            if len(numeric_cols) < 2:
                self.logger.warning(
                    "Insufficient numeric columns for correlation analysis"
                )
                return {"error": "Insufficient numeric columns"}

            self.logger.info("Analyzing feature correlations")

            # Calculate correlation matrix
            correlation_matrix = data[numeric_cols].corr()

            # Create correlation heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=0.5,
                ax=ax,
            )
            ax.set_title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(
                self.plots_dir / "correlations" / "correlation_heatmap.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Analyze correlations with target if target is numeric
            target_correlations = {}
            if target_col in data.columns:
                if target_col in numeric_cols:
                    target_corr = (
                        correlation_matrix[target_col]
                        .drop(target_col)
                        .sort_values(key=abs, ascending=False)
                    )
                    target_correlations = target_corr.to_dict()

                    # Plot top correlations with target
                    top_corr = target_corr.head(10)
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x=top_corr.values, y=top_corr.index, ax=ax)
                    ax.set_title(f"Top Correlations with {target_col}")
                    ax.set_xlabel("Correlation Coefficient")
                    plt.tight_layout()
                    plt.savefig(
                        self.plots_dir
                        / "correlations"
                        / f"top_correlations_{target_col}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(fig)

            analysis = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "target_correlations": target_correlations,
                "high_correlations": self._find_high_correlations(correlation_matrix),
            }

            self.logger.info("Correlation analysis completed")
            return analysis

        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
            return {"error": str(e)}

    def _find_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.8
    ) -> Dict:
        """Find pairs of features with high correlation."""
        high_corr_pairs = {}

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) > threshold:
                    high_corr_pairs[f"{col1}_vs_{col2}"] = round(corr_value, 3)

        return high_corr_pairs

    def analyze_missing_values(self, data: pd.DataFrame) -> Dict:
        """Comprehensive missing values analysis."""
        try:
            self.logger.info("Analyzing missing values")

            missing_summary = data.isnull().sum()
            missing_percentage = (missing_summary / len(data) * 100).round(2)

            analysis = {
                "total_missing_values": int(missing_summary.sum()),
                "columns_with_missing": int((missing_summary > 0).sum()),
                "missing_by_column": missing_summary[missing_summary > 0].to_dict(),
                "missing_percentage": missing_percentage[
                    missing_percentage > 0
                ].to_dict(),
                "rows_with_missing": int(data.isnull().any(axis=1).sum()),
                "complete_rows": int((~data.isnull().any(axis=1)).sum()),
            }

            # Create missing values heatmap if there are missing values
            if analysis["total_missing_values"] > 0:
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(
                    data.isnull(), cbar=True, yticklabels=False, cmap="viridis", ax=ax
                )
                ax.set_title("Missing Values Heatmap")
                plt.tight_layout()
                plt.savefig(
                    self.plots_dir / "missing_values_heatmap.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig)

            self.logger.info("Missing values analysis completed")
            return analysis

        except Exception as e:
            self.logger.error(f"Error in missing values analysis: {str(e)}")
            return {"error": str(e)}

    def run_full_analysis(self, data: pd.DataFrame, target_col: str = None) -> Dict:
        """Run comprehensive EDA analysis on the dataset."""
        self.logger.info("🔍 Starting comprehensive EDA analysis...")

        # Initialize report
        self.report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_shape": data.shape,
        }

        try:
            # Check for empty dataset early and raise exception
            if data.empty:
                error_msg = "Cannot perform EDA analysis on empty dataset"
                self.logger.error(error_msg)
                self.report["analysis_failed"] = True
                self.report["error"] = error_msg
                self._write_report()
                raise ValueError(error_msg)

            # Get target column from config if not provided
            if target_col is None:
                target_col = self.config.get("features", {}).get(
                    "target_column", "readmitted"
                )

            # Run all analyses
            self.report["dataset_description"] = self.describe_dataset(data)
            self.report["target_analysis"] = self.analyze_target_variable(
                data, target_col
            )
            self.report["feature_distributions"] = self.analyze_feature_distributions(
                data
            )
            self.report["correlation_analysis"] = self.analyze_correlations(
                data, target_col
            )
            self.report["missing_values_analysis"] = self.analyze_missing_values(data)

            # Summary statistics
            self.report["summary"] = {
                "total_features": len(data.columns),
                "numeric_features": len(self._get_numeric_columns(data)),
                "categorical_features": len(self._get_categorical_columns(data)),
                "missing_data_percentage": round(
                    (data.isnull().sum().sum() / data.size) * 100, 2
                ),
                "analysis_completed": True,
            }

            # Write report
            self._write_report()

            self.logger.info("Comprehensive EDA analysis completed successfully!")
            return self.report

        except Exception as e:
            self.logger.error(f"❌ EDA analysis failed: {str(e)}")
            self.report["analysis_failed"] = True
            self.report["error"] = str(e)
            self._write_report()
            raise

    def _write_report(self):
        """Write EDA report to file."""
        try:
            report_dir = Path("logs")
            report_dir.mkdir(exist_ok=True)

            report_file = report_dir / "eda_report.json"
            with open(report_file, "w") as f:
                json.dump(self.report, f, indent=2, default=str)

            self.logger.info(f"EDA report written to: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to write EDA report: {str(e)}")


if __name__ == "__main__":
    """Demonstrate EDAAnalyzer functionality."""
    print("EDAAnalyzer module loaded successfully.")
    print("Usage examples:")
    print("  from src.eda.eda import EDAAnalyzer")
    print("  eda = EDAAnalyzer()")
    print("  report = eda.run_full_analysis(data)")
    print("  target_analysis = eda.analyze_target_variable(data, 'target_column')")

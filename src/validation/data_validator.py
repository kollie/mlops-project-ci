import pandas as pd
import numpy as np
import yaml
import logging
import json
import os
from pathlib import Path
from typing import Dict


class DataValidator:
    """
    Comprehensive data validator for the MLOps pipeline.

    Validates data schema, types, missing values, and target distribution.
    Can be used for both validation-only and validation-with-cleaning modes.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.report = {}

        # Initialize required columns from config
        try:
            self.required_columns = (
                self.config["features"]["numerical_features"]
                + self.config["features"]["categorical_features"]
                + [self.config["features"]["target_column"]]
            )
        except KeyError as e:
            self.logger.warning(
                f"Features configuration not found: {e}. Using basic validation."
            )
            self.required_columns = []

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
            log_file = log_config.get("file", "logs/validation.log")
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

    def validate_schema(self, data: pd.DataFrame) -> bool:
        """Validate that all required columns are present."""
        try:
            if not self.required_columns:
                self.logger.info(
                    "No schema requirements defined. Skipping schema validation."
                )
                return True

            required = set(self.required_columns)
            available = set(data.columns)

            # Special handling for common column substitutions
            if (
                "age_group" in required
                and "age_group" not in available
                and "age" in available
            ):
                self.logger.warning(
                    "Expected 'age_group' not found. Using 'age' as acceptable substitute."
                )
                available = available.union({"age_group"})

            missing = required - available
            if missing:
                self.logger.error(f"Missing required columns: {missing}")
                self.report["missing_columns"] = list(missing)
                return False

            self.logger.info("Schema validation passed.")
            self.report["schema_valid"] = True
            return True

        except Exception as e:
            self.logger.error(f"Schema validation error: {str(e)}")
            self.report["schema_error"] = str(e)
            return False

    def validate_data_types(self, data: pd.DataFrame) -> bool:
        """Validate that columns have expected data types."""
        try:
            if "features" not in self.config:
                self.logger.info(
                    "No data type requirements defined. Skipping type validation."
                )
                return True

            type_issues = {}

            # Check numerical features
            for col in self.config["features"].get("numerical_features", []):
                if col in data.columns and not np.issubdtype(
                    data[col].dtype, np.number
                ):
                    type_issues[col] = f"Expected numeric, got {data[col].dtype}"

            # Check categorical features
            for col in self.config["features"].get("categorical_features", []):
                if col in data.columns and not (
                    pd.api.types.is_string_dtype(data[col])
                    or pd.api.types.is_categorical_dtype(data[col])
                    or pd.api.types.is_object_dtype(data[col])
                ):
                    type_issues[col] = f"Expected categorical, got {data[col].dtype}"

            if type_issues:
                self.logger.error(f"Data type validation failed: {type_issues}")
                self.report["type_issues"] = type_issues
                return False

            self.logger.info("Data type validation passed.")
            self.report["data_types_valid"] = True
            return True

        except Exception as e:
            self.logger.error(f"Data type validation error: {str(e)}")
            self.report["data_type_error"] = str(e)
            return False

    def validate_missing_values(
        self, data: pd.DataFrame, threshold: float = 0.5
    ) -> bool:
        """Validate that missing values are within acceptable limits."""
        try:
            missing_ratio = data.isnull().mean()
            high_missing = missing_ratio[missing_ratio > threshold]

            self.report["missing_value_ratios"] = missing_ratio.to_dict()

            if not high_missing.empty:
                self.logger.error(
                    f"Columns with >{threshold*100}% missing values: {list(high_missing.index)}"
                )
                self.report["high_missing_columns"] = list(high_missing.index)
                return False

            self.logger.info(
                f"Missing value validation passed. Max missing ratio: {missing_ratio.max():.2%}"
            )
            self.report["missing_values_valid"] = True
            return True

        except Exception as e:
            self.logger.error(f"Missing value validation error: {str(e)}")
            self.report["missing_value_error"] = str(e)
            return False

    def validate_target_distribution(
        self, data: pd.DataFrame, min_class_ratio: float = 0.01
    ) -> bool:
        """Validate target variable distribution."""
        try:
            if (
                "features" not in self.config
                or "target_column" not in self.config["features"]
            ):
                self.logger.info(
                    "No target column defined. Skipping target distribution validation."
                )
                return True

            target_col = self.config["features"]["target_column"]

            if target_col not in data.columns:
                self.logger.error(f"Target column '{target_col}' not found in data")
                return False

            dist = data[target_col].value_counts(normalize=True)
            self.report["target_distribution"] = dist.to_dict()

            if (dist < min_class_ratio).any():
                self.logger.error(
                    f"Target class imbalance detected. Min class ratio: {dist.min():.2%}"
                )
                self.report["target_imbalance"] = True
                return False

            self.logger.info(
                f"Target distribution validation passed. Classes: {dist.to_dict()}"
            )
            self.report["target_distribution_valid"] = True
            return True

        except Exception as e:
            self.logger.error(f"Target distribution validation error: {str(e)}")
            self.report["target_distribution_error"] = str(e)
            return False

    def validate_all(self, data: pd.DataFrame, **kwargs) -> Dict[str, bool]:
        """Run all validation checks and return results."""
        self.logger.info("Starting comprehensive data validation...")

        # Initialize report
        self.report = {
            "initial_rows": len(data),
            "initial_columns": len(data.columns),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        # Run all validations
        results = {
            "schema": self.validate_schema(data),
            "data_types": self.validate_data_types(data),
            "missing_values": self.validate_missing_values(
                data, kwargs.get("missing_threshold", 0.5)
            ),
            "target_distribution": self.validate_target_distribution(
                data, kwargs.get("min_class_ratio", 0.01)
            ),
        }

        all_passed = all(results.values())
        self.report["all_validations_passed"] = all_passed
        self.report["validation_results"] = results

        # Write validation report
        self._write_report()

        if all_passed:
            self.logger.info("All data validations passed successfully!")
        else:
            failed = [k for k, v in results.items() if not v]
            self.logger.error(f"Validation failed for: {failed}")

        return results

    def validate_and_clean(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Validate data and apply cleaning operations."""
        self.logger.info("Starting data validation and cleaning...")

        # Initialize report
        self.report = {
            "initial_rows": len(data),
            "initial_columns": len(data.columns),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

        try:
            # Step 1: Schema validation (must pass)
            self.logger.info("Step 1: Validating data schema...")
            if not self.validate_schema(data):
                raise ValueError(
                    "Schema validation failed. Cannot proceed with cleaning."
                )

            # Step 2: Data type validation (log issues but continue)
            self.logger.info("Step 2: Validating data types...")
            self.validate_data_types(data)

            # Step 3: Handle missing values
            self.logger.info("Step 3: Handling missing values...")
            strategy = kwargs.get("strategy", "drop")
            self.logger.info(f"Using missing value strategy: '{strategy}'")
            data_cleaned = self._handle_missing_values(data, strategy)

            # Step 4: Validate target distribution on cleaned data
            self.logger.info("Step 4: Validating target distribution...")
            self.validate_target_distribution(data_cleaned)

            # Update final report
            self.report["final_rows"] = len(data_cleaned)
            self.report["final_columns"] = len(data_cleaned.columns)
            self.report["rows_dropped"] = (
                self.report["initial_rows"] - self.report["final_rows"]
            )
            self.report["columns_dropped"] = (
                self.report["initial_columns"] - self.report["final_columns"]
            )
            self.report["cleaning_completed"] = True

            # Write final report
            self._write_report()

            # Summary log
            rows_change = self.report["initial_rows"] - self.report["final_rows"]
            cols_change = self.report["initial_columns"] - self.report["final_columns"]

            self.logger.info("=" * 50)
            self.logger.info("DATA VALIDATION AND CLEANING COMPLETED")
            self.logger.info("=" * 50)
            self.logger.info("Data summary:")
            self.logger.info(
                f"   Rows: {self.report['initial_rows']} → {self.report['final_rows']} ({-rows_change:+d})"
            )
            self.logger.info(
                f"   Columns: {self.report['initial_columns']} → {self.report['final_columns']} ({-cols_change:+d})"
            )

            if rows_change > 0:
                rows_pct = (rows_change / self.report["initial_rows"]) * 100
                self.logger.info(f"   Rows removed: {rows_change} ({rows_pct:.1f}%)")

            if cols_change > 0:
                cols_pct = (cols_change / self.report["initial_columns"]) * 100
                self.logger.info(f"   Columns removed: {cols_change} ({cols_pct:.1f}%)")

            self.logger.info("=" * 50)

            return data_cleaned

        except Exception as e:
            self.logger.error(f"Validation and cleaning failed: {str(e)}")
            self.report["cleaning_failed"] = True
            self.report["error"] = str(e)
            self._write_report()
            raise

    def _handle_missing_values(
        self, data: pd.DataFrame, strategy: str = "drop"
    ) -> pd.DataFrame:
        """Handle missing values based on specified strategy."""
        missing_summary = data.isnull().sum()
        missing_ratios = data.isnull().mean()

        self.report["missing_values_before_cleaning"] = missing_summary.to_dict()
        self.report["missing_ratios_before_cleaning"] = missing_ratios.to_dict()

        rows_before = len(data)
        cols_before = len(data.columns)

        if strategy == "drop":
            # Log which rows will be dropped and why
            rows_with_missing = data.isnull().any(axis=1).sum()
            if rows_with_missing > 0:
                self.logger.info(f"Found {rows_with_missing} rows with missing values")

                # Log missing value details by column
                for col in data.columns:
                    missing_count = missing_summary[col]
                    if missing_count > 0:
                        missing_pct = missing_ratios[col] * 100
                        self.logger.info(
                            f"  Column '{col}': {missing_count} missing values ({missing_pct:.1f}%)"
                        )

            # Drop rows with any missing values
            data_cleaned = data.dropna()
            rows_dropped = rows_before - len(data_cleaned)

            if rows_dropped > 0:
                self.logger.info(
                    f"Dropped {rows_dropped} rows with missing values (strategy: {strategy})"
                )
                self.logger.info(
                    f"   Rows: {rows_before} → {len(data_cleaned)} ({rows_dropped} removed)"
                )
            else:
                self.logger.info("No rows dropped - no missing values found")

        elif strategy == "drop_columns":
            # Drop columns with high missing value ratio
            threshold = 0.5
            high_missing_cols = missing_summary[
                missing_summary > len(data) * threshold
            ].index

            if len(high_missing_cols) > 0:
                self.logger.warning(
                    f"Dropping {len(high_missing_cols)} columns with >{threshold*100}% missing values:"
                )
                for col in high_missing_cols:
                    missing_count = missing_summary[col]
                    missing_pct = missing_ratios[col] * 100
                    self.logger.warning(
                        f"   - Column '{col}': {missing_count}/{len(data)} missing ({missing_pct:.1f}%)"
                    )

                data_cleaned = data.drop(columns=high_missing_cols)
                self.logger.info(
                    f"   Columns: {cols_before} → {len(data_cleaned.columns)} ({len(high_missing_cols)} removed)"
                )

                # Store dropped columns info in report
                self.report["dropped_columns"] = {
                    col: {
                        "missing_count": int(missing_summary[col]),
                        "missing_percentage": float(missing_ratios[col] * 100),
                        "reason": f">{threshold*100}% missing values",
                    }
                    for col in high_missing_cols
                }
            else:
                self.logger.info(
                    f"No columns dropped - no columns exceed {threshold*100}% missing threshold"
                )
                data_cleaned = data.copy()

        else:
            # For now, default to dropping rows
            self.logger.warning(
                f"Unknown missing value strategy '{strategy}'. Using 'drop' as fallback."
            )

            rows_with_missing = data.isnull().any(axis=1).sum()
            if rows_with_missing > 0:
                # Log details before fallback
                for col in data.columns:
                    missing_count = missing_summary[col]
                    if missing_count > 0:
                        missing_pct = missing_ratios[col] * 100
                        self.logger.info(
                            f"  Column '{col}': {missing_count} missing values ({missing_pct:.1f}%)"
                        )

            data_cleaned = data.dropna()
            rows_dropped = rows_before - len(data_cleaned)

            if rows_dropped > 0:
                self.logger.info(
                    f"Dropped {rows_dropped} rows with missing values (fallback strategy)"
                )

        self.report["missing_value_strategy"] = strategy
        self.report["rows_before_cleaning"] = rows_before
        self.report["rows_after_cleaning"] = len(data_cleaned)
        self.report["columns_before_cleaning"] = cols_before
        self.report["columns_after_cleaning"] = len(data_cleaned.columns)

        return data_cleaned

    def _write_report(self):
        """Write validation report to file."""
        try:
            report_dir = Path("logs")
            report_dir.mkdir(exist_ok=True)

            report_file = report_dir / "validation_report.json"
            with open(report_file, "w") as f:
                json.dump(self.report, f, indent=2, default=str)

            self.logger.info(f"Validation report written to: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to write validation report: {str(e)}")


if __name__ == "__main__":
    """Demonstrate DataValidator functionality."""
    print("DataValidator module loaded successfully.")
    print("Usage examples:")
    print("  from src.validation.data_validator import DataValidator")
    print("  validator = DataValidator()")
    print("  results = validator.validate_all(data)  # Validation only")
    print("  clean_data = validator.validate_and_clean(data)  # Validation + cleaning")

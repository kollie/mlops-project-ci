"""
Feature Engineering Module for MLOps Project.

Handles advanced feature creation and selection following the same patterns
as other modules in the pipeline.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """
    Feature engineering pipeline for the MLOps project.

    Creates new features and applies feature selection following the same
    patterns as other modules in the pipeline.
    """

    def __init__(self, config_path: str = "src/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.report = {}

        # Feature selection components
        self.feature_selector: Optional[SelectKBest] = None
        self.selected_features: List[str] = []
        self.original_features: List[str] = []
        self.engineered_features: List[str] = []

        # Categorical encoding components
        self.label_encoders: Dict[str, LabelEncoder] = {}

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate required config sections
            required_sections = ["data", "model", "logging", "features"]
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
            log_file = log_config.get("file", "logs/feature_engineering.log").replace(
                "main.log", "feature_engineering.log"
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

    def _create_age_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create age group categorical feature."""
        try:
            if "age" not in data.columns:
                self.logger.warning(
                    "'age' column not found, skipping age group creation"
                )
                return data

            # Convert age categories to numeric for grouping
            age_mapping = {
                "[0-10)": 5,
                "[10-20)": 15,
                "[20-30)": 25,
                "[30-40)": 35,
                "[40-50)": 45,
                "[50-60)": 55,
                "[60-70)": 65,
                "[70-80)": 75,
                "[80-90)": 85,
                "[90-100)": 95,
            }

            data = data.copy()
            data["age_numeric"] = data["age"].map(age_mapping)

            # Create age groups
            data["age_group"] = pd.cut(
                data["age_numeric"],
                bins=[0, 30, 50, 70, 100],
                labels=["Young", "Adult", "Senior", "Elderly"],
                include_lowest=True,
            )

            # Drop temporary column
            data = data.drop(columns=["age_numeric"])

            self.logger.info("✅ Created age_group feature")
            return data

        except Exception as e:
            self.logger.error(f"Error creating age groups: {str(e)}")
            return data

    def _create_length_of_stay_groups(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create length of stay categorical feature."""
        try:
            if "time_in_hospital" not in data.columns:
                self.logger.warning(
                    "'time_in_hospital' column not found, skipping LOS groups"
                )
                return data

            data = data.copy()
            data["los_group"] = pd.cut(
                data["time_in_hospital"],
                bins=[0, 3, 7, 14, float("inf")],
                labels=["Short", "Medium", "Long", "Extended"],
                include_lowest=True,
            )

            self.logger.info("✅ Created los_group feature")
            return data

        except Exception as e:
            self.logger.error(f"Error creating LOS groups: {str(e)}")
            return data

    def _create_total_visits(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create total previous visits feature."""
        try:
            visit_columns = [
                "number_outpatient",
                "number_emergency",
                "number_inpatient",
            ]
            missing_cols = [col for col in visit_columns if col not in data.columns]

            if missing_cols:
                self.logger.warning(
                    f"Missing visit columns {missing_cols}, skipping total visits"
                )
                return data

            data = data.copy()
            data["total_visits"] = data[visit_columns].sum(axis=1)

            self.logger.info("✅ Created total_visits feature")
            return data

        except Exception as e:
            self.logger.error(f"Error creating total visits: {str(e)}")
            return data

    def _create_medication_intensity(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create medication intensity feature."""
        try:
            required_cols = ["num_medications", "time_in_hospital"]
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                self.logger.warning(
                    f"Missing columns {missing_cols}, skipping medication intensity"
                )
                return data

            data = data.copy()
            # Avoid division by zero
            data["medication_intensity"] = data["num_medications"] / (
                data["time_in_hospital"].replace(0, 1)
            )

            self.logger.info("✅ Created medication_intensity feature")
            return data

        except Exception as e:
            self.logger.error(f"Error creating medication intensity: {str(e)}")
            return data

    def _create_binary_flags(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create useful binary flag features."""
        try:
            data = data.copy()
            created_flags = []

            # Has emergency visits
            if "number_emergency" in data.columns:
                data["has_emergency_visits"] = (data["number_emergency"] > 0).astype(
                    int
                )
                created_flags.append("has_emergency_visits")

            # Was medicated
            if "num_medications" in data.columns:
                data["was_medicated"] = (data["num_medications"] > 0).astype(int)
                created_flags.append("was_medicated")

            # Many diagnoses (high complexity)
            if "number_diagnoses" in data.columns:
                data["many_diagnoses"] = (data["number_diagnoses"] >= 9).astype(int)
                created_flags.append("many_diagnoses")

            # Had medication change
            if "change" in data.columns:
                data["medication_changed"] = (data["change"] == "Ch").astype(int)
                created_flags.append("medication_changed")

            # Is diabetic medication user
            if "diabetesMed" in data.columns:
                data["uses_diabetes_med"] = (data["diabetesMed"] == "Yes").astype(int)
                created_flags.append("uses_diabetes_med")

            if created_flags:
                self.logger.info(f"✅ Created binary flags: {created_flags}")

            return data

        except Exception as e:
            self.logger.error(f"Error creating binary flags: {str(e)}")
            return data

    def _create_diagnosis_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create diagnosis-based features."""
        try:
            diag_columns = ["diag_1", "diag_2", "diag_3"]
            available_diag_cols = [col for col in diag_columns if col in data.columns]

            if not available_diag_cols:
                self.logger.warning(
                    "No diagnosis columns found, skipping diagnosis features"
                )
                return data

            data = data.copy()
            created_features = []

            def extract_numeric_code(code):
                """Extract numeric part from diagnosis code."""
                try:
                    code_str = str(code).strip()
                    if code_str in ["", "nan", "None"]:
                        return None
                    # Remove E and V prefixes, extract first numeric part
                    if code_str.startswith(("E", "V")):
                        return None
                    return float(code_str.split(".")[0])
                except Exception:
                    return None

            # Check for diabetes (250.xx codes)
            diabetes_flags = []
            for col in available_diag_cols:
                data[f"{col}_numeric"] = data[col].apply(extract_numeric_code)
                diabetes_flag = (data[f"{col}_numeric"] == 250).astype(int)
                diabetes_flags.append(diabetes_flag)
                data = data.drop(columns=[f"{col}_numeric"])  # Clean up temp column

            if diabetes_flags:
                data["has_diabetes_diagnosis"] = pd.concat(diabetes_flags, axis=1).max(
                    axis=1
                )
                created_features.append("has_diabetes_diagnosis")

            # Check for circulatory system diseases (390-459)
            circulatory_flags = []
            for col in available_diag_cols:
                data[f"{col}_numeric"] = data[col].apply(extract_numeric_code)
                circulatory_flag = data[f"{col}_numeric"].apply(
                    lambda x: 1 if x is not None and 390 <= x <= 459 else 0
                )
                circulatory_flags.append(circulatory_flag)
                data = data.drop(columns=[f"{col}_numeric"])  # Clean up temp column

            if circulatory_flags:
                data["has_circulatory_diagnosis"] = pd.concat(
                    circulatory_flags, axis=1
                ).max(axis=1)
                created_features.append("has_circulatory_diagnosis")

            if created_features:
                self.logger.info(f"✅ Created diagnosis features: {created_features}")

            return data

        except Exception as e:
            self.logger.error(f"Error creating diagnosis features: {str(e)}")
            return data

    def _handle_missing_values_smart(self, X: pd.DataFrame) -> None:
        """Handle missing values intelligently based on column types."""
        try:
            for column in X.columns:
                if X[column].isna().any():
                    if X[column].dtype.name == "category":
                        # For categorical columns, add 'Unknown' category and fill
                        if "Unknown" not in X[column].cat.categories:
                            X[column] = X[column].cat.add_categories(["Unknown"])
                        X[column] = X[column].fillna("Unknown")
                        self.logger.info(
                            f"Filled categorical column '{column}' with 'Unknown'"
                        )
                    elif X[column].dtype in ["int64", "float64", "int32", "float32"]:
                        # For numerical columns, fill with 0
                        X[column] = X[column].fillna(0)
                        self.logger.info(f"Filled numerical column '{column}' with 0")
                    else:
                        # For other types, convert to string and fill with 'Unknown'
                        X[column] = X[column].astype(str).fillna("Unknown")
                        self.logger.info(f"Filled column '{column}' with 'Unknown'")

        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            # Fallback: try to handle each column type separately
            for column in X.columns:
                if X[column].isna().any():
                    if X[column].dtype.name == "category":
                        X[column] = (
                            X[column].cat.add_categories(["Unknown"]).fillna("Unknown")
                        )
                    else:
                        X[column] = X[column].fillna(
                            0
                            if np.issubdtype(X[column].dtype, np.number)
                            else "Unknown"
                        )

    def _encode_categorical_features(
        self, X: pd.DataFrame, is_training: bool = True
    ) -> pd.DataFrame:
        """Encode categorical features for machine learning models."""
        try:
            X_encoded = X.copy()

            # Get categorical columns (including category dtype and object dtype)
            categorical_cols = X_encoded.select_dtypes(
                include=["category", "object"]
            ).columns.tolist()

            if len(categorical_cols) > 0:
                self.logger.info(
                    f"Encoding {len(categorical_cols)} categorical features: {categorical_cols}"
                )

                for col in categorical_cols:
                    if is_training:
                        # Fit new encoder during training
                        if col not in self.label_encoders:
                            le = LabelEncoder()
                            # Handle NaN values by converting to string
                            X_encoded[col] = (
                                X_encoded[col].astype(str).fillna("Unknown")
                            )

                            # Fit the encoder
                            le.fit(X_encoded[col])
                            self.label_encoders[col] = le

                            self.logger.info(
                                f"Fitted label encoder for column '{col}' with {len(le.classes_)} categories"
                            )

                        # Transform using fitted encoder
                        le = self.label_encoders[col]
                        X_encoded[col] = X_encoded[col].astype(str).fillna("Unknown")
                        X_encoded[col] = le.transform(X_encoded[col])

                    else:
                        # Transform using existing encoder during inference
                        if col in self.label_encoders:
                            le = self.label_encoders[col]
                            X_encoded[col] = (
                                X_encoded[col].astype(str).fillna("Unknown")
                            )

                            # Handle unknown categories
                            unknown_mask = ~X_encoded[col].isin(le.classes_)
                            if unknown_mask.any():
                                self.logger.warning(
                                    f"Found {unknown_mask.sum()} unknown categories in column '{col}'. Replacing with 'Unknown'."
                                )
                                X_encoded.loc[unknown_mask, col] = "Unknown"

                                # Add 'Unknown' to encoder classes if not present
                                if "Unknown" not in le.classes_:
                                    le.classes_ = np.append(le.classes_, "Unknown")

                            X_encoded[col] = le.transform(X_encoded[col])

                        else:
                            # If encoder doesn't exist, create a simple one (fallback)
                            self.logger.warning(
                                f"No encoder found for column '{col}'. Creating fallback encoder."
                            )
                            le = LabelEncoder()
                            X_encoded[col] = (
                                X_encoded[col].astype(str).fillna("Unknown")
                            )
                            X_encoded[col] = le.fit_transform(X_encoded[col])
                            self.label_encoders[col] = le

                    # Ensure the column is numeric
                    X_encoded[col] = X_encoded[col].astype(int)
                    self.logger.info(f"Encoded categorical column '{col}' to integer")

            self.logger.info(
                f"✅ Categorical encoding completed. Final shape: {X_encoded.shape}"
            )
            return X_encoded

        except Exception as e:
            self.logger.error(f"Error encoding categorical features: {str(e)}")
            # Fallback: simple label encoding
            try:
                X_encoded = X.copy()
                categorical_cols = X_encoded.select_dtypes(
                    include=["category", "object"]
                ).columns

                for col in categorical_cols:
                    le = LabelEncoder()
                    X_encoded[col] = X_encoded[col].astype(str).fillna("Unknown")
                    X_encoded[col] = le.fit_transform(X_encoded[col])
                    self.logger.info(f"Applied fallback encoding to column '{col}'")

                return X_encoded
            except Exception as fallback_error:
                self.logger.error(
                    f"Fallback encoding also failed: {str(fallback_error)}"
                )
                raise

    def _apply_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply feature selection to reduce dimensionality."""
        try:
            feature_config = self.config.get("feature_engineering", {})
            n_features = feature_config.get("n_features_to_select")

            if not n_features or n_features >= X.shape[1]:
                self.logger.info("Skipping feature selection (k >= total features)")
                self.selected_features = list(X.columns)
                return X

            self.logger.info(
                f"Selecting top {n_features} features using ANOVA F-test..."
            )

            # All features should be numerical at this point after encoding
            # But let's double-check
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                self.logger.warning(
                    f"Found non-numeric columns after encoding: {list(non_numeric_cols)}"
                )
                # Try to convert them
                for col in non_numeric_cols:
                    try:
                        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
                    except ValueError:
                        self.logger.error(
                            f"Could not convert column {col} to numeric. Dropping it."
                        )
                        X = X.drop(columns=[col])

            # If we don't have enough features for selection, keep all
            if X.shape[1] <= n_features:
                self.logger.info(
                    f"Not enough features ({X.shape[1]}) for selection (k={n_features}). Keeping all features."
                )
                self.selected_features = list(X.columns)
                return X

            # Encode target variable if it's categorical
            y_encoded = y.copy()
            if y.dtype == "object":
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)

            # Apply feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected_array = self.feature_selector.fit_transform(X, y_encoded)

            # Get selected feature names
            selected_mask = self.feature_selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()

            # Create final DataFrame
            X_selected = pd.DataFrame(
                X_selected_array, columns=self.selected_features, index=X.index
            )

            self.logger.info(
                f"✅ Selected {len(self.selected_features)} features from {X.shape[1]} total"
            )
            self.logger.info(f"Selected features: {self.selected_features}")

            return X_selected

        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            # Return original data if feature selection fails
            self.logger.warning(
                "Returning original features due to feature selection error"
            )
            self.selected_features = list(X.columns)
            return X

    def fit_transform(
        self, data: pd.DataFrame, target_col: str = "readmitted"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit the feature engineering pipeline and transform the data."""
        try:
            self.logger.info("🔧 Starting feature engineering pipeline...")

            # Validate input
            if data.empty:
                raise ValueError(
                    "Cannot perform feature engineering on empty DataFrame"
                )
            if target_col not in data.columns:
                raise KeyError(f"Target column '{target_col}' not found in data")

            # Initialize report
            self.report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "original_shape": data.shape,
                "target_column": target_col,
                "original_features": list(data.columns),
            }

            # Separate target
            X = data.drop(columns=[target_col])
            y = data[target_col]

            self.original_features = list(X.columns)

            # Apply feature engineering steps
            self.logger.info("Creating new features...")

            # Create new features
            X = self._create_age_groups(X)
            X = self._create_length_of_stay_groups(X)
            X = self._create_total_visits(X)
            X = self._create_medication_intensity(X)
            X = self._create_binary_flags(X)
            X = self._create_diagnosis_features(X)

            # Track engineered features
            self.engineered_features = [
                col for col in X.columns if col not in self.original_features
            ]

            self.logger.info(
                f"Created {len(self.engineered_features)} new features: {self.engineered_features}"
            )

            # Handle missing values in new features
            self._handle_missing_values_smart(X)

            # Encode categorical features for ML models
            X = self._encode_categorical_features(X, is_training=True)

            # Apply feature selection
            if self.config.get("feature_engineering", {}).get("apply_selection", True):
                X_selected = self._apply_feature_selection(X, y)
            else:
                X_selected = X
                self.selected_features = list(X.columns)
                self.logger.info("Feature selection disabled in config")

            # Update report
            self.report.update(
                {
                    "features_after_engineering": X.shape[1],
                    "engineered_features": self.engineered_features,
                    "features_after_selection": X_selected.shape[1],
                    "selected_features": self.selected_features,
                    "final_shape": X_selected.shape,
                    "engineering_completed": True,
                }
            )

            # Write report
            self._write_report()

            self.logger.info("✅ Feature engineering completed successfully!")
            self.logger.info(f"Final feature shape: {X_selected.shape}")

            return X_selected, y

        except Exception as e:
            self.logger.error(f"❌ Feature engineering failed: {str(e)}")
            self.report["engineering_failed"] = True
            self.report["error"] = str(e)
            self._write_report()
            raise

    def transform(
        self, data: pd.DataFrame, target_col: str = "readmitted"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Transform new data using the fitted feature engineering pipeline."""
        try:
            self.logger.info(
                "🔄 Transforming data with fitted feature engineering pipeline..."
            )

            # Validate input
            if data.empty:
                raise ValueError("Cannot transform empty DataFrame")
            if target_col not in data.columns:
                raise KeyError(f"Target column '{target_col}' not found in data")

            # Separate target
            X = data.drop(columns=[target_col])
            y = data[target_col]

            # Apply the same feature engineering steps
            X = self._create_age_groups(X)
            X = self._create_length_of_stay_groups(X)
            X = self._create_total_visits(X)
            X = self._create_medication_intensity(X)
            X = self._create_binary_flags(X)
            X = self._create_diagnosis_features(X)

            # Handle missing values
            self._handle_missing_values_smart(X)

            # Encode categorical features using fitted encoders
            X = self._encode_categorical_features(X, is_training=False)

            # Apply feature selection if fitted
            if self.feature_selector is not None and self.selected_features:
                # Ensure we only select from features that exist
                available_features = [
                    f for f in self.selected_features if f in X.columns
                ]
                if len(available_features) != len(self.selected_features):
                    self.logger.warning(
                        f"Some features missing in transform data. Using {len(available_features)} of {len(self.selected_features)} features."
                    )

                X_selected = X[available_features]
            else:
                # If no feature selection was applied during fit
                X_selected = X

            self.logger.info(
                f"✅ Data transformed successfully. Shape: {X_selected.shape}"
            )

            return X_selected, y

        except Exception as e:
            self.logger.error(f"❌ Data transformation failed: {str(e)}")
            raise

    def get_feature_names(self) -> List[str]:
        """Get the names of the final selected features."""
        if not self.selected_features:
            raise ValueError("Pipeline not fitted yet. Feature names unavailable.")
        return self.selected_features.copy()

    def get_engineered_features(self) -> List[str]:
        """Get the names of the newly created features."""
        return self.engineered_features.copy()

    def _write_report(self):
        """Write feature engineering report to file."""
        try:
            report_dir = Path("logs")
            report_dir.mkdir(exist_ok=True)

            report_file = report_dir / "feature_engineering_report.json"
            with open(report_file, "w") as f:
                json.dump(self.report, f, indent=2, default=str)

            self.logger.info(f"📋 Feature engineering report written to: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to write feature engineering report: {str(e)}")


if __name__ == "__main__":
    """Demonstrate FeatureEngineer functionality."""
    print("Feature engineering module loaded successfully.")
    print("Usage examples:")
    print("  from src.features.feature_engineering import FeatureEngineer")
    print("  engineer = FeatureEngineer()")
    print("  X_train, y_train = engineer.fit_transform(train_data)")
    print("  X_test, y_test = engineer.transform(test_data)")

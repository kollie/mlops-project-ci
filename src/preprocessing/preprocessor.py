import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
from pathlib import Path
from typing import List, Optional, Tuple
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


class Preprocessor:
    """
    Data preprocessing pipeline for the MLOps project.

    Handles missing values, feature scaling, encoding, and feature selection
    following the same patterns as other modules in the pipeline.
    """

    def __init__(self, config_path: str = "conf/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.report = {}

        # Initialize sklearn components
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.feature_selector: Optional[SelectKBest] = None

        # Feature tracking
        self._numerical_features: List[str] = []
        self._categorical_features: List[str] = []
        self._feature_names: List[str] = []
        self._target_column: str = ""

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
            log_file = log_config.get("file", "logs/preprocessing.log").replace(
                "main.log", "preprocessing.log"
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

    def _get_feature_columns(self, data: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Get numerical and categorical feature columns from config and data."""
        try:
            features_config = self.config.get("features", {})

            # Get columns from config
            config_numerical = features_config.get("numerical_features", [])
            config_categorical = features_config.get("categorical_features", [])
            drop_columns = features_config.get("drop_columns", [])

            # Filter to only include columns that exist in data and aren't dropped
            numerical_features = [
                col
                for col in config_numerical
                if col in data.columns and col not in drop_columns
            ]
            categorical_features = [
                col
                for col in config_categorical
                if col in data.columns and col not in drop_columns
            ]

            self.logger.info(f"Identified {len(numerical_features)} numerical features")
            self.logger.info(
                f"Identified {len(categorical_features)} categorical features"
            )

            return numerical_features, categorical_features

        except Exception as e:
            self.logger.error(f"Error identifying feature columns: {str(e)}")
            raise

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        try:
            if data.empty:
                return data

            self.logger.info("Handling missing values...")
            data_clean = data.copy()

            # Replace '?' with NaN (common in some datasets)
            for col in data_clean.select_dtypes(include=["object"]).columns:
                data_clean.loc[data_clean[col] == "?", col] = np.nan

            # Log missing value statistics
            missing_counts = data_clean.isnull().sum()
            missing_percentages = (missing_counts / len(data_clean) * 100).round(2)

            if missing_counts.sum() > 0:
                self.logger.info("Missing values per column:")
                for col, count in missing_counts[missing_counts > 0].items():
                    pct = missing_percentages[col]
                    self.logger.info(f"  {col}: {count} ({pct}%)")
            else:
                self.logger.info("No missing values found")

            # Store in report
            self.report["missing_values_before"] = missing_counts.to_dict()
            self.report["missing_percentages_before"] = missing_percentages.to_dict()

            return data_clean

        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise

    def _create_preprocessing_pipeline(
        self, numerical_features: List[str], categorical_features: List[str]
    ) -> ColumnTransformer:
        """Create and configure the preprocessing pipeline."""
        try:
            self.logger.info("Creating preprocessing pipeline...")

            transformers = []

            # Numerical features pipeline
            if numerical_features:
                self.logger.info(
                    f"Adding numerical pipeline for {len(numerical_features)} features"
                )
                num_pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                )
                transformers.append(("num", num_pipeline, numerical_features))

            # Categorical features pipeline
            if categorical_features:
                self.logger.info(
                    f"Adding categorical pipeline for {len(categorical_features)} features"
                )
                cat_pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                drop="first",  # This ensures consistent feature count
                            ),
                        ),
                    ]
                )
                transformers.append(("cat", cat_pipeline, categorical_features))

            if not transformers:
                raise ValueError(
                    "No valid features available for preprocessing pipeline"
                )

            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder="drop",  # Drop any columns not specified
            )

            self.logger.info("Preprocessing pipeline created successfully")
            return preprocessor

        except Exception as e:
            self.logger.error(f"Error creating preprocessing pipeline: {str(e)}")
            raise

    def _encode_target(self, target_series: pd.Series) -> pd.Series:
        """Encode target variable using LabelEncoder."""
        try:
            self.logger.info(f"Encoding target variable: {target_series.name}")

            # Initialize and fit label encoder
            self.label_encoder = LabelEncoder()
            encoded_values = self.label_encoder.fit_transform(target_series)

            # Create encoded series
            encoded_series = pd.Series(
                encoded_values, index=target_series.index, name=target_series.name
            )

            # Log encoding mapping
            classes = self.label_encoder.classes_
            self.logger.info(f"Target encoding mapping: {dict(enumerate(classes))}")

            # Store in report
            self.report["target_encoding"] = {
                "original_classes": classes.tolist(),
                "encoding_mapping": dict(enumerate(classes)),
            }

            return encoded_series

        except Exception as e:
            self.logger.error(f"Error encoding target variable: {str(e)}")
            raise

    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select features using statistical feature selection."""
        try:
            feature_selection_config = self.config.get("feature_selection", {})
            k = feature_selection_config.get("n_features")

            if not k or k >= X.shape[1]:
                self.logger.info("Skipping feature selection (k >= total features)")
                return X

            self.logger.info(f"Selecting top {k} features using ANOVA F-test...")

            # Check for zero variance features
            variances = np.var(X, axis=0)
            zero_var_mask = variances == 0

            if zero_var_mask.any():
                zero_var_count = zero_var_mask.sum()
                self.logger.warning(
                    f"Found {zero_var_count} zero variance features, removing them"
                )
                X = X[:, ~zero_var_mask]

                # Update feature names
                self._feature_names = [
                    name
                    for name, keep in zip(self._feature_names, ~zero_var_mask)
                    if keep
                ]

                # Adjust k if needed
                k = min(k, X.shape[1])

            if k >= X.shape[1]:
                self.logger.info(
                    "All remaining features selected (k >= remaining features)"
                )
                return X

            # Apply feature selection
            self.feature_selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)

            # Log feature selection results
            selected_mask = self.feature_selector.get_support()
            selected_features = [
                name
                for name, selected in zip(self._feature_names, selected_mask)
                if selected
            ]

            self.logger.info(
                f"Selected {len(selected_features)} features from {X.shape[1]} total"
            )
            self.logger.info(f"Selected features: {selected_features}")

            # Store in report
            self.report["feature_selection"] = {
                "method": "ANOVA F-test",
                "k": k,
                "original_features": len(self._feature_names),
                "selected_features": len(selected_features),
                "selected_feature_names": selected_features,
            }

            return X_selected

        except Exception as e:
            self.logger.error(f"Error in feature selection: {str(e)}")
            raise

    def _update_feature_names_after_fit(
        self, numerical_features: List[str], categorical_features: List[str]
    ):
        """Update feature names after fitting the pipeline."""
        try:
            feature_names = list(numerical_features)  # Start with numerical features

            # Add categorical feature names (after one-hot encoding)
            if categorical_features and "cat" in self.preprocessor.named_transformers_:
                onehot_encoder = self.preprocessor.named_transformers_[
                    "cat"
                ].named_steps["onehot"]

                for feat_idx, feature in enumerate(categorical_features):
                    categories = onehot_encoder.categories_[feat_idx]
                    feature_names.extend([f"{feature}_{cat}" for cat in categories])

            self._feature_names = feature_names
            self.logger.info(
                f"Updated feature names: {len(self._feature_names)} total features"
            )

        except Exception as e:
            self.logger.error(f"Error updating feature names: {str(e)}")
            raise

    def fit(self, data: pd.DataFrame, target_col: str = None) -> None:
        """Fit the preprocessing pipeline on the data."""
        try:
            if data.empty:
                raise ValueError("Cannot fit on empty DataFrame")

            if target_col is None:
                target_col = self.config["features"]["target_column"]

            self.logger.info(
                f"🔧 Fitting preprocessing pipeline on data shape: {data.shape}"
            )

            # Update report
            self.report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "original_shape": data.shape,
                "target_column": target_col,
                "fitting_completed": False,
            }

            # Handle missing values
            data = self._handle_missing_values(data)

            # Separate target and features
            if target_col not in data.columns:
                raise KeyError(f"Target column '{target_col}' not found in data")

            y = data[target_col]
            X = data.drop(columns=[target_col])

            # Get feature columns from actual data
            numerical_features, categorical_features = self._get_feature_columns(X)

            self.logger.info(f"   Numerical features: {len(numerical_features)}")
            self.logger.info(f"   Categorical features: {len(categorical_features)}")

            # Create and fit preprocessing pipeline
            self.preprocessor = self._create_preprocessing_pipeline(
                numerical_features, categorical_features
            )
            self.preprocessor.fit(X)

            # Fit label encoder for target
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)

            # Apply preprocessing to get data for feature selection
            X_processed = self.preprocessor.transform(X)
            y_encoded = self.label_encoder.transform(y)

            # Set feature names from the fitted transformer
            try:
                feature_names = []
                for name, transformer, columns in self.preprocessor.transformers:
                    if name == "num" and len(columns) > 0:
                        feature_names.extend([f"num__{col}" for col in columns])
                    elif name == "cat" and len(columns) > 0:
                        if hasattr(transformer, "get_feature_names_out"):
                            cat_names = transformer.get_feature_names_out(columns)
                            feature_names.extend([f"cat__{name}" for name in cat_names])
                        else:
                            feature_names.extend([f"cat__{col}" for col in columns])

                # Ensure we have the right number of feature names
                if len(feature_names) != X_processed.shape[1]:
                    feature_names = [
                        f"feature_{i}" for i in range(X_processed.shape[1])
                    ]

                self._feature_names = feature_names

            except Exception as e:
                self.logger.warning(f"Could not set feature names: {e}")
                self._feature_names = [
                    f"feature_{i}" for i in range(X_processed.shape[1])
                ]

            # Apply feature selection
            X_processed = self._select_features(X_processed, y_encoded)

            self.logger.info("Preprocessing pipeline fitted successfully")
            self.logger.info(f"   Final feature count: {X_processed.shape[1]}")

            # Update report
            self.report["fitting_completed"] = True
            self.report["final_shape"] = (X_processed.shape[0], X_processed.shape[1])
            self.report["numerical_features_count"] = len(numerical_features)
            self.report["categorical_features_count"] = len(categorical_features)

            # Write report
            self._write_report()

        except Exception as e:
            self.logger.error(f"❌ Error fitting preprocessing pipeline: {str(e)}")
            self.report["fitting_completed"] = False
            self.report["error"] = str(e)
            self._write_report()
            raise

    def transform(
        self, data: pd.DataFrame, target_col: str = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Transform data using fitted preprocessor."""
        try:
            if self.preprocessor is None:
                raise ValueError(
                    "Preprocessing pipeline not fitted yet. Call fit() first."
                )

            if data.empty:
                raise ValueError("Cannot transform empty DataFrame")

            if target_col is None:
                target_col = self.config["features"]["target_column"]

            self.logger.info(f"🔄 Transforming data with shape: {data.shape}")

            # Handle missing values
            data = self._handle_missing_values(data)

            # Separate target and features
            if target_col not in data.columns:
                raise KeyError(f"Target column '{target_col}' not found in data")

            y = data[target_col]
            X = data.drop(columns=[target_col])

            # Get feature columns for this data (not from config)
            numerical_features, categorical_features = self._get_feature_columns(X)

            # Apply preprocessing pipeline
            X_processed = self.preprocessor.transform(X)

            # Get the actual feature names from the preprocessor
            try:
                # Get feature names from the fitted ColumnTransformer
                feature_names = []
                for name, transformer, columns in self.preprocessor.transformers:
                    if name == "num" and len(columns) > 0:
                        # For numerical features, use original names
                        feature_names.extend([f"num__{col}" for col in columns])
                    elif name == "cat" and len(columns) > 0:
                        # For categorical features, get names from OneHotEncoder
                        if hasattr(transformer, "get_feature_names_out"):
                            cat_names = transformer.get_feature_names_out(columns)
                            feature_names.extend([f"cat__{name}" for name in cat_names])
                        else:
                            # Fallback for older sklearn versions
                            feature_names.extend([f"cat__{col}" for col in columns])

                # If we still don't have the right number of feature names, create generic ones
                if len(feature_names) != X_processed.shape[1]:
                    feature_names = [
                        f"feature_{i}" for i in range(X_processed.shape[1])
                    ]

            except Exception as e:
                self.logger.warning(
                    f"Could not get feature names from transformer: {e}"
                )
                # Create generic feature names
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

            # Create DataFrame with proper feature names
            X_df = pd.DataFrame(X_processed, columns=feature_names, index=data.index)

            # Apply feature selection if fitted
            if self.feature_selector is not None:
                X_df = pd.DataFrame(
                    self.feature_selector.transform(X_df),
                    columns=[
                        feature_names[i]
                        for i in self.feature_selector.get_support(indices=True)
                    ],
                    index=X_df.index,
                )

            # Encode target
            if self.label_encoder is not None:
                y_encoded = pd.Series(
                    self.label_encoder.transform(y), name=target_col, index=y.index
                )
            else:
                y_encoded = y

            self.logger.info("Data transformed successfully")
            self.logger.info(f"   Features shape: {X_df.shape}")
            self.logger.info(f"   Target shape: {y_encoded.shape}")

            return X_df, y_encoded

        except Exception as e:
            self.logger.error(f"❌ Error transforming data: {str(e)}")
            raise

    def fit_transform(
        self, data: pd.DataFrame, target_col: str = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit the pipeline and transform the data in one step."""
        self.fit(data, target_col)
        return self.transform(data, target_col)

    def get_feature_names(self) -> List[str]:
        """Get the names of the final selected features."""
        if not self._feature_names:
            raise ValueError("Pipeline not fitted yet. Feature names unavailable.")

        if self.feature_selector is None:
            return self._feature_names

        # Return only selected feature names
        selected_mask = self.feature_selector.get_support()
        return [
            name
            for name, selected in zip(self._feature_names, selected_mask)
            if selected
        ]

    def save_pipeline(self, filepath: str = "models/preprocessor.joblib") -> None:
        """Save the fitted preprocessing pipeline."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Save pipeline components
            pipeline_data = {
                "preprocessor": self.preprocessor,
                "label_encoder": self.label_encoder,
                "feature_selector": self.feature_selector,
                "feature_names": self._feature_names,
                "numerical_features": self._numerical_features,
                "categorical_features": self._categorical_features,
                "target_column": self._target_column,
                "config": self.config,
            }

            joblib.dump(pipeline_data, filepath)
            self.logger.info(f"💾 Preprocessing pipeline saved to: {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to save preprocessing pipeline: {str(e)}")
            raise

    def load_pipeline(self, filepath: str = "models/preprocessor.joblib") -> None:
        """Load a previously saved preprocessing pipeline."""
        try:
            # Load pipeline components
            pipeline_data = joblib.load(filepath)

            self.preprocessor = pipeline_data["preprocessor"]
            self.label_encoder = pipeline_data["label_encoder"]
            self.feature_selector = pipeline_data["feature_selector"]
            self._feature_names = pipeline_data["feature_names"]
            self._numerical_features = pipeline_data["numerical_features"]
            self._categorical_features = pipeline_data["categorical_features"]
            self._target_column = pipeline_data["target_column"]

            self.logger.info(f"📂 Preprocessing pipeline loaded from: {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to load preprocessing pipeline: {str(e)}")
            raise

    def _write_report(self):
        """Write preprocessing report to file."""
        try:
            report_dir = Path("logs")
            report_dir.mkdir(exist_ok=True)

            report_file = report_dir / "preprocessing_report.json"
            with open(report_file, "w") as f:
                json.dump(self.report, f, indent=2, default=str)

            self.logger.info(f"📋 Preprocessing report written to: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to write preprocessing report: {str(e)}")


if __name__ == "__main__":
    """Demonstrate Preprocessor functionality."""
    print("Preprocessor module loaded successfully.")
    print("Usage examples:")
    print("  from src.preprocessing.preprocessor import Preprocessor")
    print("  preprocessor = Preprocessor()")
    print("  X_train, y_train = preprocessor.fit_transform(train_data)")
    print("  X_test, y_test = preprocessor.transform(test_data)")

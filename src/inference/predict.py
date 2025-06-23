"""
Model Inference Module for MLOps Project.

Handles model loading and predictions following the same patterns
as other modules in the pipeline.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import json
import joblib
from pathlib import Path
from typing import List, Dict, Any


class ModelPredictor:
    """
    Model inference pipeline for the MLOps project.

    Handles model loading, data preprocessing, and predictions following the same
    patterns as other modules in the pipeline.
    """

    def __init__(self, config_path: str = "src/config.yaml", model_path: str = None):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.report = {}

        # Model components
        self.model = None
        self.feature_engineer = None
        self.model_metadata: Dict[str, Any] = {}

        # Prediction history
        self.prediction_history: List[Dict[str, Any]] = []

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

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
            log_file = log_config.get("file", "logs/inference.log").replace(
                "main.log", "inference.log"
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

    def load_model(self, model_path: str) -> None:
        """Load the trained model and associated components."""
        try:
            self.logger.info(f"🔄 Loading model from: {model_path}")

            # Validate model file exists
            model_file = Path(model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load model data
            model_data = joblib.load(model_path)

            # Extract components based on expected structure
            if isinstance(model_data, dict):
                # Expected structure from ModelTrainer.save()
                self.model = model_data.get("model")
                self.feature_engineer = model_data.get("feature_engineer")
                self.model_metadata = model_data.get("metadata", {})

                if self.model is None:
                    raise ValueError("No 'model' found in the saved model file")

            else:
                # Fallback: assume model_data is just the model
                self.model = model_data
                self.feature_engineer = None
                self.model_metadata = {}
                self.logger.warning(
                    "Model file doesn't contain feature engineer. Manual preprocessing may be required."
                )

            # Validate model has required methods
            if not hasattr(self.model, "predict"):
                raise ValueError("Loaded model doesn't have 'predict' method")

            # Log model information
            model_type = type(self.model).__name__
            self.logger.info("✅ Model loaded successfully:")
            self.logger.info(f"  Model type: {model_type}")
            self.logger.info(
                f"  Feature engineer: {'Available' if self.feature_engineer else 'Not available'}"
            )
            self.logger.info(f"  Metadata keys: {list(self.model_metadata.keys())}")

            # Update report
            self.report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_path": str(model_path),
                "model_type": model_type,
                "model_loaded": True,
                "feature_engineer_available": self.feature_engineer is not None,
                "metadata": self.model_metadata,
            }

        except Exception as e:
            self.logger.error(f"❌ Error loading model: {str(e)}")
            self.report = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_path": str(model_path),
                "model_loaded": False,
                "error": str(e),
            }
            raise

    def _validate_input_data(self, data: pd.DataFrame) -> None:
        """Validate input data for predictions."""
        try:
            if data.empty:
                raise ValueError("Cannot make predictions on empty DataFrame")

            # Check for required columns if metadata available
            if "required_features" in self.model_metadata:
                required_features = self.model_metadata["required_features"]
                missing_features = [
                    f for f in required_features if f not in data.columns
                ]
                if missing_features:
                    raise ValueError(f"Missing required features: {missing_features}")

            # Check for infinite or NaN values in numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if data[col].isnull().any():
                    self.logger.warning(
                        f"Found NaN values in column '{col}'. These may cause prediction errors."
                    )
                if np.isinf(data[col]).any():
                    raise ValueError(
                        f"Found infinite values in column '{col}'. Please clean the data."
                    )

            self.logger.info(f"Input data validation passed. Shape: {data.shape}")

        except Exception as e:
            self.logger.error(f"Input data validation failed: {str(e)}")
            raise

    def _preprocess_data(
        self, data: pd.DataFrame, target_col: str = "readmitted"
    ) -> pd.DataFrame:
        """Preprocess data for prediction."""
        try:
            if self.feature_engineer is None:
                self.logger.warning(
                    "No feature engineer available. Using raw data for prediction."
                )
                # Remove target column if present
                if target_col in data.columns:
                    data = data.drop(columns=[target_col])
                return data

            self.logger.info("🔧 Preprocessing data with feature engineer...")

            # If target column doesn't exist, create dummy target for processing
            if target_col not in data.columns:
                data_with_target = data.copy()
                data_with_target[target_col] = 0  # Dummy target
            else:
                data_with_target = data.copy()

            # Use feature engineer's transform method
            X_processed, y_dummy = self.feature_engineer.transform(
                data_with_target, target_col=target_col
            )

            self.logger.info(
                f"✅ Data preprocessed successfully. Shape: {X_processed.shape}"
            )
            return X_processed

        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame, preprocess: bool = True) -> np.ndarray:
        """Make predictions on input data."""
        try:
            if self.model is None:
                raise ValueError("No model loaded. Use load_model() first.")

            self.logger.info(f"🔮 Making predictions on {len(data)} samples...")

            # Validate input data
            self._validate_input_data(data)

            # Preprocess data if needed
            if preprocess:
                X_processed = self._preprocess_data(data)
            else:
                X_processed = data.copy()
                # Remove target column if present
                if "readmitted" in X_processed.columns:
                    X_processed = X_processed.drop(columns=["readmitted"])

            # Convert to numpy array to avoid feature name warnings
            X_array = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed

            # Make predictions
            predictions = self.model.predict(X_array)

            self.logger.info("✅ Predictions completed successfully")
            self.logger.info(
                f"Prediction distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}"
            )

            # Store prediction info
            prediction_info = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "n_samples": len(data),
                "prediction_distribution": dict(
                    zip(*np.unique(predictions, return_counts=True))
                ),
                "preprocessed": preprocess,
            }
            self.prediction_history.append(prediction_info)

            return predictions

        except Exception as e:
            self.logger.error(f"❌ Error making predictions: {str(e)}")
            raise

    def predict_proba(self, data: pd.DataFrame, preprocess: bool = True) -> np.ndarray:
        """Get prediction probabilities for input data."""
        try:
            if self.model is None:
                raise ValueError("No model loaded. Use load_model() first.")

            if not hasattr(self.model, "predict_proba"):
                raise ValueError(
                    f"Model {type(self.model).__name__} does not support probability predictions"
                )

            self.logger.info(
                f"🔮 Calculating prediction probabilities on {len(data)} samples..."
            )

            # Validate input data
            self._validate_input_data(data)

            # Preprocess data if needed
            if preprocess:
                X_processed = self._preprocess_data(data)
            else:
                X_processed = data.copy()
                # Remove target column if present
                if "readmitted" in X_processed.columns:
                    X_processed = X_processed.drop(columns=["readmitted"])

            # Convert to numpy array to avoid feature name warnings
            X_array = X_processed.values if isinstance(X_processed, pd.DataFrame) else X_processed

            # Get probabilities
            probabilities = self.model.predict_proba(X_array)

            self.logger.info("✅ Prediction probabilities calculated successfully")

            return probabilities

        except Exception as e:
            self.logger.error(
                f"❌ Error calculating prediction probabilities: {str(e)}"
            )
            raise

    def predict_with_confidence(
        self,
        data: pd.DataFrame,
        confidence_threshold: float = 0.8,
        preprocess: bool = True,
    ) -> Dict[str, Any]:
        """Make predictions with confidence scores and filtering."""
        try:
            self.logger.info("🔮 Making predictions with confidence analysis...")

            # Get predictions and probabilities
            predictions = self.predict(data, preprocess=preprocess)

            try:
                probabilities = self.predict_proba(data, preprocess=preprocess)

                # Calculate confidence scores (max probability for each prediction)
                confidence_scores = np.max(probabilities, axis=1)

                # Create confidence masks
                high_confidence_mask = confidence_scores >= confidence_threshold
                low_confidence_mask = confidence_scores < confidence_threshold

                results = {
                    "predictions": predictions,
                    "probabilities": probabilities,
                    "confidence_scores": confidence_scores,
                    "high_confidence_predictions": predictions[high_confidence_mask],
                    "low_confidence_predictions": predictions[low_confidence_mask],
                    "high_confidence_indices": np.where(high_confidence_mask)[0],
                    "low_confidence_indices": np.where(low_confidence_mask)[0],
                    "confidence_threshold": confidence_threshold,
                    "high_confidence_count": int(np.sum(high_confidence_mask)),
                    "low_confidence_count": int(np.sum(low_confidence_mask)),
                    "confidence_stats": {
                        "mean_confidence": float(np.mean(confidence_scores)),
                        "min_confidence": float(np.min(confidence_scores)),
                        "max_confidence": float(np.max(confidence_scores)),
                        "std_confidence": float(np.std(confidence_scores)),
                    },
                }

            except Exception as e:
                # Fallback if probabilities not available
                self.logger.warning(
                    f"Could not calculate probabilities: {e}. Returning predictions only."
                )
                results = {
                    "predictions": predictions,
                    "probabilities": None,
                    "confidence_scores": None,
                    "high_confidence_predictions": predictions,  # Assume all are high confidence
                    "low_confidence_predictions": np.array([]),
                    "high_confidence_indices": np.arange(len(predictions)),
                    "low_confidence_indices": np.array([]),
                    "confidence_threshold": confidence_threshold,
                    "high_confidence_count": len(predictions),
                    "low_confidence_count": 0,
                    "confidence_stats": None,
                }

            self.logger.info("✅ Confidence analysis completed:")
            self.logger.info(
                f"  High confidence predictions: {results['high_confidence_count']}/{len(predictions)}"
            )
            self.logger.info(
                f"  Low confidence predictions: {results['low_confidence_count']}/{len(predictions)}"
            )

            return results

        except Exception as e:
            self.logger.error(f"❌ Error in confidence prediction: {str(e)}")
            raise

    def predict_batch(
        self, data: pd.DataFrame, batch_size: int = 1000, save_path: str = None
    ) -> Dict[str, Any]:
        """Process large datasets in batches."""
        try:
            self.logger.info(
                f"🔄 Processing {len(data)} samples in batches of {batch_size}..."
            )

            total_samples = len(data)
            n_batches = (total_samples + batch_size - 1) // batch_size

            all_predictions = []
            all_probabilities = []
            batch_results = []

            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_samples)

                self.logger.info(
                    f"Processing batch {i+1}/{n_batches} (samples {start_idx}-{end_idx})"
                )

                batch_data = data.iloc[start_idx:end_idx]

                # Make predictions for this batch
                batch_predictions = self.predict(batch_data, preprocess=True)
                all_predictions.extend(batch_predictions)

                try:
                    batch_probabilities = self.predict_proba(
                        batch_data, preprocess=True
                    )
                    all_probabilities.extend(batch_probabilities)
                except Exception:
                    # If probabilities not available, continue without them
                    pass

                batch_results.append(
                    {
                        "batch_number": i + 1,
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "n_samples": end_idx - start_idx,
                        "predictions": batch_predictions,
                    }
                )

            # Combine results
            final_predictions = np.array(all_predictions)
            final_probabilities = (
                np.array(all_probabilities) if all_probabilities else None
            )

            results = {
                "predictions": final_predictions,
                "probabilities": final_probabilities,
                "total_samples": total_samples,
                "n_batches": n_batches,
                "batch_size": batch_size,
                "batch_results": batch_results,
                "prediction_distribution": dict(
                    zip(*np.unique(final_predictions, return_counts=True))
                ),
            }

            # Save results if path provided
            if save_path:
                self.save_predictions(results, save_path)

            self.logger.info("✅ Batch processing completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"❌ Error in batch prediction: {str(e)}")
            raise

    def save_predictions(self, results: Dict[str, Any], save_path: str) -> str:
        """Save prediction results to file."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Helper function to convert numpy types
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization."""
                if isinstance(obj, dict):
                    return {
                        str(key): convert_numpy_types(value)
                        for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, "item"):  # Handle numpy scalars
                    return obj.item()
                else:
                    return obj

            # Prepare data for saving
            save_data = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "model_metadata": convert_numpy_types(self.model_metadata),
                "predictions": results["predictions"].tolist(),
                "prediction_distribution": convert_numpy_types(
                    results.get("prediction_distribution", {})
                ),
                "n_samples": int(len(results["predictions"])),
            }

            # Add probabilities if available
            if results.get("probabilities") is not None:
                save_data["probabilities"] = results["probabilities"].tolist()

            # Add confidence data if available
            if (
                "confidence_scores" in results
                and results["confidence_scores"] is not None
            ):
                save_data["confidence_scores"] = results["confidence_scores"].tolist()
                save_data["confidence_stats"] = convert_numpy_types(
                    results.get("confidence_stats", {})
                )

            # Save to JSON
            with open(save_path, "w") as f:
                json.dump(save_data, f, indent=2, default=str)

            self.logger.info(f"✅ Predictions saved to: {save_path}")
            return str(save_path)

        except Exception as e:
            self.logger.error(f"Error saving predictions: {str(e)}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            return {"model_loaded": False}

        return {
            "model_loaded": True,
            "model_type": type(self.model).__name__,
            "feature_engineer_available": self.feature_engineer is not None,
            "supports_probabilities": hasattr(self.model, "predict_proba"),
            "metadata": self.model_metadata,
            "prediction_history_count": len(self.prediction_history),
        }

    def get_prediction_history(self) -> List[Dict[str, Any]]:
        """Get history of predictions made."""
        return self.prediction_history.copy()

    def clear_prediction_history(self) -> None:
        """Clear prediction history."""
        self.prediction_history.clear()
        self.logger.info("Prediction history cleared")

    def _write_report(self):
        """Write inference report to file."""
        try:
            report_dir = Path("logs")
            report_dir.mkdir(exist_ok=True)

            report_file = report_dir / "inference_report.json"

            # Helper function to convert numpy types
            def convert_numpy_types(obj):
                """Convert numpy types to native Python types for JSON serialization."""
                if isinstance(obj, dict):
                    return {
                        str(key): convert_numpy_types(value)
                        for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif hasattr(obj, "item"):  # Handle numpy scalars
                    return obj.item()
                else:
                    return obj

            # Add prediction history to report
            self.report["prediction_history"] = convert_numpy_types(
                self.prediction_history
            )
            self.report["model_info"] = convert_numpy_types(self.get_model_info())

            # Convert the entire report
            converted_report = convert_numpy_types(self.report)

            with open(report_file, "w") as f:
                json.dump(converted_report, f, indent=2, default=str)

            self.logger.info(f"📋 Inference report written to: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to write inference report: {str(e)}")


if __name__ == "__main__":
    """Demonstrate ModelPredictor functionality."""
    print("Model inference module loaded successfully.")
    print("Usage examples:")
    print("  from src.inference.predict import ModelPredictor")
    print("  predictor = ModelPredictor()")
    print("  predictor.load_model('path/to/model.joblib')")
    print("  predictions = predictor.predict(data)")
    print("  results = predictor.predict_with_confidence(data)")

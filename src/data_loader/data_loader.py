import os
import pandas as pd
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, config_path: str = "conf/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate required config sections exist
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
            log_file = log_config.get("file", "logs/data_loader.log").replace(
                "main.log", "data_loader.log"
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

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load data from file path specified in config or parameter."""
        try:
            data_path = file_path or self.config["data"]["raw_data_path"]
            data_format = self.config["data"].get("raw_data_format", "csv")

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            if data_format.lower() == "csv":
                df = pd.read_csv(data_path)
            else:
                raise ValueError(f"Unsupported data format: {data_format}")

            self.logger.info(f"Data loaded successfully from {data_path}")
            self.logger.info(f"Data shape: {df.shape}")
            self.logger.info(f"Columns: {list(df.columns)}")

            return df

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into train, validation, and test sets."""
        try:
            # Get split parameters from config
            test_size = self.config["model"].get("test_size", 0.2)
            validation_size = self.config["model"].get("validation_size", 0.2)
            random_state = self.config["model"].get("random_state", 42)

            # Calculate validation size relative to remaining data after test split
            val_size_adjusted = validation_size / (1 - test_size)

            # First split: train+val vs test
            train_val, test = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=None
            )

            # Second split: train vs val
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=None,
            )

            self.logger.info(f"Data split completed:")
            self.logger.info(f"  Train: {train.shape}")
            self.logger.info(f"  Validation: {val.shape}")
            self.logger.info(f"  Test: {test.shape}")

            return train, val, test

        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

    def save_split_data(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Save split datasets to files."""
        try:
            # Get output paths from config
            train_path = self.config["data"]["train_data_path"]
            val_path = self.config["data"]["validation_data_path"]
            test_path = self.config["data"]["test_data_path"]

            # Create output directory if it doesn't exist
            for path in [train_path, val_path, test_path]:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            # Save datasets
            train.to_csv(train_path, index=False)
            val.to_csv(val_path, index=False)
            test.to_csv(test_path, index=False)

            self.logger.info(f"Split data saved:")
            self.logger.info(f"  Train: {train_path}")
            self.logger.info(f"  Validation: {val_path}")
            self.logger.info(f"  Test: {test_path}")

        except Exception as e:
            self.logger.error(f"Error saving split data: {str(e)}")
            raise


if __name__ == "__main__":
    """Simple demonstration of DataLoader usage."""
    print("DataLoader module loaded successfully.")
    print("Usage: from src.data_loader.data_loader import DataLoader")

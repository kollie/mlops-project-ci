import os
import pandas as pd
import yaml
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(self, config_path: str = "src/config.yaml"):
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
            log_file = log_config.get("file", "logs/data_loader.log")
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
            data_path = file_path or self.config["data"].get("raw_data_path")
            data_format = self.config["data"].get("raw_data_format", "csv")

            if not data_path:
                raise ValueError(
                    "Missing 'raw_data_path' in config and no file_path provided."
                )

            data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"File not found: {data_path}")

            self.logger.info(f"Loading data from {data_path} as format: {data_format}")

            if data_format == "csv":
                data = pd.read_csv(data_path)
            elif data_format == "excel":
                data = pd.read_excel(data_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {data_format}. Supported formats: csv, excel"
                )

            self.logger.info(f"Loaded data with shape: {data.shape}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def split_data(self, data: pd.DataFrame) -> tuple:
        """Split data into train, validation, and test sets."""
        try:
            # Validate required model config parameters
            required_params = ["test_size", "validation_size", "random_state"]
            for param in required_params:
                if param not in self.config["model"]:
                    raise ValueError(f"Missing model parameter: {param}")

            test_size = self.config["model"]["test_size"]
            validation_size = self.config["model"]["validation_size"]
            random_state = self.config["model"]["random_state"]

            # First split: separate test set
            train_val, test = train_test_split(
                data, test_size=test_size, random_state=random_state
            )

            # Second split: separate train and validation
            val_size = validation_size / (1 - test_size)
            train, val = train_test_split(
                train_val, test_size=val_size, random_state=random_state
            )

            self.logger.info(
                f"Data split complete. Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}"
            )
            return train, val, test

        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

    def save_split_data(
        self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
    ):
        """Save split datasets to configured paths."""
        try:
            # Validate required data paths exist in config
            required_paths = [
                "processed_data_path",
                "train_data_path",
                "validation_data_path",
                "test_data_path",
            ]
            for path_key in required_paths:
                if path_key not in self.config["data"]:
                    raise ValueError(f"Missing data path in config: {path_key}")

            # Create directories
            Path(self.config["data"]["processed_data_path"]).mkdir(
                parents=True, exist_ok=True
            )

            # Save datasets
            train.to_csv(self.config["data"]["train_data_path"], index=False)
            val.to_csv(self.config["data"]["validation_data_path"], index=False)
            test.to_csv(self.config["data"]["test_data_path"], index=False)

            self.logger.info("Split datasets saved successfully")
            self.logger.info(
                f"Train data saved to: {self.config['data']['train_data_path']}"
            )
            self.logger.info(
                f"Validation data saved to: {self.config['data']['validation_data_path']}"
            )
            self.logger.info(
                f"Test data saved to: {self.config['data']['test_data_path']}"
            )

        except Exception as e:
            self.logger.error(f"Error saving split data: {str(e)}")
            raise


if __name__ == "__main__":
    """Simple demonstration of DataLoader usage."""
    print("DataLoader module loaded successfully.")
    print("Usage: from src.data_loader.data_loader import DataLoader")

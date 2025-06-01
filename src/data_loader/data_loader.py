import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from src.validation.data_validator import DataValidator


class DataLoader:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.validator = DataValidator(config_path)

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        log_file_path = self.config['logging']['file']
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            filename=self.config['logging']['file']
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self, file_path: str = None) -> pd.DataFrame:
        try:
            data_path = file_path or self.config['data'].get('raw_data_path')
            data_format = self.config['data'].get('raw_data_format', 'csv')

            if not data_path:
                raise ValueError("Missing 'raw_data_path' in config.")
            if not Path(data_path).exists():
                raise FileNotFoundError(f"File not found: {data_path}")

            self.logger.info(f"Loading data from {data_path} as format: {data_format}")

            if data_format == 'csv':
                data = pd.read_csv(data_path)
            elif data_format == 'excel':
                data = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_format}")

            self.logger.info(f"Loaded file shape: {data.shape}")
            return data

        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def split_data(self, data: pd.DataFrame) -> tuple:
        try:
            train_val, test = train_test_split(
                data,
                test_size=self.config['model']['test_size'],
                random_state=self.config['model']['random_state']
            )
            val_size = self.config['model']['validation_size'] / (1 - self.config['model']['test_size'])
            train, val = train_test_split(
                train_val,
                test_size=val_size,
                random_state=self.config['model']['random_state']
            )
            self.logger.info(f"Data split complete. Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
            return train, val, test
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise

    def save_split_data(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        try:
            Path(self.config['data']['processed_data_path']).mkdir(parents=True, exist_ok=True)
            Path(self.config['data']['test_data_path']).mkdir(parents=True, exist_ok=True)

            train.to_csv(self.config['data']['train_data_path'], index=False)
            val.to_csv(self.config['data']['validation_data_path'], index=False)
            test.to_csv(f"{self.config['data']['test_data_path']}/test.csv", index=False)

            self.logger.info("Split datasets saved successfully")
        except Exception as e:
            self.logger.error(f"Error saving split data: {str(e)}")
            raise


if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_data()
    validator = DataValidator()
    data_clean = validator.validate_and_clean(data)

    train, val, test = loader.split_data(data_clean)
    loader.save_split_data(train, val, test)

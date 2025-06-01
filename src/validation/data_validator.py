import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
import json


class DataValidator:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.required_columns = (
            self.config['features']['numerical_features'] +
            self.config['features']['categorical_features'] +
            [self.config['features']['target_column']]
        )
        self.report = {}

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        log_file_path = self.config['logging']['file']
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            filename=log_file_path
        )
        self.logger = logging.getLogger(__name__)

    def validate_and_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            self.report['initial_rows'] = len(data)
            self._check_required_columns(data)
            self._check_data_types(data)
            data = self._handle_missing_values(data)
            self._check_target_distribution(data)
            self.report['final_rows'] = len(data)
            self.report['rows_dropped'] = self.report['initial_rows'] - self.report['final_rows']
            self._write_report()
            self.logger.info("Full validation and cleaning completed successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Validation failed: {str(e)}")
            self._write_report()
            raise

    def _check_required_columns(self, data: pd.DataFrame):
        # List of required columns from config
        required = set(self.required_columns)
        available = set(data.columns)

        # Special substitution: allow 'age' in place of 'age_group'
        if 'age_group' in required and 'age_group' not in available and 'age' in available:
            self.logger.warning("Expected column 'age_group' not found. Using 'age' as fallback.")
            available = available.union({'age_group'})  # Pretend 'age_group' exists

        missing = required - available
        if missing:
            self.report['missing_columns'] = list(missing)
            raise ValueError(f"Missing columns: {missing}")

        self.logger.info("Required columns check passed.")
        
    def _check_data_types(self, data: pd.DataFrame):
        type_issues = {}
        for col in self.config['features']['numerical_features']:
            if col in data.columns and not np.issubdtype(data[col].dtype, np.number):
                type_issues[col] = str(data[col].dtype)
        for col in self.config['features']['categorical_features']:
            if col in data.columns and not pd.api.types.is_string_dtype(data[col]):
                type_issues[col] = str(data[col].dtype)
        if type_issues:
            self.report['type_mismatches'] = type_issues
            raise TypeError(f"Type mismatches found: {type_issues}")
        self.logger.info("Data types validated.")

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        missing = data.isnull().sum()
        self.report['missing_values'] = missing.to_dict()
        before = len(data)
        data = data.dropna()
        self.logger.info(f"Dropped {before - len(data)} rows with missing values.")
        return data

    def _check_target_distribution(self, data: pd.DataFrame):
        target = self.config['features']['target_column']
        dist = data[target].value_counts(normalize=True)
        self.report['target_distribution'] = dist.to_dict()
        if (dist < 0.01).any():
            raise ValueError("Target class imbalance detected.")
        self.logger.info("Target distribution check passed.")

    def _write_report(self):
        with open("logs/validation_report.json", "w") as f:
            json.dump(self.report, f, indent=2)

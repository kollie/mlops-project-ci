import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from typing import Dict


class LightDataValidator:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = self._load_config(config_path)
        self._setup_logging()

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

    def validate_schema(self, data: pd.DataFrame) -> bool:
        try:
            required_columns = (
                self.config['features']['numerical_features'] +
                self.config['features']['categorical_features'] +
                [self.config['features']['target_column']]
            )
            missing = set(required_columns) - set(data.columns)
            if missing:
                self.logger.error(f"Missing required columns: {missing}")
                return False
            self.logger.info("Schema validation passed.")
            return True
        except Exception as e:
            self.logger.error(f"Schema validation error: {str(e)}")
            return False

    def validate_data_types(self, data: pd.DataFrame) -> bool:
        try:
            for col in self.config['features']['numerical_features']:
                if col in data.columns and not np.issubdtype(data[col].dtype, np.number):
                    self.logger.error(f"Column {col} should be numeric but is {data[col].dtype}")
                    return False
            self.logger.info("Data type validation passed.")
            return True
        except Exception as e:
            self.logger.error(f"Data type validation error: {str(e)}")
            return False

    def validate_missing_values(self, data: pd.DataFrame, threshold: float = 0.5) -> bool:
        try:
            missing_ratio = data.isnull().mean()
            high_missing = missing_ratio[missing_ratio > threshold]
            if not high_missing.empty:
                self.logger.error(f"Columns with >{threshold*100}% missing: {list(high_missing.index)}")
                return False
            self.logger.info("Missing value ratio is acceptable.")
            return True
        except Exception as e:
            self.logger.error(f"Missing value check error: {str(e)}")
            return False

    def validate_target_distribution(self, data: pd.DataFrame) -> bool:
        try:
            target = self.config['features']['target_column']
            dist = data[target].value_counts(normalize=True)
            if (dist < 0.01).any():
                self.logger.error(f"Target class imbalance detected: {dist.to_dict()}")
                return False
            self.logger.info("Target distribution validation passed.")
            return True
        except Exception as e:
            self.logger.error(f"Target distribution validation error: {str(e)}")
            return False

    def validate_all(self, data: pd.DataFrame) -> Dict[str, bool]:
        results = {
            "schema": self.validate_schema(data),
            "data_types": self.validate_data_types(data),
            "missing_values": self.validate_missing_values(data),
            "target_distribution": self.validate_target_distribution(data)
        }
        all_passed = all(results.values())
        self.logger.info(f"Light validation complete. All passed: {all_passed}")
        return results


if __name__ == "__main__":
    from data_loader.data_loader import DataLoader

    loader = DataLoader()
    data = loader.load_data()

    validator = LightDataValidator()
    results = validator.validate_all(data)
    print("Validation Results:", results)

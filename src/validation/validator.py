import pandas as pd
import numpy as np
from typing import List, Dict, Any
import yaml
import logging
from pathlib import Path

class DataValidator:
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Initialize the DataValidator with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            filename=self.config['logging']['file']
        )
        self.logger = logging.getLogger(__name__)
    
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data contains all required columns.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            required_columns = (
                self.config['features']['categorical_columns'] +
                self.config['features']['numerical_columns'] +
                [self.config['features']['target_column']]
            )
            
            missing_columns = set(required_columns) - set(data.columns)
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            self.logger.info("Schema validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating schema: {str(e)}")
            return False
    
    def validate_data_types(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data types of columns match expected types.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            # Check numerical columns
            for col in self.config['features']['numerical_columns']:
                if not np.issubdtype(data[col].dtype, np.number):
                    self.logger.error(f"Column {col} should be numerical but is {data[col].dtype}")
                    return False
            
            self.logger.info("Data type validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data types: {str(e)}")
            return False
    
    def validate_missing_values(self, data: pd.DataFrame, threshold: float = 0.5) -> bool:
        """
        Validate that the proportion of missing values in each column is below threshold.
        
        Args:
            data (pd.DataFrame): Input dataset
            threshold (float): Maximum allowed proportion of missing values
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            missing_proportions = data.isnull().mean()
            columns_with_high_missing = missing_proportions[missing_proportions > threshold].index
            
            if len(columns_with_high_missing) > 0:
                self.logger.error(f"Columns with high missing values: {columns_with_high_missing}")
                return False
            
            self.logger.info("Missing values validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating missing values: {str(e)}")
            return False
    
    def validate_target_distribution(self, data: pd.DataFrame) -> bool:
        """
        Validate that the target variable has a reasonable distribution.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        try:
            target_col = self.config['features']['target_column']
            target_distribution = data[target_col].value_counts(normalize=True)
            
            # Check if any class has less than 1% of the data
            if (target_distribution < 0.01).any():
                self.logger.error(f"Target distribution has classes with less than 1% of data: {target_distribution}")
                return False
            
            self.logger.info("Target distribution validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating target distribution: {str(e)}")
            return False
    
    def validate_all(self, data: pd.DataFrame) -> Dict[str, bool]:
        """
        Run all validation checks on the data.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            Dict[str, bool]: Dictionary of validation check names and their results
        """
        validation_results = {
            'schema': self.validate_schema(data),
            'data_types': self.validate_data_types(data),
            'missing_values': self.validate_missing_values(data),
            'target_distribution': self.validate_target_distribution(data)
        }
        
        all_passed = all(validation_results.values())
        self.logger.info(f"All validations passed: {all_passed}")
        
        return validation_results

if __name__ == "__main__":
    # Example usage
    from data_loader.data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    data = loader.load_data()
    
    # Validate data
    validator = DataValidator()
    validation_results = validator.validate_all(data)
    
    print("Validation results:", validation_results)

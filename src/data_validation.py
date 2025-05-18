import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from .utils import setup_logging, load_config

logger = setup_logging()

class DataValidator:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = load_config(config_path)
        self.feature_config = self.config['features']
    
    def validate_schema(self, data: pd.DataFrame) -> bool:
        """Validate data schema against expected columns."""
        try:
            expected_columns = (
                self.feature_config['categorical_columns'] +
                self.feature_config['numerical_columns'] +
                [self.feature_config['target_column']]
            )
            
            missing_columns = set(expected_columns) - set(data.columns)
            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")
                return False
            
            extra_columns = set(data.columns) - set(expected_columns)
            if extra_columns:
                logger.warning(f"Extra columns found: {extra_columns}")
            
            return True
        except Exception as e:
            logger.error(f"Error validating schema: {str(e)}")
            return False
    
    def check_types(self, data: pd.DataFrame) -> bool:
        """Check data types of columns."""
        try:
            for col in self.feature_config['categorical_columns']:
                if col in data.columns and not pd.api.types.is_object_dtype(data[col]):
                    logger.warning(f"Column {col} should be categorical but is {data[col].dtype}")
            
            for col in self.feature_config['numerical_columns']:
                if col in data.columns and not pd.api.types.is_numeric_dtype(data[col]):
                    logger.warning(f"Column {col} should be numerical but is {data[col].dtype}")
            
            return True
        except Exception as e:
            logger.error(f"Error checking types: {str(e)}")
            return False
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration."""
        try:
            method = self.config['preprocessing']['handle_missing']
            
            for col in data.columns:
                if data[col].isnull().any():
                    if col in self.feature_config['categorical_columns']:
                        data[col] = data[col].fillna(data[col].mode()[0])
                    elif col in self.feature_config['numerical_columns']:
                        if method == 'mean':
                            data[col] = data[col].fillna(data[col].mean())
                        elif method == 'median':
                            data[col] = data[col].fillna(data[col].median())
                        elif method == 'zero':
                            data[col] = data[col].fillna(0)
            
            return data
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Run all validation checks."""
        schema_valid = self.validate_schema(data)
        types_valid = self.check_types(data)
        
        if not schema_valid or not types_valid:
            return False
        
        return True 
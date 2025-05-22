"""
Data validation module for the MLOps project.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

class DataValidator:
    def __init__(self):
        self.required_columns = [
            'encounter_id', 'patient_nbr', 'race', 'gender', 'age',
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'readmitted'
        ]
        self.logger = logging.getLogger(__name__)

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the input data."""
        try:
            # Check required columns
            missing_cols = set(self.required_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Check for missing values
            if data.isnull().any().any():
                raise ValueError("Data contains missing values")

            # Check data types
            if not data['encounter_id'].dtype in [np.int64, np.int32]:
                raise ValueError("encounter_id must be numeric")
            if not data['patient_nbr'].dtype in [np.int64, np.int32]:
                raise ValueError("patient_nbr must be numeric")
            if not data['age'].dtype in [np.int64, np.int32]:
                raise ValueError("age must be numeric")

            # Check target variable
            if 'readmitted' not in data.columns:
                raise ValueError("Target variable 'readmitted' not found")
            if not set(data['readmitted'].unique()).issubset({'NO', 'YES'}):
                raise ValueError("Target variable must contain only 'NO' or 'YES'")

            self.logger.info("Data validation completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise

    def validate_schema(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data schema and return statistics."""
        try:
            schema = {
                'num_rows': len(data),
                'num_columns': len(data.columns),
                'column_types': data.dtypes.to_dict(),
                'missing_values': data.isnull().sum().to_dict(),
                'unique_values': {col: data[col].nunique() for col in data.columns}
            }
            return schema
        except Exception as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            raise 
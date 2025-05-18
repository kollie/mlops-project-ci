import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from typing import Tuple, Dict, Any
from .utils import setup_logging, load_config

logger = setup_logging()

class DataPreprocessor:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = load_config(config_path)
        self.feature_config = self.config['features']
        self.preprocessing_config = self.config['preprocessing']
        
        # Initialize transformers
        self.scaler = None
        self.encoder = None
    
    def _initialize_transformers(self):
        """Initialize scaling and encoding transformers."""
        scaling_method = self.preprocessing_config['scaling_method']
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        
        encoding_method = self.preprocessing_config['encoding_method']
        if encoding_method == 'one_hot':
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data by removing duplicates and handling outliers."""
        try:
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Handle outliers in numerical columns
            for col in self.feature_config['numerical_columns']:
                if col in data.columns:
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    data[col] = data[col].clip(lower_bound, upper_bound)
            
            return data
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise
    
    def transform_data(self, data: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Transform the data using scaling and encoding."""
        try:
            if is_training:
                self._initialize_transformers()
            
            # Scale numerical features
            numerical_data = data[self.feature_config['numerical_columns']].copy()
            if self.scaler is not None:
                if is_training:
                    numerical_data = pd.DataFrame(
                        self.scaler.fit_transform(numerical_data),
                        columns=numerical_data.columns
                    )
                else:
                    numerical_data = pd.DataFrame(
                        self.scaler.transform(numerical_data),
                        columns=numerical_data.columns
                    )
            
            # Encode categorical features
            categorical_data = data[self.feature_config['categorical_columns']].copy()
            if self.encoder is not None:
                if is_training:
                    encoded_data = self.encoder.fit_transform(categorical_data)
                    encoded_columns = self.encoder.get_feature_names_out(
                        self.feature_config['categorical_columns']
                    )
                else:
                    encoded_data = self.encoder.transform(categorical_data)
                    encoded_columns = self.encoder.get_feature_names_out(
                        self.feature_config['categorical_columns']
                    )
                
                categorical_data = pd.DataFrame(
                    encoded_data,
                    columns=encoded_columns
                )
            
            # Combine transformed features
            transformed_data = pd.concat([numerical_data, categorical_data], axis=1)
            
            # Store transformers if training
            transformers = {}
            if is_training:
                transformers['scaler'] = self.scaler
                transformers['encoder'] = self.encoder
            
            return transformed_data, transformers
        except Exception as e:
            logger.error(f"Error transforming data: {str(e)}")
            raise 
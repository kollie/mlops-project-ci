import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml
import logging
from pathlib import Path

class Preprocessor:
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Initialize the Preprocessor with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.preprocessor = None
        
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
    
    def _create_preprocessing_pipeline(self):
        """
        Create the preprocessing pipeline based on configuration.
        
        Returns:
            sklearn.pipeline.Pipeline: Preprocessing pipeline
        """
        try:
            # Get column lists from config
            categorical_cols = self.config['features']['categorical_columns']
            numerical_cols = self.config['features']['numerical_columns']
            
            # Create preprocessing steps
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            # Combine preprocessing steps
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ])
            
            self.logger.info("Preprocessing pipeline created successfully")
            return preprocessor
            
        except Exception as e:
            self.logger.error(f"Error creating preprocessing pipeline: {str(e)}")
            raise
    
    def fit(self, data: pd.DataFrame):
        """
        Fit the preprocessing pipeline on the data.
        
        Args:
            data (pd.DataFrame): Input dataset
        """
        try:
            # Drop specified columns
            data = data.drop(columns=self.config['features']['drop_columns'], errors='ignore')
            
            # Create and fit the preprocessing pipeline
            self.preprocessor = self._create_preprocessing_pipeline()
            self.preprocessor.fit(data)
            
            self.logger.info("Preprocessing pipeline fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting preprocessing pipeline: {str(e)}")
            raise
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using the fitted preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            np.ndarray: Transformed data
        """
        try:
            if self.preprocessor is None:
                raise ValueError("Preprocessor must be fitted before transforming data")
            
            # Drop specified columns
            data = data.drop(columns=self.config['features']['drop_columns'], errors='ignore')
            
            # Transform the data
            transformed_data = self.preprocessor.transform(data)
            
            self.logger.info("Data transformed successfully")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit the preprocessing pipeline and transform the data.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            np.ndarray: Transformed data
        """
        self.fit(data)
        return self.transform(data)
    
    def get_feature_names(self) -> list:
        """
        Get the feature names after preprocessing.
        
        Returns:
            list: Feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        
        return self.preprocessor.get_feature_names_out()

if __name__ == "__main__":
    # Example usage
    from data_loader.data_loader import DataLoader
    
    # Load and split data
    loader = DataLoader()
    data = loader.load_data()
    train, val, test = loader.split_data(data)
    
    # Preprocess data
    preprocessor = Preprocessor()
    X_train = preprocessor.fit_transform(train.drop(columns=['readmitted']))
    X_val = preprocessor.transform(val.drop(columns=['readmitted']))
    X_test = preprocessor.transform(test.drop(columns=['readmitted']))
    
    print("Feature names:", preprocessor.get_feature_names())

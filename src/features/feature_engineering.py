import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import yaml
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Any

class FeatureEngineer:
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Initialize the FeatureEngineer with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self.feature_selector = None
        
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
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Dataset with new features
        """
        try:
            # Create age groups
            data['age_group'] = pd.cut(
                data['age'],
                bins=self.config['features']['age_bins'],
                labels=self.config['features']['age_labels']
            )
            
            # Create length of stay groups
            data['length_of_stay_group'] = pd.cut(
                data['time_in_hospital'],
                bins=self.config['features']['los_bins'],
                labels=self.config['features']['los_labels']
            )
            
            # Create number of medications feature
            medication_columns = [col for col in data.columns if 'medication' in col.lower()]
            data['num_medications'] = data[medication_columns].sum(axis=1)
            
            # Create number of diagnoses feature
            diagnosis_columns = [col for col in data.columns if 'diagnosis' in col.lower()]
            data['num_diagnoses'] = data[diagnosis_columns].sum(axis=1)
            
            self.logger.info("New features created successfully")
            return data
            
        except Exception as e:
            self.logger.error(f"Error creating features: {str(e)}")
            raise
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the most important features using statistical tests.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Selected features and their names
        """
        try:
            # Initialize feature selector
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=self.config['features']['n_features_to_select']
            )
            
            # Fit and transform the data
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            # Create DataFrame with selected features
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features)
            
            self.logger.info(f"Selected {len(selected_features)} features")
            return X_selected_df, selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def engineer_features(self, data: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, List[str]]:
        """
        Perform complete feature engineering pipeline.
        
        Args:
            data (pd.DataFrame): Input dataset
            target (pd.Series): Target variable
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Engineered features and their names
        """
        try:
            # Create new features
            data_with_features = self.create_features(data)
            
            # Select features
            X_selected, selected_features = self.select_features(
                data_with_features.drop(columns=[target.name]),
                target
            )
            
            self.logger.info("Feature engineering completed successfully")
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise 
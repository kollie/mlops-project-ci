import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import List, Tuple, Dict, Any
from .utils import setup_logging, load_config

logger = setup_logging()

class FeatureEngineer:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = load_config(config_path)
        self.feature_config = self.config['features']
        self.feature_selector = None
    
    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between numerical columns."""
        try:
            numerical_cols = self.feature_config['numerical_columns']
            interaction_data = data.copy()
            
            # Create interaction features between pairs of numerical columns
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    col1, col2 = numerical_cols[i], numerical_cols[j]
                    interaction_data[f"{col1}_{col2}_interaction"] = (
                        interaction_data[col1] * interaction_data[col2]
                    )
            
            return interaction_data
        except Exception as e:
            logger.error(f"Error creating interaction features: {str(e)}")
            raise
    
    def create_polynomial_features(self, data: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for numerical columns."""
        try:
            numerical_cols = self.feature_config['numerical_columns']
            polynomial_data = data.copy()
            
            for col in numerical_cols:
                for d in range(2, degree + 1):
                    polynomial_data[f"{col}_power_{d}"] = polynomial_data[col] ** d
            
            return polynomial_data
        except Exception as e:
            logger.error(f"Error creating polynomial features: {str(e)}")
            raise
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'f_classif', k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """Select the most important features using statistical tests."""
        try:
            if method == 'f_classif':
                self.feature_selector = SelectKBest(f_classif, k=k)
            elif method == 'mutual_info':
                self.feature_selector = SelectKBest(mutual_info_classif, k=k)
            else:
                raise ValueError(f"Unknown feature selection method: {method}")
            
            # Fit and transform the data
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
        except Exception as e:
            logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def engineer_features(self, data: pd.DataFrame, target: pd.Series = None) -> Tuple[pd.DataFrame, List[str]]:
        """Apply all feature engineering steps."""
        try:
            # Create interaction features
            data = self.create_interaction_features(data)
            
            # Create polynomial features
            data = self.create_polynomial_features(data)
            
            # Select features if target is provided
            selected_features = None
            if target is not None:
                data, selected_features = self.select_features(data, target)
            
            return data, selected_features
        except Exception as e:
            logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise 
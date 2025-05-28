import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import yaml
import logging
from pathlib import Path
import joblib

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
        self.label_encoder = None
        self.feature_selector = None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup file handler for preprocessing logs
        log_file = log_dir / "preprocessing.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.config['logging']['level'])
        file_handler.setFormatter(logging.Formatter(self.config['logging']['format']))
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config['logging']['level'])
        self.logger.addHandler(file_handler)
        
        self.logger.info("Preprocessing logging initialized")
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Replace '?' with NaN
        data = data.replace('?', np.nan)
        
        # Log missing value information
        missing_info = data.isnull().sum()
        self.logger.info("Missing values per column:\n%s", missing_info)
        
        return data
    
    def _create_preprocessing_pipeline(self) -> ColumnTransformer:
        """Create the preprocessing pipeline."""
        try:
            # Get feature lists from config
            categorical_features = self.config['features']['categorical_features']
            numerical_features = self.config['features']['numerical_features']
            
            # Create transformers
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            # Create column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )
            
            # Store feature names for later use
            self._feature_names = numerical_features.copy()
            for feature in categorical_features:
                # We'll update these with actual categories after fitting
                self._feature_names.append(f"{feature}_placeholder")
            
            return preprocessor
            
        except Exception as e:
            self.logger.error(f"Error creating preprocessing pipeline: {str(e)}")
            raise
    
    def _select_features(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select the most important features based on correlation analysis."""
        try:
            # Create feature selector
            self.feature_selector = SelectKBest(
                score_func=f_classif,
                k=self.config['feature_selection']['n_features']
            )
            
            # Fit and transform
            X_selected = self.feature_selector.fit_transform(X, y)
            
            self.logger.info(f"Selected {X_selected.shape[1]} features")
            return X_selected
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def fit(self, data: pd.DataFrame, target_col: str = 'readmitted'):
        """
        Fit the preprocessing pipeline on the data.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Name of the target column
        """
        try:
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Drop specified columns
            data = data.drop(columns=self.config['features']['drop_columns'], errors='ignore')
            
            # Separate features and target
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Encode target variable
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Create and fit the preprocessing pipeline
            self.preprocessor = self._create_preprocessing_pipeline()
            X_transformed = self.preprocessor.fit_transform(X)
            
            # Update feature names with actual categories
            self._update_feature_names(X)
            
            # Select features
            X_selected = self._select_features(X_transformed, y_encoded)
            
            self.logger.info("Preprocessing pipeline fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting preprocessing pipeline: {str(e)}")
            raise
    
    def _update_feature_names(self, X: pd.DataFrame):
        """Update feature names with actual categories after fitting."""
        try:
            # Get numerical features
            numerical_features = self.config['features']['numerical_features']
            feature_names = numerical_features.copy()
            
            # Get categorical features with their categories
            categorical_features = self.config['features']['categorical_features']
            for feature in categorical_features:
                if feature in X.columns:
                    categories = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].categories_[
                        categorical_features.index(feature)
                    ]
                    feature_names.extend([f"{feature}_{cat}" for cat in categories])
            
            self._feature_names = feature_names
            
        except Exception as e:
            self.logger.error(f"Error updating feature names: {str(e)}")
            raise
    
    def transform(self, data: pd.DataFrame, target_col: str = 'readmitted') -> tuple:
        """
        Transform the data using the fitted preprocessing pipeline.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Name of the target column
            
        Returns:
            tuple: (X_transformed_df, y_transformed) if target_col is present, else X_transformed_df
            X_transformed_df: pd.DataFrame of shape (n_samples, n_features) with feature names
            y_transformed: pd.Series of shape (n_samples,) if target_col is present
        """
        try:
            if self.preprocessor is None:
                raise ValueError("Preprocessor must be fitted before transforming data")
            
            # Handle missing values
            data = self._handle_missing_values(data)
            
            # Drop specified columns
            data = data.drop(columns=self.config['features']['drop_columns'], errors='ignore')
            
            # Check if target column is present
            has_target = target_col in data.columns
            
            if has_target:
                X = data.drop(columns=[target_col])
                y = data[target_col]
                y_transformed = pd.Series(
                    self.label_encoder.transform(y),
                    index=y.index,
                    name=target_col
                )
            else:
                X = data
            
            # Transform features
            X_transformed = self.preprocessor.transform(X)
            X_selected = self.feature_selector.transform(X_transformed)
            
            # Get feature names
            feature_names = self.get_feature_names()
            
            # Create DataFrame with feature names
            X_transformed_df = pd.DataFrame(
                X_selected,
                columns=feature_names,
                index=X.index
            )
            
            self.logger.info("Data transformed successfully")
            
            if has_target:
                return X_transformed_df, y_transformed
            return X_transformed_df
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {str(e)}")
            raise
    
    def fit_transform(self, data: pd.DataFrame, target_col: str = 'readmitted') -> tuple:
        """
        Fit the preprocessing pipeline and transform the data.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Name of the target column
            
        Returns:
            tuple: (X_transformed_df, y_transformed)
            X_transformed_df: pd.DataFrame of shape (n_samples, n_features) with feature names
            y_transformed: pd.Series of shape (n_samples,)
        """
        self.fit(data, target_col)
        return self.transform(data, target_col)
    
    def get_feature_names(self) -> list:
        """
        Get the feature names after preprocessing.
        
        Returns:
            list: Feature names
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor must be fitted before getting feature names")
        
        if not hasattr(self, '_feature_names'):
            raise ValueError("Feature names not available. Preprocessor must be fitted.")
        
        # Apply feature selection if it was used
        if self.feature_selector is not None:
            selected_indices = self.feature_selector.get_support()
            return [name for i, name in enumerate(self._feature_names) if selected_indices[i]]
        
        return self._feature_names
    
    def save(self, path: str):
        """Save the preprocessor to disk."""
        try:
            joblib.dump({
                'preprocessor': self.preprocessor,
                'label_encoder': self.label_encoder,
                'feature_selector': self.feature_selector
            }, path)
            self.logger.info(f"Preprocessor saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving preprocessor: {str(e)}")
            raise
    
    def load(self, path: str):
        """Load the preprocessor from disk."""
        try:
            saved_data = joblib.load(path)
            self.preprocessor = saved_data['preprocessor']
            self.label_encoder = saved_data['label_encoder']
            self.feature_selector = saved_data['feature_selector']
            self.logger.info(f"Preprocessor loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading preprocessor: {str(e)}")
            raise
    
    def run_preprocessing(self, data: pd.DataFrame, target_col: str = 'readmitted') -> tuple:
        """
        Run the complete preprocessing pipeline on the data.
        
        Args:
            data (pd.DataFrame): Input dataset
            target_col (str): Name of the target column
            
        Returns:
            tuple: (X_processed_df, y_processed)
            X_processed_df: pd.DataFrame of processed features
            y_processed: pd.Series of processed target
        """
        try:
            self.logger.info("Starting preprocessing pipeline...")
            
            # Handle missing values
            self.logger.info("Handling missing values...")
            data = self._handle_missing_values(data)
            
            # Drop specified columns
            self.logger.info("Dropping specified columns...")
            data = data.drop(columns=self.config['features']['drop_columns'], errors='ignore')
            
            # Separate features and target
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Encode target variable
            self.logger.info("Encoding target variable...")
            self.label_encoder = LabelEncoder()
            y_processed = pd.Series(
                self.label_encoder.fit_transform(y),
                index=y.index,
                name=target_col
            )
            
            # Create and fit preprocessing pipeline
            self.logger.info("Creating and fitting preprocessing pipeline...")
            self.preprocessor = self._create_preprocessing_pipeline()
            X_transformed = self.preprocessor.fit_transform(X)
            
            # Select features
            self.logger.info("Selecting features...")
            X_selected = self._select_features(X_transformed, y_processed)
            
            # Create DataFrame with feature names
            feature_names = self.get_feature_names()
            X_processed_df = pd.DataFrame(
                X_selected,
                columns=feature_names,
                index=X.index
            )
            
            # Log preprocessing results
            self.logger.info(f"Preprocessing completed. Final shape: {X_processed_df.shape}")
            self.logger.info(f"Selected features: {len(feature_names)}")
            self.logger.info(f"Target distribution:\n{y_processed.value_counts()}")
            
            return X_processed_df, y_processed
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise

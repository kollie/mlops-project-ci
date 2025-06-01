"""
Exploratory Data Analysis module for the MLOps project.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from pathlib import Path

class EDA:
    def __init__(self, log_file: str = "logs/eda.log"):
        """Initialize the EDA class with logging setup.
        
        Args:
            log_file (str): Path to the log file. Defaults to "logs/eda.log".
        """
        self._setup_logging(log_file)
        self.plots_dir = Path("plots")
        self.plots_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self, log_file: str):
        """Setup logging configuration.
        
        Args:
            log_file (str): Path to the log file.
        """
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file
        )
        self.logger = logging.getLogger(__name__)
    
    def describe_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate descriptive statistics for the DataFrame."""
        if df.empty:
            raise ValueError("Cannot describe an empty DataFrame")
        
        # Convert numerical columns to appropriate types
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        description = df.describe(include='all')
        self.logger.info(f"DataFrame description:\n{description}")
        return description
    
    def analyze_target_distribution(self, df: pd.DataFrame, target_col: str = 'readmitted') -> dict:
        """Analyze the distribution of the target variable."""
        if df.empty:
            raise ValueError("Cannot analyze target distribution of an empty DataFrame")
        
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")
        
        # Get value counts and percentages
        value_counts = df[target_col].value_counts()
        percentages = df[target_col].value_counts(normalize=True) * 100
        
        # Create distribution plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=target_col)
        plt.title(f'Distribution of {target_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'target_distribution.png')
        plt.close()
        
        # Log results
        self.logger.info(f"Target distribution:\n{value_counts}")
        self.logger.info(f"Target percentages:\n{percentages}")
        
        return {
            'value_counts': value_counts,
            'percentages': percentages
        }
    
    def analyze_feature_distributions(self, df: pd.DataFrame) -> dict:
        """Analyze distributions of numerical features."""
        if df.empty:
            raise ValueError("Cannot analyze feature distributions of an empty DataFrame")
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.logger.warning("No numerical columns found for distribution analysis")
            return {}
        
        # Create distribution plots
        n_cols = min(3, len(numerical_cols))
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            if idx < len(axes):
                sns.histplot(data=df, x=col, ax=axes[idx])
                axes[idx].set_title(f'Distribution of {col}')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'feature_distributions.png')
        plt.close()
        
        # Calculate statistics
        feature_stats = {}
        for col in numerical_cols:
            feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
        
        self.logger.info(f"Feature distribution statistics:\n{feature_stats}")
        return feature_stats
    
    def analyze_correlations(self, df: pd.DataFrame, target_col: str = 'readmitted') -> pd.DataFrame:
        """Analyze correlations between features and target."""
        if df.empty:
            raise ValueError("Cannot analyze correlations of an empty DataFrame")
        
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.logger.warning("No numerical columns found for correlation analysis")
            return pd.DataFrame()
        
        # Calculate correlations
        corr_matrix = df[numerical_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'correlation_matrix.png')
        plt.close()
        
        self.logger.info(f"Correlation matrix:\n{corr_matrix}")
        return corr_matrix
    
    def analyze_class_imbalance(self, df: pd.DataFrame, target_col: str = 'readmitted') -> dict:
        """Analyze class imbalance in the target variable."""
        if df.empty:
            raise ValueError("Cannot analyze class imbalance of an empty DataFrame")
        
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")
        
        # Calculate class distribution
        class_counts = df[target_col].value_counts()
        class_ratios = class_counts / len(df)
        
        # Calculate imbalance metrics
        n_classes = len(class_counts)
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        results = {
            'class_counts': class_counts,
            'class_ratios': class_ratios,
            'n_classes': n_classes,
            'imbalance_ratio': imbalance_ratio
        }
        
        self.logger.info(f"Class imbalance analysis:\n{results}")
        return results
    
    def find_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find missing values in the DataFrame."""
        if df.empty:
            return pd.DataFrame(columns=['column', 'missing_count', 'missing_percentage'])
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # Calculate missing values
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_info = pd.DataFrame({
            'column': missing_counts.index,
            'missing_count': missing_counts.values,
            'missing_percentage': missing_percentages.values
        })
        
        self.logger.info(f"Missing value analysis:\n{missing_info}")
        return missing_info
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        if df.empty:
            return df
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # Handle numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        self.logger.info("Missing values handled")
        return df
    
    def remove_low_importance_columns(self, df: pd.DataFrame, target_col: str = 'readmitted', 
                                    threshold: float = 0.1) -> pd.DataFrame:
        """Remove columns with low importance based on correlation with target."""
        if df.empty:
            raise ValueError("Cannot remove columns from an empty DataFrame")
        
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            self.logger.warning("No numerical columns found for importance analysis")
            return df
        
        # Calculate correlations with target
        correlations = df[numerical_cols].corrwith(df[target_col].map({'NO': 0, 'YES': 1}))
        
        # Select columns above threshold
        important_cols = correlations[abs(correlations) > threshold].index
        
        # Keep target column and important features
        columns_to_keep = list(important_cols) + [target_col]
        df_filtered = df[columns_to_keep]
        
        self.logger.info(f"Removed {len(df.columns) - len(columns_to_keep)} low importance columns")
        return df_filtered
    
    def run_analysis(self, df: pd.DataFrame, target_col: str = 'readmitted') -> pd.DataFrame:
        """Run the complete EDA analysis pipeline."""
        if df.empty:
            raise ValueError("Cannot run analysis on an empty DataFrame")
        
        self.logger.info("Starting EDA analysis...")
        
        # Step 1: Describe the data
        self.describe_dataframe(df)
        
        # Step 2: Analyze target distribution
        self.analyze_target_distribution(df, target_col)
        
        # Step 3: Analyze feature distributions
        self.analyze_feature_distributions(df)
        
        # Step 4: Analyze correlations
        self.analyze_correlations(df, target_col)
        
        # Step 5: Analyze class imbalance
        self.analyze_class_imbalance(df, target_col)
        
        # Step 6: Handle missing values
        df_processed = self.handle_missing_values(df)
        
        # Step 7: Remove low importance columns
        df_processed = self.remove_low_importance_columns(df_processed, target_col)
        
        self.logger.info("EDA analysis completed")
        return df_processed 
"""
Exploratory Data Analysis module for the MLOps project.
"""

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
from pathlib import Path
from scipy import stats

class EDA:
    def __init__(self, log_file: str = "logs/eda.log"):
        """Initialize the EDA class.
        
        Args:
            log_file (str): Path to the log file
        """
        self._setup_logging(log_file)
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self, log_file: str) -> None:
        """Setup logging configuration.
        
        Args:
            log_file (str): Path to the log file
        """
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def describe_dataframe(self, df: pd.DataFrame) -> Dict:
        """Generate descriptive statistics for the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict: Dictionary containing descriptive statistics
        """
        self.logger.info("Generating descriptive statistics...")
        
        # Basic statistics
        stats = {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'summary': df.describe().to_dict(),
            'memory_usage': df.memory_usage(deep=True).to_dict(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'skewness': df.select_dtypes(include=['int64', 'float64']).skew().to_dict(),
            'kurtosis': df.select_dtypes(include=['int64', 'float64']).kurtosis().to_dict()
        }
        
        # Log the statistics
        self.logger.info(f"Dataframe shape: {stats['shape']}")
        self.logger.info(f"Memory usage: {sum(stats['memory_usage'].values()) / 1024:.2f} KB")
        self.logger.info(f"Number of unique values per column:\n{stats['unique_values']}")
        self.logger.info(f"Skewness of numerical columns:\n{stats['skewness']}")
        self.logger.info(f"Kurtosis of numerical columns:\n{stats['kurtosis']}")
        
        return stats
    
    def analyze_target_distribution(self, df: pd.DataFrame, target_col: str = 'readmitted') -> None:
        """Analyze and plot the distribution of the target variable.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
        """
        self.logger.info(f"Analyzing target distribution for column: {target_col}")
        
        # Get value counts and percentages
        value_counts = df[target_col].value_counts()
        value_percentages = df[target_col].value_counts(normalize=True) * 100
        
        self.logger.info(f"Target value counts:\n{value_counts}")
        self.logger.info(f"Target value percentages:\n{value_percentages}")
        
        # Create plots directory if it doesn't exist
        plots_dir = Path('plots')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Create count plot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x=target_col)
        plt.title('Distribution of Readmission Status')
        plt.xlabel('Readmission Status')
        plt.ylabel('Count')
        plt.savefig(plots_dir / 'target_distribution.png')
        plt.close()
        
        # Create pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        plt.title('Readmission Status Distribution')
        plt.savefig(plots_dir / 'target_distribution_pie.png')
        plt.close()
        
        self.logger.info("Target distribution plots saved to plots directory")
    
    def analyze_feature_distributions(self, df: pd.DataFrame) -> None:
        """Analyze and plot distributions of numerical features.
        
        Args:
            df (pd.DataFrame): Input dataframe
        """
        self.logger.info("Analyzing feature distributions...")
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        plots_dir = Path('plots/distributions')
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        for col in numerical_cols:
            # Create distribution plot
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col, kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(plots_dir / f'{col}_distribution.png')
            plt.close()
            
            # Log statistics
            stats_dict = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }
            self.logger.info(f"Statistics for {col}:\n{stats_dict}")
    
    def analyze_correlations(self, df: pd.DataFrame, target_col: str = 'readmitted') -> None:
        """Analyze correlations between features and target.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
        """
        self.logger.info("Analyzing feature correlations...")
        
        # Calculate correlations
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        correlations = df[numerical_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('plots/correlation_heatmap.png')
        plt.close()
        
        # Analyze target correlations
        target_correlations = correlations[target_col].sort_values(ascending=False)
        self.logger.info(f"Correlations with target variable:\n{target_correlations}")
        
        # Plot top correlations with target
        top_correlations = target_correlations[1:11]  # Exclude target itself
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_correlations.values, y=top_correlations.index)
        plt.title('Top 10 Feature Correlations with Target')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig('plots/top_correlations.png')
        plt.close()
    
    def analyze_class_imbalance(self, df: pd.DataFrame, target_col: str = 'readmitted') -> Dict:
        """Analyze class imbalance in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            
        Returns:
            Dict: Dictionary containing imbalance metrics
        """
        self.logger.info("Analyzing class imbalance...")
        
        # Calculate class distribution
        class_counts = df[target_col].value_counts()
        class_ratios = class_counts / len(df)
        
        # Calculate imbalance ratio
        imbalance_ratio = class_counts.max() / class_counts.min()
        
        metrics = {
            'class_counts': class_counts.to_dict(),
            'class_ratios': class_ratios.to_dict(),
            'imbalance_ratio': imbalance_ratio
        }
        
        self.logger.info(f"Class distribution:\n{class_counts}")
        self.logger.info(f"Class ratios:\n{class_ratios}")
        self.logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        return metrics
    
    def find_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """Find missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, int]: Dictionary containing column names and their missing value counts
        """
        self.logger.info("Analyzing missing values...")
        
        # Find actual NaN values
        nan_counts = df.isna().sum()
        nan_cols = nan_counts[nan_counts > 0]
        
        # Find '?' placeholders
        question_mark_counts = (df == '?').sum()
        question_mark_cols = question_mark_counts[question_mark_counts > 0]
        
        # Combine results
        missing_values = {
            'nan_values': nan_cols.to_dict(),
            'question_mark_values': question_mark_cols.to_dict()
        }
        
        # Log results
        if nan_cols.empty and question_mark_cols.empty:
            self.logger.info("No missing values found in the dataframe")
        else:
            if not nan_cols.empty:
                self.logger.info(f"Columns with NaN values:\n{nan_cols}")
            if not question_mark_cols.empty:
                self.logger.info(f"Columns with '?' values:\n{question_mark_cols}")
        
        return missing_values
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataframe.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        self.logger.info("Handling missing values...")
        
        # Replace '?' with NaN
        df = df.replace('?', np.nan)
        
        # Handle numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            self.logger.info(f"Filled missing values in {col} with median: {median_value}")
        
        # Handle categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            self.logger.info(f"Filled missing values in {col} with mode: {mode_value}")
        
        return df
    
    def remove_low_importance_columns(self, df: pd.DataFrame, 
                                    target_col: str = 'readmitted',
                                    threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """Remove columns that don't contribute significantly to the model.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            threshold (float): Threshold for feature importance
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Tuple containing the cleaned dataframe and list of removed columns
        """
        self.logger.info("Analyzing column importance...")
        
        # Calculate correlation with target
        correlations = df.corr()[target_col].abs()
        low_importance_cols = correlations[correlations < threshold].index.tolist()
        
        # Remove low importance columns
        df_cleaned = df.drop(columns=low_importance_cols)
        
        # Log results
        self.logger.info(f"Removed {len(low_importance_cols)} columns with low importance:")
        for col in low_importance_cols:
            self.logger.info(f"- {col}: correlation = {correlations[col]:.4f}")
        
        return df_cleaned, low_importance_cols
    
    def run_analysis(self, df: pd.DataFrame, target_col: str = 'readmitted') -> pd.DataFrame:
        """Run complete EDA pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            
        Returns:
            pd.DataFrame: Processed dataframe
        """
        self.logger.info("Starting EDA pipeline...")
        
        # Step 1: Describe dataframe
        self.describe_dataframe(df)
        
        # Step 2: Analyze target distribution
        self.analyze_target_distribution(df, target_col)
        
        # Step 3: Analyze feature distributions
        self.analyze_feature_distributions(df)
        
        # Step 4: Analyze correlations
        self.analyze_correlations(df, target_col)
        
        # Step 5: Analyze class imbalance
        self.analyze_class_imbalance(df, target_col)
        
        # Step 6: Find missing values
        missing_values = self.find_missing_values(df)
        
        # Step 7: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 8: Remove low importance columns
        df, removed_cols = self.remove_low_importance_columns(df, target_col)
        
        self.logger.info("EDA pipeline completed successfully")
        return df 
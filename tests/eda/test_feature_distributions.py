"""
Tests for the analyze_feature_distributions method.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_analyze_feature_distributions_basic(eda_instance, sample_data, plots_dir):
    """Test basic functionality of analyze_feature_distributions."""
    eda_instance.analyze_feature_distributions(sample_data)
    
    # Get numerical columns
    numerical_cols = sample_data.select_dtypes(include=['int64', 'float64']).columns
    
    # Check if plots were created for each numerical column
    for col in numerical_cols:
        assert (plots_dir / 'distributions' / f'{col}_distribution.png').exists()

def test_analyze_feature_distributions_no_numerical(eda_instance, plots_dir):
    """Test analyze_feature_distributions with no numerical columns."""
    # Create DataFrame with only categorical columns
    data = pd.DataFrame({
        'cat1': ['A', 'B', 'C'] * 10,
        'cat2': ['X', 'Y', 'Z'] * 10
    })
    
    eda_instance.analyze_feature_distributions(data)
    
    # Check that no distribution plots were created
    dist_dir = plots_dir / 'distributions'
    assert not dist_dir.exists() or len(list(dist_dir.glob('*.png'))) == 0

def test_analyze_feature_distributions_empty(eda_instance, plots_dir):
    """Test analyze_feature_distributions with empty DataFrame."""
    empty_df = pd.DataFrame()
    eda_instance.analyze_feature_distributions(empty_df)
    
    # Check that no distribution plots were created
    dist_dir = plots_dir / 'distributions'
    assert not dist_dir.exists() or len(list(dist_dir.glob('*.png'))) == 0

def test_analyze_feature_distributions_with_missing(eda_instance, plots_dir):
    """Test analyze_feature_distributions with missing values."""
    # Create DataFrame with missing values
    data = pd.DataFrame({
        'col1': [1, 2, np.nan, 4, 5],
        'col2': [1.1, 2.2, 3.3, np.nan, 5.5]
    })
    
    eda_instance.analyze_feature_distributions(data)
    
    # Check if plots were created
    assert (plots_dir / 'distributions' / 'col1_distribution.png').exists()
    assert (plots_dir / 'distributions' / 'col2_distribution.png').exists()

def test_analyze_feature_distributions_with_inf(eda_instance, plots_dir):
    """Test analyze_feature_distributions with infinite values."""
    # Create DataFrame with infinite values
    data = pd.DataFrame({
        'col1': [1, 2, np.inf, 4, 5],
        'col2': [1.1, 2.2, 3.3, -np.inf, 5.5]
    })
    
    eda_instance.analyze_feature_distributions(data)
    
    # Check if plots were created
    assert (plots_dir / 'distributions' / 'col1_distribution.png').exists()
    assert (plots_dir / 'distributions' / 'col2_distribution.png').exists() 
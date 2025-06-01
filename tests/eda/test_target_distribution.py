"""
Tests for the analyze_target_distribution method.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

def test_analyze_target_distribution_basic(eda_instance, sample_data, plots_dir):
    """Test basic functionality of analyze_target_distribution."""
    eda_instance.analyze_target_distribution(sample_data)
    
    # Check if plots were created
    assert (plots_dir / 'target_distribution.png').exists()
    assert (plots_dir / 'target_distribution_pie.png').exists()

def test_analyze_target_distribution_custom_target(eda_instance, sample_data, plots_dir):
    """Test analyze_target_distribution with custom target column."""
    # Create a custom target column
    sample_data['custom_target'] = np.random.choice(['A', 'B', 'C'], len(sample_data))
    eda_instance.analyze_target_distribution(sample_data, target_col='custom_target')
    
    # Check if plots were created
    assert (plots_dir / 'target_distribution.png').exists()
    assert (plots_dir / 'target_distribution_pie.png').exists()

def test_analyze_target_distribution_binary(eda_instance, plots_dir):
    """Test analyze_target_distribution with binary target."""
    # Create binary target data
    data = pd.DataFrame({
        'target': np.random.choice([0, 1], 100)
    })
    eda_instance.analyze_target_distribution(data, target_col='target')
    
    # Check if plots were created
    assert (plots_dir / 'target_distribution.png').exists()
    assert (plots_dir / 'target_distribution_pie.png').exists()

def test_analyze_target_distribution_empty(eda_instance, plots_dir):
    """Test analyze_target_distribution with empty DataFrame."""
    empty_df = pd.DataFrame({'target': []})
    with pytest.raises(ValueError):
        eda_instance.analyze_target_distribution(empty_df, target_col='target')

def test_analyze_target_distribution_missing_target(eda_instance, sample_data):
    """Test analyze_target_distribution with missing target column."""
    with pytest.raises(KeyError):
        eda_instance.analyze_target_distribution(sample_data, target_col='nonexistent_column') 
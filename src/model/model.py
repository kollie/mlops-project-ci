"""
model.py

This module acts as a factory for instantiating different machine learning models
based on configuration defined in `config.yaml`. It supports multiple model types 
including Random Forest, Logistic Regression, Decision Tree, and Naive Bayes.

Usage:
    from src.model.model import get_model
    model = get_model(config_path="config.yaml")
    
This aligns with MLOps best practices:
- Centralized model control via config
- Config-driven hyperparameters
- No hardcoding of model logic
- Supports seamless model switching
"""

import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Load model configuration
logger.info(f"Loaded model: {model_type}")
logger.info(f"Model parameters: {model_params}")



def get_model(config_path: str = "config.yaml"):
    """
    Returns an initialized ML model instance as defined in config.yaml.
    
    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        sklearn.base.BaseEstimator: Initialized model object.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_block = config["model"]
    model_type = model_block.get("active", "random_forest")
    model_params = model_block.get(model_type, {})

    if model_type == "random_forest":
        return RandomForestClassifier(**model_params)
    elif model_type == "logistic_regression":
        return LogisticRegression(**model_params)
    elif model_type == "decision_tree":
        return DecisionTreeClassifier(**model_params)
    elif model_type == "naive_bayes":
        return GaussianNB()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

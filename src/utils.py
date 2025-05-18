import logging
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

def setup_logging(config_path: str = "src/config.yaml") -> logging.Logger:
    """Set up logging configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    log_config = config['logging']
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_config['file']), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_config['level'],
        format=log_config['format'],
        handlers=[
            logging.FileHandler(log_config['file']),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str = "src/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_feature_importance(importance: list, feature_names: list, title: str = "Feature Importance"):
    """Plot feature importance."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    return plt

def plot_confusion_matrix(cm, labels, title: str = "Confusion Matrix"):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

def save_plot(plt, filename: str, directory: str = "plots"):
    """Save plot to file."""
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, filename))
    plt.close() 
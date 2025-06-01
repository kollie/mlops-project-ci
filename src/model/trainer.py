# src/model/trainer.py

"""
trainer.py

Handles model training, preprocessing on training split, and model artifact saving.

- Fits preprocessing pipeline on training split only
- Trains model using parameters from config.yaml
- Saves trained model to specified directory
- Logs each key pipeline stage for reproducibility
"""

import logging
import joblib
import os

from src.model.model import get_model
from src.preprocessing.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config):
        """
        Initializes Trainer with model, preprocessor, and config parameters.

        Args:
            config (dict): Parsed YAML configuration
        """
        self.config = config
        self.model = get_model()
        self.preprocessor = Preprocessor(config)
        self.model_path = config["data"]["model_path"]
        self.target_column = config["features"]["target_column"]

    def train(self, df_train):
        """
        Fits preprocessing pipeline and trains model on training data.

        Args:
            df_train (pd.DataFrame): Training data containing features and target
        """
        try:
            logger.info("Starting training pipeline")

            # Separate features and label
            X = df_train.drop(columns=[self.target_column])
            y = df_train[self.target_column]
            logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

            # Apply preprocessing
            X_processed = self.preprocessor.fit_transform(X)

            # Train model
            self.model.fit(X_processed, y)
            logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    def save_model(self, filename="model.pkl"):
        """
        Saves the trained model to disk.

        Args:
            filename (str): Name of the model file to save
        """
        os.makedirs(self.model_path, exist_ok=True)
        path = os.path.join(self.model_path, filename)

        try:
            joblib.dump(self.model, path)
            logger.info(f"Model saved successfully at: {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

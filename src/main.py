import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import DataLoader
from data_validation import DataValidator
from preprocessing import DataPreprocessor
from features import FeatureEngineer
from model import ModelTrainer
from evaluation import ModelEvaluator
from inference import InferencePipeline
from utils import setup_logging, load_config

logger = setup_logging()

def load_data_from_url(data_loader: DataLoader, url: str, params: dict = None) -> pd.DataFrame:
    """Load data from a URL."""
    try:
        logger.info(f"Loading data from URL: {url}")
        data = data_loader.load_api(url, params)
        return data
    except Exception as e:
        logger.error(f"Error loading data from URL: {str(e)}")
        raise

def run_training_pipeline(use_url: bool = False):
    """Run the complete training pipeline."""
    try:
        # Initialize components
        data_loader = DataLoader()
        data_validator = DataValidator()
        preprocessor = DataPreprocessor()
        feature_engineer = FeatureEngineer()
        model_trainer = ModelTrainer()
        model_evaluator = ModelEvaluator()
        
        # Load data
        logger.info("Loading data...")
        if use_url:
            # Example URL for a sample dataset (replace with your actual URL)
            url = "https://raw.githubusercontent.com/datasets/iris/master/data/iris.csv"
            # Example parameters (if needed)
            params = {
                "format": "csv",
                "download": "true"
            }
            data = load_data_from_url(data_loader, url, params)
        else:
            data = data_loader.load_csv("data/raw/train_data.csv")
        
        # Validate data
        logger.info("Validating data...")
        if not data_validator.validate_data(data):
            raise ValueError("Data validation failed")
        
        # Handle missing values
        logger.info("Handling missing values...")
        data = data_validator.handle_missing_values(data)
        
        # Preprocess data
        logger.info("Preprocessing data...")
        data, transformers = preprocessor.transform_data(data, is_training=True)
        
        # Engineer features
        logger.info("Engineering features...")
        data, selected_features = feature_engineer.engineer_features(
            data,
            target=data[load_config()['features']['target_column']]
        )
        
        # Prepare data for training
        X = data.drop(columns=[load_config()['features']['target_column']])
        y = data[load_config()['features']['target_column']]
        
        # Train model
        logger.info("Training model...")
        metrics = model_trainer.train(X, y)
        
        # Save model
        logger.info("Saving model...")
        model_trainer.save_model()
        
        # Make predictions on test set
        logger.info("Making predictions...")
        predictions = model_trainer.predict(X)
        
        # Evaluate model
        logger.info("Evaluating model...")
        evaluation_results = model_evaluator.evaluate_model(y, predictions)
        
        return metrics, evaluation_results
    
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def run_inference_pipeline(data_path: str, use_url: bool = False):
    """Run the inference pipeline on new data."""
    try:
        # Initialize inference pipeline
        pipeline = InferencePipeline()
        
        # Make predictions
        if use_url:
            logger.info(f"Running inference on URL data: {data_path}...")
            # Load data from URL
            data_loader = DataLoader()
            data = load_data_from_url(data_loader, data_path)
            # Save to temporary file for inference
            temp_path = "data/temp/test_data.csv"
            data.to_csv(temp_path, index=False)
            predictions = pipeline.predict(temp_path)
            # Clean up temporary file
            Path(temp_path).unlink(missing_ok=True)
        else:
            logger.info(f"Running inference on {data_path}...")
            predictions = pipeline.predict(data_path)
        
        return predictions
    
    except Exception as e:
        logger.error(f"Error in inference pipeline: {str(e)}")
        raise

def main():
    """Main function to run the complete pipeline."""
    try:
        # Create necessary directories
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        Path("models").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(parents=True, exist_ok=True)
        
        # Example of using URL data
        use_url_data = True  # Set to True to use URL data
        
        # Run training pipeline
        logger.info("Starting training pipeline...")
        metrics, evaluation_results = run_training_pipeline(use_url=use_url_data)
        
        # Print results
        logger.info("\nTraining Metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        logger.info("\nEvaluation Results:")
        for metric_name, value in evaluation_results['metrics'].items():
            logger.info(f"{metric_name}: {value:.4f}")
        
        # Run inference on test data
        logger.info("\nRunning inference on test data...")
        if use_url_data:
            # Example test data URL (replace with your actual URL)
            test_url = "https://raw.githubusercontent.com/datasets/iris/master/data/iris_test.csv"
            test_predictions = run_inference_pipeline(test_url, use_url=True)
        else:
            test_predictions = run_inference_pipeline("data/test/test_data.csv")
        
        logger.info(f"Number of predictions: {len(test_predictions)}")
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
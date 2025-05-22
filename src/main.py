import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from data_loader.data_loader import DataLoader
from preprocessing.preprocessor import Preprocessor
from validation.validator import DataValidator
from model.train import ModelTrainer
from evaluation.metrics import ModelEvaluator
from inference.predict import Predictor
from features.feature_engineering import FeatureEngineer

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='logs/pipeline.log'
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "src/config.yaml"):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

logger = setup_logging()

def run_pipeline():
    """Run the complete ML pipeline."""
    try:
        # Load configuration
        config = load_config()
        logger.info("Starting ML pipeline")
        
        # Step 1: Load and validate data
        logger.info("Step 1: Loading and validating data")
        loader = DataLoader()
        data = loader.load_data("data/raw/diabetic_readmission_data.csv")
        
        validator = DataValidator()
        validation_results = validator.validate_all(data)
        
        if not all(validation_results.values()):
            logger.error("Data validation failed")
            return
        
        # Step 2: Split data
        logger.info("Step 2: Splitting data")
        train, val, test = loader.split_data(data)
        loader.save_split_data(train, val, test)
        
        # Step 3: Feature Engineering
        logger.info("Step 3: Engineering features")
        feature_engineer = FeatureEngineer()
        X_train, selected_features = feature_engineer.engineer_features(
            train.drop(columns=['readmitted']),
            train['readmitted']
        )
        X_val = feature_engineer.engineer_features(
            val.drop(columns=['readmitted']),
            val['readmitted']
        )[0]
        X_test = feature_engineer.engineer_features(
            test.drop(columns=['readmitted']),
            test['readmitted']
        )[0]
        
        y_train = train['readmitted']
        y_val = val['readmitted']
        y_test = test['readmitted']
        
        # Step 4: Preprocess data
        logger.info("Step 4: Preprocessing data")
        preprocessor = Preprocessor()
        X_train = preprocessor.fit_transform(X_train)
        X_val = preprocessor.transform(X_val)
        X_test = preprocessor.transform(X_test)
        
        # Step 5: Train model
        logger.info("Step 5: Training model")
        trainer = ModelTrainer()
        trainer.train(X_train, y_train)
        trainer.save_model()
        
        # Step 6: Evaluate model
        logger.info("Step 6: Evaluating model")
        evaluator = ModelEvaluator()
        
        # Evaluate on validation set
        y_val_pred = trainer.model.predict(X_val)
        y_val_pred_proba = trainer.model.predict_proba(X_val)
        val_metrics = evaluator.calculate_metrics(y_val, y_val_pred, y_val_pred_proba)
        evaluator.log_metrics(val_metrics)
        evaluator.save_metrics(val_metrics, "validation_metrics.csv")
        
        # Evaluate on test set
        y_test_pred = trainer.model.predict(X_test)
        y_test_pred_proba = trainer.model.predict_proba(X_test)
        test_metrics = evaluator.calculate_metrics(y_test, y_test_pred, y_test_pred_proba)
        evaluator.save_metrics(test_metrics, "test_metrics.csv")
        
        # Step 7: Make predictions on test set
        logger.info("Step 7: Making predictions")
        predictor = Predictor()
        predictions = predictor.predict_with_confidence(test.drop(columns=['readmitted']))
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'true_label': y_test,
            'predicted_label': predictions['predictions'],
            'confidence_score': predictions['confidence_scores']
        })
        predictions_df.to_csv("predictions.csv", index=False)
        
        logger.info("ML pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    run_pipeline() 
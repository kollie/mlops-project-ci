import logging
import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader.data_loader import DataLoader
from src.validation.data_validator import DataValidator
from src.eda.eda import EDAAnalyzer
from src.preprocessing.preprocessor import Preprocessor
from src.features.feature_engineering import FeatureEngineer
from src.model.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from src.inference.predict import ModelPredictor

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Clear any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "main.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function to run the complete ML pipeline."""
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Starting MLOps pipeline...")
    
    try:
        # Step 1: Load and split data
        logger.info("Step 1: Loading and splitting data...")
        data_loader = DataLoader(config_path="src/config.yaml")
        df = data_loader.load_data()
        train, val, test = data_loader.split_data(df)
        data_loader.save_split_data(train, val, test)
        logger.info(f"✅ Data loaded and split - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        
        # Step 2: Validate and clean data
        logger.info("Step 2: Validating and cleaning data...")
        validator = DataValidator(config_path="src/config.yaml")
        clean_data = validator.validate_and_clean(df, strategy='drop_columns')
        logger.info(f"✅ Data validation completed - Clean data shape: {clean_data.shape}")
        
        # Step 3: Exploratory Data Analysis
        logger.info("Step 3: Performing Exploratory Data Analysis...")
        eda_analyzer = EDAAnalyzer(config_path="src/config.yaml")
        eda_report = eda_analyzer.run_full_analysis(clean_data)
        logger.info(f"✅ EDA completed - Report sections: {list(eda_report.keys())}")
        
        # Log key EDA insights
        if 'summary' in eda_report:
            summary = eda_report['summary']
            logger.info("📊 EDA Summary:")
            logger.info(f"   Total features: {summary.get('total_features', 'N/A')}")
            logger.info(f"   Numeric features: {summary.get('numeric_features', 'N/A')}")
            logger.info(f"   Categorical features: {summary.get('categorical_features', 'N/A')}")
            logger.info(f"   Missing data: {summary.get('missing_data_percentage', 'N/A')}%")
        
        if 'target_analysis' in eda_report and 'error' not in eda_report['target_analysis']:
            target_analysis = eda_report['target_analysis']
            logger.info("🎯 Target Analysis:")
            logger.info(f"   Target column: {target_analysis.get('target_column', 'N/A')}")
            logger.info(f"   Class distribution: {target_analysis.get('percentages', {})}")
            logger.info(f"   Balanced: {target_analysis.get('class_balance', {}).get('is_balanced', 'N/A')}")

        # Step 4: Preprocessing
        logger.info("Step 4: Preprocessing data...")
        preprocessor = Preprocessor(config_path="src/config.yaml")
        X_train, y_train = preprocessor.fit_transform(train)
        X_val, y_val = preprocessor.transform(val)
        X_test, y_test = preprocessor.transform(test)
        preprocessor.save_pipeline("models/preprocessor.joblib")
        logger.info(f"✅ Data preprocessed - Training shape: {X_train.shape}")
        
        # Step 5: Feature Engineering
        logger.info("Step 5: Applying feature engineering...")
        engineer = FeatureEngineer(config_path="src/config.yaml")
        X_train_eng, y_train = engineer.fit_transform(
            pd.concat([X_train, y_train], axis=1), 
            target_col='readmitted'
        )
        X_val_eng, y_val = engineer.transform(
            pd.concat([X_val, y_val], axis=1), 
            target_col='readmitted'
        )
        X_test_eng, y_test = engineer.transform(
            pd.concat([X_test, y_test], axis=1), 
            target_col='readmitted'
        )
        logger.info(f"✅ Feature engineering completed - Final shape: {X_train_eng.shape}")

        # Step 6: Train Model
        logger.info("Step 6: Training model...")
        trainer = ModelTrainer(config_path="src/config.yaml")
        trainer.fit(X_train_eng, y_train)
        model_path = trainer.save()
        logger.info(f"✅ Model trained and saved to: {model_path}")

        # Step 7: Evaluate Model
        logger.info("Step 7: Evaluating model...")
        evaluator = ModelEvaluator(config_path="src/config.yaml")
        
        # Get predictions and probabilities for evaluation
        predictions = trainer.predict(X_test_eng)
        probabilities = trainer.predict_proba(X_test_eng)
        
        # Run evaluation
        evaluation_results = evaluator.evaluate(
            y_true=y_test, 
            y_pred=predictions, 
            y_pred_proba=probabilities,
            dataset_name="test"
        )
        
        # Log key metrics
        test_metrics = evaluator.get_metrics()
        logger.info("📈 Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"   {metric}: {value:.4f}")
        
        # Save evaluation results
        metrics_path = evaluator.save_metrics()
        logger.info(f"✅ Evaluation completed - Metrics saved to: {metrics_path}")
        
        # Step 8: Test Inference
        logger.info("Step 8: Testing model inference...")
        predictor = ModelPredictor(config_path="src/config.yaml")
        predictor.load_model(model_path)
        
        # Test inference on a sample of test data
        test_sample = X_test_eng.head(10).reset_index(drop=True)
        
        # Make basic predictions
        inference_predictions = predictor.predict(test_sample, preprocess=False)
        logger.info(f"✅ Basic inference completed - {len(inference_predictions)} predictions made")
        
        # Test confidence predictions
        confidence_results = predictor.predict_with_confidence(
            test_sample, 
            confidence_threshold=0.8, 
            preprocess=False
        )
        logger.info(f"📊 Confidence Analysis:")
        logger.info(f"   High confidence predictions: {confidence_results['high_confidence_count']}/{len(test_sample)}")
        logger.info(f"   Mean confidence: {confidence_results['confidence_stats']['mean_confidence']:.4f}")
        
        # Test batch inference on full test set
        logger.info("Testing batch inference on full test set...")
        batch_results = predictor.predict_batch(
            X_test_eng.reset_index(drop=True), 
            batch_size=100, 
            save_path="data/processed/inference_results.json"
        )
        logger.info(f"✅ Batch inference completed - {batch_results['total_samples']} samples processed in {batch_results['n_batches']} batches")
        
        # Log prediction distribution
        pred_dist = batch_results['prediction_distribution']
        logger.info(f"   Prediction distribution: {pred_dist}")
        
        # Step 9: Model Comparison (if previous model exists)
        logger.info("Step 9: Checking for model comparison...")
        try:
            # Example: Compare with dummy baseline metrics
            baseline_metrics = {
                'accuracy': 0.60,
                'precision': 0.55,
                'recall': 0.50,
                'f1_score': 0.52
            }
            
            comparison = evaluator.compare_models(baseline_metrics, primary_metric='f1_score')
            
            if comparison['is_better']:
                logger.info("🎉 Current model performs better than baseline!")
                logger.info(f"   Improvement in F1-score: {comparison['differences']['f1_score']:.4f}")
            else:
                logger.info("⚠️  Current model performs worse than baseline.")
                logger.info(f"   Decrease in F1-score: {comparison['differences']['f1_score']:.4f}")
                
        except Exception as e:
            logger.warning(f"Model comparison skipped: {str(e)}")
        
        # Pipeline completion summary
        logger.info("🎊 MLOps pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("📋 Pipeline Summary:")
        logger.info(f"   Dataset size: {df.shape}")
        logger.info(f"   Training samples: {len(X_train_eng)}")
        logger.info(f"   Test samples: {len(X_test_eng)}")
        logger.info(f"   Features after engineering: {X_train_eng.shape[1]}")
        logger.info(f"   Best F1-score: {test_metrics.get('f1_score', 'N/A'):.4f}")
        logger.info(f"   Model saved to: {model_path}")
        logger.info(f"   Inference results: data/processed/inference_results.json")
        logger.info("=" * 60)
        
        return {
            'data_shapes': {
                'train': train.shape,
                'validation': val.shape,
                'test': test.shape
            },
            'metrics': test_metrics,
            'model_path': model_path,
            'inference_results': batch_results,
            'comparison': comparison if 'comparison' in locals() else None
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        logger.error("Check the logs above for detailed error information.")
        raise

if __name__ == "__main__":
    results = main()
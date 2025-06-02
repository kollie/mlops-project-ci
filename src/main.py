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
    """Main function to run the data processing pipeline."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting data processing pipeline...")
    
    try:
        # Step 1: Load data
        logger.info("Initializing DataLoader...")
        data_loader = DataLoader(config_path="src/config.yaml")
        logger.info("DataLoader initialized successfully")
        
        logger.info("Loading data...")
        df = data_loader.load_data()
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Step 2: Split data
        logger.info("Splitting data...")
        train, val, test = data_loader.split_data(df)
        logger.info(f"Data split successfully - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        
        # Step 3: Save split data
        logger.info("Saving split datasets...")
        data_loader.save_split_data(train, val, test)
        logger.info("Split datasets saved successfully")

        # Step 4: Validate and clean data
        logger.info("Validating and cleaning data...")
        validator = DataValidator(config_path="src/config.yaml")
        clean_data = validator.validate_and_clean(df, strategy='drop_columns')
        logger.info(f"Data validation completed. Clean data shape: {clean_data.shape}")
        
        # Step 5: Exploratory Data Analysis (EDA)
        logger.info("Performing Exploratory Data Analysis (EDA)...")
        eda_analyzer = EDAAnalyzer(config_path="src/config.yaml")
        eda_report = eda_analyzer.run_full_analysis(clean_data)
        logger.info(f"EDA completed successfully. Report sections: {list(eda_report.keys())}")
        
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

        # Step 6: Preprocessing
        logger.info("Preprocessing training data...")
        preprocessor = Preprocessor(config_path="src/config.yaml")
        X_train, y_train = preprocessor.fit_transform(train)
        logger.info(f"Training data preprocessed. Shape: {X_train.shape}")

        # Transform validation and test sets
        X_val, y_val = preprocessor.transform(val)
        X_test, y_test = preprocessor.transform(test)

        # Save the fitted preprocessor
        preprocessor.save_pipeline("models/preprocessor.joblib")
        
        # Step 7: Feature Engineering
        logger.info("Applying feature engineering...")
        engineer = FeatureEngineer(config_path="src/config.yaml")
        X_train_eng, y_train = engineer.fit_transform(
            pd.concat([X_train, y_train], axis=1), 
            target_col='readmitted'
        )
        logger.info(f"Training data after feature engineering. Shape: {X_train_eng.shape}")

        # Transform validation and test sets
        X_val_eng, y_val = engineer.transform(
            pd.concat([X_val, y_val], axis=1), 
            target_col='readmitted'
        )
        X_test_eng, y_test = engineer.transform(
            pd.concat([X_test, y_test], axis=1), 
            target_col='readmitted'
        )
        # Step 8: Train Model
        logger.info("Training model...")
        trainer = ModelTrainer(config_path="src/config.yaml")
        trainer.fit(X_train_eng, y_train)

        # Step 9: Save Model
        model_path = trainer.save()
        logger.info(f"Model saved to: {model_path}")

        # For inference later:
        predictions = trainer.predict(X_test_eng)
        
        # Step 9: Evaluation (when module exists)
        
        logger.info("Pipeline completed successfully")
        return {
            'train': train,
            'validation': val,
            'test': test
        }
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
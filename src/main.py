import logging
import sys
import os
import pandas as pd
from pathlib import Path
import yaml

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
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "main.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def run_load_data(config, data_source=None):
    data_loader = DataLoader(config_path="conf/config.yaml")
    df = data_loader.load_data(file_path=data_source)
    train, val, test = data_loader.split_data(df)
    data_loader.save_split_data(train, val, test)
    return df, train, val, test


def run_validate_data(config, df):
    validator = DataValidator(config_path="conf/config.yaml")
    clean_data = validator.validate_and_clean(df, strategy="drop_columns")
    return clean_data


def run_eda(config, clean_data):
    eda_analyzer = EDAAnalyzer(config_path="conf/config.yaml")
    eda_report = eda_analyzer.run_full_analysis(clean_data)
    return eda_report


def run_preprocessing(config, train, val, test):
    preprocessor = Preprocessor(config_path="conf/config.yaml")
    X_train, y_train = preprocessor.fit_transform(train)
    X_val, y_val = preprocessor.transform(val)
    X_test, y_test = preprocessor.transform(test)
    preprocessor.save_pipeline("models/preprocessor.joblib")
    return X_train, y_train, X_val, y_val, X_test, y_test


def run_feature_engineering(
    config, X_train, y_train, X_val, y_val, X_test, y_test
):
    engineer = FeatureEngineer(config_path="conf/config.yaml")
    X_train_eng, y_train = engineer.fit_transform(
        pd.concat([X_train, y_train], axis=1), target_col="readmitted"
    )
    X_val_eng, y_val = engineer.transform(
        pd.concat([X_val, y_val], axis=1), target_col="readmitted"
    )
    X_test_eng, y_test = engineer.transform(
        pd.concat([X_test, y_test], axis=1), target_col="readmitted"
    )
    return engineer, X_train_eng, y_train, X_val_eng, y_val, X_test_eng, y_test


def run_train(config, X_train_eng, y_train, engineer):
    trainer = ModelTrainer(config_path="conf/config.yaml")
    trainer.fit(X_train_eng, y_train)
    trainer.set_feature_engineer(engineer)
    model_path = trainer.save()
    return trainer, model_path


def run_evaluate(config, trainer, X_test_eng, y_test):
    evaluator = ModelEvaluator(config_path="conf/config.yaml")
    predictions = trainer.predict(X_test_eng)
    probabilities = trainer.predict_proba(X_test_eng)
    evaluator.evaluate(
        y_true=y_test,
        y_pred=predictions,
        y_pred_proba=probabilities,
        dataset_name="test",
    )
    test_metrics = evaluator.get_metrics()
    
    # Log metrics to W&B if trainer is available
    if trainer is not None:
        trainer.log_evaluation_metrics(test_metrics)
        trainer.log_feature_importance(list(X_test_eng.columns))
    
    metrics_path = evaluator.save_metrics()
    return evaluator, test_metrics, metrics_path


def run_inference(config, model_path, X_test_eng):
    predictor = ModelPredictor(config_path="conf/config.yaml")
    predictor.load_model(model_path)
    test_sample = X_test_eng.head(10).reset_index(drop=True)
    inference_predictions = predictor.predict(test_sample, preprocess=False)
    confidence_results = predictor.predict_with_confidence(
        test_sample, confidence_threshold=0.8, preprocess=False
    )
    batch_results = predictor.predict_batch(
        X_test_eng.reset_index(drop=True),
        batch_size=100,
        save_path="data/processed/inference_results.json",
    )
    return predictor, inference_predictions, confidence_results, batch_results


def run_compare(config, evaluator, trainer, test_metrics):
    baseline_metrics = {
        "accuracy": 0.55,
        "precision": 0.45,
        "recall": 0.40,
        "f1_score": 0.42,
    }
    try:
        comparison = evaluator.compare_models(
            baseline_metrics, primary_metric="f1_score"
        )
        trainer.log_evaluation_metrics({
            "baseline_f1": baseline_metrics["f1_score"],
            "current_f1": test_metrics.get("f1_score", 0),
            "improvement": comparison["differences"]["f1_score"]
        })
    except Exception as e:
        comparison = None
    return comparison


STEP_FUNCTIONS = {
    "load_data": run_load_data,
    "validate_data": run_validate_data,
    "eda": run_eda,
    "preprocess": run_preprocessing,
    "feature_engineering": run_feature_engineering,
    "train": run_train,
    "evaluate": run_evaluate,
    "inference": run_inference,
    "compare": run_compare,
}


def main(data_source: str = None):
    """Main function to run the complete ML pipeline. Accepts a data_source file path."""
    # Setup logging
    logger = setup_logging()
    logger.info("🚀 Starting MLOps pipeline...")

    # Load steps from config.yaml
    with open("conf/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    steps_to_run = config.get("main", {}).get("steps", [])

    try:
        # Step 1: Load and split data
        if "load_data" in steps_to_run:
            logger.info("Step 1: Loading and splitting data...")
            df, train, val, test = run_load_data(config, data_source)
            logger.info(
                f"✅ Data loaded and split - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}"
            )
        else:
            df = train = val = test = None

        # Step 2: Validate and clean data
        if "validate_data" in steps_to_run:
            logger.info("Step 2: Validating and cleaning data...")
            clean_data = run_validate_data(config, df)
            logger.info(
                f"✅ Data validation completed - Clean data shape: {clean_data.shape}"
            )
        else:
            clean_data = df

        # Step 3: Exploratory Data Analysis
        if "eda" in steps_to_run:
            logger.info("Step 3: Performing Exploratory Data Analysis...")
            eda_report = run_eda(config, clean_data)
            logger.info(f"✅ EDA completed - Report sections: {list(eda_report.keys())}")

            # Log key EDA insights
            if "summary" in eda_report:
                summary = eda_report["summary"]
                logger.info("📊 EDA Summary:")
                logger.info(f"   Total features: {summary.get('total_features', 'N/A')}")
                logger.info(
                    f"   Numeric features: {summary.get('numeric_features', 'N/A')}"
                )
                logger.info(
                    f"   Categorical features: {summary.get('categorical_features', 'N/A')}"
                )
                logger.info(
                    f"   Missing data: {summary.get('missing_data_percentage', 'N/A')}%"
                )

            if (
                "target_analysis" in eda_report
                and "error" not in eda_report["target_analysis"]
            ):
                target_analysis = eda_report["target_analysis"]
                logger.info("🎯 Target Analysis:")
                logger.info(
                    f"   Target column: {target_analysis.get('target_column', 'N/A')}"
                )
                logger.info(
                    f"   Class distribution: {target_analysis.get('percentages', {})}"
                )
                logger.info(
                    f"   Balanced: {target_analysis.get('class_balance', {}).get('is_balanced', 'N/A')}"
                )

        # Step 4: Preprocessing
        if "preprocess" in steps_to_run:
            logger.info("Step 4: Preprocessing data...")
            X_train, y_train, X_val, y_val, X_test, y_test = run_preprocessing(config, train, val, test)
            logger.info(f"✅ Data preprocessed - Training shape: {X_train.shape}")
        else:
            X_train = y_train = X_val = y_val = X_test = y_test = None

        # Step 5: Feature Engineering
        if "feature_engineering" in steps_to_run:
            logger.info("Step 5: Applying feature engineering...")
            engineer, X_train_eng, y_train, X_val_eng, y_val, X_test_eng, y_test = run_feature_engineering(config, X_train, y_train, X_val, y_val, X_test, y_test)
            logger.info(
                f"✅ Feature engineering completed - Final shape: {X_train_eng.shape}"
            )
        else:
            X_train_eng = X_val_eng = X_test_eng = None

        # Step 6: Train Model
        if "train" in steps_to_run:
            logger.info("Step 6: Training model...")
            trainer, model_path = run_train(config, X_train_eng, y_train, engineer)
            logger.info(f"✅ Model trained and saved to: {model_path}")
        else:
            trainer = model_path = None

        # Step 7: Evaluate Model
        if "evaluate" in steps_to_run:
            logger.info("Step 7: Evaluating model...")
            evaluator, test_metrics, metrics_path = run_evaluate(config, trainer, X_test_eng, y_test)
            logger.info("📈 Test Results:")
            for metric, value in test_metrics.items():
                logger.info(f"   {metric}: {value:.4f}")
            logger.info(f"✅ Evaluation completed - Metrics saved to: {metrics_path}")
        else:
            evaluator = test_metrics = metrics_path = None

        # Step 8: Test Inference
        if "inference" in steps_to_run:
            logger.info("Step 8: Testing model inference...")
            predictor, inference_predictions, confidence_results, batch_results = run_inference(config, model_path, X_test_eng)
            logger.info(
                f"✅ Basic inference completed - {len(inference_predictions)} predictions made"
            )
            logger.info("📊 Confidence Analysis:")
            logger.info(
                f"   High confidence predictions: {confidence_results['high_confidence_count']}/{len(inference_predictions)}"
            )
            logger.info(
                f"   Mean confidence: {confidence_results['confidence_stats']['mean_confidence']:.4f}"
            )
            logger.info("Testing batch inference on full test set...")
            logger.info(
                f"✅ Batch inference completed - {batch_results['total_samples']} samples processed in {batch_results['n_batches']} batches"
            )
            pred_dist = batch_results["prediction_distribution"]
            logger.info(f"   Prediction distribution: {pred_dist}")
        else:
            predictor = batch_results = None

        # Step 9: Model Comparison (if previous model exists)
        if "compare" in steps_to_run:
            logger.info("Step 9: Checking for model comparison...")
            try:
                comparison = run_compare(config, evaluator, trainer, test_metrics)
                if comparison["is_better"]:
                    logger.info("🎉 Current model performs better than baseline!")
                    logger.info(
                        f"   Improvement in F1-score: {comparison['differences']['f1_score']:.4f}"
                    )
                else:
                    logger.info("⚠️  Current model performs worse than baseline.")
                    logger.info(
                        f"   Decrease in F1-score: {comparison['differences']['f1_score']:.4f}"
                    )
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
        logger.info("   Inference results: data/processed/inference_results.json")
        logger.info("=" * 60)

        # Finish wandb run
        trainer.finish_wandb()

        return {
            "data_shapes": {
                "train": train.shape,
                "validation": val.shape,
                "test": test.shape,
            },
            "metrics": test_metrics,
            "model_path": model_path,
            "inference_results": batch_results,
            "comparison": comparison if "comparison" in locals() else None,
            "data_source": data_source if data_source else config["data"].get("raw_data_path"),
        }

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        logger.error("Check the logs above for detailed error information.")
        
        # Ensure wandb run is finished even on failure
        if 'trainer' in locals():
            trainer.finish_wandb()
        
        raise


if __name__ == "__main__":
    results = main()

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import mlflow
import logging
import pandas as pd
from pathlib import Path

# Import pipeline modules desde src
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
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "main.log"), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


@hydra.main(config_path="src", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = setup_logging()
    logger.info("Starting MLOps pipeline...")

    # Set MLflow experiment
    mlflow.set_experiment("diabetic_readmission_pipeline")
    with mlflow.start_run(run_name="full_pipeline"):
        # Log config.yaml usado
        mlflow.log_dict(OmegaConf.to_container(cfg), "config.yaml")

        # 1. Data Loading
        logger.info("Step 1: Loading and splitting data...")
        data_loader = DataLoader(config_path=to_absolute_path("src/config.yaml"))
        df = data_loader.load_data(file_path=to_absolute_path(cfg.data.raw_data_path))
        train, val, test = data_loader.split_data(df)
        data_loader.save_split_data(train, val, test)
        logger.info(f"Data loaded and split - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        mlflow.log_metric("n_train_samples", train.shape[0])
        mlflow.log_metric("n_val_samples", val.shape[0])
        mlflow.log_metric("n_test_samples", test.shape[0])
        # Log artifacts de data loader
        for artifact in [
            "logs/data_loader_report.json",
            "data/processed/train.csv",
            "data/processed/val.csv",
            "data/processed/test.csv",
        ]:
            path = to_absolute_path(artifact)
            if Path(path).exists():
                mlflow.log_artifact(path)

        # 2. Data Validation
        logger.info("Step 2: Validating and cleaning data...")
        validator = DataValidator(config_path=to_absolute_path("src/config.yaml"))
        clean_data = validator.validate_and_clean(df, strategy="drop_columns")
        logger.info(f"Data validation completed - Clean data shape: {clean_data.shape}")
        # Log artifacts de validación
        for artifact in [
            "logs/validation_report.json",
            "data/processed/clean_data.csv",
        ]:
            path = to_absolute_path(artifact)
            if Path(path).exists():
                mlflow.log_artifact(path)

        # 3. EDA
        logger.info("Step 3: Performing Exploratory Data Analysis...")
        eda_analyzer = EDAAnalyzer(config_path=to_absolute_path("src/config.yaml"))
        eda_report = eda_analyzer.run_full_analysis(clean_data)
        logger.info(f"EDA completed - Report sections: {list(eda_report.keys())}")
        if "summary" in eda_report:
            summary = eda_report["summary"]
            logger.info("EDA Summary:")
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
            logger.info("Target Analysis:")
            logger.info(
                f"   Target column: {target_analysis.get('target_column', 'N/A')}"
            )
            logger.info(
                f"   Class distribution: {target_analysis.get('percentages', {})}"
            )
            logger.info(
                f"   Balanced: {target_analysis.get('class_balance', {}).get('is_balanced', 'N/A')}"
            )
        # Log artifacts de EDA
        for artifact in [
            "logs/eda_report.json",
            "plots/eda_plot.png",
            "plots/feature_distributions.png",
            "plots/correlation_matrix.png",
        ]:
            path = to_absolute_path(artifact)
            if Path(path).exists():
                mlflow.log_artifact(path)

        # 4. Preprocessing
        logger.info("Step 4: Preprocessing data...")
        preprocessor = Preprocessor(config_path=to_absolute_path("src/config.yaml"))
        X_train, y_train = preprocessor.fit_transform(train)
        X_val, y_val = preprocessor.transform(val)
        X_test, y_test = preprocessor.transform(test)
        preprocessor.save_pipeline(to_absolute_path("models/preprocessor.joblib"))
        logger.info(f"Data preprocessed - Training shape: {X_train.shape}")
        # Log artifacts de preprocesado
        for artifact in [
            "models/preprocessor.joblib",
            "logs/preprocessing_report.json",
        ]:
            path = to_absolute_path(artifact)
            if Path(path).exists():
                mlflow.log_artifact(path)

        # 5. Feature Engineering
        logger.info("Step 5: Applying feature engineering...")
        engineer = FeatureEngineer(config_path=to_absolute_path("src/config.yaml"))
        X_train_eng, y_train = engineer.fit_transform(pd.concat([X_train, y_train], axis=1), target_col=cfg.features.target_column)
        X_val_eng, y_val = engineer.transform(pd.concat([X_val, y_val], axis=1), target_col=cfg.features.target_column)
        X_test_eng, y_test = engineer.transform(pd.concat([X_test, y_test], axis=1), target_col=cfg.features.target_column)
        logger.info(f"Feature engineering completed - Final shape: {X_train_eng.shape}")
        # Log artifacts de feature engineering
        for artifact in [
            "logs/feature_engineering_report.json",
        ]:
            path = to_absolute_path(artifact)
            if Path(path).exists():
                mlflow.log_artifact(path)

        # 6. Model Training
        logger.info("Step 6: Training model...")
        trainer = ModelTrainer(config_path=to_absolute_path("src/config.yaml"))
        trainer.fit(X_train_eng, y_train)
        model_path = trainer.save(to_absolute_path("models/trained_model.joblib"))
        logger.info(f"Model trained and saved to: {model_path}")
        mlflow.log_artifact(to_absolute_path("models/trained_model.joblib"))
        # Log artifacts de training
        for artifact in [
            "logs/training_report.json",
        ]:
            path = to_absolute_path(artifact)
            if Path(path).exists():
                mlflow.log_artifact(path)

        # 7. Model Evaluation
        logger.info("Step 7: Evaluating model...")
        evaluator = ModelEvaluator(config_path=to_absolute_path("src/config.yaml"))
        predictions = trainer.predict(X_test_eng)
        probabilities = trainer.predict_proba(X_test_eng)
        evaluation_results = evaluator.evaluate(
            y_true=y_test,
            y_pred=predictions,
            y_pred_proba=probabilities,
            dataset_name="test",
        )
        test_metrics = evaluator.get_metrics()
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
            logger.info(f"   {metric}: {value:.4f}")
        metrics_path = evaluator.save_metrics()
        logger.info(f"Evaluation completed - Metrics saved to: {metrics_path}")
        if metrics_path:
            mlflow.log_artifact(to_absolute_path(metrics_path))
        # Log artifacts de evaluación
        for artifact in [
            "logs/evaluation_report.json",
            "plots/roc_curve.png",
            "plots/pr_curve.png",
            "plots/confusion_matrix.png",
        ]:
            path = to_absolute_path(artifact)
            if Path(path).exists():
                mlflow.log_artifact(path)

        # 8. Inference
        logger.info("Step 8: Testing model inference...")
        predictor = ModelPredictor(config_path=to_absolute_path("src/config.yaml"))
        predictor.load_model(model_path)
        test_sample = X_test_eng.head(10).reset_index(drop=True)
        inference_predictions = predictor.predict(test_sample, preprocess=False)
        logger.info(f"Basic inference completed - {len(inference_predictions)} predictions made")
        confidence_results = predictor.predict_with_confidence(
            test_sample, confidence_threshold=0.8, preprocess=False
        )
        logger.info(f"   High confidence predictions: {confidence_results['high_confidence_count']}/{len(test_sample)}")
        logger.info(f"   Mean confidence: {confidence_results['confidence_stats']['mean_confidence']:.4f}")
        batch_results = predictor.predict_batch(
            X_test_eng.reset_index(drop=True),
            batch_size=100,
            save_path=to_absolute_path("data/processed/inference_results.json"),
        )
        logger.info(f"Batch inference completed - {batch_results['total_samples']} samples processed in {batch_results['n_batches']} batches")
        mlflow.log_artifact(to_absolute_path("data/processed/inference_results.json"))
        # Log artifacts de inference
        for artifact in [
            "logs/inference_report.json",
        ]:
            path = to_absolute_path(artifact)
            if Path(path).exists():
                mlflow.log_artifact(path)

        # 9. Model Comparison (opcional)
        logger.info("Step 9: Checking for model comparison...")
        try:
            baseline_metrics = {
                "accuracy": 0.60,
                "precision": 0.55,
                "recall": 0.50,
                "f1_score": 0.52,
            }
            comparison = evaluator.compare_models(baseline_metrics, primary_metric="f1_score")
            if comparison["is_better"]:
                logger.info("Current model performs better than baseline!")
                logger.info(
                    f"   Improvement in F1-score: {comparison['differences']['f1_score']:.4f}"
                )
            else:
                logger.info("Current model performs worse than baseline.")
                logger.info(
                    f"   Decrease in F1-score: {comparison['differences']['f1_score']:.4f}"
                )
        except Exception as e:
            logger.warning(f"Model comparison skipped: {str(e)}")

        # Pipeline completion summary
        logger.info("MLOps pipeline completed successfully!")
        logger.info("=" * 60)
        logger.info("Pipeline Summary:")
        logger.info(f"   Dataset size: {df.shape}")
        logger.info(f"   Training samples: {len(X_train_eng)}")
        logger.info(f"   Test samples: {len(X_test_eng)}")
        logger.info(f"   Features after engineering: {X_train_eng.shape[1]}")
        logger.info(f"   Best F1-score: {test_metrics.get('f1_score', 'N/A'):.4f}")
        logger.info(f"   Model saved to: {model_path}")
        logger.info("   Inference results: data/processed/inference_results.json")
        logger.info("=" * 60)

if __name__ == "__main__":
    main() 
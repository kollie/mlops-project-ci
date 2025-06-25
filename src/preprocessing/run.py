import argparse
import mlflow
import pandas as pd
from preprocessor import Preprocessor
import os

def run_preprocessing(config_path, run_name, log_to_mlflow=True):
    """
    Run the preprocessing pipeline and log parameters, metrics, and artifacts to MLflow.
    If log_to_mlflow is True, starts a run if not already active (supports nested runs).
    """
    def _run():
        preprocessor = Preprocessor(config_path=config_path)
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("run_name", run_name)
        # Load training data
        train_path = preprocessor.config["data"]["train_data_path"]
        mlflow.log_param("train_data_path", train_path)
        df = pd.read_csv(train_path)
        target_col = preprocessor.config.get("features", {}).get("target_column", "readmitted")
        # Fit and transform
        X, y = preprocessor.fit_transform(df, target_col=target_col)
        # Save pipeline
        pipeline_path = "models/preprocessor.joblib"
        preprocessor.save_pipeline(pipeline_path)
        mlflow.log_param("pipeline_path", pipeline_path)
        # Log key metrics from report
        report = preprocessor.report
        mlflow.log_metrics({
            "numerical_features": report.get("numerical_features_count", 0),
            "categorical_features": report.get("categorical_features_count", 0),
            "final_features": report.get("final_shape", (0, 0))[1]
        })
        # Log preprocessing report
        if os.path.exists("logs/preprocessing_report.json"):
            mlflow.log_artifact("logs/preprocessing_report.json")
        # Log pipeline and config files if exist
        if os.path.exists(pipeline_path):
            mlflow.log_artifact(pipeline_path)
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
    experiment_name = "preprocessing_experiment"
    if log_to_mlflow:
        mlflow.set_experiment(experiment_name)
        if mlflow.active_run() is None:
            with mlflow.start_run(run_name=run_name):
                _run()
        else:
            _run()
    else:
        _run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="conf/config.yaml", help="Path to config file")
    parser.add_argument("--run_name", type=str, default="preprocessing_run", help="Name for MLflow run")
    args = parser.parse_args()
    run_preprocessing(args.config_path, args.run_name, log_to_mlflow=True)

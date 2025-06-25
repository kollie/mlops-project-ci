import argparse
import mlflow
import pandas as pd
from feature_engineering import FeatureEngineer
import os

def run_feature_engineering(config_path, run_name, log_to_mlflow=True):
    """
    Run the feature engineering pipeline and log parameters, metrics, and artifacts to MLflow.
    If log_to_mlflow is True, starts a run if not already active (supports nested runs).
    """
    def _run():
        engineer = FeatureEngineer(config_path=config_path)
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("run_name", run_name)
        # Load training data
        train_path = engineer.config["data"]["train_data_path"]
        mlflow.log_param("train_data_path", train_path)
        df = pd.read_csv(train_path)
        target_col = engineer.config.get("features", {}).get("target_column", "readmitted")
        # Run feature engineering
        X, y = engineer.fit_transform(df, target_col=target_col)
        # Log key metrics from report
        report = engineer.report
        mlflow.log_metrics({
            "original_features": len(report.get("original_features", [])),
            "engineered_features": len(report.get("engineered_features", [])),
            "selected_features": len(report.get("selected_features", [])),
            "final_features": report.get("features_after_selection", 0)
        })
        # Log feature engineering report
        if os.path.exists("logs/feature_engineering_report.json"):
            mlflow.log_artifact("logs/feature_engineering_report.json")
        # Log config file if exists
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
    experiment_name = "feature_engineering_experiment"
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
    parser.add_argument("--run_name", type=str, default="feature_engineering_run", help="Name for MLflow run")
    args = parser.parse_args()
    run_feature_engineering(args.config_path, args.run_name, log_to_mlflow=True)

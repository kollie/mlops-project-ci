import argparse
import mlflow
import pandas as pd
from eda import EDAAnalyzer
from pathlib import Path
import os

def run_eda(config_path, run_name, log_to_mlflow=True):
    """
    Run the EDA pipeline and log parameters, metrics, and artifacts to MLflow.
    If log_to_mlflow is True, starts a run if not already active (supports nested runs).
    """
    def _run():
        analyzer = EDAAnalyzer(config_path=config_path)
        experiment_name = analyzer.config["logging"].get("mlflow_experiment", "eda_experiment")
        # Log config parameters
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("experiment_name", experiment_name)
        # Load dataset
        data_path = analyzer.config["data"]["train_data_path"]
        mlflow.log_param("data_path", data_path)
        df = pd.read_csv(data_path)
        # Run EDA
        report = analyzer.run_full_analysis(df)
        # Log key metrics from report summary
        summary = report.get("summary", {})
        mlflow.log_metrics({
            "total_features": summary.get("total_features", 0),
            "numeric_features": summary.get("numeric_features", 0),
            "categorical_features": summary.get("categorical_features", 0),
            "missing_data_percentage": summary.get("missing_data_percentage", 0.0)
        })
        # Log generated plots
        for path in Path("plots").rglob("*.*"):
            if path.is_file():
                mlflow.log_artifact(str(path))
        # Log final EDA report
        if os.path.exists("logs/eda_report.json"):
            mlflow.log_artifact("logs/eda_report.json")
        # Log config file if exists
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)

    if log_to_mlflow:
        analyzer = EDAAnalyzer(config_path=config_path)
        experiment_name = analyzer.config["logging"].get("mlflow_experiment", "eda_experiment")
        mlflow.set_experiment(experiment_name)
        if mlflow.active_run() is None:
            with mlflow.start_run(run_name=run_name):
                _run()
        else:
            _run()
    else:
        _run()

if __name__ == "__main__":
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="src/config.yaml", help="Path to config file")
    parser.add_argument("--run_name", type=str, default="eda_run", help="Name for MLflow run")
    args = parser.parse_args()
    run_eda(args.config_path, args.run_name, log_to_mlflow=True)

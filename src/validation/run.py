import argparse
import mlflow
import pandas as pd
from data_validator import DataValidator
import os

def run_validation(config_path, data_path, run_name, clean=False, log_to_mlflow=True):
    """
    Run the data validation pipeline and log parameters, metrics, and artifacts to MLflow.
    If log_to_mlflow is True, starts a run if not already active (supports nested runs).
    """
    def _run():
        validator = DataValidator(config_path=config_path)
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("clean", clean)
        # Load data
        df = pd.read_csv(data_path)
        # Run validation (with or without cleaning)
        if clean:
            df_cleaned = validator.validate_and_clean(df)
            # Optionally save cleaned data
            cleaned_path = data_path.replace('.csv', '_cleaned.csv')
            df_cleaned.to_csv(cleaned_path, index=False)
            mlflow.log_param("cleaned_data_path", cleaned_path)
            if os.path.exists(cleaned_path):
                mlflow.log_artifact(cleaned_path)
        else:
            validator.validate_all(df)
        # Log key metrics from report
        report = validator.report
        mlflow.log_metrics({
            "initial_rows": report.get("initial_rows", 0),
            "initial_columns": report.get("initial_columns", 0),
            "final_rows": report.get("final_rows", report.get("initial_rows", 0)),
            "final_columns": report.get("final_columns", report.get("initial_columns", 0)),
            "rows_dropped": report.get("rows_dropped", 0),
            "columns_dropped": report.get("columns_dropped", 0),
        })
        # Log validation report
        if os.path.exists("logs/validation_report.json"):
            mlflow.log_artifact("logs/validation_report.json")
        # Log config file if exists
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
    experiment_name = "validation_experiment"
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
    parser.add_argument("--data_path", type=str, required=True, help="Path to data file to validate")
    parser.add_argument("--run_name", type=str, default="validation_run", help="Name for MLflow run")
    parser.add_argument("--clean", action="store_true", help="Whether to clean data after validation")
    args = parser.parse_args()
    run_validation(args.config_path, args.data_path, args.run_name, clean=args.clean, log_to_mlflow=True)

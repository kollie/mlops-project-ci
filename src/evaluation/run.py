import argparse
import mlflow
import pandas as pd
from evaluator import ModelEvaluator
import os

def run_evaluation(config_path, run_name, log_to_mlflow=True):
    """
    Run the model evaluation pipeline and log parameters, metrics, and artifacts to MLflow.
    If log_to_mlflow is True, starts a run if not already active (supports nested runs).
    """
    def _run():
        evaluator = ModelEvaluator(config_path=config_path)
        # Log config parameters
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("run_name", run_name)
        # Load predictions and true values
        y_true = pd.read_csv(evaluator.config["data"]["y_test_path"]).values.ravel()
        y_pred = pd.read_csv(evaluator.config["data"]["y_pred_path"]).values.ravel()
        y_pred_proba = None
        if "y_pred_proba_path" in evaluator.config["data"]:
            y_pred_proba = pd.read_csv(evaluator.config["data"]["y_pred_proba_path"]).values
        # Run evaluation
        results = evaluator.evaluate(y_true, y_pred, y_pred_proba)
        # Log main metrics
        mlflow.log_metrics(results["metrics"])
        # Log evaluation report
        if os.path.exists("logs/evaluation_report.json"):
            mlflow.log_artifact("logs/evaluation_report.json")
        # Log config file if exists
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
    # Set experiment name
    experiment_name = "evaluation_experiment"
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
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="conf/config.yaml", help="Path to YAML config")
    parser.add_argument("--run_name", type=str, default="evaluation_run", help="MLflow run name")
    args = parser.parse_args()
    run_evaluation(args.config_path, args.run_name, log_to_mlflow=True)

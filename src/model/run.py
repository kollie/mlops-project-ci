import argparse
import mlflow
import pandas as pd
from trainer import ModelTrainer
import os

def run_training(config_path, run_name, log_to_mlflow=True):
    """
    Run the model training pipeline and log parameters, metrics, and artifacts to MLflow.
    If log_to_mlflow is True, starts a run if not already active (supports nested runs).
    """
    def _run():
        trainer = ModelTrainer(config_path=config_path)
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("run_name", run_name)
        # Load training data
        train_path = trainer.config["data"]["train_data_path"]
        mlflow.log_param("train_data_path", train_path)
        df = pd.read_csv(train_path)
        target_col = trainer.config.get("features", {}).get("target_column", "readmitted")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        # Train model
        trainer.fit(X, y)
        # Save model
        model_dir = trainer.config["data"].get("model_path", "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{run_name}_model.pkl")
        trainer.save(model_path)
        mlflow.log_param("model_path", model_path)
        # Log training report
        if os.path.exists("logs/training_report.json"):
            mlflow.log_artifact("logs/training_report.json")
        # Log config and model files if exist
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
    experiment_name = "model_training_experiment"
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
    parser.add_argument("--run_name", type=str, default="model_training_run", help="Name for MLflow run")
    args = parser.parse_args()
    run_training(args.config_path, args.run_name, log_to_mlflow=True)

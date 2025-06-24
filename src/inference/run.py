import argparse
import mlflow
import pandas as pd
from predict import ModelPredictor
import os

def run_inference(config_path, model_path, data_path, run_name, batch_size=1000, log_to_mlflow=True):
    """
    Run the inference pipeline and log parameters, metrics, and artifacts to MLflow.
    If log_to_mlflow is True, starts a run if not already active (supports nested runs).
    """
    def _run():
        predictor = ModelPredictor(config_path=config_path, model_path=model_path)
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("batch_size", batch_size)
        # Load input data
        df = pd.read_csv(data_path)
        # Run batch prediction
        results = predictor.predict_batch(df, batch_size=batch_size, save_path="data/predictions/inference_results.json")
        # Log key metrics
        mlflow.log_metrics({
            "total_samples": results.get("total_samples", 0),
            "n_batches": results.get("n_batches", 0),
            "batch_size": results.get("batch_size", batch_size),
            "unique_predictions": len(results.get("prediction_distribution", {})),
        })
        # Log prediction distribution as params
        for label, count in results.get("prediction_distribution", {}).items():
            mlflow.log_param(f"pred_count_{label}", count)
        # Log inference report
        if os.path.exists("logs/inference_report.json"):
            mlflow.log_artifact("logs/inference_report.json")
        # Log predictions file
        if os.path.exists("data/predictions/inference_results.json"):
            mlflow.log_artifact("data/predictions/inference_results.json")
        # Log config and model files if exist
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)
        if os.path.exists(model_path):
            mlflow.log_artifact(model_path)
    experiment_name = "inference_experiment"
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
    parser.add_argument("--config_path", type=str, default="src/config.yaml", help="Path to config file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data for inference")
    parser.add_argument("--run_name", type=str, default="inference_run", help="Name for MLflow run")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for inference")
    args = parser.parse_args()
    run_inference(args.config_path, args.model_path, args.data_path, args.run_name, args.batch_size, log_to_mlflow=True)

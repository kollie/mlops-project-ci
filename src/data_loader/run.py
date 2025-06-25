import argparse
from data_loader import DataLoader
import mlflow
import os

def run_data_loader(config_path, run_name, log_to_mlflow=True):
    
    def _run():
        loader = DataLoader(config_path=config_path)
        df = loader.load_data()
        train, val, test = loader.split_data(df)
        loader.save_split_data(train, val, test)

        # Log principal parameters
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("run_name", run_name)
        mlflow.log_param("test_size", loader.config["model"].get("test_size"))
        mlflow.log_param("validation_size", loader.config["model"].get("validation_size"))
        mlflow.log_param("random_state", loader.config["model"].get("random_state"))

        # Log métricmetrics
        mlflow.log_metrics({
            "train_size": len(train),
            "val_size": len(val),
            "test_size": len(test)
        })

        # Log artifacts
        mlflow.log_artifact(loader.config["data"]["train_data_path"])
        mlflow.log_artifact(loader.config["data"]["validation_data_path"])
        mlflow.log_artifact(loader.config["data"]["test_data_path"])
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path)

    if log_to_mlflow:
        mlflow.set_experiment("data_loader_experiment")
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
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()
    run_data_loader(args.config_path, args.run_name, log_to_mlflow=True)

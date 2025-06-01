
import pytest
import pandas as pd
from src.model.trainer import Trainer

@pytest.fixture
def mock_config():
    return {
        "data": {"model_path": "/tmp/models"},
        "features": {"target_column": "target"},
        "model": {
            "active": "random_forest",
            "random_forest": {
                "n_estimators": 10,
                "max_depth": 5,
                "random_state": 42
            }
        },
        "preprocessing": {
            "numerical_features": ["num1", "num2"],
            "categorical_features": ["cat1"],
            "drop_columns": []
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "file": None
        }
    }

@pytest.fixture
def mock_training_data():
    return pd.DataFrame({
        "num1": [1, 2, 3, 4],
        "num2": [4, 3, 2, 1],
        "cat1": ["A", "B", "A", "B"],
        "target": [0, 1, 0, 1]
    })

def test_trainer_pipeline_runs(mock_config, mock_training_data, tmp_path):
    # Arrange
    mock_config["data"]["model_path"] = str(tmp_path)
    trainer = Trainer(config=mock_config)

    # Act
    trainer.train(mock_training_data)
    trainer.save_model("test_model.pkl")

    # Assert
    model_file = tmp_path / "test_model.pkl"
    assert model_file.exists(), "Model file was not saved successfully."


def test_training_logs_shape(mock_config, mock_training_data, caplog):
    trainer = Trainer(config=mock_config)
    with caplog.at_level("INFO"):
        trainer.train(mock_training_data)
    assert any("Training data shape" in msg for msg in caplog.messages)

def test_model_save_failure(mock_config, mock_training_data):
    trainer = Trainer(config=mock_config)
    trainer.train(mock_training_data)
    with mock.patch("joblib.dump", side_effect=Exception("Save error")), pytest.raises(Exception):
        trainer.save_model("fail_model.pkl")

def test_missing_target_column(mock_config, mock_training_data):
    trainer = Trainer(config=mock_config)
    mock_training_data.drop(columns=["target"], inplace=True)
    with pytest.raises(KeyError):
        trainer.train(mock_training_data)


# Diabetic Readmission Prediction MLOps Project

This project implements a machine learning pipeline for predicting diabetic patient readmission using MLOps best practices.

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── diabetic_readmission_data.csv
│   ├── processed/
│   └── test/
├── src/
│   ├── data_loader/
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── preprocessor.py
│   ├── validation/
│   │   ├── __init__.py
│   │   └── validator.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── train.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── config.yaml
│   └── main.py
├── models/
├── logs/
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create necessary directories:

```bash
mkdir -p data/{raw,processed,test} models logs
```

4. Place your dataset in `data/raw/diabetic_readmission_data.csv`

## Running the Pipeline

To run the complete ML pipeline:

```bash
python src/main.py
```

The pipeline will:

1. Load and validate the data
2. Split and preprocess the data
3. Train the model
4. Evaluate the model
5. Make predictions

## Output

The pipeline generates several outputs:

- Processed data in `data/processed/`
- Trained model in `models/`
- Evaluation metrics in `validation_metrics.csv` and `test_metrics.csv`
- Predictions in `predictions.csv`
- Logs in `logs/pipeline.log`

## MLflow Tracking

The project uses MLflow for experiment tracking. To view the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Configuration

The pipeline configuration can be modified in `src/config.yaml`. Key parameters include:

- Data paths
- Model parameters
- Feature engineering settings
- Evaluation metrics
- Logging configuration

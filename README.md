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

# Code comments

1. `data_loader.py` – Loads, splits, and saves data
2. `data_validator.py` – Validates schema, types, completeness, and cleans the data
3. `validator.py` – Lightweight validation for fast pre-checks (e.g., in CI)

The pipeline uses `config.yaml` to control paths, expectations, and logging.

## Scripts Overview

### `data_loader.py`

**Location:** `src/data_loader/data_loader.py`

**Responsibilities:**

- Load dataset from a local path or Google Drive link (via config)
- Validate and clean the data (by calling `DataValidator`)
- Split data into train/validation/test using config-defined ratios
- Save processed outputs to defined paths

**Key Features:**

- Reads file type (`csv`, `excel`) from config
- Logs every step and outcome
- Drops nothing by itself – relies on the validator
- Safe default fallback behavior (if configured)

### `data_validator.py`

**Location:** `src/validation/data_validator.py`

**Responsibilities:**

- Validate schema, data types, and target distribution
- Drop rows with missing values
- Log issues, write full validation report to `logs/validation_report.json`

**Key Features:**

- Uses fallback: treats `'age'` as `'age_group'` if needed
- Entirely driven by `config.yaml`
- Logs warnings and errors
- Designed for full pipeline integration

### `validator.py`

**Location:** `src/validation/validator.py`

**Responsibilities:**

- Lightweight validation for use in fast CI checks
- Boolean-based: schema, types, missing values, target balance
- No cleaning or mutation — just checks

**Key Features:**

- Very fast and simple
- Good for test suites or minimal CLI usage
- Logs only issues, does not raise exceptions

## Tests Overview

Each script has a dedicated test suite under `tests/`.

### `tests/test_data_loader.py`

**Tests the following:**

- Loading data from Google Drive (or fallback to mock file)
- Splitting data into train/val/test with correct proportions
- Saving datasets to correct config-defined paths
- Using `tmp_path` to avoid real file system side effects

### `tests/test_data_validation.py`

**Tests the following:**

- Cleans rows with missing values
- Detects type mismatches and raises `TypeError`
- Fails on missing required columns (`ValueError`)
- Fails on target distribution imbalance
- Passes full validation on clean data

### `tests/test_validator.py`

**Tests the following:**

- Schema validation with fallback to `age` if `age_group` is missing
- Detects missing columns and fails cleanly
- Accepts correct numeric and categorical types
- Detects type mismatches
- Validates acceptable missing value ratios
- Detects high-missing columns
- Checks target distribution balance
- Returns expected booleans from `validate_all()`

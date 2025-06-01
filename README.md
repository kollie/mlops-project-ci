# MLOps Group Project

This project implements a machine learning pipeline for predicting hospital readmission rates using patient data.

## Project Structure

```
.
├── data/
│   ├── raw/               # Raw data files
│   ├── processed/         # Processed data files
│   └── interim/          # Intermediate data files
├── logs/                 # Log files
├── models/              # Trained models
├── notebooks/           # Jupyter notebooks
├── plots/              # Generated plots
├── src/
│   ├── data/           # Data loading and validation
│   ├── eda/            # Exploratory Data Analysis
│   ├── preprocessing/  # Data preprocessing
│   └── main.py         # Main pipeline script
└── tests/              # Test files
```

## Components

### 1. Data Loading (`src/data/`)

- Handles data loading from various sources
- Implements data validation
- Manages data splitting into train/validation/test sets

### 2. Exploratory Data Analysis (`src/eda/`)

The EDA module provides comprehensive data analysis capabilities:

#### Features

- **Data Description**

  - Generates descriptive statistics
  - Analyzes data types and distributions
  - Identifies missing values and outliers

- **Target Analysis**

  - Analyzes target variable distribution
  - Generates visualizations for target patterns
  - Calculates class imbalance metrics

- **Feature Analysis**

  - Analyzes numerical and categorical features
  - Generates distribution plots
  - Identifies correlations between features

- **Missing Value Analysis**

  - Detects missing values and placeholders
  - Analyzes patterns in missing data
  - Provides strategies for handling missing values

- **Low Importance Feature Detection**
  - Identifies features with low predictive power
  - Calculates feature importance scores
  - Recommends features for removal

#### Outputs

- Detailed analysis reports in `logs/eda.log`
- Visualizations saved in `plots/` directory
- Processed DataFrame ready for preprocessing

### 3. Preprocessing (`src/preprocessing/`)

The preprocessing module implements a comprehensive data preprocessing pipeline:

#### Features

- **Missing Value Handling**

  - Replaces placeholders with NaN
  - Imputes missing values using appropriate strategies
  - Handles both numerical and categorical features

- **Feature Engineering**

  - Encodes categorical variables
  - Scales numerical features
  - Creates interaction features

- **Feature Selection**

  - Selects most important features
  - Removes low-importance features
  - Maintains feature interpretability

- **Target Processing**
  - Encodes target variable
  - Handles class imbalance
  - Prepares target for model training

#### Outputs

- Processed features (X) and target (y)
- Preprocessing pipeline for consistent transformations
- Logs of preprocessing steps in `logs/preprocessing.log`

## Pipeline Flow

1. **Data Loading**

   ```python
   data_loader = DataLoader(config_path="src/config.yaml")
   df = data_loader.load_data()
   ```

2. **Exploratory Data Analysis**

   ```python
   eda = EDA()
   df_after_eda = eda.run_analysis(df)
   ```

3. **Preprocessing**
   ```python
   preprocessor = Preprocessor(config_path="src/config.yaml")
   X_processed, y_processed = preprocessor.run_preprocessing(df_after_eda)
   ```

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   python src/main.py
   ```

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## Logging

The project implements comprehensive logging:

- Main pipeline logs: `logs/main.log`
- EDA logs: `logs/eda.log`
- Preprocessing logs: `logs/preprocessing.log`

## Configuration

Configuration is managed through `src/config.yaml`:

- Feature definitions
- Preprocessing parameters
- Logging settings
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

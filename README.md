# MLOps Hospital Readmission Prediction Project

This project implements a comprehensive machine learning pipeline for predicting hospital readmission rates using patient data. The pipeline follows MLOps best practices with automated testing, logging, evaluation, and inference capabilities.

## Project Structure

```
.
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── logs/                      # Log files from all modules
├── models/                    # Trained models and artifacts
├── plots/                     # Generated visualizations
├── src/
│   ├── config.yaml           # Central configuration file
│   ├── main.py              # Main pipeline orchestrator
│   ├── data_loader/         # Data loading and management
│   ├── validation/          # Data validation and quality checks
│   ├── eda/                 # Exploratory Data Analysis
│   ├── preprocessing/       # Data preprocessing pipeline
│   ├── features/            # Feature engineering
│   ├── model/              # Model training and selection
│   ├── evaluation/         # Model evaluation and metrics
│   └── inference/          # Model inference and prediction
├── tests/                    # Comprehensive test suite
├── environment.yml          # Conda environment specification
└── requirements.txt         # Pip requirements (backup)
```

## Features

### 🚀 Complete MLOps Pipeline
- **Data Loading & Validation**: Robust data ingestion with quality checks
- **Exploratory Data Analysis**: Comprehensive automated analysis with visualizations
- **Feature Engineering**: Advanced feature creation and selection
- **Model Training**: Multiple algorithms with hyperparameter optimization
- **Model Evaluation**: Comprehensive metrics and performance analysis
- **Model Inference**: Production-ready prediction service
- **CI/CD Ready**: Automated testing and validation

### 📊 Advanced Analytics
- **Interactive Visualizations**: Automated plot generation for all analysis steps
- **Statistical Analysis**: In-depth statistical testing and correlation analysis
- **Performance Monitoring**: Detailed logging and reporting at every step
- **Model Comparison**: Automated comparison with baseline models

## Installation

### Using Conda (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kollie/mlops-project-ci.git
   cd mlops-project-ci
   ```

2. **Create and activate conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate mlops
   ```

3. **Verify installation:**
   ```bash
   python -c "import pandas, scikit-learn, numpy; print('Environment ready!')"
   ```

### Alternative: Using pip

1. **Create virtual environment:**
   ```bash
   python -m venv mlops_env
   source mlops_env/bin/activate  # On Windows: mlops_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Run the Complete Pipeline

```bash
# Activate your environment
conda activate mlops

# Run the full pipeline
python src/main.py

# Or run it as a module
python -m src.main
```

### Run Individual Components

```bash
# Just EDA
python -c "from src.eda.eda import EDAAnalyzer; eda = EDAAnalyzer(); eda.run_full_analysis(data)"

# Just model training
python -c "from src.model.trainer import ModelTrainer; trainer = ModelTrainer(); trainer.fit(X, y)"

# Just inference
python -c "from src.inference.predict import ModelPredictor; predictor = ModelPredictor(); predictor.load_model('path/to/model')"
```

## Module Documentation

### 1. Data Loading (`src/data_loader/`)

Handles data ingestion, validation, and splitting with robust error handling.

#### Features
- **Multi-source Support**: Local files, Google Drive links, URLs
- **Automatic Validation**: Integrated data quality checks
- **Smart Splitting**: Stratified train/validation/test splits
- **Flexible Formats**: CSV, Excel, Parquet support

#### Usage
```python
from src.data_loader.data_loader import DataLoader

data_loader = DataLoader(config_path="src/config.yaml")
df = data_loader.load_data()
train, val, test = data_loader.split_data(df)
data_loader.save_split_data(train, val, test)
```

#### Outputs
- Split datasets in `data/processed/`
- Loading logs in `logs/data_loader.log`
- Data quality reports

### 2. Data Validation (`src/validation/`)

Comprehensive data quality assurance with multiple validation strategies.

#### Components
- **`data_validator.py`**: Full validation pipeline with cleaning
- **`validator.py`**: Lightweight validation for CI/CD

#### Features
- **Schema Validation**: Column presence, types, constraints
- **Quality Checks**: Missing values, outliers, distributions
- **Target Analysis**: Class balance, target validity
- **Automated Cleaning**: Configurable data cleaning strategies

#### Usage
```python
from src.validation.data_validator import DataValidator

validator = DataValidator(config_path="src/config.yaml")
clean_data = validator.validate_and_clean(df, strategy='drop_columns')
```

#### Outputs
- Validation reports in `logs/validation_report.json`
- Cleaned datasets
- Quality metrics and recommendations

### 3. Exploratory Data Analysis (`src/eda/`)

Automated comprehensive data analysis with rich visualizations.

#### Features
- **Statistical Analysis**: Descriptive statistics, distributions, correlations
- **Target Analysis**: Class balance, target relationships
- **Feature Analysis**: Univariate and bivariate analysis
- **Missing Data Analysis**: Patterns and impact assessment
- **Automated Visualizations**: 20+ plot types automatically generated

#### Key Analyses
- Data description and summary statistics
- Target variable distribution and balance
- Numerical feature distributions and outliers
- Categorical feature analysis
- Correlation matrices and feature relationships
- Missing value patterns and heatmaps
- Feature importance estimation

#### Usage
```python
from src.eda.eda import EDAAnalyzer

eda_analyzer = EDAAnalyzer(config_path="src/config.yaml")
eda_report = eda_analyzer.run_full_analysis(df)
```

#### Outputs
- Comprehensive analysis report
- 20+ visualizations in `plots/eda/`
- Analysis logs in `logs/eda.log`
- Recommendations for preprocessing

### 4. Preprocessing (`src/preprocessing/`)

Production-ready data preprocessing pipeline with sklearn integration.

#### Features
- **Missing Value Handling**: Multiple imputation strategies
- **Categorical Encoding**: One-hot, label, target encoding
- **Numerical Scaling**: StandardScaler, MinMaxScaler, RobustScaler
- **Outlier Treatment**: Detection and handling strategies
- **Pipeline Persistence**: Save/load preprocessing pipelines

#### Usage
```python
from src.preprocessing.preprocessor import Preprocessor

preprocessor = Preprocessor(config_path="src/config.yaml")
X_train, y_train = preprocessor.fit_transform(train)
X_test, y_test = preprocessor.transform(test)
preprocessor.save_pipeline("models/preprocessor.joblib")
```

#### Outputs
- Preprocessed features and targets
- Saved preprocessing pipeline
- Preprocessing logs with transformation details

### 5. Feature Engineering (`src/features/`)

Advanced feature creation and selection for improved model performance.

#### Features
- **Automated Feature Creation**: Polynomial, interaction, statistical features
- **Feature Selection**: Statistical tests, importance-based, correlation-based
- **Domain-Specific Features**: Healthcare-specific feature engineering
- **Feature Validation**: Importance scoring and selection
- **Pipeline Integration**: Seamless integration with preprocessing

#### Feature Types Created
- **Statistical Features**: Mean, median, std, percentiles
- **Interaction Features**: Feature combinations and ratios
- **Polynomial Features**: Higher-order polynomial combinations
- **Domain Features**: Healthcare-specific calculated features
- **Temporal Features**: Age groups, admission patterns

#### Usage
```python
from src.features.feature_engineering import FeatureEngineer

engineer = FeatureEngineer(config_path="src/config.yaml")
X_engineered, y = engineer.fit_transform(df, target_col='readmitted')
```

#### Outputs
- Enhanced feature sets
- Feature importance rankings
- Feature engineering pipeline
- Engineering logs and statistics

### 6. Model Training (`src/model/`)

Comprehensive model training with multiple algorithms and hyperparameter optimization.

#### Supported Models
- **Random Forest**: With grid search optimization
- **Gradient Boosting**: XGBoost, LightGBM support
- **Logistic Regression**: With regularization
- **Support Vector Machine**: With kernel optimization
- **Neural Networks**: Basic MLP support

#### Features
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Model Persistence**: Save/load trained models
- **Feature Engineering Integration**: Automatic pipeline creation
- **Performance Tracking**: Training metrics and validation scores
- **Early Stopping**: Prevent overfitting

#### Usage
```python
from src.model.trainer import ModelTrainer

trainer = ModelTrainer(config_path="src/config.yaml")
trainer.fit(X_train, y_train)
model_path = trainer.save()
```

#### Outputs
- Trained model artifacts in `models/`
- Training logs with performance metrics
- Hyperparameter optimization results
- Model metadata and configuration

### 7. Model Evaluation (`src/evaluation/`)

Comprehensive model evaluation with detailed metrics and visualizations.

#### Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score
- **Probabilistic Metrics**: ROC-AUC, PR-AUC, Brier Score
- **Advanced Metrics**: Matthews Correlation, Balanced Accuracy
- **Confusion Matrix**: Detailed classification analysis
- **Probability Calibration**: Reliability diagrams

#### Features
- **ROC/PR Curves**: Performance visualization
- **Model Comparison**: Compare against baseline models
- **MLflow Integration**: Experiment tracking
- **Comprehensive Reports**: JSON reports with all metrics
- **Visualization Suite**: Automated evaluation plots

#### Usage
```python
from src.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(config_path="src/config.yaml")
results = evaluator.evaluate(y_true, y_pred, y_pred_proba)
comparison = evaluator.compare_models(other_metrics)
```

#### Outputs
- Detailed evaluation reports in `logs/evaluation_report.json`
- ROC and PR curve visualizations
- Confusion matrix plots
- Model comparison results
- MLflow experiment tracking

### 8. Model Inference (`src/inference/`)

Production-ready inference service with confidence scoring and batch processing.

#### Features
- **Single Predictions**: Individual sample prediction
- **Batch Processing**: Efficient large dataset processing
- **Confidence Scoring**: Prediction confidence analysis
- **Probability Outputs**: Class probability estimates
- **Model Loading**: Load saved models with preprocessing
- **Input Validation**: Robust input data validation

#### Inference Types
- **Basic Prediction**: Simple class predictions
- **Probability Prediction**: Class probability estimates
- **Confidence Analysis**: Predictions with confidence scores
- **Batch Inference**: Process large datasets efficiently

#### Usage
```python
from src.inference.predict import ModelPredictor

predictor = ModelPredictor(config_path="src/config.yaml")
predictor.load_model("models/trained_model.joblib")

# Single predictions
predictions = predictor.predict(data)

# Confidence analysis
results = predictor.predict_with_confidence(data, confidence_threshold=0.8)

# Batch processing
batch_results = predictor.predict_batch(large_data, batch_size=1000)
```

#### Outputs
- Prediction results with confidence scores
- Batch processing summaries
- Inference logs and performance metrics
- Saved prediction results

## Configuration

The entire pipeline is controlled through `src/config.yaml`:

```yaml
data:
  raw_data_path: "data/raw/diabetic_readmission_data.csv"
  processed_data_path: "data/processed"
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15

model:
  type: "random_forest"
  parameters:
    n_estimators: 100
    max_depth: 10
    random_state: 42

feature_engineering:
  enable_polynomial_features: true
  polynomial_degree: 2
  enable_interaction_features: true

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
  cross_validation_folds: 5

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## Testing

Comprehensive test suite with 95%+ code coverage:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_data_loader.py
pytest tests/test_evaluation.py
pytest tests/test_inference.py

# Run with coverage
pytest --cov=src tests/

# Run tests verbosely
pytest -v
```

### Test Coverage
- **Unit Tests**: Individual function testing
- **Integration Tests**: Module interaction testing
- **End-to-End Tests**: Full pipeline testing
- **Mock Testing**: External dependency testing
- **Error Handling**: Exception and edge case testing

## Logging and Monitoring

Comprehensive logging system across all modules:

### Log Files
- `logs/main.log` - Main pipeline execution
- `logs/data_loader.log` - Data loading operations
- `logs/validation_report.json` - Data validation results
- `logs/eda.log` - EDA analysis logs
- `logs/preprocessing.log` - Preprocessing operations
- `logs/feature_engineering.log` - Feature engineering logs
- `logs/model_training.log` - Model training progress
- `logs/evaluation_report.json` - Model evaluation results
- `logs/inference_report.json` - Inference operations

### Monitoring Features
- **Performance Metrics**: Execution time tracking
- **Error Tracking**: Detailed error logging and stack traces
- **Progress Indicators**: Visual progress bars and status updates
- **Resource Usage**: Memory and computation monitoring

## Results and Artifacts

### Generated Artifacts
- **Trained Models**: `models/trained_model.joblib`
- **Preprocessing Pipelines**: `models/preprocessor.joblib`
- **Feature Engineering Pipelines**: `models/feature_engineer.joblib`
- **Evaluation Reports**: `logs/evaluation_report.json`
- **Prediction Results**: `data/processed/inference_results.json`

### Visualizations
- **EDA Plots**: 20+ analysis visualizations in `plots/eda/`
- **Model Performance**: ROC curves, confusion matrices in `plots/evaluation/`
- **Feature Analysis**: Feature importance and correlation plots

## MLflow Integration

Optional MLflow integration for experiment tracking:

```bash
# Start MLflow UI
mlflow ui

# View experiments at http://localhost:5000
```

### Tracked Metrics
- Model performance metrics
- Hyperparameter configurations  
- Training artifacts
- Model comparison results

## Pipeline Execution Flow

1. **Data Loading** → Load and validate raw data
2. **Data Validation** → Quality checks and cleaning
3. **EDA** → Comprehensive data analysis
4. **Preprocessing** → Data cleaning and transformation
5. **Feature Engineering** → Advanced feature creation
6. **Model Training** → Train and optimize models
7. **Model Evaluation** → Comprehensive performance analysis
8. **Model Inference** → Production-ready predictions
9. **Reporting** → Generate comprehensive reports

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Performance Metrics

The pipeline generates comprehensive performance metrics:

### Model Performance
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **PR-AUC**: Area under the Precision-Recall curve

### Pipeline Performance  
- **Execution Time**: Time for each pipeline step
- **Memory Usage**: Peak memory consumption
- **Data Quality**: Missing value and outlier statistics
- **Feature Importance**: Feature contribution rankings

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with scikit-learn, pandas, and MLflow
- Follows MLOps best practices and patterns
- Designed for production deployment and monitoring
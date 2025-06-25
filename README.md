![CI](https://github.com/kollie/mlops-project-ci/actions/workflows/ci.yml/badge.svg)

# MLOps Hospital Readmission Prediction Project

A comprehensive machine learning pipeline for predicting hospital readmission rates using patient data. This project implements a complete MLOps workflow with automated data processing, model training, evaluation, and inference capabilities.

## 🎯 Project Overview

This project demonstrates a production-ready MLOps pipeline for hospital readmission prediction, featuring:

- **Complete Data Pipeline**: From raw data loading to model deployment
- **Automated Feature Engineering**: Advanced feature creation and selection
- **Model Training & Evaluation**: Multiple algorithms with comprehensive metrics
- **Production API**: FastAPI-based inference service
- **Comprehensive Logging**: Detailed tracking of all pipeline steps
- **Error Handling**: Robust error management and recovery
- **Clean Output**: Resolved all model-related warnings

## 🏷️ Baseline Checkpoint

The initial baseline for this project is tagged as [`v0.1.0-notebook-to-mlops`](https://github.com/kollie/mlops-project-ci/releases/tag/v0.1.0-notebook-to-mlops).
This tag marks the transition from the original notebook to a reproducible MLOps pipeline.

To revert to this baseline at any time:

```bash
git checkout v0.1.0-notebook-to-mlops
```

## 🏗️ Architecture

```
src/
├── api/                    # FastAPI application
│   └── app.py             # Main API endpoints
├── data_loader/           # Data loading and splitting
├── validation/            # Data validation and cleaning
├── eda/                   # Exploratory data analysis
├── preprocessing/         # Data preprocessing pipeline
├── features/              # Feature engineering
├── model/                 # Model training and saving
├── evaluation/            # Model evaluation and metrics
├── inference/             # Model inference and predictions
└── config.yaml           # Configuration file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd mlops_group_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
# Run the full MLOps pipeline
python src/main.py

# Or run with custom data source
python src/main.py --data_source "path/to/your/data.csv"
```

### 3. Start the FastAPI Service

```bash
# Start the API server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# Or run directly
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### 4. Test the API

```bash
# Test the pipeline API
python test_pipeline_api.py

# Or test manually
curl http://localhost:8000/health
```

## 📊 Pipeline Components

### 1. Data Loading & Validation

- **DataLoader**: Loads and splits data into train/validation/test sets
- **DataValidator**: Validates schema, data types, and missing values
- **Smart Cleaning**: Handles missing values and data type conversions

### 2. Exploratory Data Analysis (EDA)

- **Comprehensive Analysis**: Statistical summaries, distributions, correlations
- **Target Analysis**: Class balance and distribution analysis
- **Feature Analysis**: Missing value patterns and data quality assessment

### 3. Preprocessing

- **Data Cleaning**: Handles missing values and outliers
- **Feature Scaling**: Standardization and normalization
- **Categorical Encoding**: Label encoding for categorical variables
- **Feature Selection**: SelectKBest for dimensionality reduction

### 4. Feature Engineering

- **Age Groups**: Categorical age group creation
- **Length of Stay Groups**: Hospital stay duration categorization
- **Visit Aggregations**: Total previous visits calculation
- **Medication Features**: Medication intensity and binary flags
- **Diagnosis Features**: Diabetes and circulatory disease detection
- **Binary Flags**: Emergency visits, medication changes, etc.

### 5. Model Training

- **Multiple Algorithms**: Random Forest, Logistic Regression, Decision Tree, Naive Bayes
- **Hyperparameter Tuning**: Configurable model parameters
- **Cross-Validation**: Robust model evaluation
- **Model Persistence**: Saves model and feature engineer together

### 6. Model Evaluation

- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Confusion Matrix**: Detailed classification analysis
- **Model Comparison**: Baseline comparison and improvement tracking
- **Performance Reports**: Detailed evaluation reports

### 7. Inference & Prediction

- **Batch Processing**: Efficient large dataset processing
- **Confidence Scoring**: Prediction confidence analysis
- **Real-time Predictions**: Single and batch prediction capabilities
- **Result Persistence**: Saves prediction results to files

## 🔧 Configuration

The pipeline is configured via `src/config.yaml`:

```yaml
data:
  raw_data_path: "data/raw/diabetic_readmission_data.csv"
  processed_data_path: "data/processed/"
  model_path: "models/"

model:
  active: "random_forest"
  random_forest:
    n_estimators: 200
    max_depth: 10
    min_samples_split: 5
    min_samples_leaf: 2
    max_features: "sqrt"
    bootstrap: True
    criterion: "gini"
    random_state: 42

features:
  target_column: "readmitted"
  numerical_features: ["age", "time_in_hospital", "num_medications"]
  categorical_features:
    ["gender", "admission_type_id", "discharge_disposition_id"]

feature_engineering:
  apply_selection: true
  n_features_to_select: 20

logging:
  level: "INFO"
  file: "logs/main.log"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 🚀 FastAPI API Endpoints

### Health Check

```bash
GET /health
```

Returns pipeline status and last results.

### Data Information

```bash
GET /data-info
```

Returns information about the raw data source.

### Run Pipeline

```bash
POST /run-pipeline
{
  "data_source": "data/raw/diabetic_readmission_data.csv",
  "force_retrain": false
}
```

Starts the complete MLOps pipeline asynchronously.

### Get Results

```bash
GET /results
```

Returns the latest pipeline results and metrics.

### Get Metrics

```bash
GET /metrics
```

Returns detailed model performance metrics.

### Model Information

```bash
GET /model-info
```

Returns information about the trained model.

## 📈 Performance Results

The pipeline achieves the following performance metrics:

- **F1-Score**: 0.5284 (improved from baseline 0.42)
- **Accuracy**: ~0.75
- **Precision**: ~0.65
- **Recall**: ~0.45
- **ROC-AUC**: ~0.70

### Model Comparison

- **Baseline F1-Score**: 0.42
- **Current Model F1-Score**: 0.5284
- **Improvement**: +0.1084 (25.8% improvement)

## 🛠️ Technical Features

### Error Handling & Logging

- **Comprehensive Logging**: All pipeline steps are logged with timestamps
- **Error Recovery**: Graceful handling of failures with detailed error messages
- **Progress Tracking**: Real-time progress updates during pipeline execution
- **Clean Output**: All model-related warnings have been resolved

### Data Processing

- **Smart Missing Value Handling**: Intelligent imputation based on data types
- **Automatic Type Conversion**: Converts integer columns to strings for categorical features
- **Feature Name Consistency**: Resolved all feature name warnings in model training
- **Robust Validation**: Comprehensive data validation with detailed reporting

### Model Management

- **Model Persistence**: Saves both model and feature engineer together
- **Metadata Tracking**: Detailed model metadata and training information
- **Version Control**: Model versioning and comparison capabilities
- **Feature Engineer Integration**: Seamless feature engineering pipeline integration

### Performance Optimizations

- **Batch Processing**: Efficient handling of large datasets
- **Memory Management**: Optimized memory usage for large data processing
- **Parallel Processing**: Support for parallel feature engineering
- **Caching**: Intelligent caching of intermediate results

## 🧪 Testing

### Unit Tests

```bash
# Run all unit tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_model.py
python -m pytest tests/test_inference.py
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_feature_engineering.py
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_data_validation.py
python -m pytest tests/test_eda.py
python -m pytest tests/test_evaluation.py
```

### Test Coverage

```bash
# Run with coverage
python -m pytest --cov=src tests/
```

### API Testing

Since the API now accepts dynamic file paths, you can test it using curl or any HTTP client:

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test data info
curl http://localhost:8000/data-info

# Run pipeline with custom data source
curl -X POST http://localhost:8000/run-pipeline \
  -H "Content-Type: application/json" \
  -d '{"data_source": "data/raw/diabetic_readmission_data.csv", "force_retrain": false}'

# Get pipeline results
curl http://localhost:8000/results

# Get model metrics
curl http://localhost:8000/metrics

# Get model info
curl http://localhost:8000/model-info
```

### Manual Testing

You can also test the API using Python requests:

```python
import requests
import json

# Test pipeline with custom data source
response = requests.post(
    "http://localhost:8000/run-pipeline",
    json={
        "data_source": "data/raw/diabetic_readmission_data.csv",
        "force_retrain": False
    }
)
print(response.json())

# Get results
response = requests.get("http://localhost:8000/results")
print(response.json())
```

## 📁 Project Structure

```
mlops_group_project/
├── src/                   # Source code
│   ├── api/              # FastAPI application
│   ├── data_loader/      # Data loading
│   ├── validation/       # Data validation
│   ├── eda/              # Exploratory data analysis
│   ├── preprocessing/    # Data preprocessing
│   ├── features/         # Feature engineering
│   ├── model/            # Model training
│   ├── evaluation/       # Model evaluation
│   ├── inference/        # Model inference
│   └── config.yaml       # Configuration
├── data/                 # Data files
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── models/               # Trained models
├── logs/                 # Log files
├── tests/                # Test files
├── notebooks/            # Jupyter notebooks
├── requirements.txt      # Dependencies
├── README.md            # This file
└── test_pipeline_api.py # API test script
```

## 🔍 Monitoring & Logging

### Log Files

- `logs/main.log`: Main pipeline execution logs
- `logs/validation.log`: Data validation logs
- `logs/feature_engineering.log`: Feature engineering logs
- `logs/training.log`: Model training logs
- `logs/inference.log`: Inference logs

### Reports

- `logs/validation_report.json`: Data validation results
- `logs/feature_engineering_report.json`: Feature engineering summary
- `logs/training_report.json`: Model training details
- `logs/inference_report.json`: Inference results

## 🐳 Docker Deployment

### Building the Docker Image

```bash
# Build the Docker image
docker build -t hospital-readmission-mlops .

# Build with specific tag
docker build -t hospital-readmission-mlops:v1.0.0 .
```

### Running with Docker

```bash
# Run the container
docker run -p 8000:8000 hospital-readmission-mlops

# Run with custom data volume
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  hospital-readmission-mlops

# Run in detached mode
docker run -d -p 8000:8000 --name mlops-api hospital-readmission-mlops
```

### Using Docker Compose

```bash
# Start all services
docker-compose up -d

# Start with build
docker-compose up --build -d

# View logs
docker-compose logs -f mlops-api

# Stop services
docker-compose down
```

### Testing Docker Deployment

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test API documentation
curl http://localhost:8000/docs

# Test pipeline with data
curl -X POST http://localhost:8000/run-pipeline \
  -H "Content-Type: application/json" \
  -d '{"data_source": "data/raw/diabetic_readmission_data.csv", "force_retrain": false}'
```

### Production Deployment

For production deployment, consider:

1. **Environment Variables**: Set production environment variables
2. **Resource Limits**: Configure memory and CPU limits
3. **Health Checks**: Monitor container health
4. **Logging**: Configure proper logging
5. **Security**: Run as non-root user (already configured)

```bash
# Production run with resource limits
docker run -d \
  --name mlops-api-prod \
  -p 8000:8000 \
  --memory=2g \
  --cpus=1.0 \
  --restart=unless-stopped \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=INFO \
  hospital-readmission-mlops
```

### Docker Image Features

- ✅ **Multi-stage build** for optimized image size
- ✅ **Non-root user** for security
- ✅ **Health checks** for monitoring
- ✅ **Volume mounts** for data persistence
- ✅ **Environment variables** for configuration
- ✅ **Resource limits** for production deployment
- ✅ **CORS enabled** for web integration
- ✅ **Comprehensive logging** for debugging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the diabetes dataset
- Scikit-learn for machine learning algorithms
- FastAPI for the web framework
- Pandas and NumPy for data processing

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.

## 🔬 Weights & Biases Integration

This project includes comprehensive Weights & Biases (wandb) integration for experiment tracking, model versioning, and performance monitoring.

### Setup Weights & Biases

1. **Install wandb**:

   ```bash
   pip install wandb
   ```

2. **Get your API key**:

   - Go to [https://wandb.ai/settings](https://wandb.ai/settings)
   - Copy your API key

3. **Set up environment variables**:

   ```bash
   export WANDB_API_KEY=your_api_key_here
   export WANDB_PROJECT=hospital-readmission-prediction
   export WANDB_ENTITY=your_username_or_team_name
   ```

4. **Or use the setup script**:
   ```bash
   python setup_wandb.py
   ```

### What Gets Tracked

- **Training Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Model Configuration**: Hyperparameters, model type, training parameters
- **Data Information**: Dataset size, feature count, data shapes
- **Model Artifacts**: Trained models, feature importance, evaluation reports
- **Experiment Metadata**: Tags, notes, timestamps, environment info

### Dashboard Features

- **Experiment Comparison**: Compare different model runs
- **Model Lineage**: Track model versions and training data
- **Performance Monitoring**: Real-time metric tracking
- **Artifact Management**: Version control for models and data
- **Collaboration**: Share experiments with team members

### Usage

```python
# The integration is automatic when running the pipeline
python src/main.py

# Check your dashboard at: https://wandb.ai
```

### Configuration

Edit `src/config.yaml` to customize wandb settings:

```yaml
wandb:
  project: "hospital-readmission-prediction"
  entity: "your_username_or_team_name"
  experiment_name: "mlops_pipeline_v1"
  enabled: true
  tags:
    - "mlops"
    - "hospital-readmission"
    - "fastapi"
    - "production"
  notes: "Hospital readmission prediction MLOps pipeline with FastAPI"
```

## 🧪 Testing

# MLOps Group Project

A comprehensive MLOps project implementing a complete machine learning pipeline with modular design and essential components for a production-ready ML system.

## Project Overview

This project implements a complete machine learning pipeline with the following components:

- Data Loading and Validation
- Data Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Inference Pipeline
- Automated Testing
- CI/CD Integration

## Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── main.py              # Main pipeline script
│   ├── data_loader.py       # Data loading utilities
│   ├── data_validation.py   # Data validation
│   ├── preprocessing.py     # Data preprocessing
│   ├── features.py          # Feature engineering
│   ├── model.py            # Model training
│   ├── evaluation.py       # Model evaluation
│   ├── inference.py        # Inference pipeline
│   ├── utils.py            # Utility functions
│   └── config.yaml         # Configuration file
├── data/
│   ├── raw/                # Raw data
│   └── test/               # Test data
├── tests/                  # Test files
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd mlops_project
```

2. Create and activate a virtual environment:

```bash

python -m venv venv
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The project uses a `config.yaml` file for configuration. Key configuration parameters include:

- Data paths
- Model parameters
- Feature engineering settings
- Preprocessing options
- Evaluation metrics

## Usage

### Running the Complete Pipeline

1. Using local data:

```bash
python src/main.py
```

2. Using data from URLs:

```python
# In src/main.py, set:
use_url_data = True
```

### Running Individual Components

1. Data Loading:

```python
from src import DataLoader

# Load from local file
data_loader = DataLoader()
data = data_loader.load_csv("data/raw/train_data.csv")

# Load from URL
data = data_loader.load_api("https://example.com/data.csv")
```

2. Model Training:

```python
from src import ModelTrainer

trainer = ModelTrainer()
metrics = trainer.train(X, y)
```

3. Inference:

```python
from src import InferencePipeline

pipeline = InferencePipeline()
predictions = pipeline.predict("data/test/test_data.csv")
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src/
```

## Data Format

The project expects data in CSV format with the following structure:

### Training Data

```
age,income,category,location,score,target
25,50000,A,NY,85,0
30,60000,B,CA,90,1
...
```

### Test Data

```
age,income,category,location,score
26,51000,A,NY,86
31,61000,B,CA,89
...
```

## Features

- Modular design for easy maintenance and extension
- Comprehensive error handling and logging
- Automated testing with pytest
- CI/CD integration with GitHub Actions
- Support for both local and URL-based data loading
- Configurable pipeline components
- Detailed logging and metrics reporting

## License

This project is licensed under the MIT License - see the LICENSE file for details.

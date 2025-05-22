from setuptools import setup, find_packages

setup(
    name="mlops-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "mlflow>=2.0.0",
        "joblib>=1.1.0",
        "python-dotenv>=0.19.0",
        "sqlalchemy>=1.4.23",
        "requests>=2.26.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
        ],
    },
    python_requires=">=3.8",
    author="MLOps Team",
    description="MLOps Project for Diabetic Readmission Prediction",
    keywords="mlops, machine learning, healthcare",
) 
from setuptools import setup, find_packages

setup(
    name="mlops_group_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pyyaml>=6.0.0",
        "joblib>=1.1.0",
        "tqdm>=4.64.0",
    ],
    python_requires=">=3.8",
)

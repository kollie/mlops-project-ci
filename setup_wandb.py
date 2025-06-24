#!/usr/bin/env python3
"""
Weights & Biases Setup Script for Hospital Readmission MLOps Pipeline

This script helps users set up their wandb API key and test the integration.
"""

import os
from pathlib import Path
import dotenv


def setup_wandb():
    """Setup Weights & Biases configuration."""
    print("🔧 Setting up Weights & Biases for Hospital Readmission MLOps Pipeline")
    print("=" * 70)

    # Load .env variables
    dotenv.load_dotenv()

    # Check if wandb is installed
    try:
        import wandb
        print("✅ Weights & Biases is installed")
    except ImportError:
        print("❌ Weights & Biases is not installed")
        print("   Install it with: pip install wandb")
        return False

    # Check if API key is already set
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        print(f"✅ WANDB_API_KEY is already set (starts with: {api_key[:8]}...)")
    else:
        print("❌ WANDB_API_KEY is not set")
        print("\n📋 To get your API key:")
        print("1. Go to https://wandb.ai/settings")
        print("2. Copy your API key")
        print("3. Set it as an environment variable:")
        print("   export WANDB_API_KEY=your_api_key_here")
        print("\n   Or add it to your .env file:")
        print("   WANDB_API_KEY=your_api_key_here")
        return False

    # Test wandb connection and log a sample run
    print("\n🧪 Testing wandb connection and logging...")
    try:
        import random
        import wandb
        # Initialize a test run with config from .env
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "ml_ops_project"),
            entity=os.getenv("WANDB_ENTITY"),
            name="setup_test",
            config={
                "learning_rate": 0.02,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 10
            },
        )
        # Simulate training and log metrics
        epochs = 10
        offset = random.random() / 5
        for epoch in range(2, epochs):
            acc = 1 - 2 ** -epoch - random.random() / epoch - offset
            loss = 2 ** -epoch + random.random() / epoch + offset
            run.log({
                "acc": acc,
                "loss": loss,
                "epoch": epoch
            })
        run.finish()
        print("✅ Weights & Biases connection and logging successful!")
    except Exception as e:
        print(f"❌ Weights & Biases connection failed: {str(e)}")
        return False

    # Check configuration
    print("\n📋 Configuration Summary:")
    print(f"   Project: {os.getenv('WANDB_PROJECT', 'Not set (will use default)')}")
    print(f"   Entity: {os.getenv('WANDB_ENTITY', 'Not set (will use your username)')}")
    print(f"   API Key: {api_key[:8]}...")

    # Create .env file if it doesn't exist
    env_file = Path(".env")
    if not env_file.exists():
        print("\n📝 Creating .env file...")
        env_content = (
            f"# Weights & Biases Configuration\n"
            f"WANDB_API_KEY={api_key}\n"
            f"WANDB_PROJECT={os.getenv('WANDB_PROJECT', 'ml_ops_project')}\n"
            f"WANDB_ENTITY={os.getenv('WANDB_ENTITY', 'your_username_or_team_name')}\n"
            "\n# Application Configuration\n"
            "ENVIRONMENT=development\n"
            "LOG_LEVEL=INFO\n"
            "PORT=8000\n"
            "\n# Model Configuration\n"
            "MODEL_NAME=hospital_readmission_model\n"
            "EXPERIMENT_NAME=mlops_pipeline_v1\n"
            "\n# Data Configuration\n"
            "DATA_SOURCE=data/raw/diabetic_readmission_data.csv\n"
        )
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file")
    else:
        print("✅ .env file already exists")

    print("\n🎉 Weights & Biases setup completed!")
    print("\n📚 Next steps:")
    print("1. Run the pipeline: python src/main.py")
    print("2. Check your wandb dashboard: https://wandb.ai")
    print("3. View experiment tracking and model artifacts")

    return True


def test_wandb_integration():
    """Test the wandb integration with a simple model."""
    print("\n🧪 Testing wandb integration with simple model...")
    try:
        import wandb
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        # Create dummy data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        # Train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        # Test predictions
        y_pred = model.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        # Initialize wandb
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", "ml_ops_project"),
            entity=os.getenv("WANDB_ENTITY"),
            name="integration_test",
            config={
                "model_type": "RandomForest",
                "n_estimators": 10,
                "test_size": 0.2
            }
        )
        # Log metrics
        wandb.log({
            "test/accuracy": accuracy,
            "test/samples": len(y_test)
        })
        # Finish run
        run.finish()
        print(f"✅ Integration test successful! Accuracy: {accuracy:.4f}")
        print("   Check your wandb dashboard for the test run")
    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        return False
    return True


if __name__ == "__main__":
    print("🚀 Weights & Biases Setup for Hospital Readmission MLOps Pipeline")
    print("=" * 70)
    # Setup wandb
    if setup_wandb():
        # Test integration
        test_wandb_integration()
    print("\n📖 For more information:")
    print("   - Weights & Biases docs: https://docs.wandb.ai")
    print("   - Project README: README.md")
    print("   - Configuration: src/config.yaml")
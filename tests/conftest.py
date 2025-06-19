import pytest
import os
from pathlib import Path


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Create necessary directories
    directories = ["logs", "plots", "data/raw", "data/processed"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    # Set up test environment variables
    os.environ["PYTHONPATH"] = str(Path(__file__).parent.parent)

    yield

    # Cleanup after tests - but preserve important data files
    cleanup_patterns = {
        "logs": ["*.log", "*.json"],  # Only remove log files
        "plots": ["*.png", "*.jpg", "*.pdf"],  # Only remove plot files
        "data/processed": ["*"],  # Remove all processed files
        "data/raw": [],  # DON'T remove anything from raw data directory
    }

    for directory, patterns in cleanup_patterns.items():
        dir_path = Path(directory)
        if dir_path.exists():
            for pattern in patterns:
                for file in dir_path.glob(pattern):
                    if file.is_file():
                        try:
                            file.unlink()
                        except Exception:
                            pass  # Ignore errors during cleanup

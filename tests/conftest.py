import pytest
import os
from pathlib import Path

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test."""
    # Create necessary directories
    directories = ['logs', 'plots', 'data/raw', 'data/processed']
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set up test environment variables
    os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)
    
    yield
    
    # Cleanup after tests
    for directory in directories:
        if Path(directory).exists():
            for file in Path(directory).glob('*'):
                if file.is_file():
                    file.unlink() 
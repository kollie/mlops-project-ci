import logging
import sys
import os
from pathlib import Path
import yaml

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader.data_loader import DataLoader

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Clear any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "main.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "src/config.yaml") -> dict:
    """Load configuration from yaml file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")

def main():
    """Main function to run the data processing pipeline."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting data processing pipeline...")
    
    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")
        
        # Step 1: Load data
        logger.info("Loading data...")
        data_loader = DataLoader(config_path="src/config.yaml")
        df = data_loader.load_data()
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Step 2: Split data
        logger.info("Splitting data...")
        train, val, test = data_loader.split_data(df)
        logger.info(f"Data split successfully - Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
        
        # Step 3: Save split data
        logger.info("Saving split datasets...")
        data_loader.save_split_data(train, val, test)
        logger.info("Split datasets saved successfully")
        
        # Future steps would go here:
        # Step 4: EDA (when module exists)
        # Step 5: Preprocessing (when module exists)
        # Step 6: Feature engineering (when module exists)
        # Step 7: Model training (when module exists)
        # Step 8: Evaluation (when module exists)
        
        logger.info("Pipeline completed successfully")
        return {
            'train': train,
            'validation': val,
            'test': test
        }
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
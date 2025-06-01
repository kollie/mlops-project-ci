import logging
from pathlib import Path
import yaml
from src.data.data_loader import DataLoader
from src.eda.eda import EDA
from src.preprocessing.preprocessor import Preprocessor

def setup_logging():
    """Setup logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
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
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

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
        
        # Step 2: Perform EDA
        logger.info("Starting Exploratory Data Analysis...")
        eda = EDA()
        df_after_eda = eda.run_analysis(df)
        logger.info("EDA completed successfully")
        
        # Step 3: Preprocess data
        logger.info("Starting data preprocessing...")
        preprocessor = Preprocessor(config_path="src/config.yaml")
        X_processed, y_processed = preprocessor.run_preprocessing(df_after_eda)
        logger.info(f"Preprocessing completed. Features shape: {X_processed.shape}, Target shape: {y_processed.shape}")
        
        # Log final shapes
        logger.info("Pipeline completed successfully")
        logger.info(f"Final processed features shape: {X_processed.shape}")
        logger.info(f"Final processed target shape: {y_processed.shape}")
        
        return X_processed, y_processed
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
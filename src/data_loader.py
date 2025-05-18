import pandas as pd
import sqlalchemy as sa
import requests
from typing import Union, Dict, Any
from .utils import load_config, setup_logging

logger = setup_logging()

class DataLoader:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config = load_config(config_path)
        self.db_config = self.config['database']
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        try:
            logger.info(f"Loading data from CSV: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
    
    def load_database(self, query: str) -> pd.DataFrame:
        """Load data from database using SQL query."""
        try:
            logger.info("Connecting to database")
            connection_string = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
            engine = sa.create_engine(connection_string)
            
            logger.info(f"Executing query: {query}")
            return pd.read_sql(query, engine)
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            raise
    
    def load_api(self, url: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Load data from API endpoint."""
        try:
            logger.info(f"Fetching data from API: {url}")
            response = requests.get(url, params=params)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except Exception as e:
            logger.error(f"Error loading from API: {str(e)}")
            raise
    
    def save_data(self, data: pd.DataFrame, file_path: str) -> None:
        """Save data to CSV file."""
        try:
            logger.info(f"Saving data to: {file_path}")
            data.to_csv(file_path, index=False)
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise 
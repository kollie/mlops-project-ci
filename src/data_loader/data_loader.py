import pandas as pd
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging

class DataLoader:
    def __init__(self, config_path: str = "src/config.yaml"):
        """
        Initialize the DataLoader with configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from yaml file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        #Check path and directory if needed
        log_file_path = self.config['logging']['file']
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=self.config['logging']['level'],
            format=self.config['logging']['format'],
            filename=self.config['logging']['file']
        )
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the raw data from the specified path or configured path.
        
        Args:
            file_path (str, optional): Path to the data file. If None, uses the path from config.
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            data_path = file_path if file_path is not None else self.config['data']['raw_data_path']
            self.logger.info(f"Loading data from {data_path}")
            return pd.read_csv(data_path)
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def split_data(self, data: pd.DataFrame) -> tuple:
        """
        Split the data into train, validation, and test sets.
        
        Args:
            data (pd.DataFrame): Input dataset
            
        Returns:
            tuple: (train_data, validation_data, test_data)
        """
        try:
            # First split: separate test set
            train_val, test = train_test_split(
                data,
                test_size=self.config['model']['test_size'],
                random_state=self.config['model']['random_state']
            )
            
            # Second split: separate validation set from training set
            val_size = self.config['model']['validation_size'] / (1 - self.config['model']['test_size'])
            train, val = train_test_split(
                train_val,
                test_size=val_size,
                random_state=self.config['model']['random_state']
            )
            
            self.logger.info(f"Data split complete. Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
            
            return train, val, test
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            raise
    
    def save_split_data(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """
        Save the split datasets to their respective paths.
        
        Args:
            train (pd.DataFrame): Training data
            val (pd.DataFrame): Validation data
            test (pd.DataFrame): Test data
        """
        try:
            # Create directories if they don't exist
            Path(self.config['data']['processed_data_path']).mkdir(parents=True, exist_ok=True)
            Path(self.config['data']['test_data_path']).mkdir(parents=True, exist_ok=True)
            
            # Save the datasets
            train.to_csv(self.config['data']['train_data_path'], index=False)
            val.to_csv(self.config['data']['validation_data_path'], index=False)
            test.to_csv(f"{self.config['data']['test_data_path']}/test.csv", index=False)
            
            self.logger.info("Split datasets saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving split data: {str(e)}")
            raise

    def data_preparation(self, data: pd.DataFrame):
        """
        Data will be cleaned to make sure all ingested data is valid for the model.
        It will follow a line by line approach and log the result with all the drops and fixes made

        """

        try:

            initial_rows = len(data)

            # Drop rows where there is no diagnosis, == '?'
            diag_mask = (
                (data['diag_1'] == '?') &
                (data['diag_2'] == '?') &
                (data['diag_3'] == '?')
            )
            dropped_diag = diag_mask.sum()
            data = data[~diag_mask]
            self.logger.info(f"Dropped {dropped_diag} rows without diagnoses (diag_1, diag_2, diag_3).")

            # Drop rows without race, == '?'
            dropped_race = (data['race'] == '?').sum()
            data = data[data['race'] != '?']
            self.logger.info(f"Dropped {dropped_race} rows with unknown race.")

            # Drop rows where discharge_disposition_id == 11, which is death.
            dropped_discharge = (data['discharge_disposition_id'] == 11).sum()
            data = data[data['discharge_disposition_id'] != 11]
            self.logger.info(f"Dropped {dropped_discharge} rows where patients expired (discharge_disposition_id == 11).")

            # Drop rows where gender is 'Unknown/Invalid'
            dropped_gender = (data['gender'] == 'Unknown/Invalid').sum()
            data = data[data['gender'] != 'Unknown/Invalid']
            self.logger.info(f"Dropped {dropped_gender} rows with unknown/invalid gender.")

            # Total dropped rows
            total_dropped = initial_rows - len(data)
            self.logger.info(f"Total rows dropped during cleaning: {total_dropped}")

            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    data = loader.load_data('https://drive.google.com/uc?id=1WCn9cjVQH2LP8E5ZC1-cNKVKBE8OrFGy')
    data_clean = loader.data_preparation(data)
    
    print(data_clean.head())
    
    train, val, test = loader.split_data(data)
    loader.save_split_data(train, val, test)

   

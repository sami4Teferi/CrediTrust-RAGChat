"""
Module for loading and validating the complaints dataset.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/task2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataLoader:
    """Handles loading and validation of the complaints dataset."""
    
    def __init__(self, data_path: Path):
        """
        Initialize the DataLoader.
        
        Args:
            data_path (Path): Path to the cleaned complaints CSV.
        """
        self.data_path = data_path

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load and validate the complaints dataset.
        
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if loading fails.
        """
        try:
            if not self.data_path.exists():
                logger.error(f"Data file not found at {self.data_path}")
                raise FileNotFoundError(f"Data file not found at {self.data_path}")
            df = pd.read_csv(self.data_path)
            required_columns = ['complaint_id', 'product', 'consumer_complaint_narrative']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns: {required_columns}")
                raise ValueError(f"Missing required columns: {required_columns}")
            logger.info(f"Loaded dataset with {len(df)} complaints")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return None
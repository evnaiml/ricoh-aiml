"""Reading data from various file formats"""
# %% [markdown]
# * Summary:
# - The module contains data readers for various file formats
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com, evgeni.v.nikolaev.ai@gmail.com
# -----------------------------------------------------------------------------
# * Updated on: 2025-06-16
# * Created on: 2025-06-16
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# %% [markdown]
# -----------------------------------------------------------------------------
# * Load python modules
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import os
import re
from typing import Optional, Dict, List, Any
# -----------------------------------------------------------------------------
class DataReader(ABC):
    """ Abstract class for reading data from a file

        Args:
            ABC (abc.ABCMeta): Abstract Base Class
    """

@abstractmethod
def read_data(self, path:Path) -> pd.DataFrame:
    """ Abstract method to read data from a file

        Args:
            path (Path): a path to the file

        Returns:
            pd.DataFrame: A pandas dataframe
    """
pass
# -----------------------------------------------------------------------------
class DataReaderCSV(DataReader):
    """ Class for reading data from a CSV file"""

    def read_data(self, path:Path) -> pd.DataFrame:
        """ Read data from a CSV file and use the file name as the index name.
            If the CSV file contains an "Unnamed: 0" column, it is dropped.
            This method is useful for reading CSV files into a pandas DataFrame
            and ensuring that the index name is set to the file name for easy reference.
            The file name is converted to uppercase and the ".csv" extension is removed.

        Args:
            path (Path): a path to the file

        Returns:
            pd.DataFrame: A pandas dataframe
        """
        df = pd.read_csv(path)

        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        file_name = Path(path).stem.lower()
        file_name.replace(".csv","")
        df.index.name = file_name.upper()

        return df

# -----------------------------------------------------------------------------
class DataReaderEXCEL(DataReader):
    """Class for reading data from an EXCEL file"""

    def read_data(self, path:Path) -> pd.DataFrame:
        """ Read data from an EXCEL file and use the file name as the index name.
            If the EXCEL file contains an "Unnamed: 0" column, it is dropped.
            This method is useful for reading EXCEL files into a pandas DataFrame
            and ensuring that the index name is set to the file name for easy reference.
            The file name is converted to uppercase and the ".xlsx" extension is removed.

        Args:
            path (Path): a path to the file

        Returns:
            pd.DataFrame: A pandas dataframe
        """
        df = pd.read_excel(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        file_name = Path(path).stem.lower()
        file_name.replace(".csv","")
        df.index.name = file_name.upper()
        return df

# -----------------------------------------------------------------------------
class DataReader(ABC):
    """Abstract class for reading data from a file

    Args:
        ABC (abc.ABCMeta): Abstract Base Class
    """

    @abstractmethod
    def read_data(self, path: Path) -> pd.DataFrame:
        """Abstract method to read data from a file

        Args:
            path (Path): a path to the file

        Returns:
            pd.DataFrame: A pandas dataframe
        """
        pass


class DataReaderCSVMetadata(DataReader):
    """DataReader implementation that reads CSV files with metadata-based column filtering"""

    def __init__(self, csv_dir: Path, metadata_dir: Path):
        """Initialize with CSV and metadata directories

        Args:
            csv_dir (Path): Directory containing CSV files
            metadata_dir (Path): Directory containing metadata txt files
        """
        self.csv_dir = Path(csv_dir)
        self.metadata_dir = Path(metadata_dir)

        # Setup logging
        self._setup_logging()

        self.logger.info(f"Initialized DataReaderCSVMetadata with CSV dir: {csv_dir}, Metadata dir: {metadata_dir}")

    def _setup_logging(self):
        """Setup logging to console and file"""
        # Create logs directory in the same directory as the main executing file
        import sys
        if hasattr(sys.modules['__main__'], '__file__'):
            # Get the directory of the main script
            main_file = sys.modules['__main__'].__file__
            script_dir = Path(main_file).parent
        else:
            # Fallback to current working directory if running in interactive mode
            script_dir = Path.cwd()

        log_dir = script_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_dir / 'data_reader.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def read_data(self, path: Path) -> pd.DataFrame:
        """Read data from CSV file using metadata for column filtering

        Args:
            path (Path): Path to the CSV file

        Returns:
            pd.DataFrame: Filtered and processed pandas dataframe
        """
        csv_path = Path(path)

        # If path is relative to csv_dir, make it absolute
        if not csv_path.is_absolute():
            csv_path = self.csv_dir / csv_path

        self.logger.info(f"Reading CSV file: {csv_path}")

        # Find corresponding metadata file
        metadata_path = self._find_metadata_file(csv_path)
        if not metadata_path:
            self.logger.error(f"No metadata file found for CSV: {csv_path}")
            raise FileNotFoundError(f"No metadata file found for CSV: {csv_path}")

        self.logger.info(f"Using metadata file: {metadata_path}")

        # Read CSV file
        try:
            df = pd.read_csv(csv_path)
            self.logger.info(f"Successfully read CSV with shape: {df.shape}")
        except Exception as e:
            self.logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
            raise

        # Read metadata and determine which columns to keep
        columns_to_keep = self._parse_metadata(metadata_path, df.columns.tolist())

        # Filter columns
        df_filtered = df[columns_to_keep]
        self.logger.info(f"Filtered dataframe shape: {df_filtered.shape}")
        self.logger.info(f"Kept columns: {columns_to_keep}")

        # Convert data types
        df_processed = self._convert_data_types(df_filtered)

        self.logger.info("Data processing completed successfully")
        return df_processed

    def read_all_data(self) -> List[pd.DataFrame]:
        """Read all CSV files in the directory and return list of dataframes

        Returns:
            List[pd.DataFrame]: List of processed dataframes with index names set to file stems
        """
        self.logger.info("Starting to read all CSV files")

        # Find all CSV files in the directory
        csv_files = list(self.csv_dir.glob("*.csv"))

        if not csv_files:
            self.logger.warning(f"No CSV files found in directory: {self.csv_dir}")
            return []

        self.logger.info(f"Found {len(csv_files)} CSV files to process")

        dataframes = []

        for csv_file in csv_files:
            try:
                # Read the CSV file
                df = self.read_data(csv_file)

                # Set the index name to the file stem
                df.index.name = csv_file.stem

                dataframes.append(df)
                self.logger.info(f"Successfully processed {csv_file.name} with index name: {csv_file.stem}")

            except Exception as e:
                self.logger.error(f"Failed to process {csv_file.name}: {str(e)}")
                # Continue processing other files instead of failing completely
                continue

        self.logger.info(f"Successfully processed {len(dataframes)} out of {len(csv_files)} CSV files")
        return dataframes

    def _find_metadata_file(self, csv_path: Path) -> Optional[Path]:
        """Find metadata file corresponding to CSV file

        Args:
            csv_path (Path): Path to CSV file

        Returns:
            Optional[Path]: Path to metadata file if found, None otherwise
        """
        csv_name = csv_path.stem  # filename without extension

        # Look for metadata files that start with the CSV filename
        for metadata_file in self.metadata_dir.glob("*.txt"):
            if metadata_file.stem.startswith(csv_name):
                return metadata_file

        return None

    def _parse_metadata(self, metadata_path: Path, csv_columns: List[str]) -> List[str]:
        """Parse metadata file to determine which columns to keep

        Args:
            metadata_path (Path): Path to metadata file
            csv_columns (List[str]): List of column names from CSV

        Returns:
            List[str]: List of column names to keep
        """
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_lines = f.readlines()
        except Exception as e:
            self.logger.error(f"Error reading metadata file {metadata_path}: {str(e)}")
            raise

        columns_to_keep = []

        for column in csv_columns:
            should_keep = self._should_keep_column(column, metadata_lines)
            if should_keep:
                columns_to_keep.append(column)
                self.logger.debug(f"Column '{column}' will be kept")
            else:
                self.logger.debug(f"Column '{column}' will be skipped")

        return columns_to_keep

    def _should_keep_column(self, column_name: str, metadata_lines: List[str]) -> bool:
        """Determine if a column should be kept based on metadata

        Args:
            column_name (str): Name of the column
            metadata_lines (List[str]): Lines from metadata file

        Returns:
            bool: True if column should be kept, False otherwise
        """
        for line in metadata_lines:
            line = line.strip()
            if column_name in line:
                # Check prefix at the beginning of the line
                if line.startswith('(+)') or line.startswith('(=)'):
                    return True
                elif line.startswith('(-)') or line.startswith('(?)'):
                    return False

        # If no line found with column name, skip the column
        return False

    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types intelligently

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Dataframe with converted data types
        """
        df_converted = df.copy()

        for column in df_converted.columns:
            self.logger.debug(f"Processing column: {column}")
            df_converted[column] = df_converted[column].apply(self._convert_value)

        return df_converted

    def _convert_value(self, value: Any) -> Any:
        """Convert a single value to appropriate data type

        Args:
            value (Any): Input value

        Returns:
            Any: Converted value
        """
        # Handle NaN and None values
        if pd.isna(value) or value is None:
            return np.nan

        # If already a number, return as is
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            # Handle float with .0 -> convert to int
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return value

        # Convert string values
        if isinstance(value, str):
            value = value.strip()

            # Empty string -> NaN
            if not value:
                return np.nan

            # Try to convert to number
            try:
                # Try integer first
                if '.' not in value and 'e' not in value.lower():
                    return int(value)
                else:
                    # Try float
                    float_val = float(value)
                    # If it's a whole number, convert to int
                    if float_val.is_integer():
                        return int(float_val)
                    return float_val
            except ValueError:
                # If conversion fails, return NaN
                return np.nan

        # For other types, return as is
        return value

# Example usage
# if __name__ == "__main__":
#     # Example usage
#     csv_dir = Path("data/csv")
#     metadata_dir = Path("data/metadata")

#     # Create reader instance
#     reader = DataReaderCSVMetadata(csv_dir, metadata_dir)

#     # Read a specific CSV file
#     try:
#         df = reader.read_data(Path("sample_data.csv"))
#         print(f"Loaded dataframe with shape: {df.shape}")
#         print(f"Columns: {df.columns.tolist()}")
#         print(f"Data types:\n{df.dtypes}")
#     except Exception as e:
#         print(f"Error: {e}")

#     # Read all CSV files in the directory
#     try:
#         dataframes = reader.read_all_data()
#         print(f"\nLoaded {len(dataframes)} dataframes:")

#         for i, df in enumerate(dataframes):
#             print(f"DataFrame {i+1}:")
#             print(f"  Index name: {df.index.name}")
#             print(f"  Shape: {df.shape}")
#             print(f"  Columns: {df.columns.tolist()}")
#             print()

#     except Exception as e:
#         print(f"Error reading all data: {e}")
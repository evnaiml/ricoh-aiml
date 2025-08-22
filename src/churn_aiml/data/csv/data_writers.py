"""Writering outputs with various file formats"""
# %% [markdown]
# * Summary:
# - The module contains data writers for various file formats
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
from typing import Optional, List
from pathlib import Path
import pandas as pd
# -----------------------------------------------------------------------------
class DataWriter(ABC):
    """ Abstract class for writing data to a file
    
        Args:
            ABC (abc.ABCMeta): Abstract Base Class
    """
    
    @abstractmethod
    def write_data(self, path:Path, colnames: Optional[List[str]] = None) -> None:        
        """ Abstract method to write data to a file        
            Args:
                path (Path): A path to the file.
                colnames (Optional[List[str]], optional): A subset to columns to write. Defaults to None.
        """
        pass
# -----------------------------------------------------------------------------
class DataWriterCSV(DataWriter):    
    """ The class for writing data to a csv file
        Args:
            DataWriter: Abstract Base Class
    """

def write_data(        
        self, 
        data_frame:pd.DataFrame, 
        output_dir, 
        colnames: Optional[List[str]] = None,
        index: Optional[bool] = False
    ) -> None:
    """ The method to write data to a file.
        
        Args:
            data_frame (pd.DataFrame): Dataframe to write.
            output_dir (Path): Path to the output directory.
            colnames (Optional[List[str]], optional): A subset to columns to write. Defaults to None.
            index (Optional[bool], optional): Flag to write index. Defaults to False.
    """
    # Copy input dataframe
    df = data_frame.copy(deep=True)
    
    # Create the output file name
    file_name = df.index.name + ".csv" 
    
    # Create the directory where to write the data if does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write data or specified columns to a file 
    if colnames is not None:
        data_frame.to_csv(output_dir / file_name, columns=colnames, index=index)
    else:
        data_frame.to_csv(output_dir / file_name, index=index)

# -----------------------------------------------------------------------------
class DataWriterExcel(DataWriter):    
    """ The class for writing data to an excel file
        
        Args:
            DataWriter: Abstract Base Class
    """

    def write_data(            
            self,
            data_frame:pd.DataFrame,
            output_dir:str,
            colnames: Optional[List[str]] = None,
            index: Optional[bool] = False
        ) -> None:

        """ The method to write data to a file.
            Args:
                data_frame (pd.DataFrame): Dataframe to write.
                output_dir (Path): Path to the output directory.
                colnames (Optional[List[str]], optional): A subset to columns to write. Defaults to None.
                index (Optional[bool], optional): Flag to write index. Defaults to False.
        """
        # Copy input dataframe
        df = data_frame.copy(deep=True)

        # Create the output file name
        file_name = df.index.name + ".xlsx" 

        # Create the directory where to write the data if does not exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write data or specified columns to a file 
        if colnames is not None:
            data_frame.to_excel(output_dir / file_name, columns=colnames, index=index)
        else:
            data_frame.to_excel(output_dir / file_name, index=index) 
# %%
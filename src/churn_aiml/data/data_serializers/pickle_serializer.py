"""
The module contains PickleSerializer Class
* Pickle pros:
  - Universal - part of Python standard library, no extra dependency
  - Broader compatibility - works with any Python object
  - Simpler for small data - less overhead for tiny datasets

* Joblib pros:
  - Better compression - joblib automatically compresses large numpy
  arrays, resulting in smaller file sizes
  - More efficient for numerical data - optimized for scientific computing
   and numpy/pandas objects
  - Parallel processing support - can leverage multiple cores for large
arrays
  - Memory mapping - can read large files without loading everything into
  memory

  For ML-industrial use case (DataFrames with numerical data, datetime, categories):
  - Joblib is likely better because it's specifically optimized for
  scientific data structures
  - File sizes will typically be 20-50% smaller with joblib
  - Loading can be faster for large DataFrames

  Both preserve dtypes perfectly, so it comes down to performance. For
  DataFrames with lots of numerical data (like TSFRESH features),
  joblib is generally the better choice.
"""
# -----------------------------------------------------------------------------
# Author: Evgeni Nikoolaev
# email: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# UPDATED ON: 2025-08-11
# CREATED ON: 2025-08-09
# -----------------------------------------------------------------------------
# COPYRIGHT@2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh
# -----------------------------------------------------------------------------
# %%
import pandas as pd
from pathlib import Path
from typing import Union
import pickle
import gzip
import json
from datetime import datetime

# %%
# ============= class PickleSerializer =============
class PickleSerializer:
    """
    A simple class to save and load complex data (e.g. pandas data and encoders, etc.) using pickle format.

    Suffix rules:
    - .pkl or .pickle → saves without compression
    - .pkl.gz or .pickle.gz → saves with compression
    - no suffix or other → defaults to .pkl.gz (compressed)

    Automatically creates directories if they don't exist.
    """

    def __init__(self):
        pass

    @staticmethod
    def _ensure_directory(file_path: Path) -> None:
        """Ensure the parent directory exists, create if it doesn't."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def save_data(data, output_path: Union[str, Path]) -> Path:
        """
        Save data to pickle format with perfect preservation of all data and dtypes.

        Suffix behavior:
        - file.pkl or file.pickle → saves without compression
        - file.pkl.gz or file.pickle.gz → saves with compression
        - file or file.other → saves as file.pkl.gz (compressed)

        Parameters
        ----------
        data : any
            Data to save
        output_path : Union[str, Path]
            Path where to save the pickle file

        Returns
        -------
        Path
            Path where the file was saved
        """
        output_path = Path(output_path)
        output_path_str = str(output_path)

        # Ensure directory exists
        PickleSerializer._ensure_directory(output_path)

        # Check existing extensions and handle accordingly
        if output_path_str.endswith(('.pkl.gz', '.pickle.gz')):
            # Already has .pkl.gz or .pickle.gz - save with compression
            with gzip.open(output_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved compressed data to: {output_path}")
        elif output_path_str.endswith(('.pkl', '.pickle')):
            # Has .pkl or .pickle - save without compression
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved data to: {output_path}")
        else:
            # No pickle extension - add .pkl.gz by default
            if output_path.suffix:
                # Has some other extension, replace with .pkl.gz
                output_path = output_path.with_suffix('.pkl.gz')
            else:
                # No extension, add .pkl.gz
                output_path = Path(output_path_str + '.pkl.gz')

            # Ensure directory exists for new path
            PickleSerializer._ensure_directory(output_path)

            # Save with compression
            with gzip.open(output_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"✓ Saved compressed data to: {output_path}")
        if hasattr(data, 'shape'):
            print(f"  Shape: {data.shape}")
        
        # Save metadata
        metadata = {
            'data_type': type(data).__name__,
            'saved_at': datetime.now().isoformat(),
            'format': 'pickle',
            'compressed': str(output_path).endswith('.gz'),
            'compression_type': 'gzip' if str(output_path).endswith('.gz') else None
        }
        
        # Add shape info if available
        if hasattr(data, 'shape'):
            metadata['shape'] = data.shape
        
        # Add dtype info for DataFrames
        if hasattr(data, 'dtypes'):
            metadata['dtypes'] = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Save metadata as JSON
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return output_path

    @staticmethod
    def load_data(file_path: Union[str, Path]):
        """
        Load data from pickle file with perfect preservation of all data and dtypes.
        Automatically handles decompression if file ends with .gz.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the pickle file

        Returns
        -------
        any
            Loaded data with perfectly preserved data and dtypes

        Raises
        ------
        FileNotFoundError
            If the pickle file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")

        # Check if file is compressed (ends with .gz)
        if str(file_path).endswith('.gz'):
            # Load with gzip decompression
            with gzip.open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded compressed data from: {file_path}")
        else:
            # Load without decompression
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            print(f"✓ Loaded data from: {file_path}")
        if hasattr(data, 'shape'):
            print(f"  Shape: {data.shape}")

        return data


# ============= EXAMPLE FUNCTIONS =============

def example_basic_usage():
    """Example 1: Basic Data Save/Load"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Data Save/Load")
    print("=" * 60)

    # Create sample data
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'value': [100.5, 200.3, 300.7, 400.1, 500.9],
        'category': ['A', 'B', 'A', 'C', 'B']
    })

    print("Original data:")
    print(df)
    print("\nOriginal dtypes:")
    print(df.dtypes)

    # Save and load
    persister = PickleSerializer()
    saved_path = persister.save_data(df, 'outputs/basic_data.pkl')
    loaded_df = persister.load_data('outputs/basic_data.pkl')

    print("\nLoaded data:")
    print(loaded_df)
    print("\nDtypes match: ", df.dtypes.equals(loaded_df.dtypes))
    print("Data identical: ", df.equals(loaded_df))

    return loaded_df


def example_complex_dtypes():
    """Example 2: Complex Dtypes Preservation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Complex Dtypes Preservation")
    print("=" * 60)

    import numpy as np

    # Create data with various dtypes
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=5),
        'datetime_tz': pd.date_range('2024-01-01', periods=5, tz='UTC'),
        'int32': pd.array([1, 2, 3, 4, 5], dtype='int32'),
        'nullable_int': pd.array([1, 2, None, 4, 5], dtype='Int64'),
        'float32': pd.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype='float32'),
        'category': pd.Categorical(['cat1', 'cat2', 'cat1', 'cat3', 'cat2']),
        'string': pd.array(['a', 'b', 'c', 'd', 'e'], dtype='string'),
        'bool': [True, False, True, False, True],
        'timedelta': pd.timedelta_range('1 day', periods=5),
        'period': pd.period_range('2024-01', periods=5, freq='M')
    })

    print("Original dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    # Save and load
    persister = PickleSerializer()
    persister.save_data(df, 'outputs/complex_dtypes.pkl')
    loaded_df = persister.load_data('outputs/complex_dtypes.pkl')

    print("\nLoaded dtypes:")
    for col, dtype in loaded_df.dtypes.items():
        match = "✓" if dtype == df[col].dtype else "✗"
        print(f"  {col}: {dtype} {match}")

    print(f"\nAll dtypes preserved: {df.dtypes.equals(loaded_df.dtypes)}")
    print(f"Data identical: {df.equals(loaded_df)}")

    return loaded_df


def example_tsfresh_data():
    """Example 3: TSFRESH Data (with inf/nan values)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: TSFRESH Data with inf/nan")
    print("=" * 60)

    import numpy as np

    # Create TSFRESH-like data with inf and nan
    df = pd.DataFrame({
        'CUST_ACCOUNT_NUMBER': [1001, 1002, 1003, 1004],
        'value__mean': [10.5, np.inf, 30.2, 40.1],
        'value__variance': [np.nan, 25.3, np.inf, 35.2],
        'value__skewness': [-np.inf, 0.5, 1.2, np.nan],
        'value__maximum': [100, 200, 300, 400]
    })

    print("Original data (with inf/nan):")
    print(df)

    # Save and load - preserves inf/nan exactly as is
    persister = PickleSerializer()
    persister.save_data(df, 'outputs/tsfresh_data.pkl')
    loaded_df = persister.load_data('outputs/tsfresh_data.pkl')

    print("\nLoaded data (inf/nan preserved):")
    print(loaded_df)

    print(f"\nData identical (including inf/nan): {df.equals(loaded_df)}")

    return loaded_df


def example_your_use_case():
    """Example 4: Your Specific Use Case"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Your Specific Use Case")
    print("=" * 60)

    # Simulate your merge_final data
    merge_final = pd.DataFrame({
        'CUST_ACCOUNT_NUMBER': pd.array([1, 2, 3], dtype='Int64'),
        'DOCUMENTS_OPENED': [10.5, 20.3, 30.1],
        'USED_STORAGE_MB': [100.2, 200.5, 300.8],
        'STATUS': pd.Categorical(['active', 'inactive', 'active']),
        'LAST_LOGIN': pd.date_range('2024-01-01', periods=3)
    })

    print("Original merge_final dtypes:")
    print(merge_final.dtypes)

    # Your code pattern
    persister = PickleSerializer()

    # Save
    PS_DOCUWARE_RAW_DATA_EXTRACTION = Path('outputs/PS_DOCUWARE_RAW_DATA_EXTRACTION.pkl')
    persister.save_data(merge_final, PS_DOCUWARE_RAW_DATA_EXTRACTION)

    # Load
    raw_df = persister.load_data(PS_DOCUWARE_RAW_DATA_EXTRACTION)

    print(f"\nLoaded raw_df dtypes:")
    print(raw_df.dtypes)

    # This will be True!
    print(f"\nDtypes match: {raw_df.dtypes.equals(merge_final.dtypes)}")
    print(f"Data identical: {raw_df.equals(merge_final)}")

    return raw_df


def main():
    """Run all examples"""

    # Run Example 1: Basic usage
    loaded_basic = example_basic_usage()

    # Run Example 2: Complex dtypes
    loaded_complex = example_complex_dtypes()

    # Run Example 3: TSFRESH data with inf/nan
    loaded_tsfresh = example_tsfresh_data()

    # Run Example 4: Your use case
    loaded_your_case = example_your_use_case()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
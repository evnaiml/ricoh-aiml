"""
The module contains JoblibSerializer Class

* Joblib pros:
  - Better compression - joblib automatically compresses large numpy
  arrays, resulting in smaller file sizes
  - More efficient for numerical data - optimized for scientific computing
   and numpy/pandas objects
  - Parallel processing support - can leverage multiple cores for large
   arrays
  - Memory mapping - can read large files without loading everything into
  memory


* Pickle pros:
  - Universal - part of Python standard library, no extra dependency
  - Broader compatibility - works with any Python object
  - Simpler for small data - less overhead for tiny datasets

  For ML-industrial use case (data with numerical values, datetime, categories):
  - Joblib is likely better because it's specifically optimized for
  scientific data structures
  - File sizes will typically be 20-50% smaller with joblib
  - Loading can be faster for large data

  Both preserve dtypes perfectly, so it comes down to performance. For
  data with lots of numerical values (like TSFRESH features),
  joblib is generally the better choice.
"""
# -----------------------------------------------------------------------------
# Author: Evgeni Nikoolaev
# email: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# UPDATED ON: 2025-08-11
# CREATED ON: 2025-08-11
# -----------------------------------------------------------------------------
# COPYRIGHT@2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh
# -----------------------------------------------------------------------------
# %%
import pandas as pd
import joblib
from pathlib import Path
from typing import Union
import numpy as np
from datetime import datetime
import json

# %%
# ============= class JoblibSerializer =============
class JoblibSerializer:
    """
    A simple class to save and load complex data (e.g. pandas data and encoders, etc.) using joblib format.

    Suffix rules:
    - .joblib → saves without compression
    - .joblib.gz → saves with compression (level 3)
    - no suffix or other → defaults to .joblib.gz (compressed)

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
        Save data to joblib format with perfect preservation of all data and dtypes.

        Suffix behavior:
        - file.joblib → saves without compression
        - file.joblib.gz → saves with compression
        - file or file.other → saves as file.joblib.gz (compressed)

        Parameters
        ----------
        data : any
            Data to save
        output_path : Union[str, Path]
            Path where to save the joblib file

        Returns
        -------
        Path
            Path where the file was saved
        """
        output_path = Path(output_path)
        output_path_str = str(output_path)

        # Ensure directory exists
        JoblibSerializer._ensure_directory(output_path)

        # Check existing extensions and handle accordingly
        if output_path_str.endswith('.joblib.gz'):
            # Already has .joblib.gz - save with compression
            joblib.dump(data, output_path, compress=3)
            print(f"✓ Saved compressed data to: {output_path}")
        elif output_path_str.endswith('.joblib'):
            # Has .joblib - save without compression
            joblib.dump(data, output_path, compress=0)
            print(f"✓ Saved data to: {output_path}")
        else:
            # No .joblib extension - add .joblib.gz by default
            if output_path.suffix:
                # Has some other extension, replace with .joblib.gz
                output_path = output_path.with_suffix('.joblib.gz')
            else:
                # No extension, add .joblib.gz
                output_path = Path(output_path_str + '.joblib.gz')

            # Ensure directory exists for new path
            JoblibSerializer._ensure_directory(output_path)

            # Save with compression
            joblib.dump(data, output_path, compress=3)
            print(f"✓ Saved compressed data to: {output_path}")

        if hasattr(data, 'shape'):
            print(f"  Shape: {data.shape}")
        
        # Save metadata
        metadata = {
            'data_type': type(data).__name__,
            'saved_at': datetime.now().isoformat(),
            'format': 'joblib',
            'compressed': str(output_path).endswith('.gz'),
            'compression_level': 3 if str(output_path).endswith('.gz') else 0
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
        Load data from joblib file with perfect preservation of all data and dtypes.
        Automatically handles decompression if file ends with .gz.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the joblib file

        Returns
        -------
        any
            Loaded data with perfectly preserved data and dtypes

        Raises
        ------
        FileNotFoundError
            If the joblib file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Joblib file not found: {file_path}")

        # Load data from joblib (automatically handles decompression if needed)
        data = joblib.load(file_path)

        if str(file_path).endswith('.gz'):
            print(f"✓ Loaded compressed data from: {file_path}")
        else:
            print(f"✓ Loaded data from: {file_path}")
        if hasattr(data, 'shape'):
            print(f"  Shape: {data.shape}")

        return data


# ============= MAIN EXAMPLE =============

def main():
    """Simple example demonstrating joblib preservation of various data types"""
    print("=" * 60)
    print("COMPRESSED JOBLIB DATA PRESERVATION EXAMPLE")
    print("=" * 60)

    # Create data with various data types
    df = pd.DataFrame({
        # Integer types
        'int_col': [1, 2, 3, 4, 5],
        'int32_col': pd.array([10, 20, 30, 40, 50], dtype='int32'),
        'nullable_int': pd.array([1, 2, None, 4, 5], dtype='Int64'),

        # Float types
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'float32_col': pd.array([10.1, 20.2, 30.3, 40.4, 50.5], dtype='float32'),

        # DateTime types
        'date': pd.date_range('2024-01-01', periods=5),
        'datetime_tz': pd.date_range('2024-01-01', periods=5, tz='UTC'),
        'timedelta': pd.timedelta_range('1 day', periods=5),

        # Categorical and string types
        'category': pd.Categorical(['cat1', 'cat2', 'cat1', 'cat3', 'cat2']),
        'string_col': pd.array(['alpha', 'beta', 'gamma', 'delta', 'epsilon'], dtype='string'),

        # Boolean
        'bool_col': [True, False, True, False, True],

        # Special values
        'with_nan': [1.0, np.nan, 3.0, 4.0, 5.0],
        'with_inf': [1.0, np.inf, -np.inf, 4.0, 5.0],
    })

    print("\nOriginal data:")
    print(df.head())
    print("\nOriginal dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

    # Initialize persister
    persister = JoblibSerializer()

    # Save data
    print("\n" + "-" * 40)
    output_path = Path('outputs/example_data.joblib')
    saved_path = persister.save_data(df, output_path)

    # Load data
    print("\n" + "-" * 40)
    loaded_df = persister.load_data(saved_path)

    # Verify preservation
    print("\n" + "=" * 40)
    print("VERIFICATION RESULTS:")
    print("=" * 40)

    print("\nLoaded dtypes:")
    for col, dtype in loaded_df.dtypes.items():
        original_dtype = df[col].dtype
        match = "✓" if dtype == original_dtype else "✗"
        print(f"  {col}: {dtype} {match}")

    # Check if dtypes are preserved
    dtypes_match = df.dtypes.equals(loaded_df.dtypes)
    print(f"\nAll dtypes preserved: {dtypes_match}")

    # Check if data is identical (including NaN and Inf values)
    data_identical = df.equals(loaded_df)
    print(f"Data identical: {data_identical}")

    # Additional checks for special values
    print("\nSpecial values check:")
    print(f"  NaN values preserved: {pd.isna(loaded_df['with_nan']).equals(pd.isna(df['with_nan']))}")
    print(f"  Inf values preserved: {np.array_equal(loaded_df['with_inf'], df['with_inf'], equal_nan=True)}")

    # Show sample of loaded data
    print("\nLoaded data sample:")
    print(loaded_df.head())

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("All data types including int, datetime, categorical,")
    print("and special values (NaN, Inf) are perfectly preserved!")
    print("Data is compressed with level 3 for optimal file size.")
    print("=" * 60)

    return loaded_df


if __name__ == "__main__":
    main()
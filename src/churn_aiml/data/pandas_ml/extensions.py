"""
Pandas ML Extensions Module

This module extends pandas DataFrame functionality with machine learning focused methods
using pandas-flavor for clean method registration. The extensions are designed to
streamline common data preprocessing tasks in ML pipelines, particularly for:

- Type conversion with memory optimization (float32 vs float64)
- Graceful handling of mixed/invalid data types
- Time series preprocessing and feature engineering
- GPU-accelerated ML model preparation (CatBoost, XGBoost, etc.)

Key Benefits:
- Memory efficient: Default to float32 for 50% memory savings
- Error resilient: Converts invalid values to NaN instead of failing
- Method chaining: Seamlessly integrates with pandas workflow
- Performance optimized: Faster training for tree-based models

Usage:
    Import this module to automatically register all extension methods:

    >>> from pandas_ml.extensions import to_float
    >>> df = pd.DataFrame({'col': ['1', '2', 'invalid']})
    >>> result = df.to_float()  # Now available on all DataFrames
    >>> chained = df.to_float().fillna(0).round(2)  # Perfect chaining

Installation Requirements:
    pip install pandas-flavor

Future Extensions:
    This module can be extended with additional ML preprocessing methods like:
    - df.encode_categorical() - Smart categorical encoding
    - df.scale_features() - Feature scaling with multiple strategies
    - df.create_lags() - Time series lag feature generation
    - df.detect_outliers() - Outlier detection and handling
"""
# %%
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-04
# * CREATED ON: 2025-08-04
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pandas_flavor import register_dataframe_method
from typing import Union

# %%
# -----------------------------------------------------------------------------
@register_dataframe_method
def to_float(df: pd.DataFrame, dtype: Union[np.dtype, type] = np.float32) -> pd.DataFrame:
    """
    Convert all DataFrame columns to specified float type, handling non-numeric values gracefully.

    This method converts all columns in a DataFrame to numeric values, replacing
    invalid/non-convertible values with NaN, then casts everything to the specified
    float dtype for memory efficiency and faster ML training.

    Parameters
    ----------
    dtype : numpy.dtype, default np.float32
        Target float dtype for conversion. Common options:
        - np.float32: Half memory usage, sufficient for most ML tasks
        - np.float64: Higher precision, standard NumPy default

    Returns
    -------
    pandas.DataFrame
        DataFrame with all columns converted to specified float dtype.
        Non-numeric values are replaced with NaN.

    Notes
    -----
    - Uses `errors='coerce'` to convert invalid values to NaN instead of raising errors
    - float32 provides sufficient precision for most ML tasks while using half the memory
    - Particularly beneficial for time series forecasting and GPU-accelerated training

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'A': ['1', '2', 'invalid', '4'],
    ...     'B': ['1.5', '2.7', '3.1', 'not_a_number'],
    ...     'C': [1, 2, 3, 4]
    ... })
    >>>
    >>> # Default float32
    >>> result = df.to_float()
    >>> print(result.dtypes)
    A    float32
    B    float32
    C    float32
    dtype: object

    >>> # Explicit float64
    >>> result_64 = df.to_float(dtype=np.float64)
    >>> print(result_64.dtypes)
    A    float64
    B    float64
    C    float64
    dtype: object

    >>> # Method chaining
    >>> result = df.to_float().fillna(0).round(2)
    """
    return df.apply(pd.to_numeric, errors='coerce').astype(dtype)


# Usage example
if __name__ == "__main__":
    # Sample data
    df = pd.DataFrame({
        'A': ['1', '2', 'invalid', '4'],
        'B': ['1.5', '2.7', '3.1', 'not_a_number'],
        'C': [1, 2, 3, 4]
    })

    print("Original DataFrame:")
    print(df)
    print(f"Original dtypes:\n{df.dtypes}\n")

    # Now you can use it directly on a DataFrame
    result = df.to_float()  # Uses float32 by default
    print("After conversion:")
    print(result)
    print(f"Result dtypes:\n{result.dtypes}")

    # With float64
    result_64 = df.to_float(dtype=np.float64)
    print(f"\nFloat64 dtypes:\n{result_64.dtypes}")

    # Method chaining
    chained_result = df.to_float().fillna(-999).round(1)
    print(f"\nChained result:\n{chained_result}")
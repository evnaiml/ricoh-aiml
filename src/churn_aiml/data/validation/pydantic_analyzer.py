"""
Pydantic Data Analyzer for advanced type inference and validation.

This module provides intelligent type inference for DataFrames, including:
- Automatic detection of integers and floats stored as strings
- DateTime pattern recognition in string columns
- Proper handling of high-cardinality categorical variables
- Binary and categorical variable classification
- NaN substitution for failed type conversions
- Removal of unnecessary decimal zeros from numeric values

Key Features:
- Converts string columns containing integers to int type
- Converts string columns containing floats to float type
- Converts string columns with datetime patterns to datetime objects
- Uses NaN for values that cannot be converted to the inferred type
- Generates Pydantic-compatible schemas with proper type annotations
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-13
# * CREATED ON: 2025-07-31
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, create_model, ValidationError
from datetime import datetime, date
import json
import re

class DataTypeInferrer:
    """Infers data types for DataFrame columns using Pydantic-compatible types"""

    def __init__(self):
        self.categorical_threshold = 10  # Max unique values for categorical

    def infer_column_type(self, series: pd.Series) -> Dict[str, Any]:
        """Infer the most appropriate data type for a pandas Series"""

        # Basic statistics
        total_count = len(series)
        non_null_count = series.count()
        missing_count = total_count - non_null_count
        unique_count = series.nunique()

        # Handle empty or all-null series
        if non_null_count == 0:
            return {
                'inferred_type': 'str',
                'pydantic_type': 'Optional[str]',
                'nullable': True,
                'missing_count': missing_count,
                'unique_count': 0,
                'sample_values': [],
                'actual_dtype': 'str'
            }

        # Get sample of non-null values
        sample_values = series.dropna().head(10).tolist()

        # Check if string column contains numeric values (check for floats first, then integers)
        if series.dtype == 'object':
            numeric_type = self._check_string_numeric_type(series)
            
            if numeric_type == 'float':
                # Convert sample values to floats, use NaN if conversion fails
                sample_values = []
                for v in series.dropna().head(10):
                    try:
                        fval = float(v)
                        if fval.is_integer():
                            sample_values.append(int(fval))
                        else:
                            sample_values.append(fval)
                    except (ValueError, TypeError):
                        sample_values.append(np.nan)
                
                return {
                    'inferred_type': 'num',
                    'pydantic_type': 'Optional[float]' if missing_count > 0 else 'float',
                    'nullable': missing_count > 0,
                    'missing_count': missing_count,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'actual_dtype': 'inferred_float'
                }
            
            elif numeric_type == 'integer':
                # Convert sample values to integers, use NaN if conversion fails
                sample_values = []
                for v in series.dropna().head(10):
                    try:
                        sample_values.append(int(float(v)))
                    except (ValueError, TypeError):
                        sample_values.append(np.nan)
            
            if unique_count == 2:
                # Integer with exactly 2 values -> bin_int
                return {
                    'inferred_type': 'bin_int',
                    'pydantic_type': 'Optional[int]' if missing_count > 0 else 'int',
                    'nullable': missing_count > 0,
                    'missing_count': missing_count,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'actual_dtype': 'inferred_int'
                }
            elif unique_count <= 10:
                # Integer with <= 10 unique values -> cat_int(count)
                return {
                    'inferred_type': f'cat_int({unique_count})',
                    'pydantic_type': 'Optional[int]' if missing_count > 0 else 'int',
                    'nullable': missing_count > 0,
                    'missing_count': missing_count,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'actual_dtype': 'inferred_int'
                }
            else:
                # Regular integer or high cardinality categorical integer
                if unique_count > self.categorical_threshold:
                    return {
                        'inferred_type': f'cat_high_card({unique_count})',
                        'pydantic_type': 'Optional[int]' if missing_count > 0 else 'int',
                        'nullable': missing_count > 0,
                        'missing_count': missing_count,
                        'unique_count': unique_count,
                        'sample_values': sample_values,
                        'actual_dtype': 'inferred_int'
                    }
                else:
                    return {
                        'inferred_type': 'int',
                        'pydantic_type': 'Optional[int]' if missing_count > 0 else 'int',
                        'nullable': missing_count > 0,
                        'missing_count': missing_count,
                        'unique_count': unique_count,
                        'sample_values': sample_values,
                        'actual_dtype': 'inferred_int'
                    }
        
        # Check if string column contains datetime patterns
        if series.dtype == 'object' and self._is_datetime_pattern(series):
            # Convert sample values to datetime format
            datetime_samples = self._convert_to_datetime_objects(series.dropna().head(10))
            
            return {
                'inferred_type': 'datetime',
                'pydantic_type': 'Optional[datetime]' if missing_count > 0 else 'datetime',
                'nullable': missing_count > 0,
                'missing_count': missing_count,
                'unique_count': unique_count,
                'sample_values': datetime_samples,
                'actual_dtype': 'inferred_datetime'
            }

        # Determine base type
        dtype_str = str(series.dtype).lower()

        # Integer type handling (check first for specific integer rules)
        if self._is_integer_like(series):
            # Remove decimal zeros from sample values if they exist, use NaN on error
            processed_samples = []
            for v in sample_values:
                try:
                    if isinstance(v, float) and v.is_integer():
                        processed_samples.append(int(v))
                    else:
                        processed_samples.append(v)
                except:
                    processed_samples.append(np.nan)
            sample_values = processed_samples
            
            if unique_count == 2:
                # Integer with exactly 2 values -> bin_int
                return {
                    'inferred_type': 'bin_int',
                    'pydantic_type': 'Optional[int]' if missing_count > 0 else 'int',
                    'nullable': missing_count > 0,
                    'missing_count': missing_count,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'actual_dtype': 'int'
                }
            elif unique_count <= 10:
                # Integer with <= 10 unique values -> cat_int(count)
                return {
                    'inferred_type': f'cat_int({unique_count})',
                    'pydantic_type': 'Optional[int]' if missing_count > 0 else 'int',
                    'nullable': missing_count > 0,
                    'missing_count': missing_count,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'actual_dtype': 'int'
                }
            else:
                # Regular integer
                return {
                    'inferred_type': 'int',
                    'pydantic_type': 'Optional[int]' if missing_count > 0 else 'int',
                    'nullable': missing_count > 0,
                    'missing_count': missing_count,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'actual_dtype': 'int'
                }

        # Boolean detection (check for general binary patterns)
        if self._is_binary(series):
            return {
                'inferred_type': 'bin',
                'pydantic_type': 'Optional[bool]' if missing_count > 0 else 'bool',
                'nullable': missing_count > 0,
                'missing_count': missing_count,
                'unique_count': unique_count,
                'sample_values': sample_values,
                'actual_dtype': 'bool'
            }

        # Float detection
        if self._is_numeric(series):
            # Remove unnecessary decimal zeros, use NaN on error
            processed_samples = []
            for v in sample_values:
                try:
                    if isinstance(v, float) and v.is_integer():
                        processed_samples.append(int(v))
                    else:
                        processed_samples.append(v)
                except:
                    processed_samples.append(np.nan)
            sample_values = processed_samples
            
            return {
                'inferred_type': 'num',
                'pydantic_type': 'Optional[float]' if missing_count > 0 else 'float',
                'nullable': missing_count > 0,
                'missing_count': missing_count,
                'unique_count': unique_count,
                'sample_values': sample_values,
                'actual_dtype': 'float'
            }

        # Date/datetime detection
        if self._is_datetime_like(series):
            # Convert datetime values to datetime objects for consistent handling
            datetime_samples = self._convert_series_to_datetime(series.dropna().head(10))
            return {
                'inferred_type': 'datetime',
                'pydantic_type': 'Optional[datetime]' if missing_count > 0 else 'datetime',
                'nullable': missing_count > 0,
                'missing_count': missing_count,
                'unique_count': unique_count,
                'sample_values': datetime_samples,
                'actual_dtype': 'datetime'
            }

        # Categorical detection (for non-integer types)
        if unique_count <= self.categorical_threshold and unique_count > 1:
            if unique_count == 2:
                # Categorical with exactly 2 values -> bin
                return {
                    'inferred_type': 'bin',
                    'pydantic_type': 'Optional[str]' if missing_count > 0 else 'str',
                    'nullable': missing_count > 0,
                    'missing_count': missing_count,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'categories': series.dropna().unique().tolist(),
                    'actual_dtype': 'str'
                }
            else:
                # Regular categorical
                return {
                    'inferred_type': f'cat({unique_count})',
                    'pydantic_type': 'Optional[str]' if missing_count > 0 else 'str',
                    'nullable': missing_count > 0,
                    'missing_count': missing_count,
                    'unique_count': unique_count,
                    'sample_values': sample_values,
                    'categories': series.dropna().unique().tolist()[:20],  # Limit categories shown
                    'actual_dtype': 'str'
                }

        # High cardinality categorical
        if unique_count > self.categorical_threshold:
            return {
                'inferred_type': f'cat_high_card({unique_count})',
                'pydantic_type': 'Optional[str]' if missing_count > 0 else 'str',
                'nullable': missing_count > 0,
                'missing_count': missing_count,
                'unique_count': unique_count,
                'sample_values': sample_values,
                'actual_dtype': 'str'
            }

        # Default to string with unique count
        return {
            'inferred_type': f'str({unique_count})',
            'pydantic_type': 'Optional[str]' if missing_count > 0 else 'str',
            'nullable': missing_count > 0,
            'missing_count': missing_count,
            'unique_count': unique_count,
            'sample_values': sample_values,
            'actual_dtype': 'str'
        }

    def _check_string_numeric_type(self, series: pd.Series) -> Optional[str]:
        """Check if string column contains numeric values and determine type"""
        if series.dtype != 'object':
            return None
        
        non_null = series.dropna()
        if len(non_null) == 0:
            return None
        
        is_integer = True
        is_float = True
        
        # Check first 100 values for performance
        for val in non_null.head(100):
            try:
                str_val = str(val).strip()
                if not str_val:
                    return None
                
                # Try to convert to float
                fval = float(str_val)
                
                # Check if it's actually an integer
                if not fval.is_integer():
                    is_integer = False
                    
            except (ValueError, TypeError):
                # Not a numeric value at all
                return None
        
        # Return float first since floats include integers
        if is_float and not is_integer:
            return 'float'
        elif is_integer:
            return 'integer'
        else:
            return None
    
    def _is_datetime_pattern(self, series: pd.Series) -> bool:
        """Check if string column contains datetime patterns"""
        if series.dtype != 'object':
            return False
        
        non_null = series.dropna()
        if len(non_null) == 0:
            return False
        
        # Common datetime patterns
        datetime_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{4}-\d{2}$',  # YYYY-MM
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',  # YYYY-MM-DD HH:MM:SS
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
        ]
        
        # Check sample values
        sample = non_null.head(20)
        date_like_count = 0
        
        for val in sample:
            str_val = str(val).strip()
            # Skip special values like "Latest Month"
            if any(keyword in str_val.lower() for keyword in ['latest', 'current', 'previous', 'next']):
                continue
            
            # Check against patterns
            for pattern in datetime_patterns:
                if re.match(pattern, str_val):
                    date_like_count += 1
                    break
            else:
                # Try parsing as datetime
                try:
                    pd.to_datetime(str_val)
                    date_like_count += 1
                except:
                    pass
        
        # If more than 50% of sample looks like dates, consider it datetime
        return date_like_count >= len(sample) * 0.5
    
    def _convert_to_datetime_objects(self, sample_series: pd.Series) -> List:
        """Convert string samples to datetime objects"""
        converted_samples = []
        
        for val in sample_series:
            str_val = str(val).strip()
            
            # Keep special values as is
            if any(keyword in str_val.lower() for keyword in ['latest', 'current', 'previous', 'next', 'last']):
                converted_samples.append(str_val)
            else:
                # Try to parse as datetime and return datetime object
                try:
                    dt = pd.to_datetime(str_val)
                    # Return as Timestamp for consistency
                    converted_samples.append(dt)
                except:
                    # Use NaN if datetime conversion fails
                    converted_samples.append(np.nan)
        
        return converted_samples
    
    def _convert_series_to_datetime(self, sample_series: pd.Series) -> List:
        """Convert pandas datetime series to datetime objects"""
        converted_samples = []
        
        for val in sample_series:
            if pd.isna(val):
                continue
            elif isinstance(val, (pd.Timestamp, datetime)):
                converted_samples.append(val)
            else:
                try:
                    dt = pd.to_datetime(val)
                    converted_samples.append(dt)
                except:
                    # Use NaN if datetime conversion fails
                    converted_samples.append(np.nan)
        
        return converted_samples

    def _is_binary(self, series: pd.Series) -> bool:
        """Check if series is binary (boolean-like)"""
        unique_vals = series.dropna().unique()
        if len(unique_vals) != 2:
            return False

        # Check for common binary patterns
        str_vals = {str(v).lower() for v in unique_vals}
        binary_patterns = [
            {'true', 'false'}, {'yes', 'no'}, {'y', 'n'},
            {'1', '0'}, {'1.0', '0.0'}, {'active', 'inactive'}
        ]

        return any(str_vals == pattern for pattern in binary_patterns) or set(unique_vals) == {0, 1}

    def _is_integer_like(self, series: pd.Series) -> bool:
        """Check if series contains integer-like values"""
        if series.dtype.kind in ['i', 'u']:  # Integer types
            return True

        if series.dtype.kind == 'f':  # Float type
            # Check if all values are whole numbers
            non_null = series.dropna()
            if len(non_null) == 0:
                return False
            return np.all(non_null == non_null.astype(int))

        return False

    def _is_numeric(self, series: pd.Series) -> bool:
        """Check if series contains numeric values"""
        return series.dtype.kind in ['i', 'u', 'f', 'c']

    def _is_datetime_like(self, series: pd.Series) -> bool:
        """Check if series contains datetime-like values"""
        if series.dtype.kind == 'M':  # Datetime type
            return True

        # Try to parse a sample as datetime
        if series.dtype == 'object':
            try:
                sample = series.dropna().head(5)
                for val in sample:
                    pd.to_datetime(str(val))
                return True
            except:
                return False

        return False

class PydanticDataAnalyzer:
    """Main analyzer class for creating Pydantic schemas and validation"""

    def __init__(self):
        self.inferrer = DataTypeInferrer()

    def analyze_dataframe(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Analyze DataFrame and create Pydantic-compatible schema"""

        features = {}

        # Analyze each column
        for column in df.columns:
            series = df[column]
            type_info = self.inferrer.infer_column_type(series)

            # Add default role
            type_info['role'] = 1
            type_info['table_name'] = table_name

            features[column] = type_info

        # Create Pydantic model definition
        schema = self._create_pydantic_schema(features, table_name)

        return {
            'table_name': table_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'schema': schema,
            'features': features,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _create_pydantic_schema(self, features: Dict[str, Dict], table_name: str) -> Dict[str, Any]:
        """Create a Pydantic model schema from feature analysis"""

        field_definitions = {}

        for column_name, feature_info in features.items():
            pydantic_type = feature_info['pydantic_type']

            # Create field definition
            if feature_info['nullable']:
                field_definitions[column_name] = {
                    'type': pydantic_type,
                    'required': False,
                    'default': None,
                    'description': f"{feature_info['inferred_type']} field with {feature_info['missing_count']} missing values"
                }
            else:
                field_definitions[column_name] = {
                    'type': pydantic_type,
                    'required': True,
                    'description': f"{feature_info['inferred_type']} field"
                }

            # Add additional metadata
            field_definitions[column_name]['metadata'] = {
                'inferred_type': feature_info['inferred_type'],
                'unique_count': feature_info['unique_count'],
                'missing_count': feature_info['missing_count'],
                'sample_values': feature_info['sample_values'][:5],  # Limit samples in schema
                'actual_dtype': feature_info.get('actual_dtype', 'unknown')
            }

        return {
            'model_name': f"{table_name}Model",
            'fields': field_definitions,
            'description': f"Pydantic model for {table_name} table",
            'created_at': datetime.now().isoformat()
        }

    def validate_dataframe(self, df: pd.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DataFrame against generated schema"""

        validation_results = {
            'valid_rows': 0,
            'invalid_rows': 0,
            'validation_errors': [],
            'summary': {}
        }

        try:
            # Create dynamic Pydantic model
            field_definitions = {}
            for field_name, field_info in schema['fields'].items():
                # Simplified field creation for validation
                if field_info['required']:
                    field_definitions[field_name] = (str, ...)  # Required string
                else:
                    field_definitions[field_name] = (Optional[str], None)  # Optional string

            DynamicModel = create_model(schema['model_name'], **field_definitions)

            # Validate sample rows (for performance with large datasets)
            sample_size = min(1000, len(df))
            sample_df = df.sample(n=sample_size) if len(df) > sample_size else df

            valid_count = 0
            for idx, row in sample_df.iterrows():
                try:
                    model_instance = DynamicModel(**row.to_dict())
                    valid_count += 1
                except ValidationError as e:
                    validation_results['validation_errors'].append({
                        'row_index': idx,
                        'errors': str(e)
                    })

            validation_results['valid_rows'] = valid_count
            validation_results['invalid_rows'] = sample_size - valid_count
            validation_results['validation_rate'] = valid_count / sample_size if sample_size > 0 else 0

        except Exception as e:
            validation_results['validation_errors'].append({
                'error': f"Schema validation failed: {str(e)}"
            })

        return validation_results
"""
Comprehensive Snowflake data fetching and analysis module with Hydra configuration integration.

This production-ready module provides a complete suite of tools for Snowflake data operations,
type validation, and comprehensive analysis capabilities suitable for enterprise environments.

## Module Organization:

### Core Classes:
1. **SnowData** (Abstract Base Class)
   - Defines the interface for Snowflake data operations

2. **SnowFetch** (Main Implementation)
   - Complete Snowflake data fetching with type enforcement
   - Configuration-based schema path resolution
   - Support for raw and join-rules schemas
   - Intelligent type conversion (string → int/float/datetime)
   - NaN/NaT handling for failed conversions

### Analysis Classes:
3. **DtypeTransformationAnalyzer**
   - Production-grade dtype transformation analysis
   - Aggregates conversion statistics across tables
   - Comprehensive reporting for data quality monitoring

4. **JoinRuleAnalyzer**
   - SQL join query analysis with performance metrics
   - Dtype transformation tracking for joined datasets
   - Memory usage and execution time monitoring

### Utility Functions:
5. **log_dtype_transformation_summary()**
   - Aggregated transformation statistics logging
   - High-level overview of type conversions

## Key Features:
- ✅ Automatic type conversion using Pydantic schemas
- ✅ Configuration-based schema path resolution via Hydra
- ✅ Comprehensive analysis methods for transformations
- ✅ Production-grade error handling and logging
- ✅ Context manager support for session management
- ✅ Performance tracking and memory optimization

## Core Methods:
- `fetch_data()` - Main method for fetching tables with filters
- `fetch_data_validation_enforced()` - Fetch with automatic type conversion
- `fetch_custom_query()` - Execute any custom SQL
- `analyze_dtype_transformations()` - Detailed type change analysis
- `get_schema_info()` - Schema metadata retrieval
- `log_join_query_analysis()` - Comprehensive query result analysis

## Usage:
```python
from churn_aiml.data.db.snowflake.fetchdata import (
    SnowFetch,
    DtypeTransformationAnalyzer,
    JoinRuleAnalyzer,
    log_dtype_transformation_summary
)

# Basic usage with context manager
with SnowFetch(config=cfg, environment="development") as fetcher:
    df = fetcher.fetch_data("your_table_name")

# Production analysis
analyzer = DtypeTransformationAnalyzer(logger)
analyzer.analyze_table(fetcher, df_before, df_after, table_name)
```

Updated: 2025-08-13
- Added DtypeTransformationAnalyzer for production-grade analysis
- Added JoinRuleAnalyzer for SQL join query analysis
- Added log_dtype_transformation_summary utility function
- Enhanced with comprehensive docstrings throughout
- Centralized all general-purpose analysis functionality
"""
# %%
# -----------------------------------------------------------------------------
# ✅ What this file contains:
#
# Abstract Base Class: SnowData - defines the interface
# Main Implementation: SnowFetch - complete Snowflake data fetching class
# Configuration Integration: Automatically extracts session parameters from your Hydra config
# Environment Support: Handles both development and production environments
# Comprehensive Methods:
#
# fetch_data() - Main method for fetching tables with filters
# fetch_data_validation_enforced() - Fetch with automatic type conversion using Pydantic schema
# fetch_custom_query() - Execute any custom SQL
# get_table_info() - Get table structure information
# Session management with context manager support
#
# 🔧 Key Features:
#
# Auto-configuration: Reads from config.products.DOCUWARE.db.snowflake.sessions
# Logging Integration: Uses your loguru logger configuration
# Error Handling: Comprehensive exception handling
# Performance Tracking: Logs execution times and metrics
# Flexible Queries: Support for WHERE, ORDER BY, LIMIT, custom columns
# %%
# 📋 Usage:
# # Basic usage
# from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
# # With context manager (recommended)
# with SnowFetch(config=cfg, environment="development") as fetcher:
#     df = fetcher.fetch_data("your_table_name")
# -----------------------------------------------------------------------------
# Author: Evgeni Nikolaev
# emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# UPDATED ON: 2025-08-13
# CREATED ON: 2025-06-13
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh-USA. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh-USA and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh-USA
# -----------------------------------------------------------------------------
# %%
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import time
import json
import os
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from omegaconf import DictConfig

# Suppress the pkg_resources deprecation warning from snowflake-snowpark-python
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message=".*pkg_resources is deprecated.*")
    from snowflake.snowpark import Session
    from snowflake.snowpark.exceptions import SnowparkSQLException

from churn_aiml.loggers.loguru.config import get_logger
from churn_aiml.utils.find_paths import ProjectRootFinder


# =============================================================================
# CORE CLASSES
# =============================================================================

class SnowData(ABC):
    """Abstract class to fetch data from Snowflake with configuration support.

    Args:
        ABC (abc.ABCMeta): Abstract base class
    """

    @abstractmethod
    def fetch_data(self, table_name: str, schema: str = None, **kwargs) -> pd.DataFrame:
        """An abstract method to fetch data from Snowflake.

        Args:
            table_name (str): Snowflake table name
            schema (str, optional): Schema name (uses config default if None)
            **kwargs: Additional query parameters

        Returns:
            pd.DataFrame: A pandas dataframe with Snowflake data
        """
        pass


class SnowFetch(SnowData):
    """Enhanced Snowflake data fetching with type validation and comprehensive analysis.

    This class extends SnowData to provide comprehensive data fetching capabilities
    from Snowflake with automatic type validation, schema enforcement, and built-in
    data analysis methods. It integrates seamlessly with Hydra configuration and
    Pydantic schemas for robust data type management.

    Key Features:
        - Automatic session configuration from Hydra config
        - Type enforcement using Pydantic schemas (raw and join_rules contexts)
        - Dtype transformation analysis with detailed logging
        - Schema information retrieval and validation
        - Query result analysis with memory and null value profiling
        - Support for custom SQL queries and table fetching

    Args:
        SnowData (abc.ABCMeta): Abstract base class for Snowflake data operations
    """

    def __init__(self, config: DictConfig, environment: str = "development"):
        """Initialize SnowFetch with configuration.

        Args:
            config (DictConfig): Hydra configuration object
            environment (str, optional): Environment to use ('development' or 'production').
                                       Defaults to "development".
        """
        self.config = config
        self.environment = environment
        self.logger = get_logger()
        self.session = None
        self._session_config = None

        # Extract session configuration
        self._extract_session_config()

        # Initialize Snowflake session
        self._create_session()

    def _extract_session_config(self):
        """Extract session configuration from the config object."""
        try:
            # Navigate to the session configuration
            session_configs = None

            # Try the new structure: products.DOCUWARE.db.snowflake.sessions
            if hasattr(self.config, 'products') and hasattr(self.config.products, 'DOCUWARE'):
                session_configs = self.config.products.DOCUWARE.db.snowflake.sessions

            # Fallback to old structure: db.snowflake.sessions
            elif hasattr(self.config, 'db'):
                session_configs = self.config.db.snowflake.sessions

            if session_configs is None:
                raise AttributeError("Could not find session configuration in any expected location")

            if not hasattr(session_configs, self.environment):
                raise ValueError(f"Environment '{self.environment}' not found in session configurations")

            self._session_config = getattr(session_configs, self.environment)
            self.logger.info(f"Successfully extracted session config for environment: {self.environment}")

        except AttributeError as e:
            self.logger.error(f"Failed to extract session configuration: {e}")
            self.logger.error("Expected config structure: config.products.DOCUWARE.db.snowflake.sessions or config.db.snowflake.sessions")
            raise RuntimeError(f"Invalid configuration structure: {e}")

    def _create_session(self):
        """Create Snowflake session using configuration parameters."""
        try:
            # Build connection parameters from config
            connection_parameters = {
                'user': self._session_config.user,
                'password': self._session_config.password,
                'account': self._session_config.account,
                'warehouse': self._session_config.warehouse,
                'database': self._session_config.database,
            }

            # Create session
            self.session = Session.builder.configs(connection_parameters).create()

            # Set schema and role
            self.session.use_schema(self._session_config.schema)
            self.session.use_role(self._session_config.role)

            self.logger.info(f"Successfully created Snowflake session for {self.environment} environment")
            self.logger.info(f"Connected to database: {self._session_config.database}, schema: {self._session_config.schema}")

        except Exception as e:
            self.logger.error(f"Failed to create Snowflake session: {e}")
            raise RuntimeError(f"Snowflake connection failed: {e}")

    def fetch_data(self, table_name: str, schema: str = None, **kwargs) -> pd.DataFrame:
        """Fetch data from Snowflake table.

        Args:
            table_name (str): Snowflake table name
            schema (str, optional): Schema name (uses config default if None)
            **kwargs: Additional query parameters like:
                - limit (int): Limit number of rows
                - where_clause (str): WHERE condition
                - columns (List[str]): Specific columns to select
                - order_by (str): ORDER BY clause

        Returns:
            pd.DataFrame: A pandas dataframe with Snowflake data
        """
        if not self.session:
            raise RuntimeError("Snowflake session not initialized")

        try:
            start_time = time.time()

            # Use provided schema or default from config
            target_schema = schema if schema else self._session_config.schema

            # Build base query
            columns = kwargs.get('columns', ['*'])
            if isinstance(columns, list):
                columns_str = ', '.join(columns)
            else:
                columns_str = '*'

            query = f"SELECT {columns_str} FROM {target_schema}.{table_name}"

            # Add WHERE clause if provided
            where_clause = kwargs.get('where_clause')
            if where_clause:
                query += f" WHERE {where_clause}"

            # Add ORDER BY clause if provided
            order_by = kwargs.get('order_by')
            if order_by:
                query += f" ORDER BY {order_by}"

            # Add LIMIT clause if provided
            limit = kwargs.get('limit')
            if limit:
                query += f" LIMIT {limit}"

            self.logger.info(f"Executing query: {query}")

            # Execute query and convert to pandas DataFrame
            snowpark_df = self.session.sql(query)
            pandas_df = snowpark_df.to_pandas()

            end_time = time.time()
            duration = end_time - start_time

            self.logger.info(f"Successfully fetched {len(pandas_df)} rows from {target_schema}.{table_name}")
            self.logger.info(f"Query execution time: {duration:.3f} seconds")

            # Log performance metrics if the method exists
            if hasattr(self.logger, 'log_performance'):
                self.logger.log_performance(
                    operation=f"fetch_data_{table_name}",
                    duration=duration,
                    rows_fetched=len(pandas_df),
                    columns_count=len(pandas_df.columns),
                    table_name=table_name,
                    schema=target_schema
                )

            return pandas_df

        except SnowparkSQLException as e:
            self.logger.error(f"Snowflake SQL error while fetching data from {table_name}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error while fetching data from {table_name}: {e}")
            raise

    def fetch_custom_query(self, query: str) -> pd.DataFrame:
        """Execute a custom SQL query.

        Args:
            query (str): Custom SQL query to execute

        Returns:
            pd.DataFrame: Query results as pandas DataFrame
        """
        if not self.session:
            raise RuntimeError("Snowflake session not initialized")

        try:
            start_time = time.time()

            self.logger.info(f"Executing custom query: {query}")

            # Execute query and convert to pandas DataFrame
            snowpark_df = self.session.sql(query)
            pandas_df = snowpark_df.to_pandas()

            end_time = time.time()
            duration = end_time - start_time

            self.logger.info(f"Successfully executed custom query, fetched {len(pandas_df)} rows")
            self.logger.info(f"Query execution time: {duration:.3f} seconds")

            return pandas_df

        except SnowparkSQLException as e:
            self.logger.error(f"Snowflake SQL error in custom query: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in custom query: {e}")
            raise

    def fetch_custom_query_validation_enforced(self, sql_query: str, query_name: str,
                                              schema_path: Optional[Path] = None,
                                              context: Optional[str] = "join_rules") -> pd.DataFrame:
        """Execute a custom SQL query with enforced data type validation.
        
        This method executes a custom query and applies type conversions based on
        a Pydantic schema JSON file, ensuring consistent data types across all queries.
        
        Args:
            sql_query (str): SQL query to execute
            query_name (str): Name of the query (used to find schema file)
            schema_path (Optional[Path]): Path to the schema JSON file.
                If None, looks for {query_name}_schema.json in configured locations.
            context (Optional[str]): Context for schema resolution ('raw' or 'join_rules').
                Defaults to 'join_rules' for custom queries.
        
        Returns:
            pd.DataFrame: Query results with enforced data types based on schema
        
        Raises:
            FileNotFoundError: If schema file is not found
            ValueError: If schema is invalid or conversion fails
        """
        start_time = time.time()
        
        # First, execute the custom query
        self.logger.info(f"Executing custom query '{query_name}' with type validation enforcement")
        df = self.fetch_custom_query(sql_query)
        
        # Find schema file if not provided
        if schema_path is None:
            schema_path = self._find_schema_path(query_name, context)
        
        # If schema exists, apply type conversions
        if schema_path and schema_path.exists():
            self.logger.info(f"Using schema file: {schema_path}")
            
            # Load the schema
            with open(schema_path, 'r') as f:
                schema_json = json.load(f)
            
            # Apply type conversions based on schema fields
            if 'fields' in schema_json:
                for column_name, field_info in schema_json['fields'].items():
                    if column_name in df.columns:
                        field_type = field_info.get('type', 'str')
                        validator = field_info.get('validator', None)
                        
                        # Handle bool_to_int validator for int fields
                        if field_type == 'int' and validator == 'bool_to_int':
                            df[column_name] = self._convert_bool_to_int(df[column_name])
                        elif field_type == 'int':
                            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('Int64')
                        elif field_type == 'float':
                            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').astype('float64')
                        elif field_type == 'bool':
                            df[column_name] = self._convert_to_bool(df[column_name])
                        elif field_type == 'datetime':
                            df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        else:
            self.logger.warning(f"No schema found for query '{query_name}'. Applying default conversions.")
            # Apply basic conversions for known problematic columns
            if 'CHURNED_FLAG' in df.columns:
                # Convert CHURNED_FLAG to boolean
                df['CHURNED_FLAG'] = self._convert_to_bool(df['CHURNED_FLAG'])
            if 'CUST_ACCOUNT_NUMBER' in df.columns:
                # Ensure CUST_ACCOUNT_NUMBER is Int64
                df['CUST_ACCOUNT_NUMBER'] = pd.to_numeric(df['CUST_ACCOUNT_NUMBER'], errors='coerce').astype('Int64')
        
        end_time = time.time()
        duration = end_time - start_time
        
        self.logger.info(f"Query '{query_name}' completed with type validation. "
                        f"Processing time: {duration:.3f} seconds")
        
        return df

    def get_table_info(self, table_name: str, schema: str = None) -> pd.DataFrame:
        """Get information about a table structure.

        Args:
            table_name (str): Table name
            schema (str, optional): Schema name (uses config default if None)

        Returns:
            pd.DataFrame: Table structure information
        """
        target_schema = schema if schema else self._session_config.schema
        query = f"DESCRIBE TABLE {target_schema}.{table_name}"

        self.logger.info(f"Getting table info for {target_schema}.{table_name}")
        return self.fetch_custom_query(query)

    def fetch_data_validation_enforced(self, table_name: str, schema_path: Optional[Path] = None,
                                      context: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Fetch data with enforced data type validation using Pydantic schema.

        This method fetches data from Snowflake and applies type conversions based on
        a previously generated Pydantic schema JSON file. It ensures that:
        - String columns containing integers are converted to int
        - String columns containing floats are converted to float
        - String columns with datetime patterns are converted to datetime
        - Invalid conversions result in NaN/None values

        Args:
            table_name (str): Name of the table to fetch
            schema_path (Optional[Path]): Path to the schema JSON file.
                If None, looks for {table_name}_schema.json in configured locations.
            context (Optional[str]): Context for schema resolution ('raw' or 'join_rules').
                If None, auto-detects based on current directory or uses default.
            **kwargs: Additional arguments passed to fetch_data (where, order_by, limit, etc.)

        Returns:
            pd.DataFrame: DataFrame with enforced data types based on schema

        Raises:
            FileNotFoundError: If schema file is not found
            ValueError: If schema is invalid or conversion fails
        """
        start_time = time.time()

        # First, fetch the raw data
        self.logger.info(f"Fetching raw data from {table_name} with type validation enforcement")
        df = self.fetch_data(table_name, **kwargs)

        # Find schema file if not provided
        if schema_path is None:
            schema_path = self._find_schema_path(table_name, context)

        self.logger.info(f"Using schema file: {schema_path}")

        # Load the schema
        with open(schema_path, 'r') as f:
            schema = json.load(f)

        if 'fields' not in schema:
            raise ValueError(f"Invalid schema format: 'fields' key not found in {schema_path}")

        # Apply type conversions based on schema
        self.logger.info(f"Applying type conversions for {len(schema['fields'])} columns")

        for column_name, field_info in schema['fields'].items():
            if column_name not in df.columns:
                self.logger.warning(f"Column {column_name} in schema but not in DataFrame")
                continue

            metadata = field_info.get('metadata', {})
            actual_dtype = metadata.get('actual_dtype', 'unknown')
            field_type = field_info.get('type', 'str')
            validator = field_info.get('validator', None)
            inferred_type = metadata.get('inferred_type', '')

            # Check for bool_to_int validator first (highest priority)
            if field_type == 'int' and validator == 'bool_to_int':
                self.logger.debug(f"Converting {column_name} from bool to int (0/1) using validator")
                df[column_name] = self._convert_bool_to_int(df[column_name])
            # Apply conversions based on actual_dtype
            elif actual_dtype == 'inferred_int' or (actual_dtype == 'int' and df[column_name].dtype == 'object'):
                # Convert string to integer
                self.logger.debug(f"Converting {column_name} from string to int")
                df[column_name] = self._convert_to_int(df[column_name])

            elif actual_dtype == 'inferred_float' or (actual_dtype == 'float' and df[column_name].dtype == 'object'):
                # Convert string to float
                self.logger.debug(f"Converting {column_name} from string to float")
                df[column_name] = self._convert_to_float(df[column_name])

            elif actual_dtype == 'inferred_datetime' or (
                actual_dtype == 'datetime' and df[column_name].dtype == 'object'
            ):
                # Convert string to datetime
                self.logger.debug(f"Converting {column_name} from string to datetime")
                df[column_name] = self._convert_to_datetime(df[column_name])

            elif actual_dtype == 'int' and df[column_name].dtype in ['float64', 'float32']:
                # Convert float to int if all values are whole numbers
                self.logger.debug(f"Converting {column_name} from float to int")
                df[column_name] = self._convert_float_to_int(df[column_name])

            elif actual_dtype == 'int' and field_info.get('validator') == 'bool_to_int':
                # Handle bool_to_int validator for int fields
                self.logger.debug(f"Converting {column_name} from bool to int (0/1)")
                df[column_name] = self._convert_bool_to_int(df[column_name])
            elif actual_dtype == 'bool':
                # Handle boolean conversions
                self.logger.debug(f"Converting {column_name} to bool")
                df[column_name] = self._convert_to_bool(df[column_name])

        end_time = time.time()
        duration = end_time - start_time

        self.logger.info(
            f"Successfully applied type validation to {table_name}. "
            f"Processing time: {duration:.3f} seconds"
        )

        # Log performance metrics if available
        if hasattr(self.logger, 'log_performance'):
            self.logger.log_performance(
                operation=f"fetch_data_validation_enforced_{table_name}",
                duration=duration,
                rows_fetched=len(df),
                columns_count=len(df.columns),
                table_name=table_name,
                schema_path=str(schema_path)
            )

        return df

    def _find_schema_path(self, table_name: str, context: Optional[str] = None) -> Path:
        """Find schema path using Hydra configuration.

        Args:
            table_name (str): Name of the table to find schema for
            context (Optional[str]): Context for schema resolution ('raw' or 'join_rules').
                If None, auto-detects based on current directory or uses default.

        Returns:
            Path: Path to the schema file

        Raises:
            FileNotFoundError: If schema file is not found
        """
        # Check for environment variable override first
        env_override = os.environ.get('PYDANTIC_SCHEMA_PATH')
        if env_override:
            env_path = Path(env_override)
            if env_path.exists():
                self.logger.info(f"Using schema from environment override: {env_path}")
                return env_path

        # Get project root
        try:
            project_root = ProjectRootFinder().find_path()
        except Exception as e:
            self.logger.warning(f"Could not find project root: {e}")
            project_root = Path.cwd()

        # Try to get schema configuration from Hydra config
        schema_paths_to_try = []

        # Check if we have schema_paths configuration (check both possible locations)
        schema_config = None
        if hasattr(self.config, 'validation') and hasattr(self.config.validation, 'pydantic') and hasattr(self.config.validation.pydantic, 'schema_paths'):
            if hasattr(self.config.validation.pydantic.schema_paths, 'DOCUWARE'):
                schema_config = self.config.validation.pydantic.schema_paths
        elif hasattr(self.config, 'schema_paths') and hasattr(self.config.schema_paths, 'DOCUWARE'):
            schema_config = self.config.schema_paths

        if schema_config and hasattr(schema_config, 'DOCUWARE'):
            docuware_config = schema_config.DOCUWARE
            base_dir = project_root / docuware_config.base_dir

            # Determine which search path to use
            search_order = []

            # Use explicit context if provided
            if context:
                if context == 'raw':
                    search_order = ['raw', 'join_rules']
                elif context == 'join_rules':
                    search_order = ['join_rules', 'raw']
                else:
                    self.logger.warning(f"Unknown context '{context}', using auto-detection")
                    context = None

            # Auto-detect based on current directory if no context provided
            if not context:
                cwd = Path.cwd()
                if 'no_rules' in str(cwd) or 'raw' in str(cwd):
                    # Prioritize raw schemas
                    search_order = ['raw', 'join_rules']
                elif 'with_rules' in str(cwd) or 'join' in str(cwd):
                    # Prioritize join_rules schemas
                    search_order = ['join_rules', 'raw']
                else:
                    # Use default order from config
                    default_type = docuware_config.resolution.get('default_type', 'raw')
                    if default_type == 'raw':
                        search_order = ['raw', 'join_rules']
                    else:
                        search_order = ['join_rules', 'raw']

            # Build paths to try based on search order
            schema_name = docuware_config.schema_name_pattern.format(table_name=table_name)

            for search_type in search_order:
                if search_type in docuware_config.search_paths:
                    search_dir = docuware_config.search_paths[search_type]
                    schema_paths_to_try.append(base_dir / search_dir / schema_name)

            # Add fallback paths
            if 'fallback_paths' in docuware_config:
                for fallback in docuware_config.fallback_paths:
                    if fallback == '.':
                        schema_paths_to_try.append(Path.cwd() / schema_name)
                    else:
                        schema_paths_to_try.append(project_root / fallback / schema_name)

        else:
            # Fallback to hardcoded paths if configuration not available
            self.logger.warning("Schema paths configuration not found in Hydra config, using defaults")
            schema_paths_to_try = [
                project_root / f"data/products/DOCUWARE/DB/snowflake/csv/raw/{table_name}_schema.json",
                project_root / f"data/products/DOCUWARE/DB/snowflake/csv/join_rules/{table_name}_schema.json",
                Path(f"{table_name}_schema.json"),
            ]

        # Try to find the schema file
        for path in schema_paths_to_try:
            if path.exists():
                self.logger.info(f"Found schema at: {path}")
                return path

        # If not found, raise error with helpful message
        searched_paths = '\n  - '.join(str(p) for p in schema_paths_to_try)
        raise FileNotFoundError(
            f"Schema file not found for table '{table_name}'. "
            f"Searched in:\n  - {searched_paths}\n"
            f"Please provide schema_path explicitly or ensure {table_name}_schema.json exists in one of these locations."
        )

    def _convert_to_int(self, series: pd.Series) -> pd.Series:
        """Convert series to integer, using NaN for failed conversions."""
        def safe_int_convert(val):
            if pd.isna(val):
                return np.nan
            try:
                # Convert to float first to handle decimal strings, then to int
                return int(float(str(val)))
            except (ValueError, TypeError):
                return np.nan

        converted = series.apply(safe_int_convert)
        # Try to convert to nullable integer dtype
        try:
            return converted.astype('Int64')
        except:
            return converted

    def _convert_to_float(self, series: pd.Series) -> pd.Series:
        """Convert series to float, using NaN for failed conversions."""
        def safe_float_convert(val):
            if pd.isna(val):
                return np.nan
            try:
                fval = float(str(val))
                # Return as int if it's a whole number
                if fval.is_integer():
                    return int(fval)
                return fval
            except (ValueError, TypeError):
                return np.nan

        return series.apply(safe_float_convert)

    def _convert_to_datetime(self, series: pd.Series) -> pd.Series:
        """Convert series to datetime, using NaT for failed conversions."""
        def safe_datetime_convert(val):
            if pd.isna(val):
                return pd.NaT

            str_val = str(val).strip()
            # Keep special values as is (they'll become NaT)
            if any(keyword in str_val.lower() for keyword in ['latest', 'current', 'previous', 'next', 'last']):
                # For special keywords, you might want to handle them differently
                # For now, we'll return NaT
                return pd.NaT

            try:
                return pd.to_datetime(str_val)
            except:
                return pd.NaT

        return series.apply(safe_datetime_convert)

    def _convert_float_to_int(self, series: pd.Series) -> pd.Series:
        """Convert float series to int if all values are whole numbers."""
        # Check if all non-null values are whole numbers
        non_null = series.dropna()
        if len(non_null) == 0:
            return series

        if all(val.is_integer() for val in non_null):
            try:
                # Use nullable integer type to preserve NaN
                return series.astype('Int64')
            except:
                # If conversion fails, keep as float but remove decimals where possible
                return series.apply(lambda x: int(x) if pd.notna(x) and x.is_integer() else x)

        return series

    def _convert_to_bool(self, series: pd.Series) -> pd.Series:
        """Convert series to boolean."""
        # Common boolean mappings
        bool_map = {
            'true': True, 'false': False,
            'yes': True, 'no': False,
            'y': True, 'n': False,
            '1': True, '0': False,
            1: True, 0: False,
            'active': True, 'inactive': False
        }

        def safe_bool_convert(val):
            if pd.isna(val):
                return np.nan

            # Convert to lowercase string for comparison
            str_val = str(val).lower().strip()

            if str_val in bool_map:
                return bool_map[str_val]

            # If not in map, try to evaluate as truthy/falsy
            try:
                return bool(val)
            except:
                return np.nan

        return series.apply(safe_bool_convert)

    def _convert_bool_to_int(self, series: pd.Series) -> pd.Series:
        """Convert series to integer with boolean encoding (1=True/Y, 0=False/N)."""
        # Common boolean to int mappings
        bool_to_int_map = {
            # String representations
            'true': 1, 'false': 0,
            'yes': 1, 'no': 0,
            'y': 1, 'n': 0,
            '1': 1, '0': 0,
            # Python boolean values
            True: 1, False: 0,
            # Numeric values
            1: 1, 0: 0
        }

        def safe_bool_to_int_convert(val):
            if pd.isna(val):
                return np.nan

            # Convert to lowercase string for comparison
            if isinstance(val, str):
                str_val = val.lower().strip()
                if str_val in bool_to_int_map:
                    return bool_to_int_map[str_val]
            elif val in bool_to_int_map:
                return bool_to_int_map[val]

            # If not in map, try to convert to boolean first, then to int
            try:
                if isinstance(val, (bool, np.bool_)):
                    return 1 if val else 0
                # Try numeric conversion
                num_val = float(val)
                return 1 if num_val != 0 else 0
            except:
                return np.nan

        result = series.apply(safe_bool_to_int_convert)
        return result.astype('Int64')  # Use nullable integer type

    def log_join_query_analysis(self, query_name: str, sql_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Log detailed analysis of join query results.

        This method analyzes and logs comprehensive information about query results,
        including data types, memory usage, null values, and column statistics.

        Args:
            query_name (str): Name or identifier of the query
            sql_query (str): The SQL query that was executed
            df (pd.DataFrame): The resulting DataFrame from the query

        Returns:
            Dict[str, Any]: Analysis report containing:
                - query_name: Query identifier
                - query_length: Length of SQL query
                - shape: DataFrame shape (rows, columns)
                - dtype_distribution: Count of each data type
                - memory_mb: Memory usage in MB
                - null_analysis: Null value statistics
                - column_summary: Summary of first 10 columns
        """

        # Build analysis report
        analysis_report = {
            'query_name': query_name,
            'query_length': len(sql_query),
            'shape': df.shape,
            'rows': df.shape[0],
            'columns': df.shape[1],
            'dtype_distribution': {},
            'memory_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'null_analysis': {
                'columns_with_nulls': 0,
                'total_nulls': 0,
                'null_details': []
            },
            'column_summary': []
        }

        # Log header
        self.logger.info("\n" + "="*80)
        self.logger.info(f"JOIN QUERY ANALYSIS: {query_name}")
        self.logger.info("="*80)

        # Log basic info
        self.logger.info(f"Query name: {query_name}")
        self.logger.info(f"SQL query length: {len(sql_query)} characters")
        self.logger.info(f"Result shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Analyze data types
        dtype_counts = df.dtypes.value_counts()
        analysis_report['dtype_distribution'] = {str(k): int(v) for k, v in dtype_counts.items()}

        self.logger.info("\nData type distribution:")
        for dtype, count in dtype_counts.items():
            self.logger.info(f"  {dtype}: {count} columns")

        # Log memory usage
        self.logger.info(f"\nMemory usage: {analysis_report['memory_mb']:.2f} MB")

        # Null analysis
        null_counts = df.isnull().sum()
        columns_with_nulls = null_counts[null_counts > 0]
        analysis_report['null_analysis']['columns_with_nulls'] = len(columns_with_nulls)
        analysis_report['null_analysis']['total_nulls'] = int(null_counts.sum())

        if len(columns_with_nulls) > 0:
            self.logger.info(f"\nColumns with null values: {len(columns_with_nulls)}/{len(df.columns)}")
            for col in columns_with_nulls.head(10).index:
                null_pct = (null_counts[col] / len(df)) * 100
                self.logger.info(f"  {col}: {null_counts[col]:,} nulls ({null_pct:.1f}%)")
                analysis_report['null_analysis']['null_details'].append({
                    'column': col,
                    'null_count': int(null_counts[col]),
                    'null_percentage': null_pct
                })
            if len(columns_with_nulls) > 10:
                self.logger.info(f"  ... and {len(columns_with_nulls) - 10} more columns with nulls")

        # Sample column info
        self.logger.info("\nFirst 10 columns:")
        for col in df.columns[:10]:
            self.logger.info(f"  {col}: {df[col].dtype}")
            analysis_report['column_summary'].append({
                'name': col,
                'dtype': str(df[col].dtype),
                'unique_values': int(df[col].nunique()),
                'null_count': int(df[col].isna().sum())
            })

        self.logger.info("="*80)

        return analysis_report

    def get_schema_info(self, table_name: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed schema information for a table.

        This method retrieves schema information from the Pydantic schema JSON files
        based on the specified context (raw or join_rules). It provides detailed
        information about field definitions, data types, and schema location.

        Args:
            table_name (str): Name of the table to get schema info for
            context (Optional[str]): Schema context ('raw' or 'join_rules').
                If None, auto-detects based on current directory or uses default.

        Returns:
            Dict[str, Any]: Schema information containing:
                - table: Table name
                - context: Context used
                - schema_found: Whether schema file was found
                - schema_path: Path to schema file (if found)
                - fields: Dictionary of field definitions with types
        """
        schema_info = {
            'table': table_name,
            'context': context or 'auto-detected',
            'schema_found': False,
            'schema_path': None,
            'fields': {},
            'total_fields': 0
        }

        try:
            # Get project root
            project_root = ProjectRootFinder().find_path()

            # Determine schema path based on context
            schema_paths_to_try = []

            # Check if we have schema_paths configuration
            schema_config = None
            if hasattr(self.config, 'validation') and hasattr(self.config.validation, 'pydantic'):
                if hasattr(self.config.validation.pydantic, 'schema_paths'):
                    schema_config = self.config.validation.pydantic.schema_paths

            if schema_config and hasattr(schema_config, 'DOCUWARE'):
                docuware_config = schema_config.DOCUWARE

                # Determine context if not provided
                if not context:
                    # Auto-detect based on current directory
                    cwd = Path.cwd()
                    if 'no_rules' in str(cwd) or 'raw' in str(cwd):
                        context = 'raw'
                    elif 'with_rules' in str(cwd) or 'join' in str(cwd):
                        context = 'join_rules'
                    else:
                        context = 'raw'  # Default to raw
                    schema_info['context'] = f'auto-detected:{context}'

                # Build schema path based on context
                if context == 'raw' and hasattr(docuware_config.search_paths, 'raw'):
                    base_path = project_root / docuware_config.search_paths.raw
                elif context == 'join_rules' and hasattr(docuware_config.search_paths, 'join_rules'):
                    base_path = project_root / docuware_config.search_paths.join_rules
                else:
                    # Fallback to raw
                    base_path = project_root / docuware_config.search_paths.raw

                schema_path = base_path / f"{table_name}_schema.json"

                if schema_path.exists():
                    with open(schema_path, 'r') as f:
                        schema_data = json.load(f)

                    schema_info['schema_found'] = True
                    schema_info['schema_path'] = str(schema_path)

                    # Extract field information
                    if 'fields' in schema_data:
                        for field_name, field_info in schema_data['fields'].items():
                            metadata = field_info.get('metadata', {})
                            schema_info['fields'][field_name] = {
                                'type': field_info.get('type', 'unknown'),
                                'actual_dtype': metadata.get('actual_dtype', 'unknown'),
                                'inferred_dtype': metadata.get('inferred_dtype', 'unknown'),
                                'nullable': field_info.get('nullable', True),
                                'description': field_info.get('description', '')
                            }

                    schema_info['total_fields'] = len(schema_info['fields'])

                    # Log schema information
                    self.logger.info(f"Schema loaded from: {schema_path}")
                    self.logger.info(f"Schema contains {schema_info['total_fields']} field definitions")
                    self.logger.info(f"Context: {schema_info['context']}")
                else:
                    self.logger.info(f"No schema found at: {schema_path}")
            else:
                self.logger.warning("Schema configuration not found in config")

        except Exception as e:
            self.logger.warning(f"Could not load schema info for {table_name}: {e}")

        return schema_info

    def analyze_dtype_transformations(self, df_original: pd.DataFrame, df_enforced: pd.DataFrame,
                                     table_name: str) -> Dict[str, Any]:
        """Analyze and log dtype transformations between original and enforced DataFrames.

        This method compares two DataFrames to identify and report dtype transformations
        that occurred during schema enforcement. It provides detailed logging of:
        - Column-by-column dtype changes
        - Conversion failures (new NaN/NaT values)
        - Summary statistics by transformation type

        Args:
            df_original (pd.DataFrame): Original DataFrame before type enforcement
            df_enforced (pd.DataFrame): DataFrame after type enforcement
            table_name (str): Name of the table for logging purposes

        Returns:
            Dict[str, Any]: Detailed transformation report containing:
                - table: Table name
                - timestamp: Analysis timestamp
                - total_columns: Total number of columns
                - transformations: List of column transformations
                - summary: Aggregated transformation statistics
        """

        transformation_report = {
            'table': table_name,
            'timestamp': datetime.now().isoformat(),
            'total_columns': len(df_original.columns),
            'transformations': [],
            'summary': {
                'columns_transformed': 0,
                'to_int64': 0,
                'to_float64': 0,
                'to_datetime': 0,
                'to_bool': 0,
                'unchanged': 0,
                'conversion_failures': {}
            }
        }

        for col in df_original.columns:
            if col not in df_enforced.columns:
                continue

            original_dtype = str(df_original[col].dtype)
            enforced_dtype = str(df_enforced[col].dtype)

            if original_dtype != enforced_dtype:
                transformation_report['transformations'].append({
                    'column': col,
                    'original_dtype': original_dtype,
                    'enforced_dtype': enforced_dtype,
                    'null_count_before': int(df_original[col].isna().sum()),
                    'null_count_after': int(df_enforced[col].isna().sum()),
                    'conversion_failures': int(df_enforced[col].isna().sum() - df_original[col].isna().sum())
                })

                transformation_report['summary']['columns_transformed'] += 1

                # Categorize transformation
                if 'int' in enforced_dtype.lower():
                    transformation_report['summary']['to_int64'] += 1
                elif 'float' in enforced_dtype.lower():
                    transformation_report['summary']['to_float64'] += 1
                elif 'datetime' in enforced_dtype.lower():
                    transformation_report['summary']['to_datetime'] += 1
                elif 'bool' in enforced_dtype.lower():
                    transformation_report['summary']['to_bool'] += 1
            else:
                transformation_report['summary']['unchanged'] += 1

        # Log detailed transformation report
        self.logger.info("="*80)
        self.logger.info(f"DTYPE TRANSFORMATION REPORT - {table_name}")
        self.logger.info("="*80)
        self.logger.info(f"Total columns: {transformation_report['total_columns']}")
        self.logger.info(f"Columns transformed: {transformation_report['summary']['columns_transformed']}")
        self.logger.info(f"Columns unchanged: {transformation_report['summary']['unchanged']}")

        if transformation_report['summary']['columns_transformed'] > 0:
            self.logger.info("\nTransformation breakdown:")
            self.logger.info(f"  → Int64: {transformation_report['summary']['to_int64']} columns")
            self.logger.info(f"  → Float64: {transformation_report['summary']['to_float64']} columns")
            self.logger.info(f"  → DateTime: {transformation_report['summary']['to_datetime']} columns")
            self.logger.info(f"  → Boolean: {transformation_report['summary']['to_bool']} columns")

            self.logger.info("\nDetailed transformations:")
            for trans in transformation_report['transformations'][:20]:  # Log first 20
                self.logger.info(f"  • {trans['column']}:")
                self.logger.info(f"    {trans['original_dtype']} → {trans['enforced_dtype']}")
                if trans['conversion_failures'] > 0:
                    self.logger.warning(f"    ⚠️ {trans['conversion_failures']} conversion failures (NaN/NaT)")

            if len(transformation_report['transformations']) > 20:
                self.logger.info(f"  ... and {len(transformation_report['transformations']) - 20} more transformations")

        self.logger.info("="*80)

        return transformation_report

    def close_session(self):
        """Close the Snowflake session."""
        if self.session:
            try:
                self.session.close()
                self.logger.info("Snowflake session closed successfully")
            except Exception as e:
                self.logger.error(f"Error closing Snowflake session: {e}")
        else:
            self.logger.warning("No active session to close")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close_session()


# =============================================================================
# ANALYSIS CLASSES
# =============================================================================

class DtypeTransformationAnalyzer:
    """Production-grade analyzer for dtype transformations with comprehensive reporting.

    This class provides enterprise-level analysis of data type transformations,
    tracking conversions across multiple tables and generating detailed reports
    for data quality monitoring and validation in production environments.

    Features:
        - Tracks all dtype conversions across tables
        - Aggregates conversion statistics by type
        - Identifies and reports conversion failures
        - Generates comprehensive final summaries
        - Provides performance and memory metrics

    Attributes:
        logger: Logger instance for detailed reporting
        all_transformations: List of all transformation reports
        total_conversions: Aggregated conversion statistics
    """

    def __init__(self, logger):
        """Initialize the analyzer with a logger instance.

        Args:
            logger: Logger instance for detailed reporting
        """
        self.logger = logger
        self.all_transformations = []
        self.total_conversions = {
            'to_int64': 0,
            'to_float64': 0,
            'to_datetime': 0,
            'to_bool': 0,
            'unchanged': 0,
            'conversion_failures': 0
        }

    def analyze_table(self, fetcher: 'SnowFetch', df_before: pd.DataFrame, df_after: pd.DataFrame,
                     table_name: str) -> Dict[str, Any]:
        """Analyze dtype transformations for a single table using SnowFetch's built-in method.

        This method leverages SnowFetch's analyze_dtype_transformations() to perform
        detailed analysis and extends it with production-specific metrics including
        dtype distributions and memory usage.

        Args:
            fetcher: SnowFetch instance with analysis methods
            df_before: Original DataFrame before type enforcement
            df_after: DataFrame after type enforcement
            table_name: Name of the table being analyzed

        Returns:
            Dict containing comprehensive transformation report
        """

        # Use SnowFetch's built-in method for dtype analysis
        transformation_report = fetcher.analyze_dtype_transformations(df_before, df_after, table_name)

        # Build extended report with additional statistics
        report = {
            'table': table_name,
            'timestamp': datetime.now().isoformat(),
            'total_columns': len(df_before.columns),
            'columns_transformed': transformation_report['summary']['columns_transformed'],
            'dtype_distribution_before': {},
            'dtype_distribution_after': {},
            'transformations': transformation_report['transformations'],
            'conversion_stats': {
                'to_int64': transformation_report['summary']['to_int64'],
                'to_float64': transformation_report['summary']['to_float64'],
                'to_datetime': transformation_report['summary']['to_datetime'],
                'to_bool': transformation_report['summary']['to_bool'],
                'unchanged': transformation_report['summary']['unchanged'],
                'conversion_failures': sum(t.get('conversion_failures', 0) for t in transformation_report['transformations'])
            }
        }

        # Analyze dtype distributions
        dtype_counts_before = df_before.dtypes.value_counts()
        dtype_counts_after = df_after.dtypes.value_counts()

        report['dtype_distribution_before'] = {str(k): int(v) for k, v in dtype_counts_before.items()}
        report['dtype_distribution_after'] = {str(k): int(v) for k, v in dtype_counts_after.items()}

        # Update totals
        self.total_conversions['to_int64'] += report['conversion_stats']['to_int64']
        self.total_conversions['to_float64'] += report['conversion_stats']['to_float64']
        self.total_conversions['to_datetime'] += report['conversion_stats']['to_datetime']
        self.total_conversions['to_bool'] += report['conversion_stats']['to_bool']
        self.total_conversions['unchanged'] += report['conversion_stats']['unchanged']
        self.total_conversions['conversion_failures'] += report['conversion_stats']['conversion_failures']

        self.all_transformations.append(report)
        self._log_table_report(report)

        return report

    def _log_table_report(self, report: Dict[str, Any]):
        """Log detailed transformation report for a single table.

        Generates comprehensive log output for production monitoring,
        including dtype distributions, transformation breakdowns,
        and conversion failure warnings.

        Args:
            report: Transformation report dictionary for a table
        """

        self.logger.info("\n" + "="*80)
        self.logger.info(f"DTYPE TRANSFORMATION REPORT: {report['table']}")
        self.logger.info("="*80)

        self.logger.info(f"Total columns: {report['total_columns']}")
        self.logger.info(f"Columns transformed: {report['columns_transformed']}")
        self.logger.info(f"Columns unchanged: {report['conversion_stats']['unchanged']}")

        if report['columns_transformed'] > 0:
            self.logger.info("\nTransformation breakdown:")
            self.logger.info(f"  → Int64: {report['conversion_stats']['to_int64']}")
            self.logger.info(f"  → Float64: {report['conversion_stats']['to_float64']}")
            self.logger.info(f"  → DateTime: {report['conversion_stats']['to_datetime']}")
            self.logger.info(f"  → Boolean: {report['conversion_stats']['to_bool']}")

            if report['conversion_stats']['conversion_failures'] > 0:
                self.logger.warning(f"  ⚠️ Conversion failures: {report['conversion_stats']['conversion_failures']}")

            self.logger.info("\nDtype distribution before:")
            for dtype, count in report['dtype_distribution_before'].items():
                self.logger.info(f"  {dtype}: {count} columns")

            self.logger.info("\nDtype distribution after:")
            for dtype, count in report['dtype_distribution_after'].items():
                self.logger.info(f"  {dtype}: {count} columns")

            self.logger.info("\nDetailed transformations (first 15):")
            for i, trans in enumerate(report['transformations'][:15]):
                self.logger.info(f"  {i+1}. {trans['column']}:")
                # Map the keys from analyze_dtype_transformations output
                dtype_before = trans.get('original_dtype', trans.get('dtype_before', 'unknown'))
                dtype_after = trans.get('enforced_dtype', trans.get('dtype_after', 'unknown'))
                self.logger.info(f"     {dtype_before} → {dtype_after}")
                if trans.get('conversion_failures', 0) > 0:
                    self.logger.warning(f"     ⚠️ {trans['conversion_failures']} conversion failures")

            if len(report['transformations']) > 15:
                self.logger.info(f"  ... and {len(report['transformations']) - 15} more transformations")

        self.logger.info("="*80)

    def log_final_summary(self):
        """Generate and log comprehensive production summary of all transformations.

        This method produces a detailed final report suitable for production
        monitoring, including:
        - Global transformation statistics
        - Performance metrics
        - Data quality indicators
        - Top tables by transformation count
        - Conversion failure summary
        """

        self.logger.info("\n" + "="*100)
        self.logger.info("PRODUCTION DTYPE TRANSFORMATION SUMMARY - ALL TABLES")
        self.logger.info("="*100)

        total_tables = len(self.all_transformations)
        tables_with_transformations = sum(1 for t in self.all_transformations if t['columns_transformed'] > 0)
        total_columns_transformed = sum(t['columns_transformed'] for t in self.all_transformations)

        self.logger.info(f"Tables processed: {total_tables}")
        self.logger.info(f"Tables with transformations: {tables_with_transformations}")
        self.logger.info(f"Total columns transformed: {total_columns_transformed}")

        self.logger.info("\nGlobal transformation statistics:")
        self.logger.info(f"  → Int64 conversions: {self.total_conversions['to_int64']}")
        self.logger.info(f"  → Float64 conversions: {self.total_conversions['to_float64']}")
        self.logger.info(f"  → DateTime conversions: {self.total_conversions['to_datetime']}")
        self.logger.info(f"  → Boolean conversions: {self.total_conversions['to_bool']}")
        self.logger.info(f"  → Unchanged columns: {self.total_conversions['unchanged']}")

        if self.total_conversions['conversion_failures'] > 0:
            self.logger.warning(f"\n⚠️ Total conversion failures: {self.total_conversions['conversion_failures']}")

        # Top tables by transformations
        sorted_tables = sorted(self.all_transformations,
                              key=lambda x: x['columns_transformed'],
                              reverse=True)

        self.logger.info("\nTop 10 tables by number of dtype transformations:")
        for i, table in enumerate(sorted_tables[:10]):
            if table['columns_transformed'] > 0:
                self.logger.info(f"  {i+1}. {table['table']}: {table['columns_transformed']} columns")
                if table['conversion_stats']['conversion_failures'] > 0:
                    self.logger.warning(f"     ⚠️ {table['conversion_stats']['conversion_failures']} failures")

        # Performance metrics
        if self.all_transformations:
            avg_transformations = total_columns_transformed / total_tables
            self.logger.info(f"\nAverage transformations per table: {avg_transformations:.1f}")

        self.logger.info("="*100)


class JoinRuleAnalyzer:
    """Analyzer for join rule execution and dtype transformations.

    This class provides comprehensive analysis of SQL join queries with detailed
    reporting capabilities for production environments. It leverages SnowFetch's
    built-in methods for consistent analysis and extends them with join-specific
    metrics and aggregated reporting.

    Features:
        - Query result analysis with performance metrics
        - Dtype transformation tracking across queries
        - Memory usage and execution time monitoring
        - Comprehensive final summaries
        - Production-grade error handling and reporting

    Attributes:
        logger: Logger instance for detailed reporting
        query_reports: List of all query analysis reports
        transformation_summary: Aggregated transformation statistics
    """

    def __init__(self, logger):
        """Initialize the analyzer with a logger instance.

        Args:
            logger: Logger instance for detailed reporting
        """
        self.logger = logger
        self.query_reports = []
        self.transformation_summary = {
            'total_queries': 0,
            'queries_with_transformations': 0,
            'total_columns_transformed': 0,
            'dtype_conversion_stats': {
                'to_int64': 0,
                'to_float64': 0,
                'to_datetime': 0,
                'to_bool': 0,
                'conversion_failures': 0
            }
        }

    def analyze_query_result(self, fetcher: 'SnowFetch', query_name: str, sql_query: str,
                            description: str, df: pd.DataFrame,
                            fetch_time: float) -> Dict[str, Any]:
        """Analyze a query result using SnowFetch's built-in method and extend with additional info.

        This method leverages SnowFetch's log_join_query_analysis() for core analysis
        and extends the report with production-specific metrics including timing,
        descriptions, and performance indicators.

        Args:
            fetcher: SnowFetch instance with analysis methods
            query_name: Name identifier for the query
            sql_query: SQL query string that was executed
            description: Human-readable description of the query
            df: Resulting DataFrame from query execution
            fetch_time: Time taken to execute the query in seconds

        Returns:
            Dict containing comprehensive query analysis report
        """

        # Use SnowFetch's built-in method for the core analysis
        base_report = fetcher.log_join_query_analysis(query_name, sql_query, df)

        # Extend the report with additional fields
        report = {
            **base_report,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'fetch_time_seconds': fetch_time,
            'column_info': base_report.get('column_summary', [])
        }

        self.query_reports.append(report)

        # Log additional production-specific info
        self.logger.info(f"\nProduction metrics:")
        self.logger.info(f"  Fetch time: {fetch_time:.2f} seconds")
        self.logger.info(f"  Description: {description}")

        return report

    def analyze_dtype_enforcement(self, fetcher: 'SnowFetch', table_name: str, context: str,
                                 df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dtype enforcement using SnowFetch's built-in method.

        This method uses SnowFetch's analyze_dtype_transformations() for detailed
        analysis and tracks aggregate statistics for final reporting.

        Args:
            fetcher: SnowFetch instance with analysis methods
            table_name: Name of the table/query being analyzed
            context: Schema context used (join_rules, raw, etc.)
            df_before: DataFrame before type enforcement
            df_after: DataFrame after type enforcement

        Returns:
            Dict containing dtype enforcement analysis report
        """

        # Use SnowFetch's built-in method for dtype analysis
        transformation_report = fetcher.analyze_dtype_transformations(df_before, df_after, table_name)

        # Extract key metrics
        enforcement_report = {
            'table': table_name,
            'context': context,
            'columns_transformed': transformation_report['summary']['columns_transformed'],
            'transformations': transformation_report['transformations']
        }

        # Update summary statistics
        self.transformation_summary['total_columns_transformed'] += transformation_report['summary']['columns_transformed']
        self.transformation_summary['dtype_conversion_stats']['to_int64'] += transformation_report['summary']['to_int64']
        self.transformation_summary['dtype_conversion_stats']['to_float64'] += transformation_report['summary']['to_float64']
        self.transformation_summary['dtype_conversion_stats']['to_datetime'] += transformation_report['summary']['to_datetime']
        self.transformation_summary['dtype_conversion_stats']['to_bool'] += transformation_report['summary']['to_bool']

        conversion_failures = sum(t.get('conversion_failures', 0) for t in transformation_report['transformations'])
        if conversion_failures > 0:
            self.transformation_summary['dtype_conversion_stats']['conversion_failures'] += conversion_failures

        if enforcement_report['columns_transformed'] > 0:
            self.transformation_summary['queries_with_transformations'] += 1
            self._log_enforcement_report(enforcement_report)

        return enforcement_report

    def _log_enforcement_report(self, report: Dict[str, Any]):
        """Log dtype enforcement report.

        Generates detailed log output for dtype enforcement operations,
        including transformation summaries and conversion failure warnings.

        Args:
            report: Enforcement report dictionary containing transformation details
        """

        self.logger.info(f"\n🔄 Dtype enforcement for {report['table']} ({report['context']} context):")
        self.logger.info(f"  Columns transformed: {report['columns_transformed']}")

        for trans in report['transformations'][:5]:
            # Use correct keys from analyze_dtype_transformations output
            col = trans.get('column', 'unknown')
            dtype_before = trans.get('dtype_before', trans.get('original_dtype', trans.get('from', 'unknown')))
            dtype_after = trans.get('dtype_after', trans.get('enforced_dtype', trans.get('to', 'unknown')))
            self.logger.info(f"    {col}: {dtype_before} → {dtype_after}")
            if trans.get('conversion_failures', 0) > 0:
                self.logger.warning(f"      ⚠️ {trans['conversion_failures']} conversion failures")

        if len(report['transformations']) > 5:
            self.logger.info(f"    ... and {len(report['transformations']) - 5} more transformations")

    def log_final_summary(self):
        """Log comprehensive final summary.

        This method produces a detailed final report suitable for production
        monitoring, including:
        - Query execution statistics
        - Performance metrics
        - Dtype transformation summaries
        - Top queries by size and performance
        - Memory usage analysis
        """

        self.logger.info("\n" + "="*100)
        self.logger.info("PRODUCTION JOIN RULES - FINAL ANALYSIS SUMMARY")
        self.logger.info("="*100)

        self.logger.info(f"\nQuery Execution Summary:")
        self.logger.info(f"  Total queries executed: {len(self.query_reports)}")

        if self.query_reports:
            total_rows = sum(r['shape'][0] for r in self.query_reports)
            total_memory = sum(r['memory_mb'] for r in self.query_reports)
            total_time = sum(r['fetch_time_seconds'] for r in self.query_reports)

            self.logger.info(f"  Total rows fetched: {total_rows:,}")
            self.logger.info(f"  Total memory usage: {total_memory:.2f} MB")
            self.logger.info(f"  Total fetch time: {total_time:.2f} seconds")
            self.logger.info(f"  Average rows per query: {total_rows/len(self.query_reports):.0f}")
            self.logger.info(f"  Average fetch time: {total_time/len(self.query_reports):.2f} seconds")

        self.logger.info(f"\nDtype Transformation Summary:")
        self.logger.info(f"  Queries with transformations: {self.transformation_summary['queries_with_transformations']}")
        self.logger.info(f"  Total columns transformed: {self.transformation_summary['total_columns_transformed']}")

        if self.transformation_summary['total_columns_transformed'] > 0:
            self.logger.info(f"\n  Transformation breakdown:")
            for dtype, count in self.transformation_summary['dtype_conversion_stats'].items():
                if dtype != 'conversion_failures' and count > 0:
                    self.logger.info(f"    {dtype}: {count}")

            if self.transformation_summary['dtype_conversion_stats']['conversion_failures'] > 0:
                self.logger.warning(f"\n  ⚠️ Total conversion failures: "
                                  f"{self.transformation_summary['dtype_conversion_stats']['conversion_failures']}")

        # Top queries by size
        if self.query_reports:
            sorted_by_rows = sorted(self.query_reports, key=lambda x: x['shape'][0], reverse=True)
            self.logger.info("\nTop 5 queries by row count:")
            for i, report in enumerate(sorted_by_rows[:5], 1):
                self.logger.info(f"  {i}. {report['query_name']}: {report['shape'][0]:,} rows")
                self.logger.info(f"     {report['description']}")

        self.logger.info("="*100)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def log_dtype_transformation_summary(transformations: list, logger):
    """Log comprehensive summary of all dtype transformations across tables.

    This function aggregates and logs transformation statistics from multiple
    tables, providing a high-level overview of all type conversions performed
    during the data fetching process.

    Args:
        transformations (list): List of transformation reports from each table,
                               each containing columns_transformed, conversion counts
        logger: Logger instance for detailed output to log files

    Logs:
        - Total columns transformed across all tables
        - Breakdown by transformation type (Int64, Float64, DateTime, Boolean)
        - Per-table transformation summary
        - Conversion failure warnings
    """
    if not transformations:
        return

    logger.info("\n" + "="*80)
    logger.info("DTYPE TRANSFORMATION SUMMARY - ALL TABLES")
    logger.info("="*80)

    total_columns_transformed = sum(t['columns_transformed'] for t in transformations)
    total_to_int = sum(t.get('to_int64', 0) for t in transformations)
    total_to_float = sum(t.get('to_float64', 0) for t in transformations)
    total_to_datetime = sum(t.get('to_datetime', 0) for t in transformations)
    total_to_bool = sum(t.get('to_bool', 0) for t in transformations)

    logger.info(f"Total columns transformed: {total_columns_transformed}")
    logger.info(f"Transformation breakdown:")
    logger.info(f"  → Int64: {total_to_int} columns")
    logger.info(f"  → Float64: {total_to_float} columns")
    logger.info(f"  → DateTime: {total_to_datetime} columns")
    logger.info(f"  → Boolean: {total_to_bool} columns")

    logger.info("\nPer-table summary:")
    for trans in transformations:
        if trans['columns_transformed'] > 0:
            logger.info(f"  • {trans['table']}: {trans['columns_transformed']} columns transformed")
            if trans.get('conversion_failures', 0) > 0:
                logger.warning(f"    ⚠️ {trans['conversion_failures']} total conversion failures")

    logger.info("="*80)
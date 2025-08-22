"""
Enterprise Data Loading Framework for Snowflake-based Churn Prediction

This production-grade module provides a comprehensive framework for loading, processing,
and preparing customer churn data from Snowflake. It implements abstract base classes
and concrete implementations with enterprise features including type validation,
data imputation, feature engineering, and ML-ready dataset generation suitable
for large-scale churn prediction models.

✅ Core Features:
- Abstract base class (SnowDataLoader) for extensible data loading
- Concrete implementation (SnowTrainDataLoader) for training data
- Automated type validation and conversion using Pydantic
- No data imputation - raw data preserved as-is from Snowflake
- Feature engineering dataset creation optimized for tsfresh
- Memory-efficient processing with batch operations
- Extensive logging with Loguru integration
- Production-ready error handling and recovery

📊 Data Processing Capabilities:
- Multi-source data ingestion from Snowflake
- Automatic type enforcement based on schemas
- Contract line items deduplication and canonicalization
- Time series feature extraction at weekly granularity
- Static feature aggregation and transformation
- Derived features calculation (DAYS_TO_CHURN, WEEKS_TO_CHURN)
- Customer lifecycle and engagement metrics

🔍 Feature Engineering:
- Time series features (usage, documents, storage)
- Static features (risk scores, segments, credit limits)
- Customer metadata (churn flags, dates, segments)
- Aggregated features (sum, mean, std, max)
- ML-specific features (days/weeks to churn)
- tsfresh-compatible dataset structure

📝 Returned Data Structures:
- time_series_features: Weekly granular usage data
- static_features: Non-time-varying customer attributes
- customer_metadata: Customer information and churn status
- aggregated_features: Statistical summaries per customer
- combined_features: Merged dataset for EDA
- feature_engineering_dataset: tsfresh-ready dataset

📅 Business Variable Definitions (Date/Time Variables):
- TRX_DATE: Transaction processing date - when payment/invoice was recorded in system
- MONTH: Unified monthly timestamp for cross-table temporal alignment
- YYYYWK: ISO week number (YYYYWW format) - primary time grain for usage tracking
- YYYYWK_MONTH: Midpoint date of ISO week - used for datetime merging
- SLINE_START_DATE: Contract line item activation date - service commencement
- CHURN_DATE: Customer attrition date - last day of active service
- CHURNED_DATE: Alias for CHURN_DATE in some tables
- RECEIPT_DATE: Payment receipt timestamp - when payment was received
- DATE_INVOICE_GL_DATE: Invoice posting to general ledger - accounting recognition
- STARTDATECOVERAGE: Renewal coverage period start - when renewed service begins billing
- CONTRACT_END_DATE: Contract termination date - when service agreement expires
- RENEWALS_EARLIEST_DATE: First renewal start date across customer's renewal history
- RENEWALS_LATEST_DATE: Most recent renewal end date - latest service extension
- SUB_EARLIEST_DATE: First subscription start across all contracts
- SUB_LATEST_DATE: Most recent subscription end date
- EARLIEST_DATE: First interaction date across all touchpoints
- FINAL_EARLIEST_DATE: Consolidated earliest customer engagement date
- WEEK_DATE: Actual calendar date (Wednesday) of ISO week - for calculations
- CHURN_YYYYWK: ISO week when customer churned - for filtering
- CHURN_MONTH: Calendar month of churn - for cohort analysis
- CHURN_YEAR: Calendar year of churn - for yearly trends

📊 Business Metrics (Derived):
- DAYS_TO_CHURN: Countdown in days from observation to churn event
- WEEKS_TO_CHURN: Countdown in weeks from observation to churn event
- LIFESPAN_MONTHS: Total customer relationship duration in months
- LIFESPAN_WEEKS: Total customer relationship duration in weeks
- USAGE_DURATION_WEEKS: Active usage period span in weeks
- TOTAL_WEEKS: Count of weeks with recorded activity

⚙️ Configuration:
- Hydra-based configuration management
- Environment-aware settings (development/production)
- Configurable validation and conversion rules
- Flexible schema definitions
- Performance tuning parameters

💡 Usage Examples:
```python
from loaddata import SnowTrainDataLoader

# Initialize loader
loader = SnowTrainDataLoader(config, environment="production")

# Load all data
data = loader.load_data()

# Get feature engineering dataset
fe_dataset = loader.get_feature_engineering_dataset()

# Get training-ready data
training_data = loader.get_training_ready_data()
```

Updated: 2025-08-14
- Added feature engineering dataset method
- Enhanced with DAYS_TO_CHURN calculation
- Integrated ISO week converters
- Improved memory efficiency
- Added comprehensive documentation
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-14
# * CREATED ON: 2025-08-14
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union
import pandas as pd
import numpy as np
import json
from datetime import date
from pathlib import Path
from omegaconf import DictConfig

from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.data.pandas_ml.extensions import to_float
from churn_aiml.ml.datetime.iso_converters import ISOWeekDateConverter, WeekMidpointConverter
from churn_aiml.loggers.loguru.config import get_logger
import yaml


class SnowDataLoader(ABC):
    """
    Abstract base class for Snowflake data loading operations.

    This class defines the interface for loading data from Snowflake
    with support for configuration-based operations and logging.
    """

    def __init__(self, config: DictConfig, environment: str = "development"):
        """
        Initialize the data loader with configuration.

        Args:
            config: Hydra configuration object
            environment: Environment to use (development/production)
        """
        self.config = config
        self.environment = environment
        self.logger = get_logger()
        self.logger.info(f"Initialized {self.__class__.__name__} for environment: {environment}")
        
        # Load dates configuration
        self.dates_config = self._load_dates_config()
        if self.dates_config:
            self.analysis_start_date = pd.Timestamp(self.dates_config['analysis_start_date'])
            self.logger.info(f"Using analysis start date: {self.analysis_start_date}")
        else:
            # Fallback to default if config not found
            self.analysis_start_date = pd.Timestamp('2020-01-01')
            self.logger.warning(f"Dates config not found, using default start date: {self.analysis_start_date}")
    
    def _load_dates_config(self):
        """Load dates configuration from YAML file."""
        try:
            # Try to find the dates config file
            dates_config_path = Path('/home/applaimlgen/ricoh_aiml/conf/products/DOCUWARE/db/snowflake/data_config/dates_config.yaml')
            if dates_config_path.exists():
                with open(dates_config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                self.logger.warning(f"Dates config file not found at {dates_config_path}")
                return None
        except Exception as e:
            self.logger.warning(f"Error loading dates config: {e}")
            return None

    @abstractmethod
    def load_data(self) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Abstract method to load data from Snowflake.

        Returns:
            DataFrame or dictionary of DataFrames containing loaded data
        """
        pass


class SnowTrainDataLoader(SnowDataLoader):
    """
    Concrete implementation for loading and preparing training data for churn prediction.

    This class refactors and improves the data loading logic from the original
    refactored.py, providing clean, maintainable code for:
    - Loading customer and usage data
    - Preparing time series features
    - Creating feature matrices for model training
    """

    def __init__(self, config: DictConfig, environment: str = "development"):
        """
        Initialize the training data loader.

        Args:
            config: Hydra configuration object
            environment: Environment to use (development/production)
        """
        super().__init__(config, environment)

        # Initialize converters
        self.iso_converter = ISOWeekDateConverter(config, log_operations=False)
        self.midpoint_converter = WeekMidpointConverter(config, log_operations=False)

        # Initialize data containers
        self.raw_data = {}
        self.processed_data = {}

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and prepare all training data for churn prediction model.

        Returns:
            Dictionary containing:
                - 'time_series_features': Time series data for each customer
                - 'static_features': Non-time series features
                - 'customer_metadata': Customer information including churn flags
                - 'combined_features': Merged features ready for modeling
        """
        self.logger.info("Starting training data loading process")

        # Step 1: Load raw data from Snowflake
        self._load_raw_data()

        # Step 2: Process usage data with imputations
        self._process_usage_data()

        # Step 3: Merge and prepare datasets
        self._merge_datasets()

        # Step 4: Prepare time series features
        self._prepare_time_series_features()

        # Step 5: Prepare static features
        self._prepare_static_features()

        # Step 6: Calculate derived features
        self._calculate_derived_features()

        self.logger.info("Training data loading completed successfully")

        return self.processed_data

    def _load_raw_data(self):
        """Load all raw data tables from Snowflake using enhanced fetch methods.

        This method uses two different fetching strategies:
        1. fetch_custom_query() for complex join queries defined in YAML (usage_latest, churned_first_week)
           - These queries include Jaro-Winkler similarity matching and complex joins
           - Snowflake returns proper Python types (dates as datetime.date objects)
        2. fetch_data_validation_enforced() for direct table access
           - Applies Pydantic schema validation and type conversion
           - Ensures data consistency across different table sources
        """
        self.logger.info("Loading raw data from Snowflake")

        with SnowFetch(config=self.config, environment=self.environment) as fetcher:
            # Load usage_latest using custom SQL query with Jaro-Winkler matching
            # This complex query performs fuzzy matching between customer names
            # and returns data with proper types from Snowflake (dates as datetime.date)
            self.logger.info("Fetching usage_latest data using join rules")
            # Get the SQL query from the configuration
            usage_latest_sql = self.config.products.DOCUWARE.db.snowflake.sequel_rules.queries.usage_latest.sql
            # Use validation-enforced fetching to ensure consistent data types
            usage_latest = fetcher.fetch_custom_query_validation_enforced(
                sql_query=usage_latest_sql,
                query_name="usage_latest",
                context="join_rules"
            )
            self.raw_data['usage_latest'] = usage_latest
            self.logger.info(f"Loaded usage_latest: {usage_latest.shape[0]} rows")

            # Load other tables using validation-enforced fetching
            # Note: Schema (RUS_AIML) is already set in the session, so we only provide table names
            # fetch_data_validation_enforced will:
            # 1. Fetch raw data from Snowflake
            # 2. Apply Pydantic schema validation from corresponding JSON files
            # 3. Convert data types according to schema (e.g., string to int, string to datetime)
            tables_to_load = [
                ("PS_DOCUWARE_PAYMENTS", "payments"),
                ("PS_DOCUWARE_REVENUE", "revenue"),
                ("PS_DOCUWARE_TRX", "transactions"),
                ("PS_DOCUWARE_CONTRACT_SUBLINE", "contracts_sub"),
                ("PS_DOCUWARE_SSCD_RENEWALS", "renewals"),
                ("PS_DOCUWARE_L1_CUST", "customers"),  # Contains churn flags and dates
                ("DNB_RISK_BREAKDOWN", "dnb_risk")     # Risk assessment data
            ]

            for table_name, key in tables_to_load:
                self.logger.info(f"Fetching {table_name}")
                # Use validation-enforced fetching to ensure data types match schema
                # Context="raw" tells the system to look for schemas in the raw/ directory
                df = fetcher.fetch_data_validation_enforced(
                    table_name=table_name,
                    context="raw"
                )
                self.raw_data[key] = df
                self.logger.info(f"Loaded {key}: {df.shape[0]} rows")

            # Load first week data for all customers and filter to churned
            # This query identifies the earliest week (minimum YYYYWK) for customers
            # Used later for calculating customer lifecycle and engagement metrics
            self.logger.info("Fetching first week data")
            
            # Check if the optimized query exists
            if hasattr(self.config.products.DOCUWARE.db.snowflake.sequel_rules.queries, 'all_customers_first_week'):
                # Use the new optimized query that gets all customers
                all_first_week_sql = self.config.products.DOCUWARE.db.snowflake.sequel_rules.queries.all_customers_first_week.sql
                all_first_week = fetcher.fetch_custom_query_validation_enforced(
                    sql_query=all_first_week_sql,
                    query_name="all_customers_first_week",
                    context="join_rules"
                )
                # Filter to churned customers only (CHURNED_FLAG == 'Y' or 1)
                if 'CHURNED_FLAG' in all_first_week.columns:
                    churned_first_week = all_first_week[all_first_week['CHURNED_FLAG'] == 1].copy()
                    self.logger.info(f"Filtered from {len(all_first_week)} to {len(churned_first_week)} churned customer records")
                else:
                    churned_first_week = all_first_week  # Use all if no flag column
            else:
                # Fallback to the original churned-only query
                churned_first_week_sql = self.config.products.DOCUWARE.db.snowflake.sequel_rules.queries.churned_first_week.sql
                churned_first_week = fetcher.fetch_custom_query_validation_enforced(
                    sql_query=churned_first_week_sql,
                    query_name="churned_first_week",
                    context="join_rules"
                )
            
            self.raw_data['churned_first_week'] = churned_first_week
            self.logger.info(f"Loaded churned_first_week: {churned_first_week.shape[0]} rows")

    def _process_usage_data(self):
        """Process usage data WITHOUT any imputation.

        Key processing steps:
        1. Remove duplicate records
        2. Filter to contracts starting from 2020 onwards (data quality cutoff)
        3. Canonicalize contract line items for consistent deduplication

        NOTE: No data imputation is performed - data is kept as-is from Snowflake.
        Any missing data will be handled in subsequent processing steps.
        """
        self.logger.info("Processing usage data (no imputation)")

        usage_latest = self.raw_data['usage_latest'].copy()

        # Remove duplicates from the raw data
        usage_latest = usage_latest.drop_duplicates()

        # Focus only on contracts starting from analysis_start_date onwards
        # This is a data quality cutoff from dates_config.yaml - earlier data may be incomplete or unreliable
        # Note: CONTRACT_START is datetime64[ns] after validation, so we use pd.Timestamp for comparison
        usage_latest = usage_latest[usage_latest['CONTRACT_START'] >= self.analysis_start_date]
        self.logger.info(f"Filtered to contracts starting >= {self.analysis_start_date}: {usage_latest.shape[0]} rows")

        # Sort and canonicalize contract line items
        usage_latest['CONTRACT_LINE_ITEMS'] = usage_latest['CONTRACT_LINE_ITEMS'].apply(
            self._sort_contract_line_items
        )

        # Remove duplicates based on canonical form
        usage_latest = self._deduplicate_by_contract_items(usage_latest)

        self.processed_data['usage_processed'] = usage_latest
        self.logger.info(f"Processed usage data: {usage_latest.shape[0]} rows")


    def _sort_contract_line_items(self, items: Any) -> str:
        """Sort contract line items into canonical form for consistent deduplication.

        Contract line items may appear in different orders in different records
        for the same contract. Sorting them ensures we can identify duplicates.
        Example: 'A,B,C' and 'C,A,B' become 'A,B,C' after sorting.
        """
        if pd.isna(items) or items is None:
            return ""

        items_str = str(items)
        items_list = [item.strip() for item in items_str.split(',')]
        items_sorted = sorted(items_list)  # Alphabetical sorting for consistency

        return ','.join(items_sorted)

    def _deduplicate_by_contract_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates keeping the records with the most complete contract information.

        When multiple records exist for the same customer-week combination,
        keep the one with the longest CONTRACT_LINE_ITEMS string, as it likely
        contains the most complete information about the contract.
        """
        if 'CONTRACT_LINE_ITEMS' not in df.columns:
            return df

        df = df.copy()
        # Calculate length of contract items (proxy for information completeness)
        df['items_length'] = df['CONTRACT_LINE_ITEMS'].str.len()

        # Sort by customer, week, and items length (descending)
        # This ensures the most complete record appears first
        df = df.sort_values(
            ['CUST_ACCOUNT_NUMBER', 'YYYYWK', 'items_length'],
            ascending=[True, True, False]
        )

        # Keep first (most complete) record for each customer-week combination
        df = df.groupby(['CUST_ACCOUNT_NUMBER', 'YYYYWK'], as_index=False).first()

        # Remove the temporary helper column
        df = df.drop('items_length', axis=1)

        return df

    def _merge_datasets(self):
        """Merge various datasets to create comprehensive features.

        This method aggregates financial and transactional data for ALL customers (both churned and active).
        Important: When grouping and summing, we explicitly identify numeric columns
        to avoid attempting to sum datetime columns, which would cause a TypeError.
        """
        self.logger.info("Merging datasets")

        # Get ALL customers (both churned and active)
        # ALL data types are already properly handled by Pydantic schemas during fetch_data_validation_enforced
        # This includes:
        # - CHURNED_FLAG: converted to Int64 (1 for churned, 0 for active)
        # - CUST_ACCOUNT_NUMBER: converted to Int64
        # - All date columns: converted to datetime64[ns]
        customers_df = self.raw_data['customers'].drop_duplicates()
        all_customers = customers_df.copy()  # Keep all customers
        churned_customers = customers_df[customers_df['CHURNED_FLAG'] == 1].copy()  # For reference where needed

        # Process payments for ALL customers
        # Note: Data types are already consistent from Pydantic validation
        payments_df = self.raw_data['payments'].drop_duplicates()

        payments_churned = payments_df.merge(
            all_customers[['CUST_ACCOUNT_NUMBER']],
            left_on='CUSTOMER_NO',
            right_on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        payments_churned = payments_churned.drop('CUSTOMER_NO', axis=1)

        # Identify numeric columns for aggregation
        # This prevents TypeError when trying to sum datetime columns
        numeric_cols = payments_churned.select_dtypes(include=['number']).columns.tolist()
        group_cols = ['CUST_ACCOUNT_NUMBER', 'RECEIPT_DATE']
        agg_cols = [col for col in numeric_cols if col not in group_cols]

        if agg_cols:
            # Create aggregation dictionary for numeric columns only
            agg_dict = {col: 'sum' for col in agg_cols}
            payments_churned = payments_churned.groupby(group_cols).agg(agg_dict).reset_index()
        else:
            # If no numeric columns to aggregate, just get unique combinations
            payments_churned = payments_churned.groupby(group_cols).size().reset_index(name='count')

        # RECEIPT_DATE is already datetime from Pydantic validation
        # MONTH serves as unified temporal key for cross-table joining
        # Business meaning: Monthly aggregation point for financial reconciliation
        payments_churned['MONTH'] = payments_churned['RECEIPT_DATE']

        # Process revenue
        # Note: Data types are already consistent from Pydantic validation
        revenue_df = self.raw_data['revenue'].drop_duplicates()

        revenue_churned = revenue_df.merge(
            all_customers[['CUST_ACCOUNT_NUMBER']],
            on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        # Identify numeric columns for aggregation
        numeric_cols = revenue_churned.select_dtypes(include=['number']).columns.tolist()
        group_cols = ['CUST_ACCOUNT_NUMBER', 'DATE_INVOICE_GL_DATE']
        agg_cols = [col for col in numeric_cols if col not in group_cols]

        if agg_cols:
            agg_dict = {col: 'sum' for col in agg_cols}
            revenue_churned = revenue_churned.groupby(group_cols).agg(agg_dict).reset_index()
        else:
            revenue_churned = revenue_churned.groupby(group_cols).size().reset_index(name='count')

        # DATE_INVOICE_GL_DATE is already datetime from Pydantic validation
        # MONTH aligns revenue recognition with payment and usage cycles
        # Business meaning: Revenue recognition timestamp for accounting
        revenue_churned['MONTH'] = revenue_churned['DATE_INVOICE_GL_DATE']

        # Process transactions
        # Note: Data types are already consistent from Pydantic validation
        trx_df = self.raw_data['transactions'].drop_duplicates()

        trx_churned = trx_df.merge(
            all_customers[['CUST_ACCOUNT_NUMBER']],
            left_on='ACCOUNT_NUMBER',
            right_on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        trx_churned = trx_churned.drop('ACCOUNT_NUMBER', axis=1)
        # Identify numeric columns for aggregation
        numeric_cols = trx_churned.select_dtypes(include=['number']).columns.tolist()
        group_cols = ['CUST_ACCOUNT_NUMBER', 'TRX_DATE']
        agg_cols = [col for col in numeric_cols if col not in group_cols]

        if agg_cols:
            agg_dict = {col: 'sum' for col in agg_cols}
            trx_churned = trx_churned.groupby(group_cols).agg(agg_dict).reset_index()
        else:
            trx_churned = trx_churned.groupby(group_cols).size().reset_index(name='count')

        # TRX_DATE: Transaction processing timestamp - critical for payment tracking
        # Business meaning: Actual transaction execution date in billing system
        trx_churned['MONTH'] = trx_churned['TRX_DATE']

        # Merge payments, revenue, and transactions
        merged = payments_churned.merge(
            revenue_churned,
            on=['CUST_ACCOUNT_NUMBER', 'MONTH'],
            how='outer'
        ).merge(
            trx_churned,
            on=['CUST_ACCOUNT_NUMBER', 'MONTH'],
            how='outer'
        )

        # Process contracts
        # Note: Data types are already consistent from Pydantic validation
        contracts_df = self.raw_data['contracts_sub'].drop_duplicates()

        contracts_df['SUB_EARLIEST_DATE'] = contracts_df.groupby(
            'CUST_ACCOUNT_NUMBER'
        )['SLINE_START_DATE'].transform('min')
        contracts_df['SUB_LATEST_DATE'] = contracts_df.groupby(
            'CUST_ACCOUNT_NUMBER'
        )['SLINE_END_DATE'].transform('max')

        contracts_churned = contracts_df.merge(
            all_customers[['CUST_ACCOUNT_NUMBER']],
            on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )

        # Remove duplicates after merging (following refactored.py pattern)
        contracts_churned = contracts_churned.drop_duplicates()

        # Continue merging with contracts
        # Note: SLINE_START_DATE is datetime from Pydantic validation
        merged = merged.merge(
            contracts_churned,
            left_on=['CUST_ACCOUNT_NUMBER', 'MONTH'],
            right_on=['CUST_ACCOUNT_NUMBER', 'SLINE_START_DATE'],
            how='outer'
        )
        # Use combine_first to maintain datetime type consistency
        merged['MONTH'] = merged['MONTH'].combine_first(merged['SLINE_START_DATE'])

        # Process renewals
        # Note: Data types are already consistent from Pydantic validation
        renewals_df = self.raw_data['renewals'].drop_duplicates()

        # Create renewal aggregates by customer
        renewals_df['RENEWALS_EARLIEST_DATE'] = renewals_df.groupby(
            'BILLTOCUSTOMERNUMBER'
        )['STARTDATECOVERAGE'].transform('min')
        renewals_df['RENEWALS_LATEST_DATE'] = renewals_df.groupby(
            'BILLTOCUSTOMERNUMBER'
        )['CONTRACT_END_DATE'].transform('max')

        # Select relevant renewal columns for merging
        renewals_churned = renewals_df[['BILLTOCUSTOMERNUMBER', 'STARTDATECOVERAGE',
                                       'CONTRACT_END_DATE', 'RENEWALS_EARLIEST_DATE',
                                       'RENEWALS_LATEST_DATE']].copy()

        # Merge renewals data with ALL customers
        # Note: BILLTOCUSTOMERNUMBER maps to CUST_ACCOUNT_NUMBER
        renewals_churned = renewals_churned.merge(
            all_customers[['CUST_ACCOUNT_NUMBER']],
            left_on='BILLTOCUSTOMERNUMBER',
            right_on='CUST_ACCOUNT_NUMBER',
            how='inner'
        ).drop('BILLTOCUSTOMERNUMBER', axis=1)

        # Merge renewals into main dataset
        merged = merged.merge(
            renewals_churned,
            on='CUST_ACCOUNT_NUMBER',
            how='left'
        )

        # Process DNB risk
        # Note: Data types are already consistent from Pydantic validation
        dnb_df = self.raw_data['dnb_risk'].drop_duplicates()

        dnb_churned = dnb_df.merge(
            all_customers[['CUST_ACCOUNT_NUMBER']],
            left_on='ACCOUNT_NUMBER',
            right_on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        dnb_churned = dnb_churned.drop('ACCOUNT_NUMBER', axis=1)

        # Add DNB risk to merged
        merged = merged.merge(dnb_churned, on='CUST_ACCOUNT_NUMBER', how='left')

        # Process usage data
        usage_processed = self.processed_data['usage_processed']
        usage_churned = usage_processed.merge(
            all_customers[['CUST_ACCOUNT_NUMBER']],
            on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )

        # YYYYWK_MONTH: Wednesday of ISO week - optimal temporal alignment point
        # Business meaning: Weekly usage data aligned to mid-week for stability
        # Using Wednesday avoids weekend/Monday biases in business activity
        usage_churned['YYYYWK_MONTH'] = usage_churned['YYYYWK'].apply(
            lambda x: pd.Timestamp(self.midpoint_converter.convert_yyyywk_to_actual_mid_date(x))
            if pd.notna(x) and x > 0 else pd.NaT
        )

        # Log data types for debugging
        self.logger.info(f"MONTH dtype in merged: {merged['MONTH'].dtype if 'MONTH' in merged.columns else 'N/A'}")
        self.logger.info(f"YYYYWK_MONTH dtype: {usage_churned['YYYYWK_MONTH'].dtype}")

        # Ensure both columns are datetime64
        if 'MONTH' in merged.columns:
            merged['MONTH'] = pd.to_datetime(merged['MONTH'])
        usage_churned['YYYYWK_MONTH'] = pd.to_datetime(usage_churned['YYYYWK_MONTH'])

        # Final merge with usage
        merged_final = merged.merge(
            usage_churned,
            left_on=['CUST_ACCOUNT_NUMBER', 'MONTH'],
            right_on=['CUST_ACCOUNT_NUMBER', 'YYYYWK_MONTH'],
            how='outer'
        )
        # Use combine_first to maintain datetime type consistency
        merged_final['MONTH'] = merged_final['MONTH'].combine_first(merged_final['YYYYWK_MONTH'])

        # Calculate derived dates
        # Only include columns that actually exist in the merged data
        date_cols = ['RECEIPT_DATE', 'DATE_INVOICE_GL_DATE', 'TRX_DATE',
                     'SLINE_START_DATE']
        # Add STARTDATECOVERAGE if it exists
        if 'STARTDATECOVERAGE' in merged_final.columns:
            date_cols.append('STARTDATECOVERAGE')

        # Filter to only existing columns
        existing_date_cols = [col for col in date_cols if col in merged_final.columns]

        merged_final['EARLIEST_DATE'] = merged_final[existing_date_cols].min(axis=1)
        merged_final['FINAL_EARLIEST_DATE'] = merged_final.groupby(
            'CUST_ACCOUNT_NUMBER'
        )['EARLIEST_DATE'].transform('min')

        # Calculate customer lifecycle metrics
        # LIFESPAN_MONTHS: Total relationship duration - key retention metric
        # Business meaning: Months from first engagement to churn (customer lifetime)
        # Ensure both columns are datetime before subtraction
        if 'CHURN_DATE' in merged_final.columns and 'FINAL_EARLIEST_DATE' in merged_final.columns:
            merged_final['CHURN_DATE'] = pd.to_datetime(merged_final['CHURN_DATE'])
            merged_final['FINAL_EARLIEST_DATE'] = pd.to_datetime(merged_final['FINAL_EARLIEST_DATE'])

            merged_final['LIFESPAN_MONTHS'] = (
                (merged_final['CHURN_DATE'] - merged_final['FINAL_EARLIEST_DATE']).dt.days / 30
            )
            # DAYS_TO_CHURN: Primary ML target - countdown to attrition event
            # Business meaning: Remaining days of customer relationship (survival time)
            merged_final['DAYS_TO_CHURN'] = (
                merged_final['CHURN_DATE'] - merged_final['FINAL_EARLIEST_DATE']
            ).dt.days
        else:
            # If columns don't exist, create them with NaN values
            merged_final['LIFESPAN_MONTHS'] = np.nan
            merged_final['DAYS_TO_CHURN'] = np.nan

        # Drop unnecessary columns
        cols_to_drop = ['SLINE_END_DATE', 'SLINE_STATUS', 'SUB_EARLIEST_DATE',
                        'SUB_LATEST_DATE', 'CONTRACT_END_DATE', 'CHURNED_FLAG',
                        'EARLIEST_DATE', 'CONTRACT_NUMBER', 'CUST_PARTY_NAME',
                        'CUSTOMER_NAME', 'CONTRACT_END',
                        'JAROWINKLER_SIMILARITY(A.CUST_PARTY_NAME, B.CUSTOMER_NAME)']

        cols_to_drop = [col for col in cols_to_drop if col in merged_final.columns]
        merged_final = merged_final.drop(columns=cols_to_drop)

        # Remove duplicates
        merged_final = merged_final.drop_duplicates()

        self.processed_data['merged_data'] = merged_final
        self.logger.info(f"Merged data: {merged_final.shape[0]} rows")

    def _prepare_time_series_features(self):
        """Prepare time series features for model training."""
        self.logger.info("Preparing time series features")

        merged_data = self.processed_data['merged_data']

        # Select time series columns
        ts_columns = ['CUST_ACCOUNT_NUMBER', 'YYYYWK', 'DOCUMENTS_OPENED',
                      'USED_STORAGE__MB', 'INVOICE_REVLINE_TOTAL',
                      'ORIGINAL_AMOUNT_DUE', 'FUNCTIONAL_AMOUNT']

        # YYYYWK: ISO week number - primary temporal grain for time series
        # Business meaning: Weekly bucketing for usage patterns and seasonality
        # Create YYYYWK from MONTH where missing to ensure complete time series
        merged_data['YYYYWK'] = merged_data.apply(
            lambda row: self.iso_converter.convert_date_to_yyyywk(row['MONTH'])
            if pd.isna(row['YYYYWK']) and pd.notna(row['MONTH'])
            else row['YYYYWK'],
            axis=1
        )

        # Extract time series data
        ts_data = merged_data[
            [col for col in ts_columns if col in merged_data.columns]
        ].copy()

        # Remove rows without YYYYWK
        ts_data = ts_data[ts_data['YYYYWK'].notna()]

        # Ensure correct types
        ts_data['YYYYWK'] = ts_data['YYYYWK'].astype('Int64')
        ts_data['CUST_ACCOUNT_NUMBER'] = ts_data['CUST_ACCOUNT_NUMBER'].astype('Int64')
        
        # Filter time series data to only include weeks from analysis_start_date onwards
        # Convert analysis_start_date to YYYYWK format for comparison
        analysis_start_yyyywk = self.iso_converter.convert_date_to_yyyywk(self.analysis_start_date)
        initial_ts_rows = len(ts_data)
        ts_data = ts_data[ts_data['YYYYWK'] >= analysis_start_yyyywk]
        filtered_rows = initial_ts_rows - len(ts_data)
        if filtered_rows > 0:
            self.logger.info(f"Filtered out {filtered_rows} rows with YYYYWK < {analysis_start_yyyywk} (before {self.analysis_start_date})")
            self.logger.info(f"Time series date range after filtering: {ts_data['YYYYWK'].min()} to {ts_data['YYYYWK'].max()}")

        # Rename columns if needed
        if 'USED_STORAGE__MB' in ts_data.columns:
            ts_data = ts_data.rename(columns={'USED_STORAGE__MB': 'USED_STORAGE_MB'})

        # Sort and remove duplicates based on customer and week
        ts_data = ts_data.sort_values(['CUST_ACCOUNT_NUMBER', 'YYYYWK'])
        # Keep only the first occurrence of each customer-week combination
        initial_rows = len(ts_data)
        ts_data = ts_data.drop_duplicates(subset=['CUST_ACCOUNT_NUMBER', 'YYYYWK'], keep='first')
        duplicates_removed = initial_rows - len(ts_data)
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate customer-week combinations from time series data")

        # Keep data intact - DO NOT impute missing values
        # Missing values are preserved as NaN for downstream processing

        self.processed_data['time_series_features'] = ts_data
        self.logger.info(f"Time series features: {ts_data.shape[0]} rows")

    def _prepare_static_features(self):
        """Prepare static (non-time series) features."""
        self.logger.info("Preparing static features")

        merged_data = self.processed_data['merged_data']

        # Select static columns (excluding CUST_ACCOUNT_NUMBER since it's the groupby key)
        static_cols = ['PROBABILITY_OF_DELINQUENCY',
                       'RICOH_CUSTOM_RISK_MODEL', 'OVERALL_BUSINESS_RISK',
                       'PAYMENT_RISK_TRIPLE_A_RATING', 'CONTRACT_LINE_ITEMS',
                       'LIFESPAN_MONTHS', 'DAYS_TO_CHURN']

        # Get first value for each customer
        # Filter to only columns that exist in merged_data
        available_static_cols = [col for col in static_cols if col in merged_data.columns]

        if available_static_cols:
            static_data = merged_data.groupby('CUST_ACCOUNT_NUMBER')[
                available_static_cols
            ].first().reset_index()
        else:
            # If no static columns available, just get unique customers
            static_data = pd.DataFrame({
                'CUST_ACCOUNT_NUMBER': merged_data['CUST_ACCOUNT_NUMBER'].unique()
            })

        # Process categorical features
        static_processed = self._process_categorical_features(static_data)

        self.processed_data['static_features'] = static_processed
        self.logger.info(f"Static features: {static_processed.shape}")

    def _process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process categorical features with encoding."""
        df = df.copy()

        # Handle missing values for categorical columns
        categorical_cols = ['OVERALL_BUSINESS_RISK', 'PAYMENT_RISK_TRIPLE_A_RATING']

        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('UNK')
                df[col] = df[col].str.replace(' ', '_', regex=False)

        # Process CONTRACT_LINE_ITEMS
        if 'CONTRACT_LINE_ITEMS' in df.columns:
            df['CONTRACT_LINE_ITEMS'] = df['CONTRACT_LINE_ITEMS'].fillna('NA')
            # Clean and standardize contract line items
            df['CONTRACT_LINE_ITEMS'] = df['CONTRACT_LINE_ITEMS'].str.replace(
                r'\d+x ', '', regex=True
            )
            # Sort items for consistent encoding
            df['CONTRACT_LINE_ITEMS'] = df['CONTRACT_LINE_ITEMS'].apply(
                self._sort_contract_line_items
            )

        return df

    def _calculate_derived_features(self):
        """Calculate additional derived features."""
        self.logger.info("Calculating derived features")

        # Get customer metadata
        customers_df = self.raw_data['customers'].copy()

        # Add churn week information
        customers_df['CHURN_YYYYWK'] = customers_df['CHURN_DATE'].apply(
            lambda x: self.iso_converter.convert_date_to_yyyywk(x)
            if pd.notna(x) else None
        )

        # Filter time series to only include data before churn
        ts_features = self.processed_data['time_series_features']

        # Merge with churn information
        ts_with_churn = ts_features.merge(
            customers_df[['CUST_ACCOUNT_NUMBER', 'CHURN_YYYYWK', 'CHURNED_FLAG']],
            on='CUST_ACCOUNT_NUMBER',
            how='left'
        )

        # For churned customers, only keep data before churn
        # CHURNED_FLAG is boolean from Pydantic schema conversion
        churned_mask = ts_with_churn['CHURNED_FLAG'] == 1
        ts_filtered = ts_with_churn[
            ~churned_mask | (ts_with_churn['YYYYWK'] <= ts_with_churn['CHURN_YYYYWK'])
        ].copy()

        # Drop helper columns
        ts_filtered = ts_filtered.drop(['CHURN_YYYYWK', 'CHURNED_FLAG'], axis=1)

        # Update time series features
        self.processed_data['time_series_features'] = ts_filtered

        # Store customer metadata
        self.processed_data['customer_metadata'] = customers_df[
            ['CUST_ACCOUNT_NUMBER', 'CUST_PARTY_NAME', 'CHURNED_FLAG',
             'CHURN_DATE', 'CUSTOMER_SEGMENT', 'CUSTOMER_SEGMENT_LEVEL']
        ].drop_duplicates()

        # Calculate time-based features
        self._calculate_time_based_features()

        self.logger.info("Derived features calculation completed")

    def _calculate_time_based_features(self):
        """Calculate time-based features from usage patterns."""
        ts_features = self.processed_data['time_series_features']

        # Group by customer to calculate aggregates
        customer_aggs = ts_features.groupby('CUST_ACCOUNT_NUMBER').agg({
            'DOCUMENTS_OPENED': ['sum', 'mean', 'std', 'max'],
            'USED_STORAGE_MB': ['sum', 'mean', 'std', 'max'],
            'INVOICE_REVLINE_TOTAL': ['sum', 'mean', 'std'],
            'ORIGINAL_AMOUNT_DUE': ['sum', 'mean', 'std'],
            'FUNCTIONAL_AMOUNT': ['sum', 'mean', 'std'],
            'YYYYWK': ['min', 'max', 'count']
        })

        # Flatten column names
        customer_aggs.columns = ['_'.join(col).strip() for col in customer_aggs.columns]
        # Only reset_index if CUST_ACCOUNT_NUMBER is in the index
        if 'CUST_ACCOUNT_NUMBER' not in customer_aggs.columns:
            customer_aggs = customer_aggs.reset_index()

        # Calculate usage duration in weeks
        customer_aggs['USAGE_DURATION_WEEKS'] = (
            customer_aggs['YYYYWK_max'] - customer_aggs['YYYYWK_min']
        )

        # Rename count column
        customer_aggs = customer_aggs.rename(columns={'YYYYWK_count': 'TOTAL_WEEKS'})

        self.processed_data['aggregated_features'] = customer_aggs
        self.logger.info(f"Calculated aggregated features for {len(customer_aggs)} customers")

    def get_training_ready_data(self) -> pd.DataFrame:
        """
        Get final training-ready dataset with all features combined.

        Returns:
            DataFrame ready for model training
        """
        # Combine static and aggregated features
        static_features = self.processed_data.get('static_features', pd.DataFrame())
        aggregated_features = self.processed_data.get('aggregated_features', pd.DataFrame())
        customer_metadata = self.processed_data.get('customer_metadata', pd.DataFrame())

        # Merge all features
        if not static_features.empty and not aggregated_features.empty:
            combined = static_features.merge(
                aggregated_features,
                on='CUST_ACCOUNT_NUMBER',
                how='outer'
            )

            if not customer_metadata.empty:
                combined = combined.merge(
                    customer_metadata,
                    on='CUST_ACCOUNT_NUMBER',
                    how='left'
                )

            self.processed_data['combined_features'] = combined
            return combined

        return pd.DataFrame()

    def get_feature_engineering_dataset(self) -> pd.DataFrame:
        """
        Prepare intact data for feature engineering tools.

        This method prepares training data for feature engineering tools like tsfresh by combining 
        time series data with customer metadata and calculating derived features like DAYS_TO_CHURN.
        
        IMPORTANT: Data is kept intact with NO imputation or dropping of missing values.
        Missing values are preserved as NaN for downstream feature engineering tools to handle.

        Returns:
            DataFrame with intact data including columns:
                - CUST_ACCOUNT_NUMBER: Customer identifier
                - YYYYWK: Week identifier
                - DOCUMENTS_OPENED: Weekly document usage (may contain NaN)
                - USED_STORAGE_MB: Weekly storage usage (may contain NaN)
                - INVOICE_REVLINE_TOTAL: Weekly invoice revenue (may contain NaN)
                - ORIGINAL_AMOUNT_DUE: Weekly amount due (may contain NaN)
                - FUNCTIONAL_AMOUNT: Weekly functional amount (may contain NaN)
                - CHURNED_FLAG: Whether customer churned (Y/N)
                - CHURN_DATE: Date of churn (if applicable)
                - DAYS_TO_CHURN: Days from current week to churn date
                - WEEKS_TO_CHURN: Weeks from current week to churn
                - CONTRACT_START: Contract start date (may contain NaN)
                - CUSTOMER_SEGMENT: Customer segment classification (may contain NaN)
                - RISK_SCORE: Customer risk score (may contain NaN)
                - FINAL_EARLIEST_DATE: Earliest relevant date for customer
        """
        self.logger.info("Creating feature engineering dataset")

        # Get time series features
        ts_features = self.processed_data.get('time_series_features', pd.DataFrame())
        if ts_features.empty:
            self.logger.warning("No time series features available")
            return pd.DataFrame()

        # Get customer metadata
        customer_metadata = self.processed_data.get('customer_metadata', pd.DataFrame())
        static_features = self.processed_data.get('static_features', pd.DataFrame())

        # Start with time series as base
        fe_dataset = ts_features.copy()

        # Merge customer metadata
        if not customer_metadata.empty:
            metadata_cols = ['CUST_ACCOUNT_NUMBER', 'CHURNED_FLAG', 'CHURN_DATE',
                           'CUSTOMER_SEGMENT', 'CUSTOMER_SEGMENT_LEVEL']
            available_cols = [col for col in metadata_cols if col in customer_metadata.columns]

            fe_dataset = fe_dataset.merge(
                customer_metadata[available_cols].drop_duplicates(),
                on='CUST_ACCOUNT_NUMBER',
                how='left'
            )

        # Merge static features (risk scores, etc.)
        if not static_features.empty:
            static_cols = ['CUST_ACCOUNT_NUMBER', 'CUST_RISK_SCORE',
                          'CUST_CREDIT_LIMIT', 'CONTRACT_START']
            available_static = [col for col in static_cols if col in static_features.columns]

            fe_dataset = fe_dataset.merge(
                static_features[available_static].drop_duplicates(),
                on='CUST_ACCOUNT_NUMBER',
                how='left'
            )

        # Calculate DAYS_TO_CHURN - the key target variable for churn prediction
        # Business meaning: Survival time remaining for each customer at each observation
        # This creates a countdown timer showing days until churn event
        if 'CHURN_DATE' in fe_dataset.columns and 'YYYYWK' in fe_dataset.columns:
            # WEEK_DATE: Actual calendar date (Wednesday) for temporal calculations
            # Business rationale: Mid-week avoids weekend effects and Monday spikes
            fe_dataset['WEEK_DATE'] = fe_dataset['YYYYWK'].apply(
                lambda x: pd.Timestamp(self.midpoint_converter.convert_yyyywk_to_actual_mid_date(x))
            )

            # Calculate days to churn for churned customers only
            # This creates a countdown: positive values = days until churn, negative = days since churn
            # CHURNED_FLAG is boolean from Pydantic schema conversion
            churned_mask = fe_dataset['CHURNED_FLAG'] == 1
            fe_dataset.loc[churned_mask, 'DAYS_TO_CHURN'] = (
                pd.to_datetime(fe_dataset.loc[churned_mask, 'CHURN_DATE']) -
                fe_dataset.loc[churned_mask, 'WEEK_DATE']
            ).dt.days

            # For active (non-churned) customers, use 9999 as a sentinel value
            # This distinguishes them from churned customers while allowing numeric operations
            fe_dataset.loc[~churned_mask, 'DAYS_TO_CHURN'] = 9999

            # Add weeks-based version for models that prefer weekly granularity
            fe_dataset['WEEKS_TO_CHURN'] = fe_dataset['DAYS_TO_CHURN'] / 7.0

            # Clean up temporary column
            fe_dataset = fe_dataset.drop('WEEK_DATE', axis=1)

        # Calculate FINAL_EARLIEST_DATE - the true start of each customer's journey
        # This is the earliest contract start date across all records for each customer
        # Used to calculate customer tenure/age at any point in time
        if 'CONTRACT_START' in fe_dataset.columns:
            fe_dataset['FINAL_EARLIEST_DATE'] = fe_dataset.groupby('CUST_ACCOUNT_NUMBER')['CONTRACT_START'].transform('min')

        # Ensure numeric types for key columns to prevent type errors in ML pipelines
        # 'coerce' converts invalid values to NaN rather than raising errors
        numeric_cols = ['DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'INVOICE_REVLINE_TOTAL',
                       'ORIGINAL_AMOUNT_DUE', 'FUNCTIONAL_AMOUNT', 'YYYYWK']
        for col in numeric_cols:
            if col in fe_dataset.columns:
                fe_dataset[col] = pd.to_numeric(fe_dataset[col], errors='coerce')

        # Sort by customer and week to ensure proper time series order
        # This is critical for time series feature extraction tools
        fe_dataset = fe_dataset.sort_values(['CUST_ACCOUNT_NUMBER', 'YYYYWK'])

        # Remove any duplicate rows that may have been introduced during merging
        fe_dataset = fe_dataset.drop_duplicates()

        # Keep data intact - DO NOT impute missing values or drop rows
        # Missing values are preserved as NaN for downstream feature engineering tools
        # This ensures intact data is used without assumptions about missing value meanings

        # Store in processed data for reference
        self.processed_data['feature_engineering_dataset'] = fe_dataset

        self.logger.info(f"Created feature engineering dataset: {fe_dataset.shape[0]} rows, {fe_dataset.shape[1]} columns")
        self.logger.info(f"Customers in dataset: {fe_dataset['CUST_ACCOUNT_NUMBER'].nunique()}")
        self.logger.info(f"Date range: {fe_dataset['YYYYWK'].min()} to {fe_dataset['YYYYWK'].max()}")

        return fe_dataset

    def get_churn_summary(self) -> pd.DataFrame:
        """
        Create a comprehensive churn property summary DataFrame with key statistics.

        This method analyzes churn patterns across the customer base and provides:
        - Overall churn metrics (count, percentage)
        - Monthly churn statistics (mean, std dev)
        - Segment-wise churn analysis
        - Time-based churn patterns
        - Customer lifecycle metrics

        Returns:
            pd.DataFrame: Summary DataFrame with churn statistics including:
                - total_customers: Total number of customers
                - churned_customers: Number of churned customers
                - active_customers: Number of active customers
                - churn_rate_pct: Overall churn percentage
                - monthly_churn_mean: Average monthly churn rate
                - monthly_churn_std: Standard deviation of monthly churn
                - avg_lifespan_days: Average customer lifespan before churn
                - segment_churn_rates: Churn rates by customer segment
                - And more detailed metrics
        """
        self.logger.info("Creating comprehensive churn summary")

        # Ensure data is loaded
        if not self.processed_data:
            self.logger.warning("No data loaded. Call load_data() first.")
            return pd.DataFrame()

        # Get customer metadata
        customers = self.raw_data.get('customers', pd.DataFrame())
        if customers.empty:
            self.logger.warning("No customer data available for churn summary")
            return pd.DataFrame()

        # Initialize summary dictionary
        summary = {}

        # 1. Overall Customer Counts
        summary['total_customers'] = len(customers)
        # CHURNED_FLAG is now int from bool_to_int validator (1=churned, 0=not churned)
        summary['churned_customers'] = len(customers[customers['CHURNED_FLAG'] == 1])
        summary['active_customers'] = len(customers[customers['CHURNED_FLAG'] == 0])
        summary['churn_rate_pct'] = (summary['churned_customers'] / summary['total_customers']) * 100

        # 2. Monthly Churn Analysis
        # CHURNED_FLAG is now int from bool_to_int validator (1=churned, 0=not churned)
        churned_customers = customers[customers['CHURNED_FLAG'] == 1].copy()
        if not churned_customers.empty and 'CHURN_DATE' in churned_customers.columns:
            # Convert churn dates to monthly periods
            churned_customers['CHURN_MONTH'] = pd.to_datetime(churned_customers['CHURN_DATE']).dt.to_period('M')

            # Calculate monthly churn counts
            monthly_churn = churned_customers.groupby('CHURN_MONTH').size()

            # Calculate all months in the data range
            if not monthly_churn.empty:
                all_months = pd.period_range(
                    start=monthly_churn.index.min(),
                    end=monthly_churn.index.max(),
                    freq='M'
                )
                monthly_churn = monthly_churn.reindex(all_months, fill_value=0)

                # Monthly statistics
                summary['monthly_churn_mean'] = float(monthly_churn.mean())
                summary['monthly_churn_std'] = float(monthly_churn.std())
                summary['monthly_churn_median'] = float(monthly_churn.median())
                summary['monthly_churn_min'] = float(monthly_churn.min())
                summary['monthly_churn_max'] = float(monthly_churn.max())

                # Monthly churn rate (as percentage of total customers)
                summary['monthly_churn_rate_mean_pct'] = (summary['monthly_churn_mean'] / summary['total_customers']) * 100
            else:
                summary['monthly_churn_mean'] = 0.0
                summary['monthly_churn_std'] = 0.0
                summary['monthly_churn_median'] = 0.0
                summary['monthly_churn_min'] = 0.0
                summary['monthly_churn_max'] = 0.0
                summary['monthly_churn_rate_mean_pct'] = 0.0

        # 3. Customer Lifespan Analysis
        if 'churned_first_week' in self.raw_data and not self.raw_data['churned_first_week'].empty:
            churned_with_dates = churned_customers.merge(
                self.raw_data['churned_first_week'],
                on='CUST_ACCOUNT_NUMBER',
                how='left'
            )

            if 'YYYYWK' in churned_with_dates.columns:
                # Calculate lifespan in weeks
                churned_with_dates['CHURN_WEEK'] = churned_with_dates['CHURN_DATE'].apply(
                    lambda x: self.iso_converter.convert_date_to_yyyywk(x) if pd.notna(x) else None
                )
                churned_with_dates['LIFESPAN_WEEKS'] = churned_with_dates['CHURN_WEEK'] - churned_with_dates['YYYYWK']

                # Filter valid lifespans
                valid_lifespans = churned_with_dates['LIFESPAN_WEEKS'].dropna()
                valid_lifespans = valid_lifespans[valid_lifespans > 0]

                if not valid_lifespans.empty:
                    summary['avg_lifespan_weeks'] = float(valid_lifespans.mean())
                    summary['avg_lifespan_days'] = float(valid_lifespans.mean() * 7)
                    summary['median_lifespan_weeks'] = float(valid_lifespans.median())
                    summary['std_lifespan_weeks'] = float(valid_lifespans.std())
                else:
                    summary['avg_lifespan_weeks'] = 0.0
                    summary['avg_lifespan_days'] = 0.0
                    summary['median_lifespan_weeks'] = 0.0
                    summary['std_lifespan_weeks'] = 0.0

        # 4. Segment Analysis
        if 'CUSTOMER_SEGMENT' in customers.columns:
            segment_churn = customers.groupby('CUSTOMER_SEGMENT').agg({
                'CHURNED_FLAG': lambda x: (x == 1).sum(),
                'CUST_ACCOUNT_NUMBER': 'count'
            })
            segment_churn.columns = ['churned_count', 'total_count']
            segment_churn['churn_rate_pct'] = (segment_churn['churned_count'] / segment_churn['total_count']) * 100

            # Add top segments to summary
            top_segments = segment_churn.nlargest(5, 'total_count')
            for i, (segment, row) in enumerate(top_segments.iterrows(), 1):
                summary[f'segment_{i}_name'] = segment
                summary[f'segment_{i}_churn_rate_pct'] = float(row['churn_rate_pct'])
                summary[f'segment_{i}_total_customers'] = int(row['total_count'])

        # 5. Time-based Patterns
        if not churned_customers.empty and 'CHURN_DATE' in churned_customers.columns:
            churned_customers['CHURN_YEAR'] = pd.to_datetime(churned_customers['CHURN_DATE']).dt.year
            yearly_churn = churned_customers.groupby('CHURN_YEAR').size()

            if not yearly_churn.empty:
                summary['churn_trend_increasing'] = yearly_churn.iloc[-1] > yearly_churn.iloc[0]
                summary['latest_year_churn'] = int(yearly_churn.iloc[-1])
                summary['earliest_year_churn'] = int(yearly_churn.iloc[0])

        # 6. Risk Score Analysis (if available)
        if 'dnb_risk' in self.raw_data and not self.raw_data['dnb_risk'].empty:
            dnb_data = self.raw_data['dnb_risk']
            if 'CUST_RISK_SCORE' in dnb_data.columns:
                risk_with_churn = customers.merge(
                    dnb_data[['ACCOUNT_NUMBER', 'CUST_RISK_SCORE']],
                    left_on='CUST_ACCOUNT_NUMBER',
                    right_on='ACCOUNT_NUMBER',
                    how='left'
                )

                # CHURNED_FLAG is boolean from Pydantic schema conversion
                churned_risk = risk_with_churn[risk_with_churn['CHURNED_FLAG'] == 1]['CUST_RISK_SCORE'].dropna()
                active_risk = risk_with_churn[risk_with_churn['CHURNED_FLAG'] == 0]['CUST_RISK_SCORE'].dropna()

                if not churned_risk.empty:
                    summary['avg_risk_score_churned'] = float(churned_risk.mean())
                    summary['avg_risk_score_active'] = float(active_risk.mean()) if not active_risk.empty else 0.0

        # 7. Create DataFrame from summary
        summary_df = pd.DataFrame([summary])

        # Add metadata
        summary_df['summary_created_at'] = pd.Timestamp.now()
        summary_df['data_environment'] = self.environment

        # Store in processed data
        self.processed_data['churn_summary'] = summary_df

        self.logger.info(f"Created churn summary with {len(summary)} metrics")
        self.logger.info(f"Overall churn rate: {summary.get('churn_rate_pct', 0):.2f}%")
        self.logger.info(f"Monthly churn mean: {summary.get('monthly_churn_mean', 0):.2f} customers")

        return summary_df

    def save_churn_summary(self, filepath: str, format: str = 'both') -> None:
        """
        Save churn summary to file(s).

        Args:
            filepath: Base filepath (without extension)
            format: 'csv', 'json', or 'both' (default)
        """
        summary = self.processed_data.get('churn_summary')
        if summary is None or summary.empty:
            summary = self.get_churn_summary()

        if summary.empty:
            self.logger.warning("No churn summary to save")
            return

        # Ensure directory exists
        from pathlib import Path
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        if format in ['csv', 'both']:
            csv_path = f"{filepath}.csv"
            summary.to_csv(csv_path, index=False)
            self.logger.info(f"Saved churn summary to {csv_path}")

        if format in ['json', 'both']:
            json_path = f"{filepath}.json"
            # Convert to JSON-serializable format
            summary_dict = summary.to_dict('records')[0]
            # Handle Timestamp serialization
            for key, value in summary_dict.items():
                if isinstance(value, pd.Timestamp):
                    summary_dict[key] = value.isoformat()

            with open(json_path, 'w') as f:
                json.dump(summary_dict, f, indent=2)
            self.logger.info(f"Saved churn summary to {json_path}")

    def get_monthly_churn_distribution(self) -> Dict:
        """
        Analyze the distribution of customer churn per month.

        Returns comprehensive statistics about monthly churn patterns including:
        - Number of customers churning each month
        - Statistical parameters (mean, median, std, min, max)
        - Monthly breakdown with year-month labels
        - Percentile distributions
        - Seasonal patterns

        Returns:
            Dictionary containing:
                - 'monthly_churn_counts': DataFrame with churn counts per month
                - 'distribution_stats': Statistical parameters of the distribution
                - 'seasonal_stats': Quarterly aggregation statistics
                - 'yearly_stats': Yearly aggregation statistics
        """
        self.logger.info("Analyzing monthly churn distribution")

        # Get customer metadata with churn information
        customers = self.processed_data.get('customer_metadata', pd.DataFrame())

        if customers.empty or 'CHURN_DATE' not in customers.columns:
            self.logger.warning("No churn data available for monthly distribution analysis")
            return {}

        # Filter to only churned customers
        churned = customers[customers['CHURNED_FLAG'] == 1].copy()

        if churned.empty:
            self.logger.warning("No churned customers found")
            return {}

        # Ensure CHURN_DATE is datetime
        churned['CHURN_DATE'] = pd.to_datetime(churned['CHURN_DATE'])
        
        # Filter churned customers to only those who churned after analysis_start_date
        initial_churned = len(churned)
        churned = churned[churned['CHURN_DATE'] >= self.analysis_start_date]
        filtered_churned = initial_churned - len(churned)
        if filtered_churned > 0:
            self.logger.info(f"Filtered out {filtered_churned} customers who churned before {self.analysis_start_date}")

        if churned.empty:
            self.logger.warning(f"No churned customers found after {self.analysis_start_date}")
            return {}

        # Extract year-month for grouping
        churned['CHURN_YEAR_MONTH'] = churned['CHURN_DATE'].dt.to_period('M')
        churned['CHURN_YEAR'] = churned['CHURN_DATE'].dt.year
        churned['CHURN_MONTH'] = churned['CHURN_DATE'].dt.month
        churned['CHURN_QUARTER'] = churned['CHURN_DATE'].dt.quarter

        # Calculate monthly churn counts
        monthly_counts = churned.groupby('CHURN_YEAR_MONTH').agg({
            'CUST_ACCOUNT_NUMBER': 'count',
            'CHURN_DATE': ['min', 'max']
        })
        monthly_counts.columns = ['customers_churned', 'first_churn_date', 'last_churn_date']
        monthly_counts = monthly_counts.reset_index()

        # Convert period to string for better display
        monthly_counts['month_label'] = monthly_counts['CHURN_YEAR_MONTH'].astype(str)

        # Calculate distribution statistics
        churn_values = monthly_counts['customers_churned'].values

        distribution_stats = {
            'mean_churns_per_month': float(churn_values.mean()),
            'median_churns_per_month': float(np.median(churn_values)),
            'std_churns_per_month': float(churn_values.std()),
            'min_churns_per_month': int(churn_values.min()),
            'max_churns_per_month': int(churn_values.max()),
            'total_months': len(monthly_counts),
            'total_churned_customers': int(churn_values.sum()),
            'coefficient_of_variation': float(churn_values.std() / churn_values.mean()) if churn_values.mean() > 0 else 0,
            'percentile_25': float(np.percentile(churn_values, 25)),
            'percentile_75': float(np.percentile(churn_values, 75)),
            'percentile_90': float(np.percentile(churn_values, 90)),
            'percentile_95': float(np.percentile(churn_values, 95)),
            'iqr': float(np.percentile(churn_values, 75) - np.percentile(churn_values, 25))
        }

        # Identify peak and low churn months
        peak_month = monthly_counts.loc[monthly_counts['customers_churned'].idxmax()]
        low_month = monthly_counts.loc[monthly_counts['customers_churned'].idxmin()]

        distribution_stats['peak_churn_month'] = peak_month['month_label']
        distribution_stats['peak_churn_count'] = int(peak_month['customers_churned'])
        distribution_stats['lowest_churn_month'] = low_month['month_label']
        distribution_stats['lowest_churn_count'] = int(low_month['customers_churned'])

        # Calculate seasonal statistics (by quarter)
        seasonal_stats = churned.groupby(['CHURN_YEAR', 'CHURN_QUARTER']).agg({
            'CUST_ACCOUNT_NUMBER': 'count'
        }).rename(columns={'CUST_ACCOUNT_NUMBER': 'customers_churned'})

        # Aggregate by quarter across all years
        quarterly_avg = churned.groupby('CHURN_QUARTER')['CUST_ACCOUNT_NUMBER'].agg(['count', 'mean'])
        quarterly_avg.columns = ['total_churns', 'avg_churns']

        # Calculate yearly statistics
        yearly_stats = churned.groupby('CHURN_YEAR').agg({
            'CUST_ACCOUNT_NUMBER': 'count',
            'CHURN_MONTH': 'nunique'
        })
        yearly_stats.columns = ['customers_churned', 'months_with_churn']
        yearly_stats['avg_monthly_churn'] = yearly_stats['customers_churned'] / yearly_stats['months_with_churn']

        # Calculate month-of-year patterns (e.g., January across all years)
        month_patterns = churned.groupby('CHURN_MONTH').agg({
            'CUST_ACCOUNT_NUMBER': ['count', 'mean']
        })
        month_patterns.columns = ['total_churns', 'avg_churns']
        month_patterns['month_name'] = pd.to_datetime(month_patterns.index, format='%m').strftime('%B')

        # Log summary statistics
        self.logger.info(f"Monthly churn distribution analysis complete:")
        self.logger.info(f"  Total months analyzed: {distribution_stats['total_months']}")
        self.logger.info(f"  Mean churns per month: {distribution_stats['mean_churns_per_month']:.2f}")
        self.logger.info(f"  Std dev: {distribution_stats['std_churns_per_month']:.2f}")
        self.logger.info(f"  Range: {distribution_stats['min_churns_per_month']} to {distribution_stats['max_churns_per_month']}")
        self.logger.info(f"  Peak month: {distribution_stats['peak_churn_month']} ({distribution_stats['peak_churn_count']} churns)")

        return {
            'monthly_churn_counts': monthly_counts,
            'distribution_stats': distribution_stats,
            'seasonal_stats': seasonal_stats.to_dict(),
            'yearly_stats': yearly_stats.to_dict(),
            'quarterly_averages': quarterly_avg.to_dict(),
            'month_of_year_patterns': month_patterns.to_dict()
        }


class SnowLiveDataLoader(SnowDataLoader):
    """
    Implementation for loading and preparing live customer data for inferencing.
    
    This class loads active customer data that will be used for churn prediction,
    following the same patterns as SnowTrainDataLoader but focused on active customers
    who haven't churned yet.
    
    Key features:
    - Loads only active (non-churned) customer data
    - Follows same data processing pipeline as training data
    - Prepares data for inferencing with trained models
    """
    
    def __init__(self, config: DictConfig, environment: str = "development"):
        """
        Initialize the live data loader.
        
        Args:
            config: Hydra configuration object
            environment: Environment to use (development/production)
        """
        super().__init__(config, environment)
        
        # Initialize converters
        self.iso_converter = ISOWeekDateConverter(config, log_operations=False)
        self.midpoint_converter = WeekMidpointConverter(config, log_operations=False)
        
        # Initialize data containers
        self.raw_data = {}
        self.processed_data = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load and prepare all live/active customer data for inferencing.
            
        Returns:
            Dictionary containing:
                - 'time_series_features': Time series data for active customers
                - 'static_features': Non-time series features
                - 'customer_metadata': Customer information for active customers
                - 'combined_features': Merged features ready for inferencing
                - 'feature_engineering_dataset': Dataset ready for feature engineering
        """
        self.logger.info("Starting live/active customer data loading process")
        
        # Step 1: Load raw data from Snowflake (active customers only)
        self._load_raw_data()
        
        # Step 2: Process usage data (no imputation)
        self._process_usage_data()
        
        # Step 3: Merge and prepare datasets
        self._merge_datasets()
        
        # Step 4: Prepare time series features
        self._prepare_time_series_features()
        
        # Step 5: Prepare static features
        self._prepare_static_features()
        
        # Step 6: Calculate derived features
        self._calculate_derived_features()
        
        
        self.logger.info("Live/active customer data loading completed successfully")
        
        return self.processed_data
    
    def _load_raw_data(self):
        """Load all raw data tables from Snowflake for active customers only."""
        self.logger.info("Loading raw data from Snowflake for active customers")
        
        with SnowFetch(config=self.config, environment=self.environment) as fetcher:
            # Load ALL customers usage data using the same query as training
            # We'll filter to active customers after loading to avoid query duplication
            self.logger.info("Fetching usage data using shared query (will filter to active customers)")
            usage_latest_sql = self.config.products.DOCUWARE.db.snowflake.sequel_rules.queries.usage_latest.sql
            usage_latest = fetcher.fetch_custom_query_validation_enforced(
                sql_query=usage_latest_sql,
                query_name="usage_latest",
                context="join_rules"
            )
            
            # Filter to only active customers (CHURNED_FLAG != 'Y' or 1)
            # Note: CHURNED_FLAG is converted to Int64 by Pydantic validation
            if 'CHURNED_FLAG' in usage_latest.columns:
                active_usage = usage_latest[usage_latest['CHURNED_FLAG'] != 1].copy()
                self.logger.info(f"Filtered from {len(usage_latest)} to {len(active_usage)} active customer records")
            else:
                # If no CHURNED_FLAG column, use all data
                active_usage = usage_latest
                self.logger.warning("No CHURNED_FLAG column found, using all records")
            
            self.raw_data['usage_latest'] = active_usage
            self.logger.info(f"Loaded active usage: {active_usage.shape[0]} rows")
            
            # Load other tables using validation-enforced fetching
            tables_to_load = [
                ("PS_DOCUWARE_PAYMENTS", "payments"),
                ("PS_DOCUWARE_REVENUE", "revenue"),
                ("PS_DOCUWARE_TRX", "transactions"),
                ("PS_DOCUWARE_CONTRACT_SUBLINE", "contracts_sub"),
                ("PS_DOCUWARE_SSCD_RENEWALS", "renewals"),
                ("PS_DOCUWARE_L1_CUST", "customers"),
                ("DNB_RISK_BREAKDOWN", "dnb_risk")
            ]
            
            for table_name, key in tables_to_load:
                self.logger.info(f"Fetching {table_name}")
                df = fetcher.fetch_data_validation_enforced(
                    table_name=table_name,
                    context="raw"
                )
                self.raw_data[key] = df
                self.logger.info(f"Loaded {key}: {df.shape[0]} rows")
            
            # Load ALL customers first week data and filter to active
            self.logger.info("Fetching first week data for all customers (will filter to active)")
            
            # Check if the new query exists, otherwise fall back to modifying the old one
            if hasattr(self.config.products.DOCUWARE.db.snowflake.sequel_rules.queries, 'all_customers_first_week'):
                all_first_week_sql = self.config.products.DOCUWARE.db.snowflake.sequel_rules.queries.all_customers_first_week.sql
                query_name = "all_customers_first_week"
            else:
                # Fallback: use churned query and modify it
                churned_first_week_sql = self.config.products.DOCUWARE.db.snowflake.sequel_rules.queries.churned_first_week.sql
                all_first_week_sql = churned_first_week_sql.replace("WHERE b.CHURNED_FLAG='Y'", "")
                query_name = "all_first_week_modified"
            
            all_first_week = fetcher.fetch_custom_query_validation_enforced(
                sql_query=all_first_week_sql,
                query_name=query_name,
                context="join_rules"
            )
            
            # Filter to active customers only (CHURNED_FLAG != 'Y' or 1)
            if 'CHURNED_FLAG' in all_first_week.columns:
                active_first_week = all_first_week[all_first_week['CHURNED_FLAG'] != 1].copy()
                self.logger.info(f"Filtered from {len(all_first_week)} to {len(active_first_week)} active customer first week records")
            else:
                # If no CHURNED_FLAG in query result, filter using customers data
                customers_df = self.raw_data['customers']
                active_customers = customers_df[customers_df['CHURNED_FLAG'] != 1]['CUST_ACCOUNT_NUMBER']
                active_first_week = all_first_week[all_first_week['CUST_ACCOUNT_NUMBER'].isin(active_customers)]
                self.logger.info(f"Filtered to {len(active_first_week)} active customer first week records")
            
            self.raw_data['active_first_week'] = active_first_week
    
    def _process_usage_data(self):
        """Process usage data with intact values (no imputation or dropping), focusing on active customers."""
        self.logger.info("Processing usage data for active customers (no imputation)")
        
        usage_latest = self.raw_data['usage_latest'].copy()
        
        # Remove duplicates
        usage_latest = usage_latest.drop_duplicates()
        
        # Focus only on contracts starting from analysis_start_date onwards
        usage_latest = usage_latest[usage_latest['CONTRACT_START'] >= self.analysis_start_date]
        self.logger.info(f"Filtered to contracts starting >= {self.analysis_start_date}: {usage_latest.shape[0]} rows")
        
        # Filter to only active customers (CHURNED_FLAG == 0 after Pydantic validation)
        customers_df = self.raw_data['customers']
        active_customers = customers_df[customers_df['CHURNED_FLAG'] == 0]['CUST_ACCOUNT_NUMBER'].unique()
        usage_latest = usage_latest[usage_latest['CUST_ACCOUNT_NUMBER'].isin(active_customers)]
        
        # Sort and canonicalize contract line items
        usage_latest['CONTRACT_LINE_ITEMS'] = usage_latest['CONTRACT_LINE_ITEMS'].apply(
            self._sort_contract_line_items
        )
        
        # Remove duplicates based on canonical form
        usage_latest = self._deduplicate_by_contract_items(usage_latest)
        
        self.processed_data['usage_processed'] = usage_latest
        self.logger.info(f"Processed active customer usage data: {usage_latest.shape[0]} rows")
    
    def _sort_contract_line_items(self, items: Any) -> str:
        """Sort contract line items into canonical form."""
        if pd.isna(items) or items is None:
            return ""
        
        items_str = str(items)
        items_list = [item.strip() for item in items_str.split(',')]
        items_sorted = sorted(items_list)
        
        return ','.join(items_sorted)
    
    def _deduplicate_by_contract_items(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates keeping the records with the most complete contract information."""
        if 'CONTRACT_LINE_ITEMS' not in df.columns:
            return df
        
        df = df.copy()
        df['items_length'] = df['CONTRACT_LINE_ITEMS'].str.len()
        
        df = df.sort_values(
            ['CUST_ACCOUNT_NUMBER', 'YYYYWK', 'items_length'],
            ascending=[True, True, False]
        )
        
        df = df.groupby(['CUST_ACCOUNT_NUMBER', 'YYYYWK'], as_index=False).first()
        df = df.drop('items_length', axis=1)
        
        return df
    
    def _merge_datasets(self):
        """Merge various datasets for active customers only."""
        self.logger.info("Merging datasets for active customers")
        
        # Get active customers only
        customers_df = self.raw_data['customers'].drop_duplicates()
        active_customers = customers_df[customers_df['CHURNED_FLAG'] == 0].copy()
        
        # Process payments for active customers
        payments_df = self.raw_data['payments'].drop_duplicates()
        
        payments_active = payments_df.merge(
            active_customers[['CUST_ACCOUNT_NUMBER']],
            left_on='CUSTOMER_NO',
            right_on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        payments_active = payments_active.drop('CUSTOMER_NO', axis=1)
        
        # Aggregate numeric columns
        numeric_cols = payments_active.select_dtypes(include=['number']).columns.tolist()
        group_cols = ['CUST_ACCOUNT_NUMBER', 'RECEIPT_DATE']
        agg_cols = [col for col in numeric_cols if col not in group_cols]
        
        if agg_cols:
            agg_dict = {col: 'sum' for col in agg_cols}
            payments_active = payments_active.groupby(group_cols).agg(agg_dict).reset_index()
        else:
            payments_active = payments_active.groupby(group_cols).size().reset_index(name='count')
        
        payments_active['MONTH'] = payments_active['RECEIPT_DATE']
        
        # Process revenue for active customers
        revenue_df = self.raw_data['revenue'].drop_duplicates()
        
        revenue_active = revenue_df.merge(
            active_customers[['CUST_ACCOUNT_NUMBER']],
            on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        
        numeric_cols = revenue_active.select_dtypes(include=['number']).columns.tolist()
        group_cols = ['CUST_ACCOUNT_NUMBER', 'DATE_INVOICE_GL_DATE']
        agg_cols = [col for col in numeric_cols if col not in group_cols]
        
        if agg_cols:
            agg_dict = {col: 'sum' for col in agg_cols}
            revenue_active = revenue_active.groupby(group_cols).agg(agg_dict).reset_index()
        else:
            revenue_active = revenue_active.groupby(group_cols).size().reset_index(name='count')
        
        revenue_active['MONTH'] = revenue_active['DATE_INVOICE_GL_DATE']
        
        # Process transactions for active customers
        trx_df = self.raw_data['transactions'].drop_duplicates()
        
        trx_active = trx_df.merge(
            active_customers[['CUST_ACCOUNT_NUMBER']],
            left_on='ACCOUNT_NUMBER',
            right_on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        trx_active = trx_active.drop('ACCOUNT_NUMBER', axis=1)
        
        numeric_cols = trx_active.select_dtypes(include=['number']).columns.tolist()
        group_cols = ['CUST_ACCOUNT_NUMBER', 'TRX_DATE']
        agg_cols = [col for col in numeric_cols if col not in group_cols]
        
        if agg_cols:
            agg_dict = {col: 'sum' for col in agg_cols}
            trx_active = trx_active.groupby(group_cols).agg(agg_dict).reset_index()
        else:
            trx_active = trx_active.groupby(group_cols).size().reset_index(name='count')
        
        trx_active['MONTH'] = trx_active['TRX_DATE']
        
        # Merge payments, revenue, and transactions
        merged = payments_active.merge(
            revenue_active,
            on=['CUST_ACCOUNT_NUMBER', 'MONTH'],
            how='outer'
        ).merge(
            trx_active,
            on=['CUST_ACCOUNT_NUMBER', 'MONTH'],
            how='outer'
        )
        
        # Process contracts
        contracts_df = self.raw_data['contracts_sub'].drop_duplicates()
        
        contracts_df['SUB_EARLIEST_DATE'] = contracts_df.groupby(
            'CUST_ACCOUNT_NUMBER'
        )['SLINE_START_DATE'].transform('min')
        contracts_df['SUB_LATEST_DATE'] = contracts_df.groupby(
            'CUST_ACCOUNT_NUMBER'
        )['SLINE_END_DATE'].transform('max')
        
        contracts_active = contracts_df.merge(
            active_customers[['CUST_ACCOUNT_NUMBER']],
            on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        
        contracts_active = contracts_active.drop_duplicates()
        
        # Continue merging with contracts
        merged = merged.merge(
            contracts_active,
            left_on=['CUST_ACCOUNT_NUMBER', 'MONTH'],
            right_on=['CUST_ACCOUNT_NUMBER', 'SLINE_START_DATE'],
            how='outer'
        )
        merged['MONTH'] = merged['MONTH'].combine_first(merged['SLINE_START_DATE'])
        
        # Process renewals
        renewals_df = self.raw_data['renewals'].drop_duplicates()
        
        renewals_df['RENEWALS_EARLIEST_DATE'] = renewals_df.groupby(
            'BILLTOCUSTOMERNUMBER'
        )['STARTDATECOVERAGE'].transform('min')
        renewals_df['RENEWALS_LATEST_DATE'] = renewals_df.groupby(
            'BILLTOCUSTOMERNUMBER'
        )['CONTRACT_END_DATE'].transform('max')
        
        renewals_active = renewals_df[['BILLTOCUSTOMERNUMBER', 'STARTDATECOVERAGE',
                                       'CONTRACT_END_DATE', 'RENEWALS_EARLIEST_DATE',
                                       'RENEWALS_LATEST_DATE']].copy()
        
        renewals_active = renewals_active.merge(
            active_customers[['CUST_ACCOUNT_NUMBER']],
            left_on='BILLTOCUSTOMERNUMBER',
            right_on='CUST_ACCOUNT_NUMBER',
            how='inner'
        ).drop('BILLTOCUSTOMERNUMBER', axis=1)
        
        # Merge renewals into main dataset
        merged = merged.merge(
            renewals_active,
            on='CUST_ACCOUNT_NUMBER',
            how='left'
        )
        
        # Process DNB risk
        dnb_df = self.raw_data['dnb_risk'].drop_duplicates()
        
        dnb_active = dnb_df.merge(
            active_customers[['CUST_ACCOUNT_NUMBER']],
            left_on='ACCOUNT_NUMBER',
            right_on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        dnb_active = dnb_active.drop('ACCOUNT_NUMBER', axis=1)
        
        # Add DNB risk to merged
        merged = merged.merge(dnb_active, on='CUST_ACCOUNT_NUMBER', how='left')
        
        # Process usage data
        usage_processed = self.processed_data['usage_processed']
        usage_active = usage_processed.merge(
            active_customers[['CUST_ACCOUNT_NUMBER']],
            on='CUST_ACCOUNT_NUMBER',
            how='inner'
        )
        
        # Create YYYYWK_MONTH column
        usage_active['YYYYWK_MONTH'] = usage_active['YYYYWK'].apply(
            lambda x: pd.Timestamp(self.midpoint_converter.convert_yyyywk_to_actual_mid_date(x))
            if pd.notna(x) and x > 0 else pd.NaT
        )
        
        # Ensure datetime types
        if 'MONTH' in merged.columns:
            merged['MONTH'] = pd.to_datetime(merged['MONTH'])
        usage_active['YYYYWK_MONTH'] = pd.to_datetime(usage_active['YYYYWK_MONTH'])
        
        # Final merge with usage
        merged_final = merged.merge(
            usage_active,
            left_on=['CUST_ACCOUNT_NUMBER', 'MONTH'],
            right_on=['CUST_ACCOUNT_NUMBER', 'YYYYWK_MONTH'],
            how='outer'
        )
        merged_final['MONTH'] = merged_final['MONTH'].combine_first(merged_final['YYYYWK_MONTH'])
        
        # Calculate derived dates
        date_cols = ['RECEIPT_DATE', 'DATE_INVOICE_GL_DATE', 'TRX_DATE', 'SLINE_START_DATE']
        if 'STARTDATECOVERAGE' in merged_final.columns:
            date_cols.append('STARTDATECOVERAGE')
        
        existing_date_cols = [col for col in date_cols if col in merged_final.columns]
        
        merged_final['EARLIEST_DATE'] = merged_final[existing_date_cols].min(axis=1)
        merged_final['FINAL_EARLIEST_DATE'] = merged_final.groupby(
            'CUST_ACCOUNT_NUMBER'
        )['EARLIEST_DATE'].transform('min')
        
        # For active customers, we don't have CHURN_DATE, so we'll calculate time elapsed instead
        merged_final['DAYS_ELAPSED'] = (pd.Timestamp.today() - merged_final['FINAL_EARLIEST_DATE']).dt.days
        merged_final['MONTHS_ELAPSED'] = merged_final['DAYS_ELAPSED'] / 30
        
        # Drop unnecessary columns
        cols_to_drop = ['SLINE_END_DATE', 'SLINE_STATUS', 'SUB_EARLIEST_DATE',
                        'SUB_LATEST_DATE', 'CONTRACT_END_DATE', 'CHURNED_FLAG',
                        'EARLIEST_DATE', 'CONTRACT_NUMBER', 'CUST_PARTY_NAME',
                        'CUSTOMER_NAME', 'CONTRACT_END', 'CHURN_DATE',
                        'JAROWINKLER_SIMILARITY(A.CUST_PARTY_NAME, B.CUSTOMER_NAME)']
        
        cols_to_drop = [col for col in cols_to_drop if col in merged_final.columns]
        merged_final = merged_final.drop(columns=cols_to_drop)
        
        # Remove duplicates
        merged_final = merged_final.drop_duplicates()
        
        self.processed_data['merged_data'] = merged_final
        self.logger.info(f"Merged active customer data: {merged_final.shape[0]} rows")
    
    def _prepare_time_series_features(self):
        """Prepare time series features for active customers."""
        self.logger.info("Preparing time series features for active customers")
        
        merged_data = self.processed_data['merged_data']
        
        # Select time series columns
        ts_columns = ['CUST_ACCOUNT_NUMBER', 'YYYYWK', 'DOCUMENTS_OPENED',
                      'USED_STORAGE__MB', 'INVOICE_REVLINE_TOTAL',
                      'ORIGINAL_AMOUNT_DUE', 'FUNCTIONAL_AMOUNT']
        
        # Create YYYYWK from MONTH where missing
        merged_data['YYYYWK'] = merged_data.apply(
            lambda row: self.iso_converter.convert_date_to_yyyywk(row['MONTH'])
            if pd.isna(row['YYYYWK']) and pd.notna(row['MONTH'])
            else row['YYYYWK'],
            axis=1
        )
        
        # Extract time series data
        ts_data = merged_data[
            [col for col in ts_columns if col in merged_data.columns]
        ].copy()
        
        # Remove rows without YYYYWK
        ts_data = ts_data[ts_data['YYYYWK'].notna()]
        
        # Ensure correct types
        ts_data['YYYYWK'] = ts_data['YYYYWK'].astype('Int64')
        ts_data['CUST_ACCOUNT_NUMBER'] = ts_data['CUST_ACCOUNT_NUMBER'].astype('Int64')
        
        # Filter time series data to only include weeks from analysis_start_date onwards
        # Convert analysis_start_date to YYYYWK format for comparison
        analysis_start_yyyywk = self.iso_converter.convert_date_to_yyyywk(self.analysis_start_date)
        initial_ts_rows = len(ts_data)
        ts_data = ts_data[ts_data['YYYYWK'] >= analysis_start_yyyywk]
        filtered_rows = initial_ts_rows - len(ts_data)
        if filtered_rows > 0:
            self.logger.info(f"Filtered out {filtered_rows} rows with YYYYWK < {analysis_start_yyyywk} (before {self.analysis_start_date})")
            self.logger.info(f"Time series date range after filtering: {ts_data['YYYYWK'].min()} to {ts_data['YYYYWK'].max()}")
        
        # Rename columns if needed
        if 'USED_STORAGE__MB' in ts_data.columns:
            ts_data = ts_data.rename(columns={'USED_STORAGE__MB': 'USED_STORAGE_MB'})
        
        # Sort and remove duplicates
        ts_data = ts_data.sort_values(['CUST_ACCOUNT_NUMBER', 'YYYYWK'])
        initial_rows = len(ts_data)
        ts_data = ts_data.drop_duplicates(subset=['CUST_ACCOUNT_NUMBER', 'YYYYWK'], keep='first')
        duplicates_removed = initial_rows - len(ts_data)
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate customer-week combinations")
        
        # Keep data intact - DO NOT impute missing values
        # Missing values are preserved as NaN for downstream processing
        
        self.processed_data['time_series_features'] = ts_data
        self.logger.info(f"Time series features for active customers: {ts_data.shape[0]} rows")
    
    def _prepare_static_features(self):
        """Prepare static features for active customers."""
        self.logger.info("Preparing static features for active customers")
        
        merged_data = self.processed_data['merged_data']
        
        # Select static columns
        static_cols = ['PROBABILITY_OF_DELINQUENCY',
                       'RICOH_CUSTOM_RISK_MODEL', 'OVERALL_BUSINESS_RISK',
                       'PAYMENT_RISK_TRIPLE_A_RATING', 'CONTRACT_LINE_ITEMS',
                       'DAYS_ELAPSED', 'MONTHS_ELAPSED']
        
        # Get first value for each customer
        available_static_cols = [col for col in static_cols if col in merged_data.columns]
        
        if available_static_cols:
            static_data = merged_data.groupby('CUST_ACCOUNT_NUMBER')[
                available_static_cols
            ].first().reset_index()
        else:
            static_data = pd.DataFrame({
                'CUST_ACCOUNT_NUMBER': merged_data['CUST_ACCOUNT_NUMBER'].unique()
            })
        
        # Process categorical features
        static_processed = self._process_categorical_features(static_data)
        
        self.processed_data['static_features'] = static_processed
        self.logger.info(f"Static features for active customers: {static_processed.shape}")
    
    def _process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process categorical features with encoding."""
        df = df.copy()
        
        # Preserve missing values for categorical columns - no imputation
        # Only standardize existing values, keep NaN values intact
        categorical_cols = ['OVERALL_BUSINESS_RISK', 'PAYMENT_RISK_TRIPLE_A_RATING']
        
        for col in categorical_cols:
            if col in df.columns:
                # Check if column has the right dtype and convert if needed
                if df[col].dtype != 'object' and df[col].dtype != 'string':
                    # Convert to string only for non-null values
                    mask = df[col].notna()
                    if mask.any():
                        df.loc[mask, col] = df.loc[mask, col].astype(str)
                
                # Only replace spaces in non-null string values
                if df[col].dtype == 'object' or df[col].dtype == 'string':
                    mask = df[col].notna()
                    if mask.any():
                        df.loc[mask, col] = df.loc[mask, col].str.replace(' ', '_', regex=False)
        
        # Process CONTRACT_LINE_ITEMS - preserve missing values
        if 'CONTRACT_LINE_ITEMS' in df.columns:
            # Check dtype and convert if needed
            if df['CONTRACT_LINE_ITEMS'].dtype != 'object' and df['CONTRACT_LINE_ITEMS'].dtype != 'string':
                mask = df['CONTRACT_LINE_ITEMS'].notna()
                if mask.any():
                    df.loc[mask, 'CONTRACT_LINE_ITEMS'] = df.loc[mask, 'CONTRACT_LINE_ITEMS'].astype(str)
            
            # Apply string operations only on non-null values
            if df['CONTRACT_LINE_ITEMS'].dtype == 'object' or df['CONTRACT_LINE_ITEMS'].dtype == 'string':
                mask = df['CONTRACT_LINE_ITEMS'].notna()
                if mask.any():
                    df.loc[mask, 'CONTRACT_LINE_ITEMS'] = df.loc[mask, 'CONTRACT_LINE_ITEMS'].str.replace(
                        r'\d+x ', '', regex=True
                    )
                    df.loc[mask, 'CONTRACT_LINE_ITEMS'] = df.loc[mask, 'CONTRACT_LINE_ITEMS'].apply(
                        self._sort_contract_line_items
                    )
        
        return df
    
    def _calculate_derived_features(self):
        """Calculate additional derived features for active customers."""
        self.logger.info("Calculating derived features for active customers")
        
        # Get customer metadata (active customers only)
        customers_df = self.raw_data['customers'].copy()
        active_customers = customers_df[customers_df['CHURNED_FLAG'] == 0]
        
        # Store customer metadata
        self.processed_data['customer_metadata'] = active_customers[
            ['CUST_ACCOUNT_NUMBER', 'CUST_PARTY_NAME', 'CHURNED_FLAG',
             'CUSTOMER_SEGMENT', 'CUSTOMER_SEGMENT_LEVEL']
        ].drop_duplicates()
        
        # Calculate time-based features
        self._calculate_time_based_features()
        
        self.logger.info("Derived features calculation completed for active customers")
    
    def _calculate_time_based_features(self):
        """Calculate time-based features from usage patterns."""
        ts_features = self.processed_data['time_series_features']
        
        # Group by customer to calculate aggregates
        customer_aggs = ts_features.groupby('CUST_ACCOUNT_NUMBER').agg({
            'DOCUMENTS_OPENED': ['sum', 'mean', 'std', 'max'],
            'USED_STORAGE_MB': ['sum', 'mean', 'std', 'max'],
            'INVOICE_REVLINE_TOTAL': ['sum', 'mean', 'std'],
            'ORIGINAL_AMOUNT_DUE': ['sum', 'mean', 'std'],
            'FUNCTIONAL_AMOUNT': ['sum', 'mean', 'std'],
            'YYYYWK': ['min', 'max', 'count']
        })
        
        # Flatten column names
        customer_aggs.columns = ['_'.join(col).strip() for col in customer_aggs.columns]
        if 'CUST_ACCOUNT_NUMBER' not in customer_aggs.columns:
            customer_aggs = customer_aggs.reset_index()
        
        # Calculate usage duration in weeks
        customer_aggs['USAGE_DURATION_WEEKS'] = (
            customer_aggs['YYYYWK_max'] - customer_aggs['YYYYWK_min']
        )
        
        # Rename count column
        customer_aggs = customer_aggs.rename(columns={'YYYYWK_count': 'TOTAL_WEEKS'})
        
        self.processed_data['aggregated_features'] = customer_aggs
        self.logger.info(f"Calculated aggregated features for {len(customer_aggs)} active customers")
    
    
    def get_training_ready_data(self) -> pd.DataFrame:
        """
        Get final inferencing-ready dataset with all features combined.
        
        Returns:
            DataFrame ready for model inferencing
        """
        # Combine static and aggregated features
        static_features = self.processed_data.get('static_features', pd.DataFrame())
        aggregated_features = self.processed_data.get('aggregated_features', pd.DataFrame())
        customer_metadata = self.processed_data.get('customer_metadata', pd.DataFrame())
        
        # Merge all features
        if not static_features.empty and not aggregated_features.empty:
            combined = static_features.merge(
                aggregated_features,
                on='CUST_ACCOUNT_NUMBER',
                how='outer'
            )
            
            if not customer_metadata.empty:
                combined = combined.merge(
                    customer_metadata,
                    on='CUST_ACCOUNT_NUMBER',
                    how='left'
                )
            
            self.processed_data['combined_features'] = combined
            return combined
        
        return pd.DataFrame()
    
    def get_feature_engineering_dataset(self) -> pd.DataFrame:
        """
        Prepare intact data for feature engineering on active customers.
        
        IMPORTANT: Data is kept intact with NO imputation or dropping of missing values.
        Missing values are preserved as NaN for downstream feature engineering tools to handle.
        
        Returns:
            DataFrame with intact data for active customers, including:
                - Time series features (may contain NaN values)
                - Customer metadata (may contain NaN values)  
                - Static features (may contain NaN values)
        """
        self.logger.info("Creating feature engineering dataset for active customers")
        
        # Get time series features
        ts_features = self.processed_data.get('time_series_features', pd.DataFrame())
        if ts_features.empty:
            self.logger.warning("No time series features available")
            return pd.DataFrame()
        
        # Get customer metadata
        customer_metadata = self.processed_data.get('customer_metadata', pd.DataFrame())
        static_features = self.processed_data.get('static_features', pd.DataFrame())
        
        # Start with time series as base
        fe_dataset = ts_features.copy()
        
        # Merge customer metadata
        if not customer_metadata.empty:
            metadata_cols = ['CUST_ACCOUNT_NUMBER', 'CHURNED_FLAG',
                           'CUSTOMER_SEGMENT', 'CUSTOMER_SEGMENT_LEVEL']
            available_cols = [col for col in metadata_cols if col in customer_metadata.columns]
            
            fe_dataset = fe_dataset.merge(
                customer_metadata[available_cols].drop_duplicates(),
                on='CUST_ACCOUNT_NUMBER',
                how='left'
            )
        
        # Merge static features
        if not static_features.empty:
            static_cols = ['CUST_ACCOUNT_NUMBER', 'DAYS_ELAPSED', 'MONTHS_ELAPSED']
            available_static = [col for col in static_cols if col in static_features.columns]
            
            if available_static:
                fe_dataset = fe_dataset.merge(
                    static_features[available_static].drop_duplicates(),
                    on='CUST_ACCOUNT_NUMBER',
                    how='left'
                )
        
        # Calculate WEEK_DATE for temporal calculations
        if 'YYYYWK' in fe_dataset.columns:
            fe_dataset['WEEK_DATE'] = fe_dataset['YYYYWK'].apply(
                lambda x: pd.Timestamp(self.midpoint_converter.convert_yyyywk_to_actual_mid_date(x))
                if pd.notna(x) and x > 0 else pd.NaT
            )
        
        # Calculate FINAL_EARLIEST_DATE from merged data
        if 'merged_data' in self.processed_data:
            merged = self.processed_data['merged_data']
            if 'FINAL_EARLIEST_DATE' in merged.columns:
                earliest_dates = merged.groupby('CUST_ACCOUNT_NUMBER')['FINAL_EARLIEST_DATE'].first()
                fe_dataset = fe_dataset.merge(
                    earliest_dates.reset_index(),
                    on='CUST_ACCOUNT_NUMBER',
                    how='left'
                )
        
        # Ensure numeric types for key columns
        numeric_cols = ['DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'INVOICE_REVLINE_TOTAL',
                       'ORIGINAL_AMOUNT_DUE', 'FUNCTIONAL_AMOUNT', 'YYYYWK']
        for col in numeric_cols:
            if col in fe_dataset.columns:
                fe_dataset[col] = pd.to_numeric(fe_dataset[col], errors='coerce')
        
        # Sort by customer and week
        fe_dataset = fe_dataset.sort_values(['CUST_ACCOUNT_NUMBER', 'YYYYWK'])
        
        # Remove duplicates
        fe_dataset = fe_dataset.drop_duplicates()
        
        # Keep data intact - DO NOT impute missing values or drop rows
        # Missing values are preserved as NaN for downstream feature engineering tools
        # This ensures intact data is used without assumptions about missing value meanings
        
        # Store in processed data
        self.processed_data['feature_engineering_dataset'] = fe_dataset
        
        self.logger.info(f"Created feature engineering dataset: {fe_dataset.shape[0]} rows, {fe_dataset.shape[1]} columns")
        self.logger.info(f"Active customers in dataset: {fe_dataset['CUST_ACCOUNT_NUMBER'].nunique()}")
        self.logger.info(f"Date range: {fe_dataset['YYYYWK'].min()} to {fe_dataset['YYYYWK'].max()}")
        
        return fe_dataset
    
    def get_active_summary(self) -> pd.DataFrame:
        """
        Create a summary of active customer statistics.
        
        Returns:
            DataFrame with active customer summary statistics
        """
        self.logger.info("Creating active customer summary")
        
        # Ensure data is loaded
        if not self.processed_data:
            self.logger.warning("No data loaded. Call load_data() first.")
            return pd.DataFrame()
        
        # Get customer metadata
        customers = self.raw_data.get('customers', pd.DataFrame())
        if customers.empty:
            self.logger.warning("No customer data available")
            return pd.DataFrame()
        
        # Initialize summary dictionary
        summary = {}
        
        # Overall Customer Counts
        summary['total_customers'] = len(customers)
        summary['active_customers'] = len(customers[customers['CHURNED_FLAG'] == 0])
        summary['churned_customers'] = len(customers[customers['CHURNED_FLAG'] == 1])
        summary['active_rate_pct'] = (summary['active_customers'] / summary['total_customers']) * 100
        
        # Active Customer Segment Analysis
        active_customers = customers[customers['CHURNED_FLAG'] == 0]
        if 'CUSTOMER_SEGMENT' in active_customers.columns:
            segment_counts = active_customers.groupby('CUSTOMER_SEGMENT').size()
            
            # Add top segments to summary
            top_segments = segment_counts.nlargest(5)
            for i, (segment, count) in enumerate(top_segments.items(), 1):
                summary[f'segment_{i}_name'] = segment
                summary[f'segment_{i}_count'] = int(count)
                summary[f'segment_{i}_pct'] = float(count / summary['active_customers'] * 100)
        
        # Risk Score Analysis for active customers
        if 'dnb_risk' in self.raw_data and not self.raw_data['dnb_risk'].empty:
            dnb_data = self.raw_data['dnb_risk']
            if 'CUST_RISK_SCORE' in dnb_data.columns:
                risk_with_active = active_customers.merge(
                    dnb_data[['ACCOUNT_NUMBER', 'CUST_RISK_SCORE']],
                    left_on='CUST_ACCOUNT_NUMBER',
                    right_on='ACCOUNT_NUMBER',
                    how='left'
                )
                
                active_risk = risk_with_active['CUST_RISK_SCORE'].dropna()
                if not active_risk.empty:
                    summary['avg_risk_score_active'] = float(active_risk.mean())
                    summary['median_risk_score_active'] = float(active_risk.median())
                    summary['std_risk_score_active'] = float(active_risk.std())
        
        # Usage statistics for active customers
        if 'time_series_features' in self.processed_data:
            ts_features = self.processed_data['time_series_features']
            summary['active_customers_with_usage'] = ts_features['CUST_ACCOUNT_NUMBER'].nunique()
            summary['avg_weeks_of_usage'] = float(ts_features.groupby('CUST_ACCOUNT_NUMBER').size().mean())
            summary['total_usage_records'] = len(ts_features)
        
        # Create DataFrame from summary
        summary_df = pd.DataFrame([summary])
        
        # Add metadata
        summary_df['summary_created_at'] = pd.Timestamp.now()
        summary_df['data_environment'] = self.environment
        
        # Store in processed data
        self.processed_data['active_summary'] = summary_df
        
        self.logger.info(f"Created active customer summary with {len(summary)} metrics")
        self.logger.info(f"Active rate: {summary.get('active_rate_pct', 0):.2f}%")
        
        return summary_df
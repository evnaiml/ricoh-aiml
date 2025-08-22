# %%
"""
Manual Snowflake data fetcher with ENFORCED type validation and comprehensive logging.

This script demonstrates manual execution of Snowflake data fetching with automatic
type enforcement using Pydantic schemas. It processes all configured tables with
full data extraction (no row limits) and provides detailed dtype transformation
analysis through the enhanced SnowFetch class methods.

✅ Key Features:
- Uses fetch_data_validation_enforced() for automatic type conversions
- Leverages SnowFetch.analyze_dtype_transformations() for detailed change analysis
- Utilizes SnowFetch.get_schema_info() for schema metadata retrieval
- Processes ALL tables with ALL rows (no limits)
- Returns pandas DataFrames with corrected dtypes (no file writing)
- Comprehensive logging of dtype transformations to log files

📊 Type Enforcement Process:
- Fetches raw data and schema-enforced data for comparison
- String columns containing integers → Int64 (nullable integer)
- String columns containing floats → float64
- Date/time strings → datetime64[ns]
- Failed conversions → NaN/NaT with detailed reporting

📝 Logging Details:
- Dtype transformation reports saved to logs/loguru/
- Includes column-by-column transformation details
- Tracks conversion failures and null value changes
- Memory usage and performance metrics

Updated: 2025-08-13
- Integrated with SnowFetch's new analysis methods
- Removed duplicate local functions
- Enhanced logging with transformation summaries
"""
# %%
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-13
# * CREATED ON: 2025-08-13
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# %%
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.utils.find_paths import ProjectRootFinder
# %%
# Set paths using ProjectRootFinder
project_root = ProjectRootFinder().find_path()
conf_dir = project_root / "conf"
print(f"Project root: {project_root}")
print(f"Config path: {conf_dir}")
# %%
# Load Hydra configuration manually
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
    cfg = compose(config_name="config")
# %%
# Setup logger (automatically creates logs in script's directory)
logger_config = setup_logger(cfg)
logger = get_logger()
logger.info("Manual Snowflake data fetching with enforced type validation started")
# %%
# ALL tables to process (no limits) - same as in 02_fetch_examples_validation_created
snow_table_list = [
    "PS_DOCUWARE_L1_CUST",
    "DNB_RISK_BREAKDOWN",
    "PS_DOCUWARE_CONTRACT_SUBLINE",
    "PS_DOCUWARE_CONTRACT_TOPLINE",
    "PS_DOCUWARE_CONTRACTS",
    "PS_DOCUWARE_CUST_SITES",
    "PS_DOCUWARE_L2_CUST_TENURE",
    "PS_DOCUWARE_LOYALTY_SURVEY",
    "PS_DOCUWARE_PAYMENTS",
    "PS_DOCUWARE_REVENUE",
    "PS_DOCUWARE_SNOW_SURVEY",
    "PS_DOCUWARE_SNOW_INC",
    "PS_DOCUWARE_SSCD_RENEWALS",
    "PS_DOCUWARE_TECH_SURVEY",
    "PS_DOCUWARE_TRX",
    "DOCUWARE_USAGE_JAPAN_V1_LATEST_V"
]
# %%
# Verify schema paths configuration
schema_config = None
if hasattr(cfg, 'validation') and hasattr(cfg.validation, 'pydantic') and hasattr(cfg.validation.pydantic, 'schema_paths'):
    schema_config = cfg.validation.pydantic.schema_paths
    logger.info("Schema paths configuration loaded from validation.pydantic")

if schema_config and hasattr(schema_config, 'DOCUWARE'):
    logger.info(f"Raw schemas path: {schema_config.DOCUWARE.search_paths.raw}")
    logger.info(f"Join rules schemas path: {schema_config.DOCUWARE.search_paths.join_rules}")
else:
    logger.warning("Schema paths configuration not found, will use defaults")
# %%
# Main Processing: Process ALL tables with ALL rows
print("\n" + "="*60)
print("Main Processing: ALL tables with ALL rows")
print("="*60)

print(f"📊 Total tables to process: {len(snow_table_list)}")
print("🚀 Processing ALL rows from ALL tables with type enforcement...")

successful_tables = []
failed_tables = []
dataframes = {}  # Store DataFrames with enforced types

# Initialize SnowFetch once for all tables
with SnowFetch(config=cfg, environment="development") as fetcher:

    for i, snow_table in enumerate(tqdm(snow_table_list, desc="Processing ALL tables")):
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing table ({i+1}/{len(snow_table_list)}): {snow_table}")
            logger.info(f"{'='*60}")

            # First fetch raw data for comparison
            logger.info(f"Fetching raw data for comparison...")
            df_raw = fetcher.fetch_data(snow_table, limit=None)

            # Get schema information using SnowFetch's built-in method
            schema_info = fetcher.get_schema_info(snow_table, "raw")

            # Fetch data with enforced type validation (NO LIMIT!)
            logger.info(f"Applying type enforcement from schema...")
            df = fetcher.fetch_data_validation_enforced(
                table_name=snow_table,
                context="raw",  # Use raw schemas
                limit=None  # NO LIMIT! Process ALL rows
            )

            # Analyze dtype transformations using SnowFetch's built-in method
            transformation_report = fetcher.analyze_dtype_transformations(df_raw, df, snow_table)

            # Store the DataFrame with enforced types
            dataframes[snow_table] = df

            logger.info(f"📊 ({i+1:2d}/{len(snow_table_list)}) {snow_table}: fetched {len(df):,} rows with enforced types")
            print(f"✅ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: {len(df):,} rows with type enforcement")

            # Display DataFrame with corrected dtypes
            print(f"\n📋 {snow_table} - First 5 rows:")
            print(df.head().to_string())

            print(f"\n📊 {snow_table} - Data types and memory usage:")
            print(df.info())

            # Log comprehensive dtype summary
            logger.info("\n" + "="*60)
            logger.info(f"FINAL DTYPE SUMMARY - {snow_table}")
            logger.info("="*60)

            dtype_counts = df.dtypes.value_counts()
            logger.info("Data type distribution:")
            for dtype, count in dtype_counts.items():
                logger.info(f"  {dtype}: {count} columns")

            # Log memory usage
            memory_usage = df.memory_usage(deep=True).sum() / (1024**2)
            logger.info(f"\nMemory usage: {memory_usage:.2f} MB")
            logger.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

            # Log sample of transformed columns for verification
            if transformation_report['summary']['columns_transformed'] > 0:
                logger.info("\nSample values from transformed columns:")
                for trans in transformation_report['transformations'][:5]:
                    col = trans['column']
                    non_null_values = df[col].dropna().head(3).tolist()
                    if non_null_values:
                        logger.info(f"  {col}: {non_null_values}")

            dtype_summary = [f"{col}:{dtype}" for col, dtype in df.dtypes.items()][:10]

            successful_tables.append({
                'table': snow_table,
                'rows': len(df),
                'columns': len(df.columns),
                'dtypes': dtype_summary[:5],  # Sample of dtypes
                'columns_transformed': transformation_report['summary']['columns_transformed'],
                'schema_used': schema_info['schema_found']
            })

            logger.info(f"✅ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: DataFrame ready with enforced types")

        except FileNotFoundError as e:
            logger.warning(f"Schema not found for {snow_table}: {e}")
            print(f"⚠️ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: Schema not found, skipping")
            failed_tables.append({'table': snow_table, 'error': 'Schema not found'})
        except Exception as e:
            logger.error(f"❌ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: failed - {str(e)}")
            print(f"❌ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: Error - {str(e)}")
            failed_tables.append({'table': snow_table, 'error': str(e)})
            continue

# %%
# Log final summary to log file
logger.info("\n" + "="*80)
logger.info("FINAL PROCESSING SUMMARY")
logger.info("="*80)
logger.info(f"Total tables processed: {len(successful_tables)}")
logger.info(f"Total tables failed: {len(failed_tables)}")

if successful_tables:
    total_transformations = sum(t['columns_transformed'] for t in successful_tables)
    tables_with_schema = sum(1 for t in successful_tables if t['schema_used'])
    logger.info(f"Total dtype transformations: {total_transformations}")
    logger.info(f"Tables with schemas applied: {tables_with_schema}/{len(successful_tables)}")

    logger.info("\nPer-table transformation summary:")
    for table in successful_tables:
        if table['columns_transformed'] > 0:
            logger.info(f"  {table['table']}: {table['columns_transformed']} columns transformed")

logger.info("="*80)

# %%
# Display summary of DataFrames in memory
print(f"\n🎉 PROCESSING COMPLETED!")
print(f"✅ Successful: {len(successful_tables)} DataFrames with enforced types")
print(f"❌ Failed: {len(failed_tables)}")

if successful_tables:
    total_rows = sum(result['rows'] for result in successful_tables)
    print(f"📊 Total data in memory: {total_rows:,} rows across {len(successful_tables)} DataFrames")

    print(f"\nDataFrames available in memory with enforced types:")
    for result in successful_tables[:5]:  # Show first 5
        print(f"  - {result['table']}: {result['rows']:,} rows, {result['columns']} columns")
        print(f"    Sample dtypes: {', '.join(result['dtypes'][:3])}")

    if len(successful_tables) > 5:
        print(f"  ... and {len(successful_tables) - 5} more DataFrames")

if failed_tables:
    print(f"\nFailed tables:")
    for failed in failed_tables:
        print(f"  - {failed['table']}: {failed['error']}")

# %%
# Example: Accessing a DataFrame with enforced types
if dataframes:
    example_table = list(dataframes.keys())[0]
    example_df = dataframes[example_table]

    print(f"\n📋 Example DataFrame: {example_table}")
    print(f"Shape: {example_df.shape}")
    print(f"\nData types (first 5 columns):")
    print(example_df.dtypes.head())
    print(f"\nFirst few rows with to_string():")
    print(example_df.head(3).to_string())
    print(f"\nDataFrame info():")
    print(example_df.info())

logger.info("Manual data fetching with type enforcement completed")
# %%
print("\n" + "="*60)
print("✨ Script execution completed!")
print(f"📊 {len(dataframes)} DataFrames available in 'dataframes' dictionary")
print("💡 Access any DataFrame using: dataframes['TABLE_NAME']")
print("="*60)
# %%

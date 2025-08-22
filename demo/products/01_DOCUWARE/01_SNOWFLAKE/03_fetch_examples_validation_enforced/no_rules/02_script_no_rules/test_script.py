"""
Hydra-based Snowflake data fetcher with enforced type validation and analysis.

This script demonstrates automated processing of Snowflake tables using Hydra
configuration with comprehensive type enforcement and logging. It utilizes the
enhanced SnowFetch class methods for dtype transformation analysis and provides
detailed logging of all type conversions.

✅ Key Features:
- Uses fetch_data_validation_enforced() for automatic type conversion
- Leverages SnowFetch.analyze_dtype_transformations() for change tracking
- Applies Pydantic schemas to enforce correct data types
- Processes all configured tables with full data extraction
- Prints DataFrames with corrected dtypes to console
- No file writing - only in-memory DataFrames

📊 Type Enforcement:
- Converts string columns to proper types based on schemas
- Returns NaN/NaT for failed conversions
- Uses Hydra configuration for all parameters
- Tracks and reports all dtype changes

📝 Logging:
- Detailed transformation reports in log files
- Summary statistics for all processed tables
- Performance metrics and memory usage

Updated: 2025-08-13
- Integrated with SnowFetch's analyze_dtype_transformations()
- Enhanced logging with aggregated transformation summaries
- Removed duplicate analysis functions
"""
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
import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch, log_dtype_transformation_summary
from churn_aiml.utils.find_paths import ProjectRootFinder

# ALL tables to process (no limits)
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

# Set paths to directories
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"

# log_dtype_transformation_summary now imported from fetchdata.py


@hydra.main(version_base=None, config_path=conf_dir.as_posix(), config_name="config")
def main(cfg: DictConfig) -> None:
    """Process all Snowflake tables with enforced type validation using Hydra.
    
    This function orchestrates the fetching and type enforcement process for all
    configured Snowflake tables. It leverages SnowFetch's built-in analysis methods
    to provide comprehensive logging of dtype transformations.
    
    Args:
        cfg (DictConfig): Hydra configuration containing:
            - Snowflake connection parameters
            - Schema paths for validation
            - Environment settings
            - Logging configuration
    
    Process Flow:
        1. Initialize SnowFetch with configuration
        2. For each table in snow_table_list:
           - Fetch raw data for comparison
           - Apply schema-based type enforcement
           - Analyze dtype transformations using SnowFetch methods
           - Log detailed results and display to console
        3. Generate comprehensive summary report
        4. Display memory usage and performance metrics
    
    Output:
        - Console: DataFrame previews with corrected dtypes
        - Log files: Detailed transformation reports and statistics
        - Memory: DataFrames dictionary with all processed tables
    """
    
    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("🚀 Processing ALL Snowflake tables with enforced type validation")
    
    # Verify schema paths configuration
    schema_config = None
    if hasattr(cfg, 'validation') and hasattr(cfg.validation, 'pydantic') and hasattr(cfg.validation.pydantic, 'schema_paths'):
        schema_config = cfg.validation.pydantic.schema_paths
        logger.info("Schema paths configuration loaded from validation.pydantic")
    
    # Get configuration parameters
    environment = cfg.get('environment', 'development')
    
    logger.info(f"📊 Processing ALL {len(snow_table_list)} tables with environment={environment}")
    logger.info("⚠️  Processing ALL rows from each table with type enforcement")
    
    # Track results
    successful_tables = []
    failed_tables = []
    dataframes = {}
    
    # Process all tables
    with SnowFetch(config=cfg, environment=environment) as fetcher:
        for i, snow_table in enumerate(tqdm(snow_table_list, desc="Processing ALL tables")):
            try:
                logger.info(f"🔄 ({i+1:2d}/{len(snow_table_list)}) Fetching ALL data from {snow_table}")
                
                # First fetch raw data for comparison
                df_raw = fetcher.fetch_data(snow_table, limit=None)
                logger.info(f"  Raw data fetched: {len(df_raw):,} rows, {len(df_raw.columns)} columns")
                
                # Fetch data with enforced type validation (NO LIMIT!)
                df = fetcher.fetch_data_validation_enforced(
                    table_name=snow_table,
                    context="raw",  # Use raw schemas
                    limit=None  # NO LIMIT! Process ALL rows
                )
                
                # Analyze dtype transformations using SnowFetch's built-in method
                transformation_report = fetcher.analyze_dtype_transformations(df_raw, df, snow_table)
                
                # Extract summary values for the successful_tables list
                transformation_summary = transformation_report['summary']
                
                # Store the DataFrame with enforced types
                dataframes[snow_table] = df
                
                logger.info(f"📊 ({i+1:2d}/{len(snow_table_list)}) {snow_table}: fetched {len(df):,} rows with enforced types")
                print(f"✅ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: {len(df):,} rows with type enforcement")
                
                # Display DataFrame with corrected dtypes
                print(f"\n📋 {snow_table} - First 5 rows:")
                print(df.head().to_string())
                
                print(f"\n📊 {snow_table} - Data types and memory usage:")
                print(df.info())
                
                successful_tables.append({
                    'table': snow_table,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                    'columns_transformed': transformation_summary['columns_transformed'],
                    'to_int64': transformation_summary['to_int64'],
                    'to_float64': transformation_summary['to_float64'],
                    'to_datetime': transformation_summary['to_datetime'],
                    'to_bool': transformation_summary['to_bool']
                })
                
            except FileNotFoundError as e:
                logger.warning(f"Schema not found for {snow_table}: {e}")
                print(f"⚠️ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: Schema not found, skipping")
                failed_tables.append({'table': snow_table, 'error': 'Schema not found'})
            except Exception as e:
                logger.error(f"❌ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: failed - {str(e)}")
                print(f"❌ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: Error - {str(e)}")
                failed_tables.append({'table': snow_table, 'error': str(e)})
                continue
    
    # Log comprehensive transformation summary
    log_dtype_transformation_summary(successful_tables, logger)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successful tables: {len(successful_tables)}")
    logger.info(f"❌ Failed tables: {len(failed_tables)}")
    
    if successful_tables:
        total_transformations = sum(t.get('columns_transformed', 0) for t in successful_tables)
        logger.info(f"Total dtype transformations: {total_transformations}")
        
        # Log tables with most transformations
        sorted_by_trans = sorted(successful_tables, 
                               key=lambda x: x.get('columns_transformed', 0), 
                               reverse=True)
        logger.info("\nTop tables by dtype transformations:")
        for table in sorted_by_trans[:5]:
            if table.get('columns_transformed', 0) > 0:
                logger.info(f"  {table['table']}: {table['columns_transformed']} columns")
    
    logger.info("="*80)
    
    print(f"\n🎉 PROCESSING COMPLETED!")
    print(f"✅ Successful: {len(successful_tables)} DataFrames with enforced types")
    print(f"❌ Failed: {len(failed_tables)}")
    
    if successful_tables:
        total_rows = sum(result['rows'] for result in successful_tables)
        total_memory = sum(result['memory_mb'] for result in successful_tables)
        print(f"📊 Total data in memory: {total_rows:,} rows, {total_memory:.2f} MB")
        
        print(f"\nDataFrames available in memory with enforced types:")
        for result in successful_tables[:5]:  # Show first 5
            print(f"  - {result['table']}: {result['rows']:,} rows, {result['columns']} columns")
        
        if len(successful_tables) > 5:
            print(f"  ... and {len(successful_tables) - 5} more DataFrames")
    
    if failed_tables:
        print(f"\nFailed tables:")
        for failed in failed_tables:
            print(f"  - {failed['table']}: {failed['error']}")
    
    print("\n" + "="*60)
    print("✨ Script execution completed!")
    print(f"📊 {len(dataframes)} DataFrames with enforced types in memory")
    print("="*60)

if __name__ == "__main__":
    main()
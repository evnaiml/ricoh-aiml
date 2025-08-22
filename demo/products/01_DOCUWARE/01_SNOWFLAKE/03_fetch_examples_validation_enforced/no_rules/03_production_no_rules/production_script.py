"""
Production-grade Snowflake data processor with comprehensive type validation and analysis.

This production script provides enterprise-ready data fetching and type enforcement
for Snowflake tables with extensive logging, error handling, and performance monitoring.
It leverages the enhanced SnowFetch class methods for robust dtype transformation
analysis and detailed reporting suitable for production environments.

✅ Production Features:
- Uses fetch_data_validation_enforced() for automatic type conversion
- Leverages SnowFetch.analyze_dtype_transformations() for detailed analysis
- Applies Pydantic schemas to enforce correct data types
- Enhanced error handling with graceful failure recovery
- Performance tracking with timing and memory metrics
- Comprehensive logging for audit and monitoring
- No file writing - only in-memory DataFrames

📊 Type Enforcement:
- String columns containing integers → Int64 (nullable)
- String columns containing floats → float64
- Date/time strings → datetime64[ns]
- Failed conversions → NaN/NaT with detailed tracking

🔍 Production Analysis:
- DtypeTransformationAnalyzer class for enterprise-grade reporting
- Aggregated statistics across all tables
- Conversion failure tracking and reporting
- Memory usage optimization with garbage collection
- Performance metrics for monitoring

Updated: 2025-08-13
- Integrated with SnowFetch's enhanced analysis methods
- Added DtypeTransformationAnalyzer for production reporting
- Enhanced logging with comprehensive transformation summaries
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
from datetime import datetime
import time
from typing import Dict, Any, List
from tqdm import tqdm
import json

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch, DtypeTransformationAnalyzer
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

# DtypeTransformationAnalyzer now imported from fetchdata.py

@hydra.main(version_base=None, config_path=conf_dir.as_posix(), config_name="config")
def main(cfg: DictConfig) -> None:
    """Production main function to process ALL Snowflake tables with enforced type validation"""
    
    start_time = time.time()
    
    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("🚀 PRODUCTION: Processing ALL Snowflake tables with enforced type validation")
    
    # Verify schema paths configuration
    schema_config = None
    if hasattr(cfg, 'validation') and hasattr(cfg.validation, 'pydantic') and hasattr(cfg.validation.pydantic, 'schema_paths'):
        schema_config = cfg.validation.pydantic.schema_paths
        logger.info("✅ Schema paths configuration loaded from validation.pydantic")
        
        if hasattr(schema_config, 'DOCUWARE'):
            logger.info(f"📁 Raw schemas path: {schema_config.DOCUWARE.search_paths.raw}")
    else:
        logger.warning("Schema paths configuration not found, will use defaults")
    
    # Get configuration parameters
    environment = cfg.get('environment', 'development')
    tables_to_process = snow_table_list  # Process ALL tables
    
    logger.info(f"📊 Processing ALL {len(tables_to_process)} tables with environment={environment}")
    logger.info("⚠️  Processing ALL rows from each table with type enforcement")
    
    # Track results
    successful_tables = []
    failed_tables = []
    total_rows_processed = 0
    dataframes = {}
    
    # Initialize transformation analyzer
    analyzer = DtypeTransformationAnalyzer(logger)
    
    # Initialize SnowFetch once for all tables
    with SnowFetch(config=cfg, environment=environment) as fetcher:
        
        for i, snow_table in enumerate(tqdm(tables_to_process, desc="Processing ALL tables")):
            try:
                table_start_time = time.time()
                
                # First fetch raw data for comparison
                logger.info(f"🔄 ({i+1:2d}/{len(tables_to_process)}) Fetching raw data from {snow_table}")
                df_raw = fetcher.fetch_data(snow_table, limit=None)
                logger.info(f"  Raw data shape: {df_raw.shape[0]:,} × {df_raw.shape[1]}")
                
                # Fetch ALL data with enforced type validation (no limits)
                logger.info(f"  Applying type enforcement from schema...")
                df = fetcher.fetch_data_validation_enforced(
                    table_name=snow_table,
                    context="raw",  # Use raw schemas
                    limit=None  # NO LIMIT - ALL ROWS
                )
                
                table_fetch_time = time.time() - table_start_time
                total_rows_processed += len(df)
                
                # Store the DataFrame with enforced types
                dataframes[snow_table] = df
                
                logger.info(f"📊 ({i+1:2d}/{len(tables_to_process)}) {snow_table}: fetched {len(df):,} rows "
                           f"in {table_fetch_time:.1f}s with enforced types")
                
                # Display DataFrame with corrected dtypes
                print(f"\n✅ ({i+1:2d}/{len(tables_to_process)}) {snow_table}: {len(df):,} rows with type enforcement")
                print(f"\n📋 {snow_table} - First 5 rows:")
                print(df.head().to_string())
                
                print(f"\n📊 {snow_table} - Data types and memory usage:")
                print(df.info())
                
                # Analyze dtype transformations using fetcher's method
                transformation_report = analyzer.analyze_table(fetcher, df_raw, df, snow_table)
                
                # Calculate memory usage
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                
                # Log sample transformed values for verification
                if transformation_report['columns_transformed'] > 0:
                    logger.info("\nSample transformed values:")
                    for trans in transformation_report['transformations'][:3]:
                        col = trans['column']
                        sample_values = df[col].dropna().head(3).tolist()
                        if sample_values:
                            logger.info(f"  {col}: {sample_values}")
                
                # Add timing and memory information
                successful_tables.append({
                    'table': snow_table,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': memory_mb,
                    'fetch_time_seconds': table_fetch_time,
                    'columns_transformed': transformation_report['columns_transformed'],
                    'conversion_failures': transformation_report['conversion_stats']['conversion_failures']
                })
                
                logger.info(f"✅ ({i+1:2d}/{len(tables_to_process)}) {snow_table}: completed successfully "
                           f"({len(df):,} rows, {memory_mb:.2f} MB, {table_fetch_time:.1f}s)")
                
                # Memory cleanup for production
                if i % 5 == 0:  # Every 5 tables, force garbage collection
                    import gc
                    gc.collect()
                
            except FileNotFoundError as e:
                logger.warning(f"Schema not found for {snow_table}: {e}")
                print(f"⚠️ ({i+1:2d}/{len(tables_to_process)}) {snow_table}: Schema not found, skipping")
                failed_tables.append({'table': snow_table, 'error': 'Schema not found'})
            except Exception as e:
                logger.error(f"❌ ({i+1:2d}/{len(tables_to_process)}) {snow_table}: failed - {str(e)}")
                failed_tables.append({'table': snow_table, 'error': str(e)})
                continue
    
    # Log comprehensive transformation summary
    analyzer.log_final_summary()
    
    # Final summary
    total_time = time.time() - start_time
    logger.info("\n" + "="*80)
    logger.info("🎉 PRODUCTION PROCESSING COMPLETED!")
    logger.info("="*80)
    logger.info(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
    logger.info(f"✅ Successful tables: {len(successful_tables)}")
    logger.info(f"❌ Failed tables: {len(failed_tables)}")
    
    if successful_tables:
        total_transformations = sum(t['columns_transformed'] for t in successful_tables)
        total_failures = sum(t.get('conversion_failures', 0) for t in successful_tables)
        logger.info(f"Total dtype transformations: {total_transformations}")
        if total_failures > 0:
            logger.warning(f"Total conversion failures: {total_failures}")
    
    logger.info("="*80)
    
    print("\n" + "="*60)
    print("🎉 PRODUCTION PROCESSING COMPLETED!")
    print("="*60)
    print(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
    print(f"✅ Successful: {len(successful_tables)} DataFrames with enforced types")
    print(f"❌ Failed: {len(failed_tables)}")
    
    if successful_tables:
        total_memory = sum(result['memory_mb'] for result in successful_tables)
        logger.info(f"📊 Total data in memory: {total_rows_processed:,} rows, {total_memory:.2f} MB")
        
        print(f"\n📊 Total data in memory: {total_rows_processed:,} rows, {total_memory:.2f} MB")
        
        # Show largest tables
        sorted_tables = sorted(successful_tables,
                              key=lambda x: x['rows'], reverse=True)
        print("\n🏆 Top 5 largest tables by row count:")
        for j, table_info in enumerate(sorted_tables[:5], 1):
            print(f"  {j}. {table_info['table']}: {table_info['rows']:,} rows, "
                  f"{table_info['memory_mb']:.2f} MB, {table_info['fetch_time_seconds']:.1f}s")
        
        # Performance statistics
        avg_fetch_time = sum(t['fetch_time_seconds'] for t in successful_tables) / len(successful_tables)
        print(f"\n⚡ Performance statistics:")
        print(f"  Average fetch time per table: {avg_fetch_time:.1f}s")
        print(f"  Total rows processed: {total_rows_processed:,}")
        print(f"  Processing speed: {total_rows_processed / total_time:.0f} rows/second")
    
    if failed_tables:
        logger.error(f"❌ Failed tables ({len(failed_tables)}):")
        print(f"\n❌ Failed tables ({len(failed_tables)}):")
        for failure in failed_tables:
            logger.error(f"  - {failure['table']}: {failure['error']}")
            print(f"  - {failure['table']}: {failure['error']}")
    
    print("\n" + "="*60)
    print("💡 DataFrames with enforced types are available in memory")
    print(f"📊 {len(dataframes)} DataFrames ready for analysis")
    print("✨ Production script execution completed!")
    print("="*60)

if __name__ == "__main__":
    main()
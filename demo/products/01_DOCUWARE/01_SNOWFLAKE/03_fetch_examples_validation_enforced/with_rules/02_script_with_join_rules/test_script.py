"""
Hydra-based Snowflake join query processor with type validation and comprehensive analysis.

This script demonstrates automated processing of complex SQL join queries using
Hydra configuration with comprehensive type enforcement and analysis. It leverages
the enhanced SnowFetch class methods for query profiling, dtype tracking, and
detailed reporting of join operations.

✅ Key Features:
- Executes SQL queries from Hydra sequel_rules configuration
- Leverages SnowFetch.log_join_query_analysis() for query profiling
- Utilizes SnowFetch.analyze_dtype_transformations() for type tracking
- Applies type enforcement based on join_rules schemas if available
- Demonstrates schema context switching (join_rules vs raw)
- Prints DataFrames with corrected dtypes to console
- No file writing - only in-memory DataFrames

📊 Type Enforcement Strategy:
- Primary: Uses join_rules schemas for joined datasets
- Fallback: Uses raw schemas when join_rules not found
- Converts data types automatically based on schema context
- Tracks and reports all dtype transformations

🔍 Analysis Capabilities:
- Query result profiling with null value analysis
- Memory usage and dtype distribution tracking
- Performance metrics for each query
- Comprehensive transformation summaries

📝 Logging:
- Detailed query execution reports
- Schema resolution tracking
- Dtype transformation analysis
- Final processing statistics

Updated: 2025-08-13
- Integrated with SnowFetch's log_join_query_analysis()
- Utilizes SnowFetch's analyze_dtype_transformations()
- Removed duplicate local functions
- Enhanced with comprehensive query analysis
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
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.utils.find_paths import ProjectRootFinder

# Set paths to directories
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"


@hydra.main(version_base=None, config_path=conf_dir.as_posix(), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to process sequel join rules with enforced type validation"""
    
    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("🚀 Processing sequel join rules with enforced type validation")
    
    # Load sequel join rules from configuration
    try:
        sequel_rules = cfg.products.DOCUWARE.db.snowflake.sequel_rules
        logger.info("✅ Successfully loaded sequel join rules configuration")
    except Exception as e:
        logger.error(f"❌ Failed to load sequel join rules: {str(e)}")
        raise
    
    # Get available queries
    available_queries = list(sequel_rules.queries.keys())
    logger.info(f"📊 Found {len(available_queries)} sequel join rules to process")
    
    # Verify schema paths configuration
    schema_config = None
    if hasattr(cfg, 'validation') and hasattr(cfg.validation, 'pydantic') and hasattr(cfg.validation.pydantic, 'schema_paths'):
        schema_config = cfg.validation.pydantic.schema_paths
        logger.info("Schema paths configuration loaded from validation.pydantic")
    
    # Get configuration parameters
    environment = cfg.get('environment', 'development')
    
    logger.info(f"📊 Processing ALL {len(available_queries)} sequel join rules with environment={environment}")
    logger.info("⚠️  Processing ALL rows from each query")
    
    # Track results
    successful_queries = []
    failed_queries = []
    query_results = {}
    
    # Initialize SnowFetch
    with SnowFetch(config=cfg, environment=environment) as fetcher:
        
        for i, query_name in enumerate(tqdm(available_queries, desc="Processing sequel join rules")):
            try:
                logger.info(f"🔄 ({i+1:2d}/{len(available_queries)}) Executing sequel join rule: {query_name}")
                
                # Get query configuration
                query_config = sequel_rules.queries[query_name]
                sql_query = query_config.sql
                description = query_config.get('description', 'No description')
                
                logger.info(f"Description: {description}")
                
                # Execute the custom query (NO LIMIT!)
                df = fetcher.fetch_custom_query(sql_query)
                
                # Log detailed execution info using SnowFetch's built-in method
                fetcher.log_join_query_analysis(query_name, sql_query, df)
                
                logger.info(f"📊 ({i+1:2d}/{len(available_queries)}) {query_name}: fetched {len(df):,} rows")
                print(f"✅ ({i+1:2d}/{len(available_queries)}) {query_name}: {len(df):,} rows fetched")
                
                # Display DataFrame
                print(f"\n📋 {query_name} - First 5 rows:")
                print(df.head().to_string())
                
                print(f"\n📊 {query_name} - Data types and memory usage:")
                print(df.info())
                
                # Store results for analysis (no CSV writing)
                query_results[query_name] = df
                
                # Display summary statistics
                print(f"\n📊 {query_name} - Summary statistics:")
                print(f"Total rows: {len(df):,}")
                print(f"Total columns: {len(df.columns)}")
                print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                successful_queries.append({
                    'query': query_name,
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                    'description': description
                })
                
                logger.info(f"✅ ({i+1:2d}/{len(available_queries)}) {query_name}: fetched successfully "
                           f"({len(df):,} rows, {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB)")
                
            except Exception as e:
                logger.error(f"❌ ({i+1:2d}/{len(available_queries)}) {query_name}: failed - {str(e)}")
                print(f"❌ ({i+1:2d}/{len(available_queries)}) {query_name}: Error - {str(e)}")
                failed_queries.append({'query': query_name, 'error': str(e)})
                continue
    
    # Demonstrate fetching a single table with join_rules context
    print("\n" + "="*60)
    print("Demonstrating table fetch with join_rules context")
    print("="*60)
    
    # Example: fetch a table that might have a schema in join_rules folder
    example_tables = ["PS_DOCUWARE_CONTRACTS"]
    
    with SnowFetch(config=cfg, environment=environment) as fetcher:
        for table_name in example_tables:
            try:
                logger.info(f"\nFetching {table_name} with join_rules schema enforcement")
                
                # First fetch raw data for comparison
                df_raw = fetcher.fetch_data(table_name, limit=1000)
                
                # Fetch with join_rules context
                df = fetcher.fetch_data_validation_enforced(
                    table_name=table_name,
                    context="join_rules",  # Explicitly use join_rules schemas
                    limit=1000  # Limit for demonstration
                )
                
                # Analyze dtype enforcement using SnowFetch's built-in method
                enforcement_report = fetcher.analyze_dtype_transformations(df_raw, df, table_name)
                
                logger.info(f"Fetched {len(df)} rows with enforced types from join_rules schema")
                print(f"✅ {table_name}: {len(df)} rows with join_rules type enforcement")
                
                # Display DataFrame with enforced types
                print(f"\n📋 {table_name} - First 5 rows with join_rules enforcement:")
                print(df.head().to_string())
                
                print(f"\n📊 {table_name} - Data types and memory usage:")
                print(df.info())
                
            except FileNotFoundError as e:
                logger.warning(f"Schema not found in join_rules for {table_name}: {e}")
                print(f"⚠️ {table_name}: No join_rules schema found, trying raw context...")
                
                # Try with raw context as fallback
                try:
                    # First fetch raw data
                    df_raw = fetcher.fetch_data(table_name, limit=1000)
                    
                    df = fetcher.fetch_data_validation_enforced(
                        table_name=table_name,
                        context="raw",
                        limit=1000
                    )
                    
                    # Analyze dtype enforcement using SnowFetch's built-in method
                    enforcement_report = fetcher.analyze_dtype_transformations(df_raw, df, table_name)
                    print(f"✅ {table_name}: {len(df)} rows with raw schema enforcement")
                    
                    # Display DataFrame with raw schema enforcement
                    print(f"\n📋 {table_name} - First 5 rows with raw schema:")
                    print(df.head().to_string())
                    
                    print(f"\n📊 {table_name} - Data types and memory usage:")
                    print(df.info())
                except Exception as e2:
                    logger.error(f"Could not fetch with any schema: {e2}")
                    
            except Exception as e:
                logger.error(f"Error processing {table_name}: {e}")
                print(f"❌ {table_name}: Error - {e}")
    
    # Log comprehensive summary
    logger.info("\n" + "="*80)
    logger.info("HYDRA SCRIPT - FINAL PROCESSING SUMMARY")
    logger.info("="*80)
    logger.info(f"Sequel join rules processed: {len(available_queries)}")
    logger.info(f"✅ Successful: {len(successful_queries)}")
    logger.info(f"❌ Failed: {len(failed_queries)}")
    
    if successful_queries:
        total_rows = sum(q['rows'] for q in successful_queries)
        total_memory = sum(q['memory_mb'] for q in successful_queries)
        
        logger.info(f"\nProcessing statistics:")
        logger.info(f"  Total rows fetched: {total_rows:,}")
        logger.info(f"  Total memory usage: {total_memory:.2f} MB")
        logger.info(f"  Average rows per query: {total_rows/len(successful_queries):.0f}")
        
        # Log queries sorted by size
        sorted_queries = sorted(successful_queries, key=lambda x: x['rows'], reverse=True)
        logger.info("\nTop queries by row count:")
        for i, query in enumerate(sorted_queries[:5], 1):
            logger.info(f"  {i}. {query['query']}: {query['rows']:,} rows")
            logger.info(f"     Description: {query['description']}")
    
    if failed_queries:
        logger.error("\nFailed queries:")
        for failed in failed_queries:
            logger.error(f"  {failed['query']}: {failed['error']}")
    
    logger.info("="*80)
    
    print(f"\n🎉 PROCESSING COMPLETED!")
    print(f"✅ Successful sequel join rules: {len(successful_queries)}")
    print(f"❌ Failed sequel join rules: {len(failed_queries)}")
    
    if successful_queries:
        total_memory = sum(result['memory_mb'] for result in successful_queries)
        total_rows = sum(result['rows'] for result in successful_queries)
        print(f"📊 Total data in memory: {total_rows:,} rows, {total_memory:.2f} MB")
        
        print(f"\n📋 DataFrames available in 'query_results' dictionary")
        
        print(f"\nSuccessful queries:")
        for result in successful_queries:
            print(f"  - {result['query']}: {result['description']}")
    
    if failed_queries:
        print(f"\nFailed queries:")
        for failed in failed_queries:
            print(f"  - {failed['query']}: {failed['error']}")
    
    print("\n" + "="*60)
    print("✨ Script execution completed!")
    print(f"📊 {len(query_results)} query results available in memory")
    print("="*60)

if __name__ == "__main__":
    main()
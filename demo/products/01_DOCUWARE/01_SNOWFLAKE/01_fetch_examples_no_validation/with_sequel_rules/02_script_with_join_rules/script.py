"""
A script to fetch data from Snowflake using Hydra with sequel join rules
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
# 🚀 Quick Start:
# -----------------------------------------------------------------------------
# * Development testing:
#       python script.py
# -----------------------------------------------------------------------------
# Execute specific query:
#       python script.py query_name=usage_latest
# -----------------------------------------------------------------------------
# * Production run:
#       python script.py environment=production disable_console_logging=true
# -----------------------------------------------------------------------------
# * Execute specific query:
#       python script.py query_name=usage_latest
# -----------------------------------------------------------------------------
# * Debug mode with verbose logging (with maximum verbosity):
#       python script.py debug=true
#       python script.py debug=true +loggers.loguru.default_level=DEBUG
# -----------------------------------------------------------------------------
# * Add row limit:
#       python script.py query_name=usage_latest row_limit=1000
# -----------------------------------------------------------------------------
# * Debug mode with maximum verbosity:
# python script.py debug=true +loggers.loguru.default_level=DEBUG
# -----------------------------------------------------------------------------
# * Check configuration:
#       python script.py --cfg job
# -----------------------------------------------------------------------------
# Key Functionality:
#
# 1. Sequel Rules Integration: Uses your complex SQL join queries instead of simple table fetching
# 2. Smart Column Inspection: Automatically analyzes data structure and shows key columns
# 3. Flexible Query Selection: Can target specific queries or run all available queries
# 4. Better Error Handling: Robust error handling with detailed logging
# 6. Results Management: Stores results in accessible variables (usage_latest, results)
# 7. Comprehensive Summaries: Detailed execution reports with row counts and success/failure rates
#
import hydra
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.utils.profiling import timer

# %%
# Set pandas display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# %%
# Set paths to directories
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"

# %%
# Define main function with decorator
@hydra.main(version_base=None, config_path=conf_dir.as_posix(), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to fetch data from Snowflake using sequel join rules"""

    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("Snowflake data fetching with sequel join rules started")

    # Get configuration parameters
    environment = cfg.get('environment', 'development')
    query_name = cfg.get('query_name', None)
    row_limit = cfg.get('row_limit', None)
    inspect_columns = cfg.get('inspect_columns', True)

    # Load sequel join rules from configuration
    try:
        sequel_rules = cfg.products.DOCUWARE.db.snowflake.sequel_rules
        logger.info("✅ Successfully loaded sequel join rules configuration")
    except Exception as e:
        logger.error(f"❌ Failed to load sequel join rules: {str(e)}")
        return

    # Display available queries
    logger.info("Available sequel join rules:")
    available_queries = list(sequel_rules.queries.keys())
    for query_name_item in available_queries:
        description = sequel_rules.queries[query_name_item].get('description', 'No description')
        logger.info(f"  - {query_name_item}: {description}")

    # Determine which queries to execute
    if query_name:
        if query_name in available_queries:
            queries_to_process = [query_name]
            logger.info(f"Processing specific query: {query_name}")
        else:
            logger.error(f"Query '{query_name}' not found. Available queries: {available_queries}")
            return
    else:
        queries_to_process = available_queries
        logger.info(f"Processing all {len(queries_to_process)} available queries")

    # Process queries
    results = {}
    total_rows = 0

    with timer():
        with SnowFetch(config=cfg, environment=environment) as fetcher:
            for i, current_query_name in tqdm(enumerate(queries_to_process, 1),
                                            desc="Executing queries",
                                            total=len(queries_to_process)):
                try:
                    logger.info(f"Executing query {i}/{len(queries_to_process)}: {current_query_name}")

                    # Get query configuration
                    query_config = sequel_rules.queries[current_query_name]
                    sql_query = query_config.sql

                    # Apply row limit if specified
                    if row_limit:
                        limited_query = f"SELECT * FROM ({sql_query}) LIMIT {row_limit}"
                        logger.info(f"Applying row limit: {row_limit}")
                        df = fetcher.session.sql(limited_query).to_pandas()
                    else:
                        df = fetcher.session.sql(sql_query).to_pandas()

                    # Store results
                    results[current_query_name] = df
                    total_rows += len(df)

                    logger.info(f"✅ {current_query_name}: Created dataframe with {len(df)} rows")

                    # Detailed inspection for first query or if specifically requested
                    if (i == 1 and inspect_columns) or len(queries_to_process) == 1:
                        logger.info(f"📋 Column inspection for {current_query_name}:")
                        logger.info(f"  Shape: {df.shape}")
                        logger.info(f"  Columns ({len(df.columns)}): {list(df.columns)}")

                        # Show key columns if they exist
                        key_cols = ['CUST_ACCOUNT_NUMBER', 'CUST_PARTY_NAME', 'CUSTOMER_NAME',
                                   'CONTRACT_NUMBER', 'CHURNED_FLAG', 'MATCH_RANK']
                        available_key_cols = [col for col in key_cols if col in df.columns]

                        if available_key_cols:
                            logger.info(f"  Key columns present: {available_key_cols}")

                        # Show sample data
                        if len(df) > 0:
                            logger.info(f"📊 Sample data preview:")
                            print(df.head(3).to_string())

                        # Show data types
                        logger.info(f"📈 Data types:")
                        for col, dtype in df.dtypes.items():
                            logger.info(f"  {col}: {dtype}")

                except Exception as e:
                    logger.error(f"❌ Failed to execute {current_query_name}: {str(e)}")
                    results[current_query_name] = None
                    continue

    # Final summary
    successful_queries = [name for name, df in results.items() if df is not None]
    failed_queries = [name for name, df in results.items() if df is None]

    logger.info("📊 Execution Summary:")
    logger.info(f"  Total queries attempted: {len(queries_to_process)}")
    logger.info(f"  Successful: {len(successful_queries)}")
    logger.info(f"  Failed: {len(failed_queries)}")
    logger.info(f"  Total rows fetched: {total_rows:,}")

    if successful_queries:
        logger.info(f"  ✅ Successful queries: {successful_queries}")

        # Show detailed results for each successful query
        for query_name in successful_queries:
            df = results[query_name]
            logger.info(f"    - {query_name}: {len(df):,} rows, {len(df.columns)} columns")

    if failed_queries:
        logger.error(f"  ❌ Failed queries: {failed_queries}")

    # Store results in global namespace for interactive use
    if len(successful_queries) == 1:
        query_name = successful_queries[0]
        globals()[query_name] = results[query_name]
        logger.info(f"💾 Result stored in variable: {query_name}")
    elif successful_queries:
        globals()['results'] = results
        logger.info(f"💾 All results stored in variable: results")

    logger.info("Snowflake data fetching with sequel join rules completed")


if __name__ == "__main__":
    main()
# %%
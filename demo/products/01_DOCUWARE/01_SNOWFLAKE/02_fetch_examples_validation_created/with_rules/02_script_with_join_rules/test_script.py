"""
Enhanced Snowflake data fetcher with sequel join rules, CSV export, Pydantic validation, and reporting
- Uses DataExporter directly for consistency with manual script
- Supports both individual query execution and batch processing
- Command-line interface with flexible parameter support
"""
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
# 🔬 DEVELOPMENT SCRIPT USAGE GUIDE
# -----------------------------------------------------------------------------
# This script provides command-line execution of sequel join rules with flexible
# parameter support, perfect for development, testing, and automated workflows.
# Features @hydra.main decorator for advanced configuration management.
#
# 📊 OUTPUT LOCATION: data/products/DOCUWARE/DB/snowflake/csv/join_rules/
# 📁 FILES CREATED: {query_name}.csv, {query_name}_schema.json, {query_name}_report.csv
# ⚙️ FEATURES: Parameter overrides, environment switching, selective processing
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 BASIC EXECUTION OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 DEFAULT EXECUTION (Process all sequel join rules):
# python test_script.py
#
# 🔸 SPECIFIC QUERY EXECUTION:
# python test_script.py query_name=usage_latest
#
# 🔸 ENVIRONMENT SWITCHING:
# python test_script.py environment=development               # Default
# python test_script.py environment=production                # Production settings
#
# 🔸 ROW LIMITING FOR TESTING:
# python test_script.py row_limit=1000                        # Limit to 1000 rows
# python test_script.py row_limit=10000                       # Limit to 10000 rows
# python test_script.py query_name=usage_latest row_limit=500 # Specific query + limit
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 DEVELOPMENT & TESTING OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 QUICK TESTING WORKFLOWS:
# python test_script.py query_name=usage_latest row_limit=100  # Fast test with 100 rows
# python test_script.py row_limit=1000 inspect_columns=true   # Inspect first 1000 rows
# python test_script.py export_results=false                  # Process without export
# python test_script.py query_name=usage_latest export_results=false  # Quick analysis only
#
# 🔹 DEVELOPMENT DEBUGGING:
# python test_script.py debug=true                           # Enable debug mode
# python test_script.py debug=true query_name=usage_latest   # Debug specific query
# python test_script.py debug=true row_limit=10              # Debug with minimal data
#
# 🔹 COLUMN INSPECTION CONTROL:
# python test_script.py inspect_columns=true                 # Enable detailed column analysis
# python test_script.py inspect_columns=false                # Disable column inspection
# python test_script.py query_name=usage_latest inspect_columns=true  # Inspect specific query
#
# 🔹 EXPORT CONTROL:
# python test_script.py export_results=true                  # Enable CSV export (default)
# python test_script.py export_results=false                 # Disable CSV export
# python test_script.py export_results=false inspect_columns=true  # Analysis only
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📝 LOGGING AND OUTPUT CONTROL
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 STANDARD LOGGING (Development):
# python test_script.py                                      # Full console output
# python test_script.py disable_console_logging=false        # Explicit console logging
# python test_script.py environment=development              # Verbose development logging
#
# 🔹 DEBUG LOGGING (Maximum Verbosity):
# python test_script.py debug=true                           # Debug mode
# python test_script.py debug=true disable_console_logging=false  # Debug with console
# python test_script.py +loggers.loguru.default_level=DEBUG  # Override log level
# python test_script.py debug=true query_name=usage_latest   # Debug specific query
#
# 🔹 MINIMAL LOGGING (Testing):
# python test_script.py disable_console_logging=true         # File logging only
# python test_script.py +loggers.loguru.default_level=ERROR  # Errors only
# python test_script.py +loggers.loguru.default_level=WARNING # Warnings and errors
#
# 🔹 CUSTOM LOG LEVELS:
# python test_script.py +loggers.loguru.default_level=INFO   # Standard info logging
# python test_script.py +loggers.loguru.default_level=CRITICAL  # Critical only
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 COMBINED PARAMETER EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 DEVELOPMENT TESTING COMBINATIONS:
# python test_script.py query_name=usage_latest row_limit=100 debug=true
# python test_script.py environment=development row_limit=1000 inspect_columns=true
# python test_script.py export_results=false inspect_columns=true debug=true
# python test_script.py query_name=usage_latest environment=development export_results=false
#
# 🔸 PRODUCTION-LIKE TESTING:
# python test_script.py environment=production row_limit=10000
# python test_script.py environment=production disable_console_logging=true
# python test_script.py environment=production query_name=usage_latest
#
# 🔸 PERFORMANCE TESTING:
# python test_script.py row_limit=50000 inspect_columns=false  # Large dataset test
# python test_script.py query_name=usage_latest               # Full dataset test
# python test_script.py environment=production               # Production performance
#
# 🔸 ANALYSIS WORKFLOWS:
# python test_script.py export_results=false inspect_columns=true debug=true  # Analysis only
# python test_script.py query_name=usage_latest inspect_columns=true         # Specific analysis
# python test_script.py row_limit=1000 inspect_columns=true export_results=true  # Sample + export
#
# ═══════════════════════════════════════════════════════════════════════════════
# ⚙️ ADVANCED HYDRA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 CONFIGURATION INSPECTION:
# python test_script.py --cfg job                            # Show job configuration
# python test_script.py --cfg hydra                          # Show Hydra configuration
# python test_script.py --help                               # Show available parameters
# python test_script.py --config-path=./conf --config-name=config  # Explicit paths
#
# 🔹 CONFIGURATION OVERRIDES:
# python test_script.py +processing.max_workers=4            # Override processing workers
# python test_script.py +data.chunk_size=15000               # Override chunk size
# python test_script.py +processing.memory_limit_gb=8        # Override memory limit
# python test_script.py +csv_processing.sequel_rules_config=custom_config  # Custom CSV config
#
# 🔹 NESTED PARAMETER OVERRIDES:
# python test_script.py +products.DOCUWARE.db.snowflake.csv_config=csv_rules_config  # CSV config
# python test_script.py products.DOCUWARE.db.snowflake.session.warehouse=TEST_WH     # Warehouse override
# python test_script.py +app.version=1.1.0                  # Application version
#
# 🔹 MULTIPLE OVERRIDES:
# python test_script.py query_name=usage_latest row_limit=1000 debug=true +data.chunk_size=5000
# python test_script.py environment=production +processing.max_workers=8 disable_console_logging=true
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 TESTING AND VALIDATION WORKFLOWS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 UNIT TESTING WORKFLOW:
# python test_script.py query_name=usage_latest row_limit=10 debug=true  # Minimal test
# python test_script.py export_results=false inspect_columns=true       # Schema validation
# python test_script.py row_limit=100 debug=true                        # Small dataset test
#
# 🔹 INTEGRATION TESTING:
# python test_script.py row_limit=1000                                   # Medium dataset
# python test_script.py environment=production row_limit=10000           # Production-like
# python test_script.py query_name=usage_latest                          # Full query test
#
# 🔹 PERFORMANCE TESTING:
# python test_script.py row_limit=100000 inspect_columns=false           # Large dataset
# python test_script.py environment=production                           # Full production test
# time python test_script.py query_name=usage_latest                     # Timing test
#
# 🔹 ERROR TESTING:
# python test_script.py query_name=nonexistent_query                     # Invalid query test
# python test_script.py row_limit=-1                                     # Invalid parameter test
# python test_script.py environment=invalid_env                          # Invalid environment
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📊 OUTPUT AND RESULTS MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 RESULT VARIABLES (Available after execution):
# # When running with single query:
# usage_latest                                                # Main dataset variable
#
# # When running multiple queries:
# query_results                                               # Dictionary of all results
# query_results['usage_latest']                               # Access specific result
# list(query_results.keys())                                  # List available queries
#
# 🔹 OUTPUT FILE VERIFICATION:
# # Check output directory for generated files:
# ls -la data/products/DOCUWARE/DB/snowflake/csv/join_rules/
# wc -l data/products/DOCUWARE/DB/snowflake/csv/join_rules/usage_latest.csv  # Row count
# du -sh data/products/DOCUWARE/DB/snowflake/csv/join_rules/  # Directory size
#
# 🔹 LOG FILE MONITORING:
# tail -f logs/loguru/info.log                               # Monitor processing
# tail -f logs/loguru/error.log                              # Monitor errors
# grep "✅" logs/loguru/info.log                              # Find successful operations
# grep "❌" logs/loguru/info.log                              # Find failed operations
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 MONITORING AND DEBUGGING
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 REAL-TIME MONITORING:
# # In another terminal, monitor progress:
# tail -f logs/loguru/info.log | grep "Processing\|✅\|❌"    # Monitor key events
# watch "ls -la data/products/DOCUWARE/DB/snowflake/csv/join_rules/"  # Watch output files
# ps aux | grep test_script                                  # Check if running
#
# 🔹 PERFORMANCE MONITORING:
# # Monitor system resources during execution:
# top -p $(pgrep -f test_script)                             # Monitor CPU/memory
# iostat -x 1                                                # Monitor disk I/O
# nvidia-smi                                                 # Monitor GPU usage (if applicable)
#
# 🔹 DEBUG OUTPUT ANALYSIS:
# # When using debug=true, analyze debug logs:
# grep "DEBUG" logs/loguru/debug.log                         # All debug messages
# grep "SQL" logs/loguru/debug.log                           # SQL-related debug info
# grep "fetch" logs/loguru/debug.log                         # Data fetching debug info
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🎓 DEVELOPMENT BEST PRACTICES
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ RECOMMENDED DEVELOPMENT WORKFLOW:
# 1. Start with small datasets: row_limit=100
# 2. Enable debugging: debug=true
# 3. Disable export initially: export_results=false
# 4. Test specific queries: query_name=usage_latest
# 5. Gradually increase data size: row_limit=1000, 10000
# 6. Enable export when satisfied: export_results=true
# 7. Test full datasets without limits
# 8. Test production environment: environment=production
#
# ⚡ PERFORMANCE OPTIMIZATION TIPS:
# - Use row_limit for faster iteration during development
# - Disable inspect_columns for large datasets
# - Use export_results=false for analysis-only runs
# - Monitor memory usage with large datasets
# - Use production environment for final testing
#
# 🔧 DEBUGGING STRATEGIES:
# - Always start with debug=true for new queries
# - Use minimal row limits for initial testing
# - Check logs/loguru/error.log for detailed error info
# - Validate configuration with --cfg job
# - Test parameter combinations incrementally
#
# 🔒 SECURITY CONSIDERATIONS:
# - Credentials are managed through Hydra configuration
# - No sensitive data in command-line parameters
# - Row limits help prevent accidental large data exports
# - Logging can be disabled for sensitive environments
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📋 PARAMETER REFERENCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 CORE PARAMETERS:
# query_name=<string>                    # Specific query to execute
# environment=<development|production>   # Execution environment
# row_limit=<integer>                    # Limit rows processed
# debug=<true|false>                     # Enable debug mode
#
# 🔸 FEATURE CONTROLS:
# export_results=<true|false>            # Enable/disable CSV export
# inspect_columns=<true|false>           # Enable/disable column inspection
# disable_console_logging=<true|false>   # Control console output
#
# 🔸 ADVANCED OVERRIDES:
# +loggers.loguru.default_level=<LEVEL>  # Custom log level
# +processing.max_workers=<integer>      # Processing parallelism
# +data.chunk_size=<integer>             # Data processing chunk size
#
# ═══════════════════════════════════════════════════════════════════════════════

import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any, List, Optional
from tqdm import tqdm

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.data.validation.data_exporter import DataExporter
from churn_aiml.utils.profiling import timer

# Set pandas display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Set paths to directories
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"

@hydra.main(version_base=None, config_path=conf_dir.as_posix(), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to fetch data from Snowflake using sequel join rules and export to CSV"""

    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("Enhanced Snowflake data fetching with sequel join rules and CSV export started")

    # Load sequel join rules from configuration
    try:
        sequel_rules = cfg.products.DOCUWARE.db.snowflake.sequel_rules
        logger.info("✅ Successfully loaded sequel join rules configuration")
    except Exception as e:
        logger.error(f"❌ Failed to load sequel join rules: {str(e)}")
        return

    # Initialize output directory for sequel join rules using csv_rules_config
    project_root = ProjectRootFinder().find_path()

    try:
        # For sequel join rules, use csv_rules_config instead of csv_raw_config
        from hydra import compose
        rules_cfg = compose(config_name="config", overrides=["products/DOCUWARE/db/snowflake/csv_config=csv_rules_config"])
        csv_config = rules_cfg.products.DOCUWARE.db.snowflake.csv_config.csv_output
        relative_path = csv_config.get('base_dir', 'data/products/DOCUWARE/DB/snowflake/csv/join_rules')
        logger.info("✅ Using csv_rules_config for sequel join rules processing")
    except Exception as e:
        # Fallback if csv_rules_config is not available
        logger.warning(f"csv_rules_config not found: {str(e)}")
        logger.info("🔄 Using fallback configuration for sequel join rules")
        relative_path = 'data/products/DOCUWARE/DB/snowflake/csv/join_rules'

    output_dir = project_root / relative_path
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")

    # Initialize DataExporter directly
    exporter = DataExporter(output_dir, logger)

    # Get configuration parameters
    environment = cfg.get('environment', 'development')
    query_name = cfg.get('query_name', None)
    row_limit = cfg.get('row_limit', None)
    inspect_columns = cfg.get('inspect_columns', True)
    export_results = cfg.get('export_results', True)

    # Display available sequel join rules
    available_queries = list(sequel_rules.queries.keys())
    logger.info("Available sequel join rules:")
    for query_name_item in available_queries:
        description = sequel_rules.queries[query_name_item].get('description', 'No description')
        logger.info(f"  - {query_name_item}: {description}")

    # Determine which queries to execute
    if query_name:
        if query_name in available_queries:
            queries_to_process = [query_name]
            logger.info(f"Processing specific sequel join rule: {query_name}")
        else:
            logger.error(f"Query '{query_name}' not found. Available queries: {available_queries}")
            return
    else:
        queries_to_process = available_queries
        logger.info(f"Processing ALL {len(queries_to_process)} sequel join rules with environment={environment}")

    if row_limit:
        logger.info(f"Row limit per query: {row_limit:,}")
    else:
        logger.info("Processing ALL rows from each sequel join rule (no row limit)")

    # Track results
    successful_exports = []
    failed_exports = []
    query_results = {}
    total_rows_processed = 0

    # Process sequel join rules
    with timer():
        with SnowFetch(config=cfg, environment=environment) as fetcher:
            for i, current_query_name in tqdm(enumerate(queries_to_process, 1),
                                            desc="Processing sequel join rules",
                                            total=len(queries_to_process)):
                try:
                    logger.info(f"🔄 ({i}/{len(queries_to_process)}) Executing sequel join rule: {current_query_name}")

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

                    # Store results for further analysis
                    query_results[current_query_name] = df
                    total_rows_processed += len(df)

                    logger.info(f"📊 ({i}/{len(queries_to_process)}) {current_query_name}: fetched {len(df):,} rows")

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
                        if len(df) > 0 and len(queries_to_process) == 1:
                            logger.info(f"📊 Sample data preview:")
                            print(df.head(3).to_string())

                        # Show data types for single query
                        if len(queries_to_process) == 1:
                            logger.info(f"📈 Data types:")
                            for col, dtype in df.dtypes.items():
                                logger.info(f"  {col}: {dtype}")

                    # Export to CSV with schema and report if enabled
                    if export_results:
                        logger.info(f"💾 Exporting {current_query_name} ({len(df):,} rows, {len(df.columns)} columns)")
                        export_result = exporter.export_with_analysis(df, current_query_name)
                        successful_exports.append({
                            'query': current_query_name,
                            'result': export_result
                        })

                        logger.info(f"✅ ({i}/{len(queries_to_process)}) {current_query_name}: exported successfully "
                                   f"({export_result['rows']:,} rows, {export_result['size_mb']:.2f} MB)")
                    else:
                        logger.info(f"✅ ({i}/{len(queries_to_process)}) {current_query_name}: processed successfully "
                                   f"({len(df):,} rows) - export skipped")

                except Exception as e:
                    logger.error(f"❌ ({i}/{len(queries_to_process)}) {current_query_name}: failed - {str(e)}")
                    failed_exports.append({'query': current_query_name, 'error': str(e)})
                    continue

    # Final summary
    logger.info("📊 Sequel Join Rules Processing Summary:")
    logger.info(f"  Total queries attempted: {len(queries_to_process)}")
    logger.info(f"  Successful: {len(query_results)}")
    logger.info(f"  Failed: {len(failed_exports)}")
    logger.info(f"  Total rows processed: {total_rows_processed:,}")

    if export_results and successful_exports:
        total_size = sum(result['result']['size_mb'] for result in successful_exports)
        logger.info(f"  Total data exported: {total_size:.2f} MB")
        logger.info(f"  Files saved to: {output_dir}")

    # Show successful queries
    if query_results:
        logger.info(f"✅ Successful sequel join rules:")
        for query_name, df in query_results.items():
            logger.info(f"    - {query_name}: {len(df):,} rows, {len(df.columns)} columns")

    # Show failed queries
    if failed_exports:
        logger.error(f"❌ Failed sequel join rules:")
        for failure in failed_exports:
            logger.error(f"    - {failure['query']}: {failure['error']}")

    # Store results in global namespace for interactive use
    if len(query_results) == 1:
        query_name = list(query_results.keys())[0]
        globals()[query_name] = query_results[query_name]
        logger.info(f"💾 Result stored in variable: {query_name}")
    elif query_results:
        globals()['query_results'] = query_results
        logger.info(f"💾 All results stored in variable: query_results")

        # Store main result for convenience
        if 'usage_latest' in query_results:
            globals()['usage_latest'] = query_results['usage_latest']
            logger.info(f"💾 Main result stored in variable: usage_latest ({len(query_results['usage_latest']):,} rows)")

    # Quick analysis if enabled
    if query_results and len(queries_to_process) == 1:
        query_name = list(query_results.keys())[0]
        df = query_results[query_name]

        logger.info(f"📊 Quick Analysis for {query_name}:")
        if 'CHURNED_FLAG' in df.columns:
            churn_counts = df['CHURNED_FLAG'].value_counts()
            logger.info(f"  Churn distribution: {dict(churn_counts)}")

        if 'CUST_ACCOUNT_NUMBER' in df.columns:
            unique_customers = df['CUST_ACCOUNT_NUMBER'].nunique()
            logger.info(f"  Unique customers: {unique_customers:,}")

        if 'MATCH_RANK' in df.columns:
            match_rank_dist = df['MATCH_RANK'].value_counts().sort_index()
            logger.info(f"  Match rank distribution: {dict(match_rank_dist)}")

    logger.info("Snowflake data fetching with sequel join rules completed")


if __name__ == "__main__":
    main()
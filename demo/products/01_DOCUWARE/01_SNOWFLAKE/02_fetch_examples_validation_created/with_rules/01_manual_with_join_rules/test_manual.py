# %%
"""
Manual Snowflake data fetcher with sequel join rules, CSV export, Pydantic validation, and reporting
- Processes ALL sequel join rules with ALL rows automatically
- Uses DataExporter class for clean separation of concerns
- Interactive execution with detailed data analysis and inspection
"""
# %%
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-13
# * CREATED ON: 2025-08-04
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# 🔬 MANUAL INTERACTIVE SCRIPT USAGE GUIDE
# -----------------------------------------------------------------------------
# This script provides interactive, step-by-step execution of sequel join rules
# with comprehensive data analysis, column inspection, and development features.
# Perfect for data exploration, debugging, and development workflows.
#
# 📊 OUTPUT LOCATION: data/products/DOCUWARE/DB/snowflake/csv/join_rules/
# 📁 FILES CREATED: {query_name}.csv, {query_name}_schema.json, {query_name}_report.csv
# 🔍 FEATURES: Full pandas display, column inspection, data quality analysis
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 EXECUTION METHODS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 JUPYTER NOTEBOOK / JUPYTER LAB:
# # Open in Jupyter and run cells interactively
# jupyter notebook test_manual.py
# jupyter lab test_manual.py
#
# 🔸 IPYTHON INTERACTIVE SESSION:
# ipython
# %run test_manual.py
#
# 🔸 PYTHON INTERACTIVE MODE:
# python -i test_manual.py                                    # Load and keep interactive
#
# 🔸 VSCODE / PYCHARM WITH CELL MODE:
# # Use # %% cell separators to run sections individually
# # Execute cells one by one for step-by-step analysis
#
# 🔸 STANDARD PYTHON EXECUTION:
# python test_manual.py                                       # Run all cells sequentially
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📊 DATA ANALYSIS FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 PANDAS DISPLAY OPTIMIZATION:
# # Automatically configured for better data viewing:
# pd.set_option('display.max_columns', None)                  # Show all columns
# pd.set_option('display.width', None)                        # No width limit
# pd.set_option('display.max_colwidth', 50)                   # Readable column width
#
# 🔹 COMPREHENSIVE COLUMN INSPECTION:
# # For each query, the script shows:
# - DataFrame shape (rows × columns)
# - Complete column list with data types
# - Key business columns (CUST_ACCOUNT_NUMBER, CHURNED_FLAG, etc.)
# - Sample data preview with proper formatting
# - Data quality metrics and statistics
#
# 🔹 INTERACTIVE DATA EXPLORATION:
# # After execution, access results via:
# usage_latest                                                # Main dataset variable
# query_results['usage_latest']                               # From results dictionary
# query_results.keys()                                        # List all available queries
#
# 🔹 CHURN ANALYSIS READY:
# # Automatic analysis includes:
# - Churn flag distribution (churned vs non-churned)
# - Unique customer counts
# - Match rank distribution for join quality
# - Customer name matching quality assessment
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🔧 DEVELOPMENT & DEBUGGING FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 STEP-BY-STEP EXECUTION:
# # Each cell (# %%) can be run independently:
# # %% Cell 1: Configuration loading and setup
# # %% Cell 2: Logger initialization
# # %% Cell 3: Sequel rules loading and inspection
# # %% Cell 4: Data processing and analysis
# # %% Cell 5: Results summary and cleanup
#
# 🔹 DETAILED LOGGING AND OUTPUT:
# # Comprehensive logging for each step:
# - Configuration loading status
# - Sequel rules discovery and description
# - SQL query execution timing
# - Data fetching progress with row counts
# - Export operations with file sizes
# - Error handling with detailed diagnostics
#
# 🔹 MEMORY AND PERFORMANCE MONITORING:
# # Built-in performance tracking:
# - Execution timing for each operation
# - Memory usage monitoring
# - Data processing speed (rows/second)
# - File size and export performance
#
# 🔹 INTERACTIVE DEBUGGING:
# # Easy debugging workflow:
# - Variables remain accessible after execution
# - Full access to intermediate results
# - Ability to re-run specific cells
# - Interactive data exploration and modification
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📋 CONFIGURATION OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 ENVIRONMENT CONFIGURATION:
# # Modify these variables in the script for different behavior:
# environment = "development"                                  # or "production"
# inspect_columns = True                                       # Enable detailed column analysis
# export_results = True                                        # Enable CSV export
# show_sample_data = True                                      # Show data samples
#
# 🔸 PANDAS DISPLAY CUSTOMIZATION:
# # Adjust these for different display needs:
# pd.set_option('display.max_rows', 100)                       # Limit row display
# pd.set_option('display.max_columns', 20)                     # Limit column display
# pd.set_option('display.precision', 3)                        # Decimal precision
#
# 🔸 PROCESSING CUSTOMIZATION:
# # Available in script variables:
# process_all_queries = True                                   # Process all vs specific query
# enable_quality_analysis = True                               # Enable data quality checks
# save_intermediate_results = True                             # Save partial results
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🎓 LEARNING AND EXPLORATION WORKFLOWS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 DATA EXPLORATION WORKFLOW:
# 1. Run configuration cells to load sequel rules
# 2. Inspect available queries and their descriptions
# 3. Execute data fetching for specific queries
# 4. Analyze column structure and data quality
# 5. Explore sample data and relationships
# 6. Export results for further analysis
#
# 🔹 DEVELOPMENT WORKFLOW:
# 1. Test configuration loading and validation
# 2. Debug SQL query execution
# 3. Validate data transformation results
# 4. Test export functionality
# 5. Verify output file generation
# 6. Analyze performance and optimize
#
# 🔹 QUERY DEVELOPMENT WORKFLOW:
# 1. Load and inspect existing sequel rules
# 2. Modify SQL queries for testing
# 3. Execute modified queries interactively
# 4. Validate results and data quality
# 5. Export test results for validation
# 6. Commit working queries to configuration
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🔍 DATA INSPECTION COMMANDS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 AFTER SCRIPT EXECUTION, USE THESE COMMANDS:
#
# # Basic data inspection:
# usage_latest.head()                                          # First 5 rows
# usage_latest.tail()                                          # Last 5 rows
# usage_latest.info()                                          # Column info and types
# usage_latest.describe()                                      # Statistical summary
# usage_latest.shape                                           # (rows, columns)
#
# # Column analysis:
# usage_latest.columns.tolist()                               # List all columns
# usage_latest.dtypes                                          # Data types
# usage_latest.isnull().sum()                                 # Missing values per column
# usage_latest.nunique()                                       # Unique values per column
#
# # Business analysis:
# usage_latest['CHURNED_FLAG'].value_counts()                 # Churn distribution
# usage_latest['CUST_ACCOUNT_NUMBER'].nunique()               # Unique customers
# usage_latest['MATCH_RANK'].value_counts()                   # Join quality
# usage_latest.groupby('CHURNED_FLAG').size()                 # Group by churn status
#
# # Advanced analysis:
# usage_latest.corr()                                          # Correlation matrix
# usage_latest.sample(10)                                      # Random sample
# usage_latest[usage_latest['CHURNED_FLAG'] == 'Y']           # Filter churned customers
# usage_latest.groupby('CUSTOMER_NAME').agg({'CONTRACT_NUMBER': 'count'})  # Aggregation
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🧪 TESTING AND VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 DATA QUALITY VALIDATION:
# # Built-in validation checks:
# - Row count validation (minimum expected rows)
# - Column completeness checks
# - Data type consistency validation
# - Join integrity verification
# - Duplicate detection and reporting
#
# 🔹 EXPORT VALIDATION:
# # Verify exports completed successfully:
# import os
# output_dir = "data/products/DOCUWARE/DB/snowflake/csv/join_rules"
# os.listdir(output_dir)                                       # List output files
# os.path.getsize(f"{output_dir}/usage_latest.csv")           # Check file size
#
# 🔹 COMPARISON TESTING:
# # Compare with existing data:
# # Load previous exports for comparison
# # Validate data consistency across runs
# # Check for unexpected changes in row counts
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📊 EXPECTED INTERACTIVE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 CONSOLE OUTPUT INCLUDES:
# ✅ Configuration loading status
# 📋 Available sequel join rules with descriptions
# 🔄 Data fetching progress with timing
# 📊 Column inspection with data types
# 🎯 Key business metrics (churn rates, customer counts)
# 💾 Export operations with file paths and sizes
# 📈 Performance metrics and processing speed
#
# 🔹 VARIABLES AVAILABLE AFTER EXECUTION:
# - usage_latest: Main dataset (pandas DataFrame)
# - query_results: Dictionary of all query results
# - exporter: DataExporter instance for additional exports
# - logger: Logger instance for additional logging
# - cfg: Hydra configuration object
#
# 🔹 FILES CREATED:
# data/products/DOCUWARE/DB/snowflake/csv/join_rules/
# ├── usage_latest.csv                                        # Main dataset
# ├── usage_latest_schema.json                                # Pydantic schema
# └── usage_latest_report.csv                                 # Data quality report
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 BEST PRACTICES FOR MANUAL EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ RECOMMENDED WORKFLOW:
# 1. Use Jupyter/IPython for best interactive experience
# 2. Run cells one by one to understand each step
# 3. Inspect data at each stage before proceeding
# 4. Save interesting findings and insights
# 5. Export results when satisfied with data quality
# 6. Document any issues or modifications needed
#
# ⚡ PERFORMANCE TIPS:
# - Use head() and sample() for large datasets preview
# - Monitor memory usage with large query results
# - Save intermediate results if processing takes long
# - Use column selection to reduce memory usage
#
# 🔧 CUSTOMIZATION TIPS:
# - Modify pandas display options for your preferences
# - Add custom analysis cells for specific business questions
# - Create additional data quality checks as needed
# - Export custom subsets of data for specific analysis
#
# ═══════════════════════════════════════════════════════════════════════════════
# %%
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.data.validation.data_exporter import DataExporter
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.utils.profiling import timer
# %%
# Set pandas display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
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
logger.info("Manual Snowflake data fetching with sequel join rules and CSV export started")
# %%
# Load sequel join rules from configuration
try:
    sequel_rules = cfg.products.DOCUWARE.db.snowflake.sequel_rules
    logger.info("✅ Successfully loaded sequel join rules configuration")
except Exception as e:
    logger.error(f"❌ Failed to load sequel join rules: {str(e)}")
    raise

# Display available sequel join rules
available_queries = list(sequel_rules.queries.keys())
print("\n" + "="*60)
print("Available Sequel Join Rules:")
print("="*60)
for query_name in available_queries:
    description = sequel_rules.queries[query_name].get('description', 'No description')
    print(f"  - {query_name}: {description}")
print(f"\n📊 Total sequel join rules to process: {len(available_queries)}")
# %%
# Initialize CSV output directory using project root finder and csv_rules_config
try:
    # Load csv_rules_config for sequel join rules processing
    from hydra import compose
    rules_cfg = compose(config_name="config", overrides=["products/DOCUWARE/db/snowflake/csv_config=csv_rules_config"])
    csv_config = rules_cfg.products.DOCUWARE.db.snowflake.csv_config.csv_output
    relative_path = csv_config.get('base_dir', 'data/products/DOCUWARE/DB/snowflake/csv/join_rules')
    logger.info("✅ Using csv_rules_config for sequel join rules processing")
except Exception as e:
    # Fallback to default path if csv_rules_config is not available
    logger.warning(f"csv_rules_config not found: {str(e)}")
    logger.info("🔄 Using fallback configuration for sequel join rules")
    relative_path = 'data/products/DOCUWARE/DB/snowflake/csv/join_rules'

output_dir = project_root / relative_path
output_dir.mkdir(parents=True, exist_ok=True)
print(f"CSV output directory (sequel join rules): {output_dir}")
# %%
# Initialize DataExporter
exporter = DataExporter(output_dir, logger)
# %%
# Main Processing: Process ALL sequel join rules with ALL rows
print("\n" + "="*60)
print("Main Processing: ALL sequel join rules with ALL rows")
print("="*60)

print(f"📊 Total sequel join rules to process: {len(available_queries)}")
print("🚀 Processing ALL rows from ALL sequel join rules automatically...")

successful_exports = []
failed_exports = []
query_results = {}  # Store results for further analysis

with timer():
    for i, query_name in tqdm(enumerate(available_queries), desc="Processing ALL sequel join rules"):
        try:
            # Get query configuration
            query_config = sequel_rules.queries[query_name]
            sql_query = query_config.sql

            # Execute SQL query to fetch ALL data (no limits)
            with SnowFetch(config=cfg, environment="development") as fetcher:
                logger.info(f"🔄 ({i+1:2d}/{len(available_queries)}) Executing sequel join rule: {query_name}")
                df = fetcher.session.sql(sql_query).to_pandas()
                logger.info(f"📊 ({i+1:2d}/{len(available_queries)}) {query_name}: fetched {len(df):,} rows")

            # Store result for further analysis
            query_results[query_name] = df

            # Column inspection for first query
            if i == 0:
                logger.info(f"📋 Column inspection for {query_name}:")
                logger.info(f"  Shape: {df.shape}")
                logger.info(f"  Columns ({len(df.columns)}): {list(df.columns)}")

                # Show key columns if they exist
                key_cols = ['CUST_ACCOUNT_NUMBER', 'CUST_PARTY_NAME', 'CUSTOMER_NAME',
                           'CONTRACT_NUMBER', 'CHURNED_FLAG', 'MATCH_RANK']
                available_key_cols = [col for col in key_cols if col in df.columns]

                if available_key_cols:
                    logger.info(f"  Key columns present: {available_key_cols}")
                    print(f"\n🎯 Key Columns Sample for {query_name}:")
                    print(df[available_key_cols].head().to_string())

            # Export with analysis using DataExporter class
            export_result = exporter.export_with_analysis(df, query_name)
            successful_exports.append({
                'query': query_name,
                'result': export_result
            })

            logger.info(f"✅ ({i+1:2d}/{len(available_queries)}) {query_name}: exported successfully "
                       f"({export_result['rows']:,} rows, {export_result['size_mb']:.2f} MB)")

        except Exception as e:
            logger.error(f"❌ ({i+1:2d}/{len(available_queries)}) {query_name}: failed - {str(e)}")
            failed_exports.append({'query': query_name, 'error': str(e)})
            continue

# Final Summary
print(f"\n🎉 SEQUEL JOIN RULES PROCESSING COMPLETED!")
print(f"✅ Successful: {len(successful_exports)}")
print(f"❌ Failed: {len(failed_exports)}")

if successful_exports:
    total_size = sum(result['result']['size_mb'] for result in successful_exports)
    total_rows = sum(result['result']['rows'] for result in successful_exports)
    print(f"📊 Total data exported: {total_rows:,} rows, {total_size:.2f} MB")

    print(f"\nAll files saved to: {output_dir}")
    print(f"Files created for each sequel join rule:")
    print(f"  - {{query_name}}.csv")
    print(f"  - {{query_name}}_schema.json")
    print(f"  - {{query_name}}_report.csv")

    # Show details for each successful query
    print(f"\n📋 Successful Query Details:")
    for export in successful_exports:
        result = export['result']
        print(f"  - {export['query']}: {result['rows']:,} rows, {result['size_mb']:.2f} MB")

if failed_exports:
    print(f"\n❌ Failed sequel join rules:")
    for failure in failed_exports:
        print(f"  - {failure['query']}: {failure['error']}")

print(f"\n🎉 Processing completed!")

# Make results available for further analysis
if query_results:
    print(f"\n💾 Query results stored in 'query_results' dictionary for further analysis")
    print(f"Access results like: query_results['usage_latest']")

    # Store the main result globally for convenience
    if 'usage_latest' in query_results:
        usage_latest = query_results['usage_latest']
        print(f"✅ usage_latest dataframe available with {len(usage_latest):,} rows")
# %%
# Optional: Quick analysis of results
if query_results:
    print(f"\n📊 Quick Analysis of Query Results:")
    for query_name, df in query_results.items():
        print(f"\n{query_name}:")
        print(f"  Shape: {df.shape}")
        if len(df) > 0:
            print(f"  Sample data types: {dict(list(df.dtypes.items())[:5])}")

            # Show churned vs non-churned if applicable
            if 'CHURNED_FLAG' in df.columns:
                churn_counts = df['CHURNED_FLAG'].value_counts()
                print(f"  Churn distribution: {dict(churn_counts)}")

            # Show unique customers if applicable
            if 'CUST_ACCOUNT_NUMBER' in df.columns:
                unique_customers = df['CUST_ACCOUNT_NUMBER'].nunique()
                print(f"  Unique customers: {unique_customers:,}")
# %%
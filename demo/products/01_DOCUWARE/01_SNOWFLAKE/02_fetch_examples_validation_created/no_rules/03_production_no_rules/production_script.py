"""
Production script to process ALL Snowflake tables with ALL rows
- Uses DataExporter directly for consistency
- Enhanced logging and error handling
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-02
# * CREATED ON: 2025-07-31
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# 🚀 Running Production Script with Different Logging Options
# ✅ With Console Logging (Default)
# Default behavior - shows all logs in console
# python process_all_tables.py
#
# Explicitly enable console logging
# python process_all_tables.py disable_console_logging=false
#
# With development environment (more verbose)
# python process_all_tables.py environment=development disable_console_logging=false
#
# With debug mode for maximum verbosity
# python process_all_tables.py debug=true disable_console_logging=false
#
# ❌ Without Console Logging (Silent Mode)
# Disable console logging - only file logging
# python process_all_tables.py disable_console_logging=true
#
# Production mode without console output
# python process_all_tables.py environment=production disable_console_logging=true
#
# Silent background processing
# nohup python process_all_tables.py disable_console_logging=true &
#
# Silent with output redirect (Linux/Mac)
# python process_all_tables.py disable_console_logging=true > /dev/null 2>&1
#
# Silent with output redirect (Windows)
# python process_all_tables.py disable_console_logging=true > NUL 2>&1
#
# 🔧 Additional Logging Control Options
#
# Minimum logging with production settings
# python process_all_tables.py environment=production disable_console_logging=true debug=false
#
# Custom log level override (if you want specific control)
# python process_all_tables.py disable_console_logging=true +loggers.loguru.default_level=ERROR
#
# Process specific table without console noise
# python process_all_tables.py table_name=PS_EXPANSION_CUST_SITES disable_console_logging=true
#
# Limited tables without console output
# python process_all_tables.py max_tables=5 disable_console_logging=true
#
# 📋 Logging Behavior Explained
# With Console Logging (disable_console_logging=false):
# 🚀 PRODUCTION: Processing ALL Snowflake tables with ALL rows
# ✅ Using CSV config from Hydra configuration
# 📁 Output directory: /path/to/ricoh_aiml/data/products/DOCUWARE/DB/snowflake/csv/raw
# 📊 Processing ALL 25 tables with environment=development
# ⚠️  Processing ALL rows from each table (no limits)
# 🔄 (01/25) Fetching ALL data from PS_EXPANSION_CUST_SITES
# 📊 (01/25) PS_EXPANSION_CUST_SITES: fetched 15,234 rows in 2.3s
# 💾 Exporting PS_EXPANSION_CUST_SITES (15,234 rows, 45 columns)
# 📁 DataExporter initialized with output directory: /path/to/output
# ✅ CSV exported: /path/to/PS_EXPANSION_CUST_SITES.csv
# ✅ Schema saved: /path/to/PS_EXPANSION_CUST_SITES_schema.json
# ✅ Report saved: /path/to/PS_EXPANSION_CUST_SITES_report.csv
# ✅ (01/25) PS_EXPANSION_CUST_SITES: completed successfully (15,234 rows, 12.5 MB, 4.1s)
# ...
#
# Without Console Logging (disable_console_logging=true):
# Even with disable_console_logging=true, logs are still written to files:
# ricoh_aiml/
# ├── logs/                    # Log files directory
# │   └── loguru/
# │       ├── debug.log        # Debug level logs
# │       ├── info.log         # Info level logs
# │       └── error.log        # Error level logs
# └── data/
#     └── products/
#         └── DOCUWARE/
#             └── DB/
#                 └── snowflake/
#                     └── csv/
#                         └── raw/     # Your CSV files still get created
#
# 🔍 Check Processing Status (When Running Silent)
# Monitor log files in real-time
# tail -f logs/loguru/info.log
#
# Check if process is still running
# ps aux | grep process_all_tables
#
# Check output directory for new files
# ls -la data/products/DOCUWARE/DB/snowflake/csv/raw/
#
# Count completed files (3 files per table = CSV + schema + report)
# ls data/products/DOCUWARE/DB/snowflake/csv/raw/ | wc -l
#
# ⏱️ Background Processing Examples
# Run in background with no console output
# nohup python process_all_tables.py disable_console_logging=true environment=production &
#
# Save process ID for later monitoring
# echo $! > process_all_tables.pid
#
# Check if still running
# kill -0 $(cat process_all_tables.pid) && echo "Running" || echo "Finished"
#
# Kill process if needed
# kill $(cat process_all_tables.pid)
#
# 🎯 Recommended Usage
# For Interactive/Development:
# python process_all_tables.py
#
# For Production/Automated:
# python process_all_tables.py environment=production disable_console_logging=true
#
# For Background Processing:
# nohup python process_all_tables.py disable_console_logging=true > process.log 2>&1 &

import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.data.validation.data_exporter import DataExporter

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

@hydra.main(version_base=None, config_path=conf_dir.as_posix(), config_name="config")
def main(cfg: DictConfig) -> None:
    """Production main function to process ALL Snowflake tables with ALL rows"""

    start_time = time.time()

    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("🚀 PRODUCTION: Processing ALL Snowflake tables with ALL rows")

    # Initialize output directory and DataExporter (same as other scripts)
    project_root = ProjectRootFinder().find_path()

    try:
        # Try to get CSV config from Hydra configuration
        csv_config = cfg.products.DOCUWARE.db.snowflake.csv_config.csv_output
        relative_path = csv_config.get('base_dir', 'data/products/DOCUWARE/DB/snowflake/csv/raw')
        logger.info("✅ Using CSV config from Hydra configuration")
    except Exception as e:
        # Fallback if csv_config is not available
        logger.warning(f"CSV config not found: {str(e)}")
        logger.info("🔄 Using fallback configuration")
        relative_path = 'data/products/DOCUWARE/DB/snowflake/csv/raw'

    output_dir = project_root / relative_path
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")

    # Initialize DataExporter directly (no wrapper class)
    exporter = DataExporter(output_dir, logger)

    # Get configuration parameters
    environment = cfg.get('environment', 'development')
    tables_to_process = snow_table_list  # Process ALL tables

    logger.info(f"📊 Processing ALL {len(tables_to_process)} tables with environment={environment}")
    logger.info("⚠️  Processing ALL rows from each table (no limits)")

    # Track results
    successful_exports = []
    failed_exports = []
    total_rows_processed = 0

    # Process all tables
    for i, snow_table in tqdm(enumerate(tables_to_process), desc="Processing ALL tables"):
        try:
            table_start_time = time.time()

            # Fetch ALL data from Snowflake (no limits)
            logger.info(f"🔄 ({i+1:2d}/{len(tables_to_process)}) Fetching ALL data from {snow_table}")
            with SnowFetch(config=cfg, environment=environment) as fetcher:
                df = fetcher.fetch_data(snow_table, limit=None)  # NO LIMIT - ALL ROWS

            table_fetch_time = time.time() - table_start_time
            total_rows_processed += len(df)

            logger.info(f"📊 ({i+1:2d}/{len(tables_to_process)}) {snow_table}: fetched {len(df):,} rows "
                       f"in {table_fetch_time:.1f}s")

            # Export to CSV with schema and report using DataExporter directly
            logger.info(f"💾 Exporting {snow_table} ({len(df):,} rows, {len(df.columns)} columns)")
            export_result = exporter.export_with_analysis(df, snow_table)

            # Add timing information
            total_time = time.time() - table_start_time
            export_result['export_time_seconds'] = total_time

            successful_exports.append({
                'table': snow_table,
                'result': export_result
            })

            logger.info(f"✅ ({i+1:2d}/{len(tables_to_process)}) {snow_table}: completed successfully "
                       f"({export_result['rows']:,} rows, {export_result['size_mb']:.2f} MB, {total_time:.1f}s)")

            # Memory cleanup
            del df

        except Exception as e:
            logger.error(f"❌ ({i+1:2d}/{len(tables_to_process)}) {snow_table}: failed - {str(e)}")
            failed_exports.append({'table': snow_table, 'error': str(e)})
            continue

    # Final summary
    total_time = time.time() - start_time
    logger.info("🎉 PRODUCTION PROCESSING COMPLETED!")
    logger.info(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
    logger.info(f"✅ Successful exports: {len(successful_exports)}")
    logger.info(f"❌ Failed exports: {len(failed_exports)}")

    if successful_exports:
        total_size = sum(result['result']['size_mb'] for result in successful_exports)
        logger.info(f"📊 Total data exported: {total_rows_processed:,} rows, {total_size:.2f} MB")
        logger.info(f"📁 All files saved to: {output_dir}")

        # Show largest tables
        sorted_exports = sorted(successful_exports,
                               key=lambda x: x['result']['rows'], reverse=True)
        logger.info("🏆 Top 5 largest tables by row count:")
        for j, export in enumerate(sorted_exports[:5], 1):
            result = export['result']
            logger.info(f"  {j}. {export['table']}: {result['rows']:,} rows, {result['size_mb']:.2f} MB")

    if failed_exports:
        logger.error(f"❌ Failed tables ({len(failed_exports)}):")
        for failure in failed_exports:
            logger.error(f"  - {failure['table']}: {failure['error']}")

    logger.info("🎯 For each table, 3 files were created:")
    logger.info("   - {table_name}.csv (data)")
    logger.info("   - {table_name}_schema.json (Pydantic schema)")
    logger.info("   - {table_name}_report.csv (analysis report)")

if __name__ == "__main__":
    main()
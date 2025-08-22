"""
Enhanced Snowflake data fetcher with CSV export, Pydantic validation, and reporting
- Uses DataExporter directly for consistency with manual script
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

# Table list to fetch
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
    """Main function to fetch data from Snowflake and export to CSV with validation"""

    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("Enhanced Snowflake data fetching with CSV export started")

    # Initialize output directory and DataExporter (same as manual script)
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

    # Initialize DataExporter directly (no wrapper class needed)
    exporter = DataExporter(output_dir, logger)

    # Get configuration parameters
    max_tables = cfg.get('max_tables', None)  # None means process all tables
    row_limit = cfg.get('row_limit', None)    # None means process all rows
    environment = cfg.get('environment', 'development')
    table_name = cfg.get('table_name', None)

    # If specific table is requested, process only that table
    if table_name:
        tables_to_process = [table_name] if table_name in snow_table_list else []
        if not tables_to_process:
            logger.error(f"Table {table_name} not found in table list")
            return
    else:
        # Process all tables or up to max_tables if specified
        if max_tables is None:
            tables_to_process = snow_table_list  # Process ALL tables
        else:
            tables_to_process = snow_table_list[:max_tables]

    logger.info(f"Processing ALL {len(tables_to_process)} tables with environment={environment}")
    if row_limit:
        logger.info(f"Row limit per table: {row_limit:,}")
    else:
        logger.info("Processing ALL rows from each table (no row limit)")

    # Track results
    successful_exports = []
    failed_exports = []

    # Fetch data from tables and export
    for i, snow_table in tqdm(enumerate(tables_to_process), desc="Processing tables"):
        try:
            # Fetch data from Snowflake
            with SnowFetch(config=cfg, environment=environment) as fetcher:
                df = fetcher.fetch_data(snow_table, limit=row_limit)
                logger.info(f"📊 ({i+1:2d}) {snow_table}: fetched {len(df)} rows")

            # Export to CSV with schema and report using DataExporter directly
            export_result = exporter.export_with_analysis(df, snow_table)
            successful_exports.append({
                'table': snow_table,
                'result': export_result
            })

            logger.info(f"✅ ({i+1:2d}) {snow_table}: exported successfully "
                       f"({export_result['rows']} rows, {export_result['size_mb']:.2f} MB)")

        except Exception as e:
            logger.error(f"❌ ({i+1:2d}) {snow_table}: failed - {str(e)}")
            failed_exports.append({'table': snow_table, 'error': str(e)})
            continue

    # Summary report
    logger.info(f"Processing completed: {len(successful_exports)} successful, {len(failed_exports)} failed")

    if successful_exports:
        total_size = sum(result['result']['size_mb'] for result in successful_exports)
        total_rows = sum(result['result']['rows'] for result in successful_exports)
        logger.info(f"Total data exported: {total_rows:,} rows, {total_size:.2f} MB")

if __name__ == "__main__":
    main()
# %%
"""
Manual Snowflake data fetcher with enhanced CSV export and Pydantic validation.

This script features:
- Automatic processing of ALL tables with ALL rows
- Advanced type inference (integers/floats in strings, datetime detection)
- DataExporter with 'Inferred_Dtype' column in reports
- Proper formatting of datetime values and removal of decimal zeros
- NaN substitution for failed type conversions
- Automatic script-local logging (logs created in this script's directory)

Output files per table:
- {table_name}.csv - Raw data export
- {table_name}_schema.json - Pydantic schema with inferred types
- {table_name}_report.csv - Analysis report with Inferred_Dtype column

Logs are automatically created in: {script_dir}/logs/loguru/
"""
# %%
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-13
# * CREATED ON: 2025-07-31
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
from datetime import datetime
from typing import Dict, Any

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.data.validation.data_exporter import DataExporter
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.utils.profiling import timer
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
logger.info("Manual Snowflake data fetching with CSV export started")
# %%
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
# %%
# Initialize CSV output directory using project root finder
csv_config = cfg.products.DOCUWARE.db.snowflake.csv_config.csv_output
relative_path = csv_config.get('base_dir', 'data/products/DOCUWARE/DB/snowflake/csv/raw')
output_dir = project_root / relative_path
output_dir.mkdir(parents=True, exist_ok=True)
print(f"CSV output directory: {output_dir}")
# %%
# Initialize DataExporter
exporter = DataExporter(output_dir, logger)
# %%
# Main Processing: Process ALL tables with ALL rows
print("\n" + "="*60)
print("Main Processing: ALL tables with ALL rows")
print("="*60)

print(f"📊 Total tables to process: {len(snow_table_list)}")
print("🚀 Processing ALL rows from ALL tables automatically...")

successful_exports = []
failed_exports = []

with timer():
    for i, snow_table in tqdm(enumerate(snow_table_list), desc="Processing ALL tables"):
        try:
            # Fetch ALL data (no limits)
            with SnowFetch(config=cfg, environment="development") as fetcher:
                df = fetcher.fetch_data(snow_table, limit=None)  # NO LIMIT!
                logger.info(f"📊 ({i+1:2d}/{len(snow_table_list)}) {snow_table}: fetched {len(df):,} rows")

            # Export with analysis using DataExporter class
            export_result = exporter.export_with_analysis(df, snow_table)
            successful_exports.append({
                'table': snow_table,
                'result': export_result
            })

            logger.info(f"✅ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: exported successfully "
                       f"({export_result['rows']:,} rows, {export_result['size_mb']:.2f} MB)")

        except Exception as e:
            logger.error(f"❌ ({i+1:2d}/{len(snow_table_list)}) {snow_table}: failed - {str(e)}")
            failed_exports.append({'table': snow_table, 'error': str(e)})
            continue

# Final Summary
print(f"\n🎉 PROCESSING COMPLETED!")
print(f"✅ Successful: {len(successful_exports)}")
print(f"❌ Failed: {len(failed_exports)}")

if successful_exports:
    total_size = sum(result['result']['size_mb'] for result in successful_exports)
    total_rows = sum(result['result']['rows'] for result in successful_exports)
    print(f"📊 Total data exported: {total_rows:,} rows, {total_size:.2f} MB")

    print(f"\nAll files saved to: {output_dir}")
    print(f"Files created for each table:")
    print(f"  - {{table_name}}.csv")
    print(f"  - {{table_name}}_schema.json")
    print(f"  - {{table_name}}_report.csv")  # Updated to CSV

if failed_exports:
    print(f"\n❌ Failed tables:")
    for failure in failed_exports:
        print(f"  - {failure['table']}: {failure['error']}")

print(f"\n🎉 Processing completed!")
# %%

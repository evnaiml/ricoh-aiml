"""
A minimal example how to fetch data from Snowflake using Hydra with decorator
"""
# %%
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-02
# * CREATED ON: 2025-07-29
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# 🚀 Quick Start:
# -----------------------------------------------------------------------------
# * Development testing: python snowflake_fetch_main.py
# -----------------------------------------------------------------------------
# * Production run: snowflake_fetch_main.py environment=production disable_console_logging=true
# -----------------------------------------------------------------------------
# * Debug issues: python snowflake_fetch_main.py debug=true max_tables=1
# -----------------------------------------------------------------------------
# * Debug mode with maximum verbosity
#
# 1. Force add the missing key (most likely to work):
# python snowflake_fetch_main.py debug=true +loggers.loguru.default_level=DEBUG
#
# 2. First check what your config actually looks like:
# python snowflake_fetch_main.py --cfg job
#
# 3. Alternative override paths:
# python snowflake_fetch_main.py debug=true +default_level=DEBUG
#
# 4. python snowflake_fetch_main.py debug=true loggers/loguru=debug_loguru
#
# 5. python snowflake_fetch_main.py debug=true +loggers.loguru.default_level=DEBUG max_tables=3 row_limit=100
# %%
import hydra
from omegaconf import DictConfig
from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from tqdm import tqdm

from churn_aiml.utils.find_paths import ProjectRootFinder

# %%
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
# %%
# Set paths to directories
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"
# %%
# Define main fucntion with decorator
@hydra.main(version_base=None, config_path=conf_dir.as_posix(), config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to fetch data from Snowflake"""

    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("Snowflake data fetching example started")

    # Get configuration parameters
    max_tables = cfg.get('max_tables', len(snow_table_list))
    row_limit = cfg.get('row_limit', None)
    environment = cfg.get('environment', 'development')
    table_name = cfg.get('table_name', None)

    # If specific table is requested, process only that table
    if table_name:
        tables_to_process = [table_name] if table_name in snow_table_list else []
        if not tables_to_process:
            logger.error(f"Table {table_name} not found in table list")
            return
    else:
        # Process up to max_tables
        tables_to_process = snow_table_list[:max_tables]

    logger.info(f"Processing {len(tables_to_process)} tables with environment={environment}")
    if row_limit:
        logger.info(f"Row limit: {row_limit}")

    # Fetch data from tables
    for i, snow_table in tqdm(enumerate(tables_to_process), desc="Fetching tables"):
        try:
            with SnowFetch(config=cfg, environment=environment) as fetcher:
                df = fetcher.fetch_data(snow_table, limit=row_limit)
                logger.info(f"✅({i+1:2d}) {snow_table}: got {len(df)} rows")

        except Exception as e:
            logger.error(f"❌({i+1:2d}) {snow_table}: failed - {str(e)}")
            continue

    logger.info("Snowflake data fetching completed")

if __name__ == "__main__":
    main()
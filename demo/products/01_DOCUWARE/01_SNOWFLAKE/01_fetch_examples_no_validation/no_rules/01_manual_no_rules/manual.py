# %%
"""
A minimal example of fetching data from Snowflake using Hydra configuration.

This script demonstrates:
- Manual Hydra configuration loading without decorators
- Automatic script-local logging (logs created in this script's directory)
- Snowflake data fetching with configurable limits
- Progress tracking with tqdm for batch operations
- Performance timing with built-in profiling utilities

Logs are automatically created in: {script_dir}/logs/loguru/
"""
# %%
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-13
# * CREATED ON: 2025-07-29
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# %%
# 📊 Key Features:
# ✅ Environment Switching: environment=development or environment=production
# ✅ Console Control: disable_console_logging=true
# ✅ Table Limiting: max_tables=5
# ✅ Row Limiting: row_limit=1000
# ✅ Debug Mode: debug=true
# ✅ Progress Tracking: Built-in with tqdm
# ✅ Error Handling: Continues processing even if some tables fail
# ✅ Summary Reports: Shows success/failure counts and largest tables
# %%
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch

from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.utils.profiling import timer
# %%
# Set paths
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"
print(f"config path: {conf_dir}")
# %%
# Load Hydra configuration manually
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
    cfg = compose(config_name="config")
# %%
# Setup logger (automatically creates logs in script's directory)
logger_config = setup_logger(cfg)
logger = get_logger()
logger.info("Snowflake data fetching example started")
# %%
# Table list
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
# Fetch 10 rows with Limit=10
with timer():
    with SnowFetch(config=cfg, environment="development") as fetcher:
        table_name = snow_table_list[0]
        df = fetcher.fetch_data(snow_table_list[0], limit=10)
        print(f"✅ Got {len(df)} rows")
        print(df.head().to_string())
# %%
# Fetch all rows with Limit=None
with timer():
    with SnowFetch(config=cfg, environment="development") as fetcher:
        df = fetcher.fetch_data(snow_table_list[0], limit=None)
        print(f"✅ Got {len(df)} rows")
# %%
# Fetch all rows with Limit=None from all datasets (~ 2 min. 30 sec)
with timer():
    for i, snow_table in tqdm(enumerate(snow_table_list)):
        with SnowFetch(config=cfg, environment="development") as fetcher:
            df = fetcher.fetch_data(snow_table, limit=None)
            print(f"✅({i+1:2d}) {snow_table}: got {len(df)} rows")
# %%

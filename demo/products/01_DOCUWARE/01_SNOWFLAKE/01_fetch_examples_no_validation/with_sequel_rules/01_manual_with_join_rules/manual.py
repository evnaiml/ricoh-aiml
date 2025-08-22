# %%
"""
A minimal example of fetching data from Snowflake using sequel join rules.

This script demonstrates:
- Manual Hydra configuration loading with sequel rules
- Automatic script-local logging (logs created in this script's directory)
- Execution of complex SQL queries defined in configuration
- Progress tracking for multiple query execution
- Comprehensive error handling and reporting

Logs are automatically created in: {script_dir}/logs/loguru/
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
# %%
# 📊 Key Features:
# ✅ Environment Switching: environment=development or environment=production
# ✅ Console Control: disable_console_logging=true
# ✅ Sequel Join Rules: Load and execute SQL queries from config
# ✅ Row Limiting: row_limit=1000
# ✅ Debug Mode: debug=true
# ✅ Progress Tracking: Built-in with tqdm
# ✅ Error Handling: Continues processing even if some queries fail
# ✅ Summary Reports: Shows success/failure counts and query results
# %%
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
import pandas as pd
from tqdm import tqdm

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch

from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.utils.profiling import timer
# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
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
logger.info("Snowflake data fetching with sequel join rules started")
# %%
# Load sequel join rules from configuration
sequel_rules = cfg.products.DOCUWARE.db.snowflake.sequel_rules
print("Available sequel join rules:")
for query_name in sequel_rules.queries.keys():
    description = sequel_rules.queries[query_name].get('description', 'No description')
    print(f"  - {query_name}: {description}")
# %%
# Execute your query and inspect columns
with timer():
    with SnowFetch(config=cfg, environment="development") as fetcher:
        query_config = sequel_rules.queries.usage_latest
        sql_query = query_config.sql

        logger.info(f"Executing query: {query_config.output_name}")
        usage_latest = fetcher.session.sql(sql_query).to_pandas()

        print(f"✅ DataFrame Shape: {usage_latest.shape}")
        print(f"✅ Total Columns: {len(usage_latest.columns)}")
        print(f"\n📋 Complete Column List:")
        for i, col in enumerate(usage_latest.columns, 1):
            print(f"  {i:2d}. {col}")

        print(f"\n📊 Data Types:")
        print(usage_latest.dtypes)

        print(f"\n🔍 First 3 rows (all columns):")
        print(usage_latest.head(3).to_string())

        # Show sample of specific key columns
        key_cols = ['CUST_ACCOUNT_NUMBER', 'CUST_PARTY_NAME', 'CUSTOMER_NAME',
                   'CONTRACT_NUMBER', 'CHURNED_FLAG', 'MATCH_RANK']
        available_key_cols = [col for col in key_cols if col in usage_latest.columns]

        if available_key_cols:
            print(f"\n🎯 Key Columns Sample:")
            print(usage_latest[available_key_cols].head().to_string())
# %%
# Example 1: Execute a specific sequel rule query (usage_latest)
with timer():
    with SnowFetch(config=cfg, environment="development") as fetcher:
        query_config = sequel_rules.queries.usage_latest
        sql_query = query_config.sql

        # Execute the SQL query
        logger.info(f"Executing query: {query_config.output_name}")
        usage_latest = fetcher.session.sql(sql_query).to_pandas()

        print(f"✅ Created {query_config.output_name} dataframe with {len(usage_latest)} rows")
        print(f"Columns: {list(usage_latest.columns)}")
        if len(usage_latest) > 0:
            print("\nFirst 5 rows:")
            print(usage_latest.head().to_string())
# %%
# Example 2: Execute a query with custom limit
with timer():
    with SnowFetch(config=cfg, environment="development") as fetcher:
        query_config = sequel_rules.queries.usage_latest
        sql_query = query_config.sql

        # Add LIMIT clause to the query for testing
        limited_query = f"SELECT * FROM ({sql_query}) LIMIT 10"

        logger.info(f"Executing limited query: {query_config.output_name} (10 rows)")
        usage_latest_limited = fetcher.session.sql(limited_query).to_pandas()

        print(f"✅ Created limited {query_config.output_name} dataframe with {len(usage_latest_limited)} rows")
        print(usage_latest_limited.head().to_string())
# %%
# Example 3: Execute all available sequel rules
with timer():
    results = {}
    with SnowFetch(config=cfg, environment="development") as fetcher:
        for query_name, query_config in tqdm(sequel_rules.queries.items()):
            try:
                logger.info(f"Executing query: {query_name}")
                sql_query = query_config.sql

                # Execute the query
                df = fetcher.session.sql(sql_query).to_pandas()
                results[query_name] = df

                print(f"✅ {query_name}: Created dataframe with {len(df)} rows")

            except Exception as e:
                logger.error(f"❌ Failed to execute {query_name}: {str(e)}")
                results[query_name] = None

    # Summary
    print(f"\n📊 Summary:")
    print(f"Total queries attempted: {len(sequel_rules.queries)}")
    print(f"Successful: {len([r for r in results.values() if r is not None])}")
    print(f"Failed: {len([r for r in results.values() if r is None])}")
# %%
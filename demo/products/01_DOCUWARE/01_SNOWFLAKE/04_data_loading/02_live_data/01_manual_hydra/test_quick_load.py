#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test to verify data loading works
"""
# Suppress known warnings before any imports
from churn_aiml.utils.suppress_warnings import suppress_known_warnings

import pandas as pd
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from churn_aiml.data.db.snowflake.loaddata import SnowLiveDataLoader, SnowTrainDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger
from churn_aiml.utils.find_paths import ProjectRootFinder

# Setup paths
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"
print(f"Config path: {conf_dir}")

# Clear and initialize Hydra configuration
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
    cfg = compose(config_name="config")

# Setup logger
logger_config = setup_logger_for_script(cfg, __file__)
logger = get_logger()

# Initialize loaders
logger.info("Initializing data loaders...")
train_data_loader = SnowTrainDataLoader(config=cfg, environment="development")
live_data_loader = SnowLiveDataLoader(config=cfg, environment="development")

# Test loading small subset
logger.info("Testing data loading...")
try:
    # Load train data (already works)
    train_data = train_data_loader.load_data()
    logger.info(f"✅ Train data loaded: {len(train_data)} datasets")
    
    # Load new data
    new_data = live_data_loader.load_data(train_data_loader=train_data_loader)
    logger.info(f"✅ New data loaded: {len(new_data)} datasets")
    
    # Check for new_data column
    for key, df in new_data.items():
        if isinstance(df, pd.DataFrame) and 'new_data' in df.columns:
            new_count = df['new_data'].sum()
            existing_count = len(df) - new_count
            logger.info(f"  {key}: {new_count} new, {existing_count} existing")
    
    print("\n✅ SUCCESS: Data loading works correctly!")
    print(f"Train datasets: {list(train_data.keys())}")
    print(f"New datasets: {list(new_data.keys())}")
    
except Exception as e:
    logger.error(f"❌ ERROR: {e}")
    raise
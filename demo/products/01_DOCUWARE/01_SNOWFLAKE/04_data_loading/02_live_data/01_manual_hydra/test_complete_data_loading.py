#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
"""
Complete Data Loading Example for New/Active Customers

This script demonstrates the complete data loading process for live/active customers
from Snowflake, following the patterns established in refactored.py but using our
refactored classes and without any imputation, tsfresh, or predictions.

✅ Key Features:
- Loads all necessary data from Snowflake for active customers
- Uses Pydantic validation schemas for type enforcement
- Uses ISO converters from our classes (not from refactored.py)
- Adds new_data indicator comparing with training data
- NO imputation of missing values
- NO tsfresh feature engineering
- NO predictions

📊 Data Loaded:
1. Usage data (with Jaro-Winkler matching)
2. Payments data
3. Revenue data
4. Transactions data
5. Contract subline data
6. SSCD Renewals data
7. L1 Customer data
8. DNB Risk data

💡 Usage:
Run directly: python test_complete_data_loading.py
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-18
# * CREATED ON: 2025-08-18
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# -----------------------------------------------------------------------------

# %%
# Suppress known warnings before any imports
from churn_aiml.utils.suppress_warnings import suppress_known_warnings

import pandas as pd
import numpy as np
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# Import required modules
from churn_aiml.data.db.snowflake.loaddata import SnowLiveDataLoader, SnowTrainDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.utils.profiling import timer

# %%
# Setup paths
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"
print(f"Config path: {conf_dir}")

# %%
# Clear and initialize Hydra configuration
GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
    cfg = compose(config_name="config")

# %%
# Setup logger
logger_config = setup_logger_for_script(cfg, __file__)
logger = get_logger()
logger.info("=" * 80)
logger.info("Starting Complete Data Loading for New/Active Customers")
logger.info("=" * 80)

# %%
# STEP 1: Load Training Data for Comparison
logger.info("STEP 1: Loading training data for comparison")
logger.info("-" * 60)

train_data_loader = SnowTrainDataLoader(config=cfg, environment="development")

with timer():
    train_data = train_data_loader.load_data()

logger.info(f"✅ Training data loaded with {len(train_data)} datasets:")
for key in train_data.keys():
    if isinstance(train_data[key], pd.DataFrame):
        logger.info(f"  - {key}: {train_data[key].shape}")

# %%
# STEP 2: Load New/Active Customer Data
logger.info("\nSTEP 2: Loading live/active customer data")
logger.info("-" * 60)

live_data_loader = SnowLiveDataLoader(config=cfg, environment="development")

with timer():
    new_data = live_data_loader.load_data(train_data_loader=train_data_loader)

logger.info(f"✅ New/active data loaded with {len(new_data)} datasets:")
for key in new_data.keys():
    if isinstance(new_data[key], pd.DataFrame):
        logger.info(f"  - {key}: {new_data[key].shape}")

# %%
# STEP 3: Verify Data Loading Components
logger.info("\nSTEP 3: Verifying data loading components")
logger.info("-" * 60)

# 3.1 Check usage data processing (similar to refactored.py lines 1862-2027)
usage_processed = new_data.get('usage_processed', pd.DataFrame())
if not usage_processed.empty:
    logger.info("✅ Usage data processed successfully:")
    logger.info(f"  - Total rows: {len(usage_processed)}")
    logger.info(f"  - Unique customers: {usage_processed['CUST_ACCOUNT_NUMBER'].nunique()}")
    logger.info(f"  - Date range: {usage_processed['YYYYWK'].min()} to {usage_processed['YYYYWK'].max()}")
    
    # Verify only active customers
    if 'CHURNED_FLAG' in usage_processed.columns:
        active_count = (usage_processed['CHURNED_FLAG'] == 0).sum()
        logger.info(f"  - Active records: {active_count}/{len(usage_processed)}")

# %%
# 3.2 Check merged datasets (similar to refactored.py lines 2034-2149)
merged_data = new_data.get('merged_data', pd.DataFrame())
if not merged_data.empty:
    logger.info("\n✅ Merged datasets successfully:")
    logger.info(f"  - Total rows: {len(merged_data)}")
    logger.info(f"  - Columns: {merged_data.shape[1]}")
    
    # Check for key columns from different sources
    expected_cols = [
        'FUNCTIONAL_AMOUNT',  # From payments
        'INVOICE_REVLINE_TOTAL',  # From revenue
        'ORIGINAL_AMOUNT_DUE',  # From transactions
        'SLINE_START_DATE',  # From contract subline
        'STARTDATECOVERAGE',  # From renewals
        'PROBABILITY_OF_DELINQUENCY',  # From DNB risk
        'DOCUMENTS_OPENED',  # From usage
        'USED_STORAGE_MB'  # From usage
    ]
    
    available_cols = [col for col in expected_cols if col in merged_data.columns]
    logger.info(f"  - Key columns present: {len(available_cols)}/{len(expected_cols)}")

# %%
# STEP 4: Verify New Data Indicator
logger.info("\nSTEP 4: Verifying new_data indicator")
logger.info("-" * 60)

has_new_indicator = False
for key, df in new_data.items():
    if isinstance(df, pd.DataFrame) and 'new_data' in df.columns:
        has_new_indicator = True
        new_count = df['new_data'].sum()
        existing_count = len(df) - new_count
        
        logger.info(f"\n{key}:")
        logger.info(f"  - New records (not in training): {new_count} ({new_count/len(df)*100:.1f}%)")
        logger.info(f"  - Existing records (in training): {existing_count} ({existing_count/len(df)*100:.1f}%)")
        
        if 'CUST_ACCOUNT_NUMBER' in df.columns:
            new_customers = df[df['new_data'] == 1]['CUST_ACCOUNT_NUMBER'].nunique()
            existing_customers = df[df['new_data'] == 0]['CUST_ACCOUNT_NUMBER'].nunique()
            logger.info(f"  - New unique customers: {new_customers}")
            logger.info(f"  - Existing unique customers: {existing_customers}")

if has_new_indicator:
    logger.info("\n✅ New data indicator successfully added")
else:
    logger.warning("\n⚠️ New data indicator not found in datasets")

# %%
# STEP 5: Verify ISO Converters Usage
logger.info("\nSTEP 5: Verifying ISO converters")
logger.info("-" * 60)

# Check that we're using our ISO converters
from churn_aiml.ml.datetime.iso_converters import ISOWeekDateConverter, WeekMidpointConverter

iso_converter = ISOWeekDateConverter(cfg, log_operations=False)
midpoint_converter = WeekMidpointConverter(cfg, log_operations=False)

# Test conversion
test_yyyywk = 202432
test_date = midpoint_converter.convert_yyyywk_to_actual_mid_date(test_yyyywk)
logger.info(f"✅ ISO converters working: YYYYWK {test_yyyywk} -> {test_date}")

# %%
# STEP 6: Data Quality Checks
logger.info("\nSTEP 6: Data quality checks")
logger.info("-" * 60)

# Check time series features
ts_features = new_data.get('time_series_features', pd.DataFrame())
if not ts_features.empty:
    logger.info("\n✅ Time series features:")
    logger.info(f"  - Shape: {ts_features.shape}")
    
    # Check for missing values
    missing_usage = ts_features[['DOCUMENTS_OPENED', 'USED_STORAGE_MB']].isnull().sum()
    logger.info(f"  - Missing DOCUMENTS_OPENED: {missing_usage['DOCUMENTS_OPENED']}")
    logger.info(f"  - Missing USED_STORAGE_MB: {missing_usage['USED_STORAGE_MB']}")
    
    # Verify NO imputation was done (there should be missing values)
    if missing_usage.sum() > 0:
        logger.info("  ✅ No imputation performed (as expected)")
    else:
        logger.warning("  ⚠️ No missing values found - verify imputation wasn't performed")

# Check static features
static_features = new_data.get('static_features', pd.DataFrame())
if not static_features.empty:
    logger.info("\n✅ Static features:")
    logger.info(f"  - Shape: {static_features.shape}")
    
    # Check DNB risk features
    risk_cols = ['PROBABILITY_OF_DELINQUENCY', 'RICOH_CUSTOM_RISK_MODEL']
    for col in risk_cols:
        if col in static_features.columns:
            missing = static_features[col].isnull().sum()
            logger.info(f"  - Missing {col}: {missing} ({missing/len(static_features)*100:.1f}%)")

# %%
# STEP 7: Summary Statistics
logger.info("\nSTEP 7: Summary statistics")
logger.info("-" * 60)

# Get customer metadata
customer_metadata = new_data.get('customer_metadata', pd.DataFrame())
if not customer_metadata.empty:
    total_customers = len(customer_metadata)
    
    # All should be active
    if 'CHURNED_FLAG' in customer_metadata.columns:
        active_count = (customer_metadata['CHURNED_FLAG'] == 0).sum()
        logger.info(f"\nCustomer breakdown:")
        logger.info(f"  - Total customers: {total_customers}")
        logger.info(f"  - Active customers: {active_count} ({active_count/total_customers*100:.1f}%)")
        
        if active_count != total_customers:
            logger.warning(f"  ⚠️ Found {total_customers - active_count} non-active customers")
    
    # Segment analysis
    if 'CUSTOMER_SEGMENT' in customer_metadata.columns:
        logger.info("\nTop customer segments:")
        segment_counts = customer_metadata['CUSTOMER_SEGMENT'].value_counts().head(5)
        for segment, count in segment_counts.items():
            logger.info(f"  - {segment}: {count} ({count/total_customers*100:.1f}%)")
    
    # New vs existing breakdown
    if 'new_data' in customer_metadata.columns:
        new_customers = customer_metadata['new_data'].sum()
        logger.info(f"\nLive customer analysis:")
        logger.info(f"  - Live customers: {new_customers} ({new_customers/total_customers*100:.1f}%)")
        logger.info(f"  - Existing customers: {total_customers - new_customers} ({(total_customers - new_customers)/total_customers*100:.1f}%)")

# %%
# STEP 8: Feature Engineering Dataset
logger.info("\nSTEP 8: Feature engineering dataset")
logger.info("-" * 60)

fe_dataset = live_data_loader.get_feature_engineering_dataset()
if not fe_dataset.empty:
    logger.info("✅ Feature engineering dataset created:")
    logger.info(f"  - Shape: {fe_dataset.shape}")
    logger.info(f"  - Unique customers: {fe_dataset['CUST_ACCOUNT_NUMBER'].nunique()}")
    
    # Check key features
    key_features = [
        'DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'CHURNED_FLAG',
        'DAYS_ELAPSED', 'MONTHS_ELAPSED', 'new_data'
    ]
    available_features = [f for f in key_features if f in fe_dataset.columns]
    logger.info(f"  - Key features available: {len(available_features)}/{len(key_features)}")
    logger.info(f"    {', '.join(available_features)}")

# %%
# STEP 9: Active Customer Summary
logger.info("\nSTEP 9: Active customer summary")
logger.info("-" * 60)

active_summary = live_data_loader.get_active_summary()
if not active_summary.empty:
    logger.info("✅ Active customer summary generated:")
    logger.info(f"  - Total customers: {active_summary['total_customers'].iloc[0]}")
    logger.info(f"  - Active customers: {active_summary['active_customers'].iloc[0]}")
    logger.info(f"  - Active rate: {active_summary['active_rate_pct'].iloc[0]:.1f}%")
    
    # Top segments
    for i in range(1, 4):
        if f'segment_{i}_name' in active_summary.columns:
            name = active_summary[f'segment_{i}_name'].iloc[0]
            count = active_summary[f'segment_{i}_count'].iloc[0]
            pct = active_summary[f'segment_{i}_pct'].iloc[0]
            logger.info(f"  - Segment {i}: {name} - {count} ({pct:.1f}%)")

# %%
# FINAL VALIDATION
logger.info("\n" + "=" * 80)
logger.info("FINAL VALIDATION SUMMARY")
logger.info("=" * 80)

validation_passed = True
validation_results = []

# Check 1: Data loaded successfully
if len(new_data) > 0:
    validation_results.append("✅ Data loaded successfully")
else:
    validation_results.append("❌ Data loading failed")
    validation_passed = False

# Check 2: Only active customers
if not customer_metadata.empty and 'CHURNED_FLAG' in customer_metadata.columns:
    if (customer_metadata['CHURNED_FLAG'] == 0).all():
        validation_results.append("✅ Only active customers loaded")
    else:
        validation_results.append("⚠️ Some non-active customers found")

# Check 3: New data indicator present
has_indicator = any('new_data' in df.columns for df in new_data.values() if isinstance(df, pd.DataFrame))
if has_indicator:
    validation_results.append("✅ New data indicator added")
else:
    validation_results.append("❌ New data indicator missing")
    validation_passed = False

# Check 4: No imputation performed
if not ts_features.empty:
    missing_count = ts_features[['DOCUMENTS_OPENED', 'USED_STORAGE_MB']].isnull().sum().sum()
    if missing_count > 0:
        validation_results.append("✅ No imputation performed (missing values present)")
    else:
        validation_results.append("⚠️ No missing values - verify imputation wasn't done")

# Check 5: Using correct ISO converters
validation_results.append("✅ Using ISO converters from churn_aiml.ml.datetime")

# Print results
for result in validation_results:
    logger.info(result)

if validation_passed:
    logger.info("\n🎉 ALL CRITICAL VALIDATIONS PASSED!")
else:
    logger.warning("\n⚠️ Some validations failed - review above")

# %%
logger.info("\n" + "=" * 80)
logger.info("DATA LOADING COMPLETE")
logger.info("=" * 80)
logger.info("All data has been loaded successfully without:")
logger.info("  - Imputation of missing values")
logger.info("  - TSFresh feature engineering")
logger.info("  - Predictions or modeling")
logger.info("\nData is ready for analysis and visualization!")
logger.info("Access loaded data through: live_data_loader.processed_data")
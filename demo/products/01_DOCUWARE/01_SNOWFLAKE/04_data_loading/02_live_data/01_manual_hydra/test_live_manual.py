# %%
"""
Interactive Live/Active Customer Data Loading and Exploration Tool

This INTERACTIVE script provides comprehensive data exploration capabilities for
active (non-churned) customer data that will be used for churn prediction inferencing.
It leverages the SnowLiveDataLoader class to load, process, and analyze active customer
data from Snowflake with real-time exploration capabilities.

✅ Key Features:
- Interactive data loading from Snowflake for active customers only
- Real-time exploration of time series and static features for active customers
- Feature engineering dataset creation for inferencing
- Comparison with training data to identify live customers not in training
- Customer segmentation and lifecycle analysis for active customers
- NO FILE OUTPUTS - purely interactive console exploration
- NO REPORTS - all analysis shown interactively via print()
- Memory-efficient processing with detailed metrics

📊 Data Processing Pipeline:
- Raw data ingestion from Snowflake with validation (active customers only)
- Usage data processing WITHOUT imputation
- Time series feature extraction at weekly granularity
- Static feature aggregation and transformation
- Live customer identification and processing

🔍 Analysis Capabilities:
- Active customer distribution and statistics
- Time series usage pattern exploration for active customers
- Segment-wise active customer behavior analysis
- Feature completeness and quality checks
- ML compatibility verification for inferencing

📝 Interactive Exploration Mode:
- Real-time data inspection with pandas DataFrames using print(df.head().to_string())
- Detailed logging of processing steps (logs only, no reports)
- Memory usage and performance metrics
- Sample data display for validation
- Comprehensive feature descriptions

⚠️ NOTE: This script is for INTERACTIVE exploration only:
- Creates only log files in logs/ directory
- NO CSV reports generated
- NO figures saved
- All data viewed interactively in console
- For automated reports, use test_live_data_analysis.py or automated_live_data_reporter.py

💡 Usage:
Run interactively in Jupyter notebook or Python console:
python test_live_manual.py

Or use with IPython for cell-by-cell execution

Updated: 2025-08-18
- Created for active/live customer data loading
- Added comparison with training data
- Enhanced for interactive exploration only
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-18
# * CREATED ON: 2025-08-18
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# %%
# Suppress known warnings before any imports
from churn_aiml.utils.suppress_warnings import suppress_known_warnings


import pandas as pd
import numpy as np
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
# %%
# Import required modules
from churn_aiml.data.db.snowflake.loaddata import SnowLiveDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.visualization.churn_plots import LiveCustomerVizSnowflake
from churn_aiml.utils.profiling import timer
import yaml
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
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
# Setup logger with local directory for logs
logger_config = setup_logger_for_script(cfg, __file__)
logger = get_logger()
logger.info("=" * 80)
logger.info("Starting Live/Active Customer Data Loading Example")
logger.info("=" * 80)

# Load and display date configuration
dates_config_path = Path('/home/applaimlgen/ricoh_aiml/conf/products/DOCUWARE/db/snowflake/data_config/dates_config.yaml')
if dates_config_path.exists():
    with open(dates_config_path, 'r') as f:
        dates_config = yaml.safe_load(f)

    logger.info("\n📅 Date Configuration for Live Data:")
    logger.info(f"  Analysis Start Date: {dates_config['analysis_start_date']}")
    logger.info(f"  Latest Data Update: {dates_config['data_update_dates'][-1]['date']}")
    logger.info("  ℹ️ Only active customers with contracts >= 2020-01-01 included")
    logger.info("  ℹ️ Red dashed lines in plots indicate data update dates")

# %%
# Initialize the new data loader
logger.info("\nInitializing SnowLiveDataLoader for active customers")
live_data_loader = SnowLiveDataLoader(config=cfg, environment="development")
# %%
# Load the live/active customer data
logger.info("Loading active customer data from Snowflake")
logger.info("This may take several minutes depending on data size...")
with timer():
    live_data = live_data_loader.load_data()
# %%
# Display loaded data information
print("\n" + "="*80)
print("LOADED DATA SUMMARY - ACTIVE CUSTOMERS")
print("="*80)

for key, df in live_data.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        print(f"\n📊 {key}:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {', '.join(df.columns[:5])}")
        if len(df.columns) > 5:
            print(f"           ... and {len(df.columns) - 5} more columns")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        if 'CUST_ACCOUNT_NUMBER' in df.columns:
            unique_customers = df['CUST_ACCOUNT_NUMBER'].nunique()
            logger.info(f"   Unique customers: {unique_customers}")
# %%
# Time Series Features Analysis for Active Customers
if 'time_series_features' in live_data:
    print("\n" + "="*80)
    print("TIME SERIES FEATURES ANALYSIS - ACTIVE CUSTOMERS")
    print("="*80)

    ts_features = live_data['time_series_features']
    print(f"\nTime series data shape: {ts_features.shape}")
    print(f"Unique active customers: {ts_features['CUST_ACCOUNT_NUMBER'].nunique()}")
    print(f"Date range: {ts_features['YYYYWK'].min()} to {ts_features['YYYYWK'].max()}")

    print("\nSample of time series data:")
    cols_to_show = ['CUST_ACCOUNT_NUMBER', 'YYYYWK', 'DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'CHURNED_FLAG']
    available_cols = [col for col in cols_to_show if col in ts_features.columns]
    print(ts_features[available_cols].head(10).to_string())


    # Average usage statistics
    usage_stats = ts_features.groupby('CUST_ACCOUNT_NUMBER').agg({
        'DOCUMENTS_OPENED': ['mean', 'sum'],
        'USED_STORAGE_MB': ['mean', 'sum'],
        'YYYYWK': 'count'
    })

    logger.info("\nUsage Statistics for Active Customers:")
    logger.info(f"  Avg documents/week/customer: {usage_stats[('DOCUMENTS_OPENED', 'mean')].mean():.2f}")
    logger.info(f"  Avg storage MB/week/customer: {usage_stats[('USED_STORAGE_MB', 'mean')].mean():.2f}")
    logger.info(f"  Avg weeks of data/customer: {usage_stats[('YYYYWK', 'count')].mean():.1f}")

# %%
# Static Features Analysis for Active Customers
if 'static_features' in live_data:
    print("\n" + "="*80)
    print("STATIC FEATURES ANALYSIS - ACTIVE CUSTOMERS")
    print("="*80)

    static_features = live_data['static_features']
    print(f"\nStatic features shape: {static_features.shape}")

    # Check for missing values
    missing_counts = static_features.isnull().sum()
    if missing_counts.any():
        print("\nMissing values:")
        for col, count in missing_counts[missing_counts > 0].items():
            pct = (count / len(static_features)) * 100
            print(f"   {col}: {count} ({pct:.1f}%)")

    print("\nData types:")
    print(static_features.dtypes)

    # Check for elapsed time features
    if 'DAYS_ELAPSED' in static_features.columns:
        logger.info(f"\nCustomer Tenure (Days Elapsed):")
        logger.info(f"  Mean: {static_features['DAYS_ELAPSED'].mean():.1f} days")
        logger.info(f"  Median: {static_features['DAYS_ELAPSED'].median():.1f} days")
        logger.info(f"  Min: {static_features['DAYS_ELAPSED'].min():.1f} days")
        logger.info(f"  Max: {static_features['DAYS_ELAPSED'].max():.1f} days")

    if 'MONTHS_ELAPSED' in static_features.columns:
        logger.info(f"\nCustomer Tenure (Months Elapsed):")
        logger.info(f"  Mean: {static_features['MONTHS_ELAPSED'].mean():.1f} months")
        logger.info(f"  Median: {static_features['MONTHS_ELAPSED'].median():.1f} months")

    # Risk score analysis for active customers
    risk_cols = ['PROBABILITY_OF_DELINQUENCY', 'RICOH_CUSTOM_RISK_MODEL']
    for col in risk_cols:
        if col in static_features.columns:
            non_null = static_features[col].notna()
            if non_null.any():
                logger.info(f"\n{col}:")
                logger.info(f"  Coverage: {non_null.sum()}/{len(static_features)} ({non_null.mean()*100:.1f}%)")
                logger.info(f"  Mean: {static_features[col].mean():.2f}")
                logger.info(f"  Std: {static_features[col].std():.2f}")
# %%
# Customer Metadata Analysis
if 'customer_metadata' in live_data:
    print("\n" + "="*80)
    print("CUSTOMER METADATA ANALYSIS - ACTIVE CUSTOMERS")
    print("="*80)

    metadata = live_data['customer_metadata']
    print(f"\nTotal active customers: {len(metadata)}")

    # All should be active (CHURNED_FLAG == 0)
    if 'CHURNED_FLAG' in metadata.columns:
        active_count = (metadata['CHURNED_FLAG'] == 0).sum()
        print(f"\nConfirmed active customers: {active_count}")

    # Segment analysis for active customers
    if 'CUSTOMER_SEGMENT' in metadata.columns:
        segment_stats = metadata['CUSTOMER_SEGMENT'].value_counts().head(10)
        print("\nTop customer segments (active only):")
        for segment, count in segment_stats.items():
            pct = (count / len(metadata)) * 100
            print(f"   {segment}: {count} ({pct:.1f}%)")

# %%
# Data Preparation for Feature Engineering - Active Customers
logger.info("\n" + "="*60)
logger.info("DATA PREPARATION FOR FEATURE ENGINEERING - ACTIVE CUSTOMERS")
logger.info("="*60)

fe_dataset = live_data_loader.get_feature_engineering_dataset()

if not fe_dataset.empty:
    logger.info(f"Data prepared for feature engineering: {fe_dataset.shape}")
    logger.info(f"Unique active customers: {fe_dataset['CUST_ACCOUNT_NUMBER'].nunique()}")
    logger.info(f"Date range: {fe_dataset['YYYYWK'].min()} to {fe_dataset['YYYYWK'].max()}")

    # Check for missing values (preserved as-is)
    missing_info = fe_dataset.isnull().sum()
    if missing_info.any():
        logger.info("\nMissing values preserved for feature engineering tools:")
        for col, count in missing_info[missing_info > 0].items():
            pct = (count / len(fe_dataset)) * 100
            logger.info(f"  {col}: {count} ({pct:.1f}%) missing values")
    else:
        logger.info("\nNo missing values in dataset")

    # Sample of the dataset
    logger.info("\nSample of data prepared for feature engineering:")
    sample_cols = ['CUST_ACCOUNT_NUMBER', 'YYYYWK', 'DOCUMENTS_OPENED',
                   'USED_STORAGE_MB']
    if 'DAYS_ELAPSED' in fe_dataset.columns:
        sample_cols.append('DAYS_ELAPSED')

    available_cols = [col for col in sample_cols if col in fe_dataset.columns]
    logger.info(fe_dataset[available_cols].head(10).to_string())
# %%
# Get Active Customer Summary
logger.info("\n" + "="*60)
logger.info("ACTIVE CUSTOMER SUMMARY STATISTICS")
logger.info("="*60)

active_summary = live_data_loader.get_active_summary()

if not active_summary.empty:
    logger.info("\nOverall Statistics:")
    logger.info(f"  Total customers: {active_summary['total_customers'].iloc[0]}")
    logger.info(f"  Active customers: {active_summary['active_customers'].iloc[0]}")
    logger.info(f"  Churned customers: {active_summary['churned_customers'].iloc[0]}")
    logger.info(f"  Active rate: {active_summary['active_rate_pct'].iloc[0]:.1f}%")

    # Top segments for active customers
    for i in range(1, 6):
        if f'segment_{i}_name' in active_summary.columns:
            name = active_summary[f'segment_{i}_name'].iloc[0]
            count = active_summary[f'segment_{i}_count'].iloc[0]
            pct = active_summary[f'segment_{i}_pct'].iloc[0]
            logger.info(f"  Top segment {i}: {name} - {count} customers ({pct:.1f}%)")
# %%
# Initialize visualization for active customers
logger.info("\n" + "="*60)
logger.info("INITIALIZING VISUALIZATION FOR ACTIVE CUSTOMERS")
logger.info("="*60)

viz = LiveCustomerVizSnowflake(figsize_scale=1.0)
logger.info("Live customer visualization class initialized for active customer analysis")

# Create sample visualizations for interactive exploration (DISPLAY ONLY - NO SAVING)
logger.info("\\nCreating live customer visualizations for interactive viewing...")

# Generate live customer distribution dashboard (interactive display only)
if 'customer_metadata' in live_data and not live_data['customer_metadata'].empty:
    logger.info("Creating active customer distribution dashboard for interactive viewing...")
    try:
        fig1 = viz.plot_active_customer_distribution(
            data=live_data['customer_metadata']
            # No save_path - display only
        )
        logger.info("✅ Active customer distribution dashboard displayed")
    except Exception as e:
        logger.warning(f"Could not create customer distribution: {e}")

# Generate live usage trends (interactive display only)
if 'time_series_features' in live_data and not live_data['time_series_features'].empty:
    logger.info("Creating live usage trends visualization for interactive viewing...")
    try:
        fig2 = viz.plot_live_usage_trends(
            data=live_data['time_series_features']
            # No save_path - display only
        )
        logger.info("✅ Live usage trends visualization displayed")
    except Exception as e:
        logger.warning(f"Could not create usage trends: {e}")

# Generate customer engagement analysis (interactive display only)
if 'time_series_features' in live_data and not live_data['time_series_features'].empty:
    logger.info("Creating customer engagement analysis for interactive viewing...")
    try:
        fig3 = viz.plot_live_customer_engagement(
            data=live_data['time_series_features']
            # No save_path - display only
        )
        logger.info("✅ Customer engagement analysis displayed")
    except Exception as e:
        logger.warning(f"Could not create engagement analysis: {e}")

# Generate ML readiness dashboard (interactive display only)
logger.info("Creating ML readiness dashboard for interactive viewing...")
try:
    fig4 = viz.plot_live_readiness_dashboard(
        data_dict=live_data
        # No save_path - display only
    )
    logger.info("✅ ML readiness dashboard displayed")
except Exception as e:
    logger.warning(f"Could not create readiness dashboard: {e}")

logger.info("\\n📊 Live customer visualizations displayed for interactive exploration")
logger.info("Live customer visualization demonstrations completed (interactive mode)")
# %%
# Numeric Features Analysis
logger.info("\n" + "="*60)
logger.info("NUMERIC FEATURES ANALYSIS - ACTIVE CUSTOMERS")
logger.info("="*60)

if 'feature_engineering_dataset' in live_data:
    fe_data = live_data['feature_engineering_dataset']
    numeric_cols = fe_data.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
        logger.info(f"Found {len(numeric_cols)} numeric columns")

        # Statistical summary for key numeric features
        key_numeric = ['DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'DAYS_ELAPSED', 'MONTHS_ELAPSED']
        available_numeric = [col for col in key_numeric if col in numeric_cols]

        for col in available_numeric:
            non_zero = fe_data[fe_data[col] > 0][col]
            if len(non_zero) > 0:
                logger.info(f"\n{col}:")
                logger.info(f"  Mean: {non_zero.mean():.2f}")
                logger.info(f"  Median: {non_zero.median():.2f}")
                logger.info(f"  Std: {non_zero.std():.2f}")
                logger.info(f"  Non-zero count: {len(non_zero)}/{len(fe_data[col])} ({len(non_zero)/len(fe_data[col])*100:.1f}%)")
# %%
# Generate comprehensive time series visualization for live data
logger.info("\n" + "="*60)
logger.info("COMPREHENSIVE TIME SERIES VISUALIZATION FOR LIVE DATA")
logger.info("="*60)

# Import the new comprehensive visualization function
from churn_aiml.visualization.churn_plots.comprehensive_time_series import plot_comprehensive_time_series_new

if 'time_series_features' in live_data and not live_data['time_series_features'].empty:
    logger.info("Creating comprehensive time series visualization for live customers...")
    try:
        fig_comprehensive = plot_comprehensive_time_series_new(
            data=live_data['time_series_features'],
            save_path=None,  # Don't save in interactive mode
            figsize_scale=1.2
        )
        logger.info("✅ Comprehensive time series visualization created for live data")
        logger.info("   Features:")
        logger.info("   - Blue scatter plots (no red/green/orange shapes)")
        logger.info("   - LOESS trends with green/red segments")
        logger.info("   - Vertical lines for data update dates")
        logger.info("   - Title color coding for trend health")
        logger.info("   - Negative values automatically dropped")
    except Exception as e:
        logger.warning(f"Could not create comprehensive visualization: {e}")

# %%
# Memory and performance summary
logger.info("\n" + "="*60)
logger.info("MEMORY AND PERFORMANCE SUMMARY")
logger.info("="*60)

total_memory = 0
for key, df in live_data.items():
    if isinstance(df, pd.DataFrame):
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        total_memory += memory_mb
        logger.info(f"{key}: {memory_mb:.2f} MB")

logger.info(f"\nTotal memory usage: {total_memory:.2f} MB")
# %%
logger.info("\n" + "="*60)
logger.info("INTERACTIVE EXPLORATION COMPLETE")
logger.info("="*60)
logger.info("Data is loaded and ready for interactive analysis")
logger.info("Access data through: live_data_loader.processed_data")
# %%

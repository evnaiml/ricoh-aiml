#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
"""
Comprehensive Train/New Data Flag Analysis and Reporting

This script performs detailed analysis of the new_data flag (0=train, 1=new)
across all datasets and generates comprehensive reports in both logs and CSV files.

✅ Key Features:
- Detailed analysis of train (0) vs new (1) data distribution
- Customer-level and record-level statistics
- Automated CSV report generation
- Comprehensive logging with statistics
- Time-based analysis of new data arrival

📊 Reports Generated:
1. live_data_summary_report.csv - Overall summary statistics
2. live_data_by_dataset.csv - Breakdown by dataset
3. live_data_by_customer_segment.csv - Segment analysis
4. live_data_time_series.csv - Time-based distribution

💡 Usage:
python test_live_data_analysis.py
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
from datetime import datetime
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
output_dir = Path(__file__).parent / "reports"
output_dir.mkdir(exist_ok=True)

print(f"Config path: {conf_dir}")
print(f"Output path: {output_dir}")

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
logger.info("TRAIN/NEW DATA FLAG ANALYSIS AND REPORTING")
logger.info("=" * 80)
logger.info(f"Report directory: {output_dir}")

# %%
def analyze_live_data_flag(df, dataset_name, logger):
    """
    Analyze the new_data flag in a dataframe and log statistics.
    
    Args:
        df: DataFrame to analyze
        dataset_name: Name of the dataset for logging
        logger: Logger instance
    
    Returns:
        dict: Analysis results
    """
    results = {
        'dataset': dataset_name,
        'total_records': len(df),
        'has_live_data_flag': False,
        'new_records': 0,
        'train_records': 0,
        'new_percentage': 0.0,
        'train_percentage': 0.0,
        'new_customers': 0,
        'train_customers': 0,
        'new_customers_pct': 0.0,
        'train_customers_pct': 0.0
    }
    
    if 'new_data' not in df.columns:
        logger.warning(f"  ⚠️ Dataset '{dataset_name}' does not have 'new_data' column")
        return results
    
    results['has_live_data_flag'] = True
    
    # Record-level analysis
    new_mask = df['new_data'] == 1
    train_mask = df['new_data'] == 0
    
    results['new_records'] = new_mask.sum()
    results['train_records'] = train_mask.sum()
    results['new_percentage'] = (results['new_records'] / results['total_records'] * 100) if results['total_records'] > 0 else 0
    results['train_percentage'] = (results['train_records'] / results['total_records'] * 100) if results['total_records'] > 0 else 0
    
    # Customer-level analysis if CUST_ACCOUNT_NUMBER exists
    if 'CUST_ACCOUNT_NUMBER' in df.columns:
        new_customers = df[new_mask]['CUST_ACCOUNT_NUMBER'].nunique()
        train_customers = df[train_mask]['CUST_ACCOUNT_NUMBER'].nunique()
        total_customers = df['CUST_ACCOUNT_NUMBER'].nunique()
        
        results['new_customers'] = new_customers
        results['train_customers'] = train_customers
        results['new_customers_pct'] = (new_customers / total_customers * 100) if total_customers > 0 else 0
        results['train_customers_pct'] = (train_customers / total_customers * 100) if total_customers > 0 else 0
    
    # Log the analysis
    logger.info(f"\n📊 {dataset_name} Analysis:")
    logger.info(f"  Total records: {results['total_records']:,}")
    logger.info(f"  ├─ Training data (0): {results['train_records']:,} ({results['train_percentage']:.1f}%)")
    logger.info(f"  └─ New data (1): {results['new_records']:,} ({results['new_percentage']:.1f}%)")
    
    if results['new_customers'] > 0 or results['train_customers'] > 0:
        logger.info(f"  Customer breakdown:")
        logger.info(f"  ├─ Training customers: {results['train_customers']:,} ({results['train_customers_pct']:.1f}%)")
        logger.info(f"  └─ Live customers: {results['new_customers']:,} ({results['new_customers_pct']:.1f}%)")
    
    return results

# %%
# STEP 1: Load Training Data
logger.info("\nSTEP 1: Loading training data for baseline")
logger.info("-" * 60)

train_data_loader = SnowTrainDataLoader(config=cfg, environment="development")

with timer():
    train_data = train_data_loader.load_data()

train_customer_count = 0
if 'customer_metadata' in train_data:
    train_customer_count = len(train_data['customer_metadata'])
    
logger.info(f"✅ Training data loaded:")
logger.info(f"  - Datasets: {len(train_data)}")
logger.info(f"  - Total customers: {train_customer_count:,}")

# %%
# STEP 2: Load New/Active Customer Data
logger.info("\nSTEP 2: Loading live/active customer data with comparison")
logger.info("-" * 60)

live_data_loader = SnowLiveDataLoader(config=cfg, environment="development")

with timer():
    new_data = live_data_loader.load_data(train_data_loader=train_data_loader)

logger.info(f"✅ New/active data loaded with {len(new_data)} datasets")

# %%
# STEP 3: Analyze Each Dataset
logger.info("\nSTEP 3: Analyzing new_data flag in each dataset")
logger.info("-" * 60)

all_results = []

for dataset_name, df in new_data.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        result = analyze_live_data_flag(df, dataset_name, logger)
        all_results.append(result)

# %%
# STEP 4: Create Summary Statistics
logger.info("\nSTEP 4: Creating summary statistics")
logger.info("-" * 60)

# Overall summary
total_new_records = sum(r['new_records'] for r in all_results)
total_train_records = sum(r['train_records'] for r in all_results)
total_all_records = sum(r['total_records'] for r in all_results)

# Find unique customers across all datasets
all_new_customers = set()
all_train_customers = set()

for dataset_name, df in new_data.items():
    if isinstance(df, pd.DataFrame) and 'new_data' in df.columns and 'CUST_ACCOUNT_NUMBER' in df.columns:
        new_custs = df[df['new_data'] == 1]['CUST_ACCOUNT_NUMBER'].unique()
        train_custs = df[df['new_data'] == 0]['CUST_ACCOUNT_NUMBER'].unique()
        all_new_customers.update(new_custs)
        all_train_customers.update(train_custs)

unique_new_customers = len(all_new_customers)
unique_train_customers = len(all_train_customers)
overlap_customers = len(all_new_customers.intersection(all_train_customers))

logger.info("📈 OVERALL SUMMARY:")
logger.info(f"  Total records across all datasets: {total_all_records:,}")
logger.info(f"  ├─ Training records (0): {total_train_records:,} ({total_train_records/total_all_records*100:.1f}%)")
logger.info(f"  └─ New records (1): {total_new_records:,} ({total_new_records/total_all_records*100:.1f}%)")
logger.info(f"\n  Unique customers:")
logger.info(f"  ├─ Training customers only: {unique_train_customers - overlap_customers:,}")
logger.info(f"  ├─ Live customers only: {unique_new_customers - overlap_customers:,}")
logger.info(f"  └─ Customers in both: {overlap_customers:,}")

# %%
# STEP 5: Segment Analysis
logger.info("\nSTEP 5: Analyzing new data by customer segment")
logger.info("-" * 60)

segment_analysis = []

if 'customer_metadata' in new_data:
    metadata = new_data['customer_metadata']
    
    if 'CUSTOMER_SEGMENT' in metadata.columns and 'new_data' in metadata.columns:
        segment_stats = metadata.groupby('CUSTOMER_SEGMENT').agg({
            'new_data': ['sum', 'count', 'mean'],
            'CUST_ACCOUNT_NUMBER': 'nunique'
        }).round(2)
        
        segment_stats.columns = ['new_count', 'total_count', 'new_ratio', 'unique_customers']
        segment_stats['train_count'] = segment_stats['total_count'] - segment_stats['new_count']
        segment_stats['new_percentage'] = (segment_stats['new_ratio'] * 100).round(1)
        segment_stats['train_percentage'] = 100 - segment_stats['new_percentage']
        
        # Log segment analysis
        logger.info("Customer Segment Analysis:")
        for segment in segment_stats.index[:10]:  # Top 10 segments
            row = segment_stats.loc[segment]
            logger.info(f"  {segment}:")
            logger.info(f"    ├─ Total: {row['total_count']:.0f} customers")
            logger.info(f"    ├─ Training: {row['train_count']:.0f} ({row['train_percentage']:.1f}%)")
            logger.info(f"    └─ New: {row['new_count']:.0f} ({row['new_percentage']:.1f}%)")
        
        # Convert to list of dicts for CSV
        for segment in segment_stats.index:
            row = segment_stats.loc[segment]
            segment_analysis.append({
                'segment': segment,
                'total_customers': int(row['unique_customers']),
                'train_count': int(row['train_count']),
                'new_count': int(row['new_count']),
                'train_percentage': row['train_percentage'],
                'new_percentage': row['new_percentage']
            })

# %%
# STEP 6: Time Series Analysis
logger.info("\nSTEP 6: Analyzing new data distribution over time")
logger.info("-" * 60)

time_series_analysis = []

if 'time_series_features' in new_data:
    ts_df = new_data['time_series_features']
    
    if 'YYYYWK' in ts_df.columns and 'new_data' in ts_df.columns:
        # Weekly distribution
        weekly_stats = ts_df.groupby('YYYYWK').agg({
            'new_data': ['sum', 'count', 'mean'],
            'CUST_ACCOUNT_NUMBER': 'nunique'
        }).round(2)
        
        weekly_stats.columns = ['new_count', 'total_count', 'new_ratio', 'unique_customers']
        weekly_stats['train_count'] = weekly_stats['total_count'] - weekly_stats['new_count']
        weekly_stats['new_percentage'] = (weekly_stats['new_ratio'] * 100).round(1)
        
        # Find the transition point (where new data starts appearing)
        first_new_week = ts_df[ts_df['new_data'] == 1]['YYYYWK'].min() if (ts_df['new_data'] == 1).any() else None
        
        if first_new_week:
            logger.info(f"  First week with new data: {first_new_week}")
            
            # Analyze last 10 weeks
            last_weeks = weekly_stats.tail(10)
            logger.info("\n  Last 10 weeks distribution:")
            for week in last_weeks.index:
                row = last_weeks.loc[week]
                logger.info(f"    Week {week}: Train={row['train_count']:.0f}, New={row['new_count']:.0f} ({row['new_percentage']:.1f}% new)")
        
        # Convert to list for CSV
        for week in weekly_stats.index:
            row = weekly_stats.loc[week]
            time_series_analysis.append({
                'week': week,
                'total_records': int(row['total_count']),
                'train_records': int(row['train_count']),
                'new_records': int(row['new_count']),
                'new_percentage': row['new_percentage'],
                'unique_customers': int(row['unique_customers'])
            })

# %%
# STEP 7: Generate CSV Reports
logger.info("\nSTEP 7: Generating CSV reports")
logger.info("-" * 60)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Summary Report
summary_data = [{
    'report_timestamp': timestamp,
    'total_records': total_all_records,
    'train_records': total_train_records,
    'new_records': total_new_records,
    'train_percentage': round(total_train_records/total_all_records*100, 1) if total_all_records > 0 else 0,
    'new_percentage': round(total_new_records/total_all_records*100, 1) if total_all_records > 0 else 0,
    'unique_train_customers': unique_train_customers,
    'unique_new_customers': unique_new_customers,
    'overlapping_customers': overlap_customers,
    'total_unique_customers': len(all_new_customers.union(all_train_customers))
}]

summary_df = pd.DataFrame(summary_data)
summary_path = output_dir / f"live_data_summary_{timestamp}.csv"
summary_df.to_csv(summary_path, index=False)
logger.info(f"  ✅ Saved summary report: {summary_path.name}")

# 2. Dataset-level Report
dataset_df = pd.DataFrame(all_results)
dataset_path = output_dir / f"live_data_by_dataset_{timestamp}.csv"
dataset_df.to_csv(dataset_path, index=False)
logger.info(f"  ✅ Saved dataset report: {dataset_path.name}")

# 3. Segment Analysis Report
if segment_analysis:
    segment_df = pd.DataFrame(segment_analysis)
    segment_path = output_dir / f"live_data_by_segment_{timestamp}.csv"
    segment_df.to_csv(segment_path, index=False)
    logger.info(f"  ✅ Saved segment report: {segment_path.name}")

# 4. Time Series Report
if time_series_analysis:
    ts_df = pd.DataFrame(time_series_analysis)
    ts_path = output_dir / f"live_data_time_series_{timestamp}.csv"
    ts_df.to_csv(ts_path, index=False)
    logger.info(f"  ✅ Saved time series report: {ts_path.name}")

# %%
# STEP 8: Create Detailed Customer List Report
logger.info("\nSTEP 8: Creating detailed customer lists")
logger.info("-" * 60)

# Create lists of new vs training customers
if 'customer_metadata' in new_data:
    metadata = new_data['customer_metadata']
    
    if 'new_data' in metadata.columns:
        # Live customers
        new_customers_df = metadata[metadata['new_data'] == 1][['CUST_ACCOUNT_NUMBER', 'CUSTOMER_SEGMENT']].copy()
        new_customers_df['data_source'] = 'new'
        
        # Training customers
        train_customers_df = metadata[metadata['new_data'] == 0][['CUST_ACCOUNT_NUMBER', 'CUSTOMER_SEGMENT']].copy()
        train_customers_df['data_source'] = 'train'
        
        # Combined list with flag
        all_customers_df = pd.concat([new_customers_df, train_customers_df], ignore_index=True)
        customers_path = output_dir / f"customer_list_{timestamp}.csv"
        all_customers_df.to_csv(customers_path, index=False)
        logger.info(f"  ✅ Saved customer list: {customers_path.name}")
        
        # Sample of live customers for verification
        if len(new_customers_df) > 0:
            logger.info("\n  Sample of NEW customers (first 5):")
            for _, row in new_customers_df.head(5).iterrows():
                logger.info(f"    - {row['CUST_ACCOUNT_NUMBER']} ({row['CUSTOMER_SEGMENT']})")

# %%
# STEP 9: Data Quality Checks
logger.info("\nSTEP 9: Data quality checks for new_data flag")
logger.info("-" * 60)

quality_issues = []

# Check 1: Verify flag values are only 0 or 1
for dataset_name, df in new_data.items():
    if isinstance(df, pd.DataFrame) and 'new_data' in df.columns:
        unique_values = df['new_data'].unique()
        valid_values = set(unique_values) == {0, 1} or set(unique_values) == {0} or set(unique_values) == {1}
        
        if not valid_values:
            quality_issues.append(f"Dataset '{dataset_name}' has invalid new_data values: {unique_values}")
            logger.warning(f"  ⚠️ {dataset_name}: Invalid values {unique_values}")
        else:
            logger.info(f"  ✅ {dataset_name}: Valid flag values {sorted(unique_values)}")

# Check 2: Consistency across datasets for same customer
customer_flag_consistency = {}
for dataset_name, df in new_data.items():
    if isinstance(df, pd.DataFrame) and 'new_data' in df.columns and 'CUST_ACCOUNT_NUMBER' in df.columns:
        customer_flags = df.groupby('CUST_ACCOUNT_NUMBER')['new_data'].first().to_dict()
        
        for cust, flag in customer_flags.items():
            if cust not in customer_flag_consistency:
                customer_flag_consistency[cust] = {}
            customer_flag_consistency[cust][dataset_name] = flag

# Check for inconsistencies
inconsistent_customers = []
for cust, flags in customer_flag_consistency.items():
    unique_flags = set(flags.values())
    if len(unique_flags) > 1:
        inconsistent_customers.append(cust)

if inconsistent_customers:
    logger.warning(f"  ⚠️ Found {len(inconsistent_customers)} customers with inconsistent flags across datasets")
    quality_issues.append(f"Inconsistent flags for {len(inconsistent_customers)} customers")
else:
    logger.info(f"  ✅ All customers have consistent flags across datasets")

# Save quality report
if quality_issues:
    quality_df = pd.DataFrame({'issue': quality_issues})
    quality_path = output_dir / f"data_quality_issues_{timestamp}.csv"
    quality_df.to_csv(quality_path, index=False)
    logger.warning(f"  ⚠️ Saved quality issues report: {quality_path.name}")

# %%
# FINAL SUMMARY
logger.info("\n" + "=" * 80)
logger.info("ANALYSIS COMPLETE - FINAL SUMMARY")
logger.info("=" * 80)

# Calculate key metrics
live_data_ratio = (total_new_records / total_all_records * 100) if total_all_records > 0 else 0
new_customer_ratio = (unique_new_customers / (unique_new_customers + unique_train_customers) * 100) if (unique_new_customers + unique_train_customers) > 0 else 0

logger.info(f"\n📊 KEY METRICS:")
logger.info(f"  Records Distribution:")
logger.info(f"    - Total: {total_all_records:,}")
logger.info(f"    - Training (0): {total_train_records:,} ({100-live_data_ratio:.1f}%)")
logger.info(f"    - New (1): {total_new_records:,} ({live_data_ratio:.1f}%)")

logger.info(f"\n  Customer Distribution:")
logger.info(f"    - Total unique: {len(all_new_customers.union(all_train_customers)):,}")
logger.info(f"    - Training only: {unique_train_customers - overlap_customers:,} ({(unique_train_customers - overlap_customers)/len(all_new_customers.union(all_train_customers))*100:.1f}%)")
logger.info(f"    - New only: {unique_new_customers - overlap_customers:,} ({(unique_new_customers - overlap_customers)/len(all_new_customers.union(all_train_customers))*100:.1f}%)")
logger.info(f"    - Both: {overlap_customers:,} ({overlap_customers/len(all_new_customers.union(all_train_customers))*100:.1f}%)")

logger.info(f"\n📁 REPORTS GENERATED:")
logger.info(f"  Location: {output_dir}")
logger.info(f"  Files created:")
for file in sorted(output_dir.glob(f"*_{timestamp}.csv")):
    logger.info(f"    - {file.name}")

logger.info("\n✅ Analysis complete! Check the reports directory for detailed CSV files.")
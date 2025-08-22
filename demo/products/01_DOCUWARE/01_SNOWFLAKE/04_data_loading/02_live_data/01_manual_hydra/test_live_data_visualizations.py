#!/usr/bin/env python
# -*- coding: utf-8 -*-
# %%
"""
New Data Visualization Script with Automated Figure Generation

This AUTOMATED script generates visualizations for train/new data distribution
and saves them to the figs/ directory for reporting purposes.

✅ Key Features:
- Generates various charts showing train vs new data distribution
- Saves all visualizations to figs/ directory
- Creates timestamped figures for tracking
- Shows vertical dashed lines at transition points
- Uses diamond symbols for new data points
- Completely automated - no interactive viewing

📊 Visualizations Generated:
1. Time series plot with train/new data markers
2. Bar chart of data distribution by dataset
3. Pie chart of overall train/new split
4. Segment-wise distribution charts
5. Weekly trend analysis

💡 Usage:
python test_live_data_visualizations.py

This script is meant for AUTOMATED runs - it saves figures to disk.
For interactive exploration, use test_new_manual.py instead.
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
import matplotlib.pyplot as plt
import seaborn as sns
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

# Create figs directory for saving visualizations
figs_dir = Path(__file__).parent / "figs"
figs_dir.mkdir(exist_ok=True)

print(f"Config path: {conf_dir}")
print(f"Figures will be saved to: {figs_dir}")

# %%
# Setup visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

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
logger.info("NEW DATA VISUALIZATION GENERATOR")
logger.info("=" * 80)
logger.info(f"Figures directory: {figs_dir}")

# %%
# Load training data
logger.info("Loading training data for comparison...")
train_data_loader = SnowTrainDataLoader(config=cfg, environment="development")

with timer():
    train_data = train_data_loader.load_data()

logger.info(f"Training data loaded with {len(train_data)} datasets")

# %%
# Load live/active customer data
logger.info("Loading live/active customer data...")
live_data_loader = SnowLiveDataLoader(config=cfg, environment="development")

with timer():
    new_data = live_data_loader.load_data(train_data_loader=train_data_loader)

logger.info(f"New data loaded with {len(new_data)} datasets")

# %%
# Generate timestamp for file naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# %%
# VISUALIZATION 1: Overall Train/New Distribution Pie Chart
logger.info("\nGenerating overall distribution pie chart...")

total_new = 0
total_train = 0

for dataset_name, df in new_data.items():
    if isinstance(df, pd.DataFrame) and 'new_data' in df.columns:
        total_new += (df['new_data'] == 1).sum()
        total_train += (df['new_data'] == 0).sum()

if total_new + total_train > 0:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = ['Training Data', 'New Data']
    sizes = [total_train, total_new]
    colors = ['#3498db', '#e74c3c']
    explode = (0, 0.1)  # Explode new data slice
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                       autopct='%1.1f%%', startangle=90,
                                       explode=explode, shadow=True)
    
    # Enhance text
    for text in texts:
        text.set_fontsize(12)
        text.set_weight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    ax.set_title('Overall Distribution: Training vs New Data', fontsize=14, weight='bold')
    
    # Add counts to legend
    legend_labels = [f'{label}: {size:,} records' for label, size in zip(labels, sizes)]
    ax.legend(legend_labels, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    fig_path = figs_dir / f"overall_distribution_pie_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✅ Saved: {fig_path.name}")

# %%
# VISUALIZATION 2: Time Series Plot with Train/New Markers
logger.info("\nGenerating time series visualization...")

if 'time_series_features' in new_data:
    ts_df = new_data['time_series_features']
    
    if 'YYYYWK' in ts_df.columns and 'new_data' in ts_df.columns:
        # Aggregate by week
        weekly_agg = ts_df.groupby(['YYYYWK', 'new_data']).agg({
            'CUST_ACCOUNT_NUMBER': 'nunique',
            'DOCUMENTS_OPENED': 'mean'
        }).reset_index()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot 1: Customer counts
        train_weeks = weekly_agg[weekly_agg['new_data'] == 0]
        new_weeks = weekly_agg[weekly_agg['new_data'] == 1]
        
        ax1.plot(train_weeks['YYYYWK'], train_weeks['CUST_ACCOUNT_NUMBER'], 
                'o-', label='Training Customers', color='#3498db', linewidth=2, markersize=6)
        
        if not new_weeks.empty:
            # Use diamond markers for new data
            ax1.plot(new_weeks['YYYYWK'], new_weeks['CUST_ACCOUNT_NUMBER'], 
                    'D-', label='New Customers', color='#e74c3c', linewidth=2, markersize=8)
            
            # Add vertical dashed line at first new data week
            first_new_week = new_weeks['YYYYWK'].min()
            ax1.axvline(x=first_new_week, color='#2ecc71', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'First New Data: Week {first_new_week}')
        
        ax1.set_ylabel('Number of Unique Customers', fontsize=11)
        ax1.set_title('Weekly Customer Distribution: Training vs New', fontsize=13, weight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average usage
        ax2.plot(train_weeks['YYYYWK'], train_weeks['DOCUMENTS_OPENED'], 
                'o-', label='Training Avg Usage', color='#3498db', linewidth=2, markersize=6)
        
        if not new_weeks.empty:
            ax2.plot(new_weeks['YYYYWK'], new_weeks['DOCUMENTS_OPENED'], 
                    'D-', label='New Avg Usage', color='#e74c3c', linewidth=2, markersize=8)
            
            # Add vertical dashed line
            ax2.axvline(x=first_new_week, color='#2ecc71', linestyle='--', 
                       linewidth=2, alpha=0.7)
        
        ax2.set_xlabel('Week (YYYYWK)', fontsize=11)
        ax2.set_ylabel('Avg Documents Opened', fontsize=11)
        ax2.set_title('Weekly Average Usage: Training vs New', fontsize=13, weight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Time Series Analysis of Train vs New Data', fontsize=14, weight='bold', y=1.02)
        plt.tight_layout()
        
        fig_path = figs_dir / f"time_series_train_new_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✅ Saved: {fig_path.name}")

# %%
# VISUALIZATION 3: Dataset-wise Distribution Bar Chart
logger.info("\nGenerating dataset distribution bar chart...")

dataset_stats = []
for dataset_name, df in new_data.items():
    if isinstance(df, pd.DataFrame) and 'new_data' in df.columns:
        stats = {
            'dataset': dataset_name,
            'train': (df['new_data'] == 0).sum(),
            'new': (df['new_data'] == 1).sum()
        }
        dataset_stats.append(stats)

if dataset_stats:
    stats_df = pd.DataFrame(dataset_stats)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(stats_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, stats_df['train'], width, label='Training Data', color='#3498db')
    bars2 = ax.bar(x + width/2, stats_df['new'], width, label='New Data', color='#e74c3c')
    
    ax.set_xlabel('Dataset', fontsize=11)
    ax.set_ylabel('Number of Records', fontsize=11)
    ax.set_title('Train vs New Data Distribution by Dataset', fontsize=13, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['dataset'], rotation=45, ha='right')
    ax.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height):,}', ha='center', va='bottom', fontsize=9)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    fig_path = figs_dir / f"dataset_distribution_bar_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✅ Saved: {fig_path.name}")

# %%
# VISUALIZATION 4: Customer Segment Distribution
logger.info("\nGenerating customer segment visualization...")

if 'customer_metadata' in new_data:
    metadata = new_data['customer_metadata']
    
    if 'CUSTOMER_SEGMENT' in metadata.columns and 'new_data' in metadata.columns:
        segment_stats = metadata.groupby(['CUSTOMER_SEGMENT', 'new_data']).size().unstack(fill_value=0)
        
        # Get top 10 segments by total customers
        segment_totals = segment_stats.sum(axis=1).sort_values(ascending=False).head(10)
        top_segments = segment_stats.loc[segment_totals.index]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_segments.plot(kind='bar', stacked=True, ax=ax, 
                         color=['#3498db', '#e74c3c'])
        
        ax.set_xlabel('Customer Segment', fontsize=11)
        ax.set_ylabel('Number of Customers', fontsize=11)
        ax.set_title('Customer Distribution by Segment: Training vs New', fontsize=13, weight='bold')
        ax.legend(['Training', 'New'], title='Data Type')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        
        fig_path = figs_dir / f"segment_distribution_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✅ Saved: {fig_path.name}")

# %%
# VISUALIZATION 5: Weekly Trend Analysis
logger.info("\nGenerating weekly trend analysis...")

if 'time_series_features' in new_data:
    ts_df = new_data['time_series_features']
    
    if 'YYYYWK' in ts_df.columns and 'new_data' in ts_df.columns:
        # Calculate weekly percentages
        weekly_pct = ts_df.groupby('YYYYWK')['new_data'].agg(['sum', 'count'])
        weekly_pct['percentage'] = (weekly_pct['sum'] / weekly_pct['count'] * 100).round(1)
        weekly_pct = weekly_pct.tail(20)  # Last 20 weeks
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        bars = ax.bar(range(len(weekly_pct)), weekly_pct['percentage'], 
                     color=['#3498db' if p == 0 else '#e74c3c' if p > 50 else '#f39c12' 
                           for p in weekly_pct['percentage']])
        
        ax.set_xlabel('Week', fontsize=11)
        ax.set_ylabel('New Data Percentage (%)', fontsize=11)
        ax.set_title('Weekly New Data Percentage Trend (Last 20 Weeks)', fontsize=13, weight='bold')
        ax.set_xticks(range(len(weekly_pct)))
        ax.set_xticklabels(weekly_pct.index, rotation=45, ha='right')
        
        # Add horizontal line at 50%
        ax.axhline(y=50, color='#95a5a6', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(len(weekly_pct)-1, 51, '50% threshold', ha='right', va='bottom', 
               fontsize=9, color='#95a5a6')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, weekly_pct['percentage'])):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., val,
                       f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
        
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
        plt.tight_layout()
        
        fig_path = figs_dir / f"weekly_trend_{timestamp}.png"
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  ✅ Saved: {fig_path.name}")

# %%
# Create summary report of generated figures
logger.info("\nCreating figure summary report...")

summary_data = {
    'timestamp': timestamp,
    'figures_generated': [],
    'total_train_records': total_train,
    'total_new_records': total_new,
    'new_percentage': round(total_new/(total_train+total_new)*100, 1) if (total_train+total_new) > 0 else 0
}

# List all generated figures
for fig_file in sorted(figs_dir.glob(f"*_{timestamp}.png")):
    summary_data['figures_generated'].append(fig_file.name)

# Save summary
summary_file = figs_dir / f"figure_summary_{timestamp}.txt"
with open(summary_file, 'w') as f:
    f.write(f"Figure Generation Summary\n")
    f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Data Statistics:\n")
    f.write(f"  - Training records: {summary_data['total_train_records']:,}\n")
    f.write(f"  - New records: {summary_data['total_new_records']:,}\n")
    f.write(f"  - New data percentage: {summary_data['new_percentage']}%\n\n")
    f.write(f"Figures Generated ({len(summary_data['figures_generated'])}):\n")
    for fig in summary_data['figures_generated']:
        f.write(f"  - {fig}\n")

logger.info(f"  ✅ Saved summary: {summary_file.name}")

# %%
# Final summary
logger.info("\n" + "="*80)
logger.info("VISUALIZATION GENERATION COMPLETE")
logger.info("="*80)
logger.info(f"📊 Generated {len(summary_data['figures_generated'])} visualizations")
logger.info(f"📁 All figures saved to: {figs_dir}")
logger.info(f"📝 Summary report: {summary_file.name}")
logger.info("\n✅ Script completed successfully!")
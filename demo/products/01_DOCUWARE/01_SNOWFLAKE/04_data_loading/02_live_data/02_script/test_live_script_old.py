"""
Development-Grade Automated Live/Active Customer Data Loading Script with Analysis

This development script provides automated data loading and analysis capabilities
for active (non-churned) customers with extensive visualization and reporting features.
It leverages the SnowLiveDataLoader class to process active customer data with detailed
logging, quality checks, comparison with training data, and comprehensive output generation.

✅ Development Features:
- Automated data loading from Snowflake for active customers only
- Feature engineering dataset creation for inferencing
- Comparison with training data to identify live customers not in training
- Comprehensive data quality analysis and validation
- Multiple visualization outputs with train/live data indicators
- Detailed JSON and CSV reports for sharing with analysts
- Performance metrics and memory usage monitoring
- Extensive logging with Hydra configuration

📊 Data Processing Pipeline:
- Raw data ingestion with type validation (active customers only)
- Usage data processing WITHOUT imputation
- Multi-dataset merging and transformation
- Time series feature preparation at weekly granularity
- Static feature aggregation and encoding
- Live customer identification through comparison

🔍 Analysis & Visualization:
- Active customer distribution analysis
- Time series usage trends with train/live separation
- Customer tenure and lifecycle analysis
- Engagement pattern visualization for active customers
- Live vs existing customer comparisons
- Comprehensive monitoring dashboards

📝 Output Organization:
- figs/: High-quality visualization exports (300 DPI)
  * active_distribution_[timestamp].png
  * usage_trends_live_[timestamp].png
  * customer_analysis_live_[timestamp].png
  * train_vs_live_comparison_[timestamp].png
- reports/: Structured data reports
  * live_data_loading_report_[timestamp].json
  * live_data_summary_[timestamp].csv
  * sample_live_[dataset]_[timestamp].csv
  * comparison_report_[timestamp].json

💡 Usage:
python test_live_script.py

Or with Hydra overrides:
python test_live_script.py product=DOCUWARE debug=true

Updated: 2025-08-18
- Created for active/live customer data processing
- Enhanced visualization with train/new indicators
- Added comparison reports between datasets
- Integrated live customer identification
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

from pathlib import Path
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from churn_aiml.data.db.snowflake.loaddata import SnowLiveDataLoader, SnowTrainDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger
from churn_aiml.visualization.churn_plots import LiveCustomerVizSnowflake


@hydra.main(config_path="/home/applaimlgen/ricoh_aiml/conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to demonstrate live/active customer data loading with detailed logging.
    
    Args:
        cfg: Hydra configuration object
    """
    # Setup logger with local directory for logs
    logger_config = setup_logger_for_script(cfg, __file__)
    logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("STARTING AUTOMATED LIVE/ACTIVE CUSTOMER DATA LOADING SCRIPT")
    logger.info("=" * 80)
    
    # Setup output directories
    script_dir = Path(__file__).parent
    figs_dir = script_dir / "figs"
    reports_dir = script_dir / "reports"
    
    # Create directories if they don't exist
    figs_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    logger.info(f"Figures will be saved to: {figs_dir}")
    logger.info(f"Reports will be saved to: {reports_dir}")
    
    # Initialize live customer visualization class
    viz = LiveCustomerVizSnowflake()
    
    try:
        # Initialize the training data loader for comparison
        logger.info("Initializing SnowTrainDataLoader for comparison...")
        train_loader = SnowTrainDataLoader(config=cfg, environment="development")
        
        # Load training data
        logger.info("Loading training data for comparison...")
        
        print("\n" + "─" * 80)
        train_data_dict = train_loader.load_data()
        print("─" * 80 + "\n")
        
        logger.info("Training data loading completed successfully!")
        
        # Initialize the new data loader
        logger.info("Initializing SnowLiveDataLoader for active customers...")
        live_loader = SnowLiveDataLoader(config=cfg, environment="development")
        
        # Load the live/active customer data with comparison
        logger.info("Beginning live/active customer data loading process...")
        logger.info("This will execute the following steps:")
        logger.info("  1. Load raw data from Snowflake (active customers only)")
        logger.info("  2. Process usage data WITHOUT imputation")
        logger.info("  3. Merge datasets")
        logger.info("  4. Prepare time series features")
        logger.info("  5. Prepare static features")
        logger.info("  6. Calculate derived features")
        
        print("\n" + "─" * 80)
        data_dict = live_loader.load_data()
        print("─" * 80 + "\n")
        
        logger.info("New/active data loading completed successfully!")
        
        # Detailed analysis of loaded data
        print("\n" + "=" * 80)
        print("📊 LIVE/ACTIVE DATA LOADING RESULTS")
        print("=" * 80)
        
        total_memory = 0
        total_rows = 0
        
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame):
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                total_memory += memory_mb
                total_rows += len(df)
                
                print(f"\n📁 {key.upper()}:")
                print(f"   Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
                print(f"   Memory usage: {memory_mb:.2f} MB")
                
                # Show column information
                print(f"   Column types:")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    print(f"      - {dtype}: {count} columns")
                
                # Check for missing values
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                print(f"   Missing values: {missing_pct:.2f}%")
                
                
                # Show sample columns
                sample_cols = list(df.columns[:5])
                if len(df.columns) > 5:
                    sample_cols.append(f"... +{len(df.columns) - 5} more")
                print(f"   Columns: {', '.join(sample_cols)}")
        
        print("\n" + "─" * 80)
        print(f"📊 TOTAL: {total_rows:,} rows across all datasets")
        print(f"💾 TOTAL MEMORY: {total_memory:.2f} MB")
        print("─" * 80)
        
        # Create feature engineering dataset
        print("\n" + "=" * 80)
        print("🔬 FEATURE ENGINEERING DATASET CREATION")
        print("=" * 80)
        
        logger.info("Creating comprehensive feature engineering dataset for active customers")
        
        # Get the feature engineering dataset for ML models
        fe_dataset = live_loader.get_feature_engineering_dataset()
        
        if not fe_dataset.empty:
            print(f"\n✅ Feature Engineering Dataset Created!")
            print(f"   Total records: {len(fe_dataset):,}")
            print(f"   Unique active customers: {fe_dataset['CUST_ACCOUNT_NUMBER'].nunique():,}")
            print(f"   Time range: {fe_dataset['YYYYWK'].min()} to {fe_dataset['YYYYWK'].max()}")
            
            
            logger.info(f"Feature engineering dataset created with {len(fe_dataset):,} records")
        else:
            logger.warning("Failed to create feature engineering dataset")
            print("\n❌ Feature engineering dataset creation failed")
        
        # Generate active customer summary
        print("\n" + "=" * 80)
        print("📊 GENERATING ACTIVE CUSTOMER SUMMARY")
        print("=" * 80)
        
        logger.info("Creating comprehensive active customer summary")
        active_summary = live_loader.get_active_summary()
        
        if not active_summary.empty:
            summary_dict = active_summary.iloc[0].to_dict()
            
            print(f"\n✅ Active Customer Summary Generated:")
            print(f"   Total Customers: {summary_dict.get('total_customers', 0):,}")
            print(f"   Active Customers: {summary_dict.get('active_customers', 0):,}")
            print(f"   Active Rate: {summary_dict.get('active_rate_pct', 0):.2f}%")
            
            # Show top segments if available
            for i in range(1, 4):
                segment_name = summary_dict.get(f'segment_{i}_name')
                segment_count = summary_dict.get(f'segment_{i}_count')
                segment_pct = summary_dict.get(f'segment_{i}_pct')
                if segment_name:
                    print(f"   Top Segment {i}: {segment_name} - {segment_count:,} ({segment_pct:.1f}%)")
            
            logger.info(f"Active customer summary created with {len(summary_dict)} metrics")
        else:
            logger.warning("Failed to create active customer summary")
        
        # Generate comprehensive visualizations with new data indicators
        print("\n" + "=" * 80)
        print("📊 GENERATING LIVE CUSTOMER VISUALIZATIONS")
        print("=" * 80)
        
        # Add feature engineering dataset to data_dict for visualization
        if not fe_dataset.empty:
            data_dict['feature_engineering_dataset'] = fe_dataset
        
        # Create custom visualizations for new/active data with train/new indicators
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Active Customer Distribution with New/Existing Breakdown
        if 'customer_metadata' in data_dict:
            metadata = data_dict['customer_metadata']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Segment distribution
            if 'CUSTOMER_SEGMENT' in metadata.columns:
                segment_counts = metadata['CUSTOMER_SEGMENT'].value_counts().head(10)
                axes[0, 0].bar(range(len(segment_counts)), segment_counts.values, color='#2ecc71')
                axes[0, 0].set_xticks(range(len(segment_counts)))
                axes[0, 0].set_xticklabels(segment_counts.index, rotation=45, ha='right')
                axes[0, 0].set_title('Active Customer Segment Distribution', fontsize=14, fontweight='bold')
                axes[0, 0].set_ylabel('Number of Customers')
                axes[0, 0].grid(True, alpha=0.3)
            
            # Customer count display
            total_customers = len(metadata)
            axes[0, 1].text(0.5, 0.5, f'Total Active\nCustomers\n{total_customers:,}', 
                           horizontalalignment='center', verticalalignment='center',
                           fontsize=24, fontweight='bold', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Active Customer Count', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].axis('off')
            
            # Customer tenure distribution
            if 'static_features' in data_dict and 'MONTHS_ELAPSED' in data_dict['static_features'].columns:
                static_df = data_dict['static_features']
                tenure = static_df['MONTHS_ELAPSED'].dropna()
                axes[1, 0].hist(tenure, bins=30, color='#2ecc71', edgecolor='black', alpha=0.7)
                
                axes[1, 0].set_xlabel('Months Since First Activity')
                axes[1, 0].set_ylabel('Number of Customers')
                axes[1, 0].set_title('Customer Tenure Distribution', fontsize=14, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Top segments by customer count
            if 'CUSTOMER_SEGMENT' in metadata.columns:
                segment_counts = metadata['CUSTOMER_SEGMENT'].value_counts().head(10)
                
                x = np.arange(len(segment_counts))
                axes[1, 1].bar(x, segment_counts.values, color='#2ecc71')
                
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(segment_counts.index, rotation=45, ha='right')
                axes[1, 1].set_ylabel('Number of Customers')
                axes[1, 1].set_title('Top 10 Segments by Customer Count', fontsize=14, fontweight='bold')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle('Active Customer Analysis Dashboard', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            fig_path = figs_dir / f"active_distribution_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved active customer distribution to {fig_path.name}")
        
        # 2. Time Series Trends for Active Customers
        if 'time_series_features' in data_dict:
            ts_features = data_dict['time_series_features']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Aggregate by week
            weekly_data = ts_features.groupby('YYYYWK').agg({
                'DOCUMENTS_OPENED': 'sum',
                'USED_STORAGE_MB': 'sum',
                'CUST_ACCOUNT_NUMBER': 'nunique'
            }).reset_index()
            
            # Documents opened over time
            axes[0, 0].plot(weekly_data['YYYYWK'], weekly_data['DOCUMENTS_OPENED'], 
                           marker='o', linewidth=2, markersize=4, color='#2ecc71')
            axes[0, 0].set_xlabel('Week (YYYYWK)')
            axes[0, 0].set_ylabel('Documents Opened')
            axes[0, 0].set_title('Weekly Document Usage', fontsize=12, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Storage usage over time
            axes[0, 1].plot(weekly_data['YYYYWK'], weekly_data['USED_STORAGE_MB'], 
                           marker='o', linewidth=2, markersize=4, color='#3498db')
            axes[0, 1].set_xlabel('Week (YYYYWK)')
            axes[0, 1].set_ylabel('Storage Used (MB)')
            axes[0, 1].set_title('Weekly Storage Usage', fontsize=12, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Active customers over time
            axes[1, 0].plot(weekly_data['YYYYWK'], weekly_data['CUST_ACCOUNT_NUMBER'], 
                          marker='o', linewidth=2, markersize=4, color='#e74c3c')
            axes[1, 0].set_xlabel('Week (YYYYWK)')
            axes[1, 0].set_ylabel('Number of Active Customers')
            axes[1, 0].set_title('Active Customers Over Time', fontsize=12, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Activity distribution by week (last 12 weeks)
            last_weeks = sorted(ts_features['YYYYWK'].unique())[-12:]  # Last 12 weeks
            recent_data = weekly_data[weekly_data['YYYYWK'].isin(last_weeks)]
            
            x = np.arange(len(recent_data))
            axes[1, 1].bar(x, recent_data['CUST_ACCOUNT_NUMBER'], color='#9b59b6')
            axes[1, 1].set_xlabel('Week')
            axes[1, 1].set_ylabel('Active Customers')
            axes[1, 1].set_title('Last 12 Weeks: Active Customers', fontsize=12, fontweight='bold')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(recent_data['YYYYWK'], rotation=45, ha='right')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle('Time Series Analysis: Active Customer Data', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            fig_path = figs_dir / f"usage_trends_live_{timestamp}.png"
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved usage trends to {fig_path.name}")
        
        print(f"\n✅ Generated custom visualizations for active customers")
        
        # Save comprehensive reports
        print("\n" + "=" * 80)
        print("📝 SAVING REPORTS")
        print("=" * 80)
        
        # Prepare comprehensive metrics
        comprehensive_metrics = {
            'timestamp': timestamp,
            'execution_date': datetime.now().isoformat(),
            'datasets': {}
        }
        
        # Collect metrics for each dataset
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                dataset_metrics = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
                    'column_names': list(df.columns),
                    'dtypes': df.dtypes.astype(str).to_dict(),
                    'missing_values': df.isnull().sum().to_dict(),
                    'duplicates': int(df.duplicated().sum())
                }
                
                
                comprehensive_metrics['datasets'][key] = dataset_metrics
        
        # Add feature engineering dataset metrics
        if not fe_dataset.empty:
            fe_metrics = {
                'total_records': len(fe_dataset),
                'unique_customers': int(fe_dataset['CUST_ACCOUNT_NUMBER'].nunique()),
                'time_range': {
                    'start': int(fe_dataset['YYYYWK'].min()),
                    'end': int(fe_dataset['YYYYWK'].max())
                },
                'ml_compatible': all(col in fe_dataset.columns for col in 
                                    ['CUST_ACCOUNT_NUMBER', 'YYYYWK', 'DOCUMENTS_OPENED'])
            }
            
            
            comprehensive_metrics['feature_engineering'] = fe_metrics
        
        # Add comparison with training data
        train_customers = set()
        if 'customer_metadata' in train_data_dict:
            train_customers = set(train_data_dict['customer_metadata']['CUST_ACCOUNT_NUMBER'].unique())
        
        new_customers = set()
        if 'customer_metadata' in data_dict:
            new_customers = set(data_dict['customer_metadata']['CUST_ACCOUNT_NUMBER'].unique())
        
        if train_customers and new_customers:
            comprehensive_metrics['comparison'] = {
                'training_customers': len(train_customers),
                'active_customers': len(new_customers),
                'overlap_customers': len(train_customers.intersection(new_customers)),
                'truly_new_customers': len(new_customers - train_customers),
                'new_customer_rate_pct': float(len(new_customers - train_customers) / len(new_customers) * 100) if new_customers else 0
            }
        
        # Save JSON report
        json_report_path = reports_dir / f"live_data_loading_report_{timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(comprehensive_metrics, f, indent=2)
        print(f"✅ JSON report saved to: {json_report_path.name}")
        
        # Save CSV report for data analysts
        csv_report_data = []
        for dataset_name, metrics in comprehensive_metrics['datasets'].items():
            csv_report_data.append({
                'dataset': dataset_name,
                'rows': metrics['rows'],
                'columns': metrics['columns'],
                'memory_mb': metrics['memory_mb'],
                'duplicates': metrics['duplicates'],
                'missing_values_total': sum(metrics['missing_values'].values()),
            })
        
        if csv_report_data:
            csv_report_df = pd.DataFrame(csv_report_data)
            csv_report_path = reports_dir / f"live_data_loading_summary_{timestamp}.csv"
            csv_report_df.to_csv(csv_report_path, index=False)
            print(f"✅ CSV summary saved to: {csv_report_path.name}")
        
        # Save active customer summary
        if not active_summary.empty:
            active_summary_path = reports_dir / f"active_customer_summary_{timestamp}.csv"
            active_summary.to_csv(active_summary_path, index=False)
            print(f"✅ Active customer summary saved to: {active_summary_path.name}")
        
        # Save sample data for each dataset
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                sample_size = min(1000, len(df))
                sample_df = df.head(sample_size)
                
                sample_file = reports_dir / f"sample_live_{key}_{timestamp}.csv"
                sample_df.to_csv(sample_file, index=False)
                logger.info(f"Sample of {key} saved to {sample_file.name}")
        
        # Save feature engineering dataset sample
        if not fe_dataset.empty:
            fe_sample = fe_dataset.head(1000)
            fe_sample_path = reports_dir / f"feature_engineering_new_sample_{timestamp}.csv"
            fe_sample.to_csv(fe_sample_path, index=False)
            print(f"✅ Feature engineering sample saved to: {fe_sample_path.name}")
        
        print("\n" + "=" * 80)
        print("✅ LIVE/ACTIVE DATA LOADING SCRIPT COMPLETED SUCCESSFULLY!")
        print(f"   Figures saved to: {figs_dir}")
        print(f"   Reports saved to: {reports_dir}")
        print(f"   Active customer summary included")
        print("=" * 80)
        
        logger.info("=" * 80)
        logger.info("LIVE/ACTIVE DATA LOADING SCRIPT COMPLETED SUCCESSFULLY!")
        logger.info(f"Figures saved to: {figs_dir}")
        logger.info(f"Reports saved to: {reports_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during data loading: {str(e)}")
        logger.exception("Full traceback:")
        print(f"\n❌ Script failed: {str(e)}")


if __name__ == "__main__":
    main()
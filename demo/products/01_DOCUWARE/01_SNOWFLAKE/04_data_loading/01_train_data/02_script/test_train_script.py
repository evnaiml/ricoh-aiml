"""
Development-Grade Automated Training Data Loading Script with Comprehensive Analysis

This development script provides automated training data loading and analysis capabilities
with extensive visualization and reporting features. It leverages the SnowTrainDataLoader
class to process customer churn training data with detailed logging, quality checks, and
comprehensive output generation suitable for development and testing environments.

✅ Development Features:
- Automated data loading from Snowflake with progress tracking
- Feature engineering dataset creation for ML models
- Comprehensive data quality analysis and validation
- Multiple visualization outputs for data exploration
- Detailed JSON and CSV reports for sharing with analysts
- Performance metrics and memory usage monitoring
- Extensive logging with Hydra configuration
- Sample data extraction for validation

📊 Data Processing Pipeline:
- Raw data ingestion with type validation
- Usage data processing with 2023 H1 imputation
- Multi-dataset merging and transformation
- Time series feature preparation at weekly granularity
- Static feature aggregation and encoding
- Derived feature calculation including DAYS_TO_CHURN

🔍 Analysis & Visualization:
- Churn distribution analysis (pie and bar charts)
- Time series usage trends (documents and storage)
- Customer lifecycle duration analysis
- Engagement pattern visualization
- Activity distribution histograms
- Comprehensive monitoring dashboards

📝 Output Organization:
- figs/: High-quality visualization exports (300 DPI)
  * churn_distribution_[timestamp].png
  * usage_trends_[timestamp].png
  * customer_analysis_[timestamp].png
- reports/: Structured data reports
  * data_loading_report_[timestamp].json
  * data_loading_summary_[timestamp].csv
  * sample_[dataset]_[timestamp].csv
  * feature_engineering_sample_[timestamp].csv

💡 Usage:
python test_train_script.py

Or with Hydra overrides:
python test_train_script.py product=DOCUWARE debug=true

Updated: 2025-08-14
- Enhanced visualization capabilities with matplotlib/seaborn
- Added CSV outputs alongside JSON for data analysts
- Integrated feature engineering dataset analysis
- Improved output organization with figs/ and reports/ directories
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-15
# * CREATED ON: 2025-08-14
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

from churn_aiml.data.db.snowflake.loaddata import SnowTrainDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger
from churn_aiml.visualization.churn_plots.churn_lifecycle import ChurnLifecycleVizSnowflake


@hydra.main(config_path="../../../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to demonstrate data loading with detailed logging.
    
    Args:
        cfg: Hydra configuration object
    """
    # Setup logger with local directory for logs
    logger_config = setup_logger_for_script(cfg, __file__)
    logger = get_logger()
    
    logger.info("=" * 80)
    logger.info("STARTING AUTOMATED TRAINING DATA LOADING SCRIPT")
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
    
    # Initialize visualization class
    viz = ChurnLifecycleVizSnowflake()
    
    try:
        # Initialize the data loader
        logger.info("Initializing SnowTrainDataLoader...")
        data_loader = SnowTrainDataLoader(config=cfg, environment="development")
        
        # Load the data with progress logging
        logger.info("Beginning data loading process...")
        logger.info("This will execute the following steps:")
        logger.info("  1. Load raw data from Snowflake")
        logger.info("  2. Process usage data with imputations")
        logger.info("  3. Merge datasets")
        logger.info("  4. Prepare time series features")
        logger.info("  5. Prepare static features")
        logger.info("  6. Calculate derived features")
        
        print("\n" + "─" * 80)
        data_dict = data_loader.load_data()
        print("─" * 80 + "\n")
        
        logger.info("Data loading completed successfully!")
        
        # Detailed analysis of loaded data
        print("\n" + "=" * 80)
        print("📊 DATA LOADING RESULTS")
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
        
        logger.info("Creating comprehensive feature engineering dataset")
        
        # Get the feature engineering dataset for ML models
        fe_dataset = data_loader.get_feature_engineering_dataset()
        
        if not fe_dataset.empty:
            print(f"\n✅ Feature Engineering Dataset Created!")
            print(f"   Total records: {len(fe_dataset):,}")
            print(f"   Unique customers: {fe_dataset['CUST_ACCOUNT_NUMBER'].nunique():,}")
            print(f"   Time range: {fe_dataset['YYYYWK'].min()} to {fe_dataset['YYYYWK'].max()}")
            
            logger.info(f"Feature engineering dataset created with {len(fe_dataset):,} records")
        else:
            logger.warning("Failed to create feature engineering dataset")
            print("\n❌ Feature engineering dataset creation failed")
        
        # Generate churn summary
        print("\n" + "=" * 80)
        print("📊 GENERATING CHURN SUMMARY")
        print("=" * 80)
        
        logger.info("Creating comprehensive churn summary")
        churn_summary = data_loader.get_churn_summary()
        
        if not churn_summary.empty:
            summary_dict = churn_summary.iloc[0].to_dict()
            
            print(f"\n✅ Churn Summary Generated:")
            print(f"   Overall Churn Rate: {summary_dict.get('churn_rate_pct', 0):.2f}%")
            print(f"   Monthly Churn Mean: {summary_dict.get('monthly_churn_mean', 0):.2f} ± {summary_dict.get('monthly_churn_std', 0):.2f} customers")
            print(f"   Average Customer Lifespan: {summary_dict.get('avg_lifespan_days', 0):.0f} days")
            
            logger.info(f"Churn summary created with {len(summary_dict)} metrics")
            
            # Get monthly churn distribution analysis
            print("\n" + "=" * 80)
            print("MONTHLY CHURN DISTRIBUTION ANALYSIS")
            print("=" * 80)
            
            churn_dist = data_loader.get_monthly_churn_distribution()
            if churn_dist:
                dist_stats = churn_dist['distribution_stats']
                
                print("\nMonthly churn count statistics:")
                print(f"   Mean: {dist_stats['mean_churns_per_month']:.1f} churns/month")
                print(f"   Median: {dist_stats['median_churns_per_month']:.1f} churns/month")
                print(f"   Min: {dist_stats['min_churns_per_month']:.1f} churns/month")
                print(f"   Max: {dist_stats['max_churns_per_month']:.1f} churns/month")
                print(f"   Std Dev: {dist_stats['std_churns_per_month']:.1f} churns/month")
                
                print(f"\nTotal months analyzed: {dist_stats['total_months']}")
                print(f"Total churned customers: {dist_stats['total_churned_customers']}")
                print(f"Average monthly churn rate: {(dist_stats['mean_churns_per_month'] / dist_stats['total_churned_customers'] * 100):.2f}% of all churned")
                print(f"Peak churn: {dist_stats['peak_churn_month']} ({dist_stats['peak_churn_count']} churns)")
                
                logger.info(f"Monthly churn distribution: mean={dist_stats['mean_churns_per_month']:.2f}, std={dist_stats['std_churns_per_month']:.2f}")
        else:
            logger.warning("Failed to create churn summary")
        
        # Generate comprehensive visualizations using standardized method
        print("\n" + "=" * 80)
        print("📊 GENERATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 80)
        
        # Add feature engineering dataset to data_dict for visualization
        if not fe_dataset.empty:
            data_dict['feature_engineering_dataset'] = fe_dataset
        
        # Generate all distribution plots using the standardized method
        logger.info("Creating all distribution plots using plot_all_distributions()")
        figures = viz.plot_all_distributions(
            data_dict=data_dict,
            churn_dist=churn_dist,
            save_dir=str(figs_dir)
        )
        
        print(f"\n✅ Generated {len(figures)} visualization figures:")
        for fig_name in figures.keys():
            print(f"   - {fig_name}.png")
        
        
        # Save comprehensive reports
        print("\n" + "=" * 80)
        print("📝 SAVING REPORTS")
        print("=" * 80)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
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
        
        # Add monthly churn distribution stats to metrics
        if churn_dist:
            comprehensive_metrics['monthly_churn_distribution'] = churn_dist['distribution_stats']
        
        # Save JSON report
        json_report_path = reports_dir / f"data_loading_report_{timestamp}.json"
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
                'missing_values_total': sum(metrics['missing_values'].values())
            })
        
        if csv_report_data:
            csv_report_df = pd.DataFrame(csv_report_data)
            csv_report_path = reports_dir / f"data_loading_summary_{timestamp}.csv"
            csv_report_df.to_csv(csv_report_path, index=False)
            print(f"✅ CSV summary saved to: {csv_report_path.name}")
        
        # Save churn summary to files
        if not churn_summary.empty:
            churn_summary_path = reports_dir / f"churn_summary_{timestamp}"
            data_loader.save_churn_summary(str(churn_summary_path), format='both')
            print(f"✅ Churn summary saved to: churn_summary_{timestamp}.csv and .json")
        
        # Save monthly churn distribution to CSV and JSON
        if churn_dist:
            # Save monthly churn counts as CSV
            monthly_counts_df = churn_dist['monthly_churn_counts']
            monthly_counts_csv_path = reports_dir / f"monthly_churn_counts_{timestamp}.csv"
            monthly_counts_df.to_csv(monthly_counts_csv_path, index=False)
            print(f"✅ Monthly churn counts saved to: {monthly_counts_csv_path.name}")
            
            # Save distribution statistics as CSV
            dist_stats = churn_dist['distribution_stats']
            dist_stats_df = pd.DataFrame([dist_stats])
            dist_stats_csv_path = reports_dir / f"monthly_churn_statistics_{timestamp}.csv"
            dist_stats_df.to_csv(dist_stats_csv_path, index=False)
            print(f"✅ Monthly churn statistics saved to: {dist_stats_csv_path.name}")
            
            # Save complete monthly churn distribution as JSON
            monthly_churn_json_path = reports_dir / f"monthly_churn_distribution_{timestamp}.json"
            with open(monthly_churn_json_path, 'w') as f:
                # Convert any numpy types to Python types for JSON serialization
                churn_dist_serializable = {}
                for key, value in churn_dist.items():
                    if isinstance(value, pd.DataFrame):
                        churn_dist_serializable[key] = value.to_dict('records')
                    elif isinstance(value, dict):
                        churn_dist_serializable[key] = value
                    else:
                        churn_dist_serializable[key] = str(value)
                json.dump(churn_dist_serializable, f, indent=2, default=str)
            print(f"✅ Monthly churn distribution saved to: {monthly_churn_json_path.name}")
        
        # Save sample data for each dataset
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                sample_size = min(1000, len(df))
                sample_df = df.head(sample_size)
                
                sample_file = reports_dir / f"sample_{key}_{timestamp}.csv"
                sample_df.to_csv(sample_file, index=False)
                logger.info(f"Sample of {key} saved to {sample_file.name}")
        
        # Save feature engineering dataset sample
        if not fe_dataset.empty:
            fe_sample = fe_dataset.head(1000)
            fe_sample_path = reports_dir / f"feature_engineering_sample_{timestamp}.csv"
            fe_sample.to_csv(fe_sample_path, index=False)
            print(f"✅ Feature engineering sample saved to: {fe_sample_path.name}")
        
        print("\n" + "=" * 80)
        print("✅ TRAINING DATA LOADING SCRIPT COMPLETED SUCCESSFULLY!")
        print(f"   Figures saved to: {figs_dir}")
        print(f"   Reports saved to: {reports_dir}")
        print(f"   Churn summary included with detailed statistics")
        print(f"   Monthly churn distribution saved as:")
        print(f"     - monthly_churn_counts_{timestamp}.csv")
        print(f"     - monthly_churn_statistics_{timestamp}.csv")
        print(f"     - monthly_churn_distribution_{timestamp}.json")
        print("=" * 80)
        
        logger.info("=" * 80)
        logger.info("TRAINING DATA LOADING SCRIPT COMPLETED SUCCESSFULLY!")
        logger.info(f"Figures saved to: {figs_dir}")
        logger.info(f"Reports saved to: {reports_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error during data loading: {str(e)}")
        logger.exception("Full traceback:")
        print(f"\n❌ Script failed: {str(e)}")


if __name__ == "__main__":
    main()
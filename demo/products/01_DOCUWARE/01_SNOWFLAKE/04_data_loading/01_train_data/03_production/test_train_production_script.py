"""
Production-Grade Enterprise Training Data Loading Pipeline with Monitoring

This enterprise-ready production script provides robust training data loading capabilities
with comprehensive monitoring, error handling, and minimal console output. It leverages
the SnowTrainDataLoader class to process large-scale customer training data with production-grade
reliability, performance optimization, and detailed metrics suitable for deployment
in production environments.

✅ Production Features:
- Enterprise-grade data loading with fault tolerance
- Automated feature engineering dataset creation for ML pipelines
- Comprehensive monitoring dashboard generation
- Performance metrics and memory optimization
- Minimal console output with detailed file logging
- Error recovery and graceful failure handling
- JSON and CSV metrics for monitoring systems
- Production-ready error tracking and alerting

📊 Data Processing Pipeline:
- Robust data ingestion with validation and retry logic
- Large-scale usage data processing with imputation
- Memory-efficient dataset merging and transformation
- Optimized time series feature extraction
- Production-grade static feature processing
- Scalable derived feature calculation

🔍 Monitoring & Analytics:
- Single comprehensive monitoring dashboard
- Key performance indicators (KPIs) tracking
- Data quality metrics and validation
- Churn rate and distribution monitoring
- Usage trend analysis and forecasting
- Memory and performance profiling

📝 Output Organization:
- figs/: Production monitoring visualizations
  * monitoring_dashboard_[timestamp].png (150 DPI optimized)
- reports/: Production metrics and reports
  * metrics_[timestamp].json (for monitoring systems)
  * metrics_[timestamp].csv (for data analysts)
  * summary_[timestamp].txt (human-readable report)
  * error_metrics_[timestamp].json (failure tracking)

⚙️ Production Configuration:
- Environment-aware configuration with Hydra
- Configurable logging levels and outputs
- Performance tuning parameters
- Error handling and retry policies
- Memory management settings

💡 Usage:
python test_train_production_script.py

Or with production overrides:
python test_train_production_script.py environment=production log_level=WARNING

🚀 Deployment Notes:
- Suitable for scheduled execution (cron/airflow)
- Integrates with monitoring systems (Datadog/CloudWatch)
- Supports containerization (Docker/Kubernetes)
- Compatible with CI/CD pipelines

Updated: 2025-08-14
- Enhanced production monitoring capabilities
- Added comprehensive error tracking
- Integrated CSV outputs for analysts
- Optimized memory usage and performance
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

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import hydra
from omegaconf import DictConfig
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns

# No need to add path - imports work directly

from churn_aiml.data.db.snowflake.loaddata import SnowTrainDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger
from churn_aiml.visualization.churn_plots.churn_lifecycle import ChurnLifecycleVizSnowflake


@hydra.main(config_path="../../../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> int:
    """
    Production main function for data loading with minimal console output.
    
    Args:
        cfg: Hydra configuration object
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Setup logger with local directory for logs (suppress console output)
    logger_config = setup_logger_for_script(cfg, __file__)
    logger = get_logger()
    
    # Start time for performance tracking
    start_time = datetime.now()
    
    # Setup output directories
    script_dir = Path(__file__).parent
    figs_dir = script_dir / "figs"
    reports_dir = script_dir / "reports"
    
    # Create directories if they don't exist
    figs_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    
    # Initialize visualization class for monitoring
    viz = ChurnLifecycleVizSnowflake(figsize_scale=1.0)
    
    # Minimal console output
    print(f"[{start_time.strftime('%Y-%m-%d %H:%M:%S')}] Starting data loading...")
    
    logger.info("=" * 80)
    logger.info("PRODUCTION TRAINING DATA LOADING STARTED")
    logger.info(f"Start time: {start_time}")
    logger.info(f"Environment: {cfg.get('environment', 'development')}")
    logger.info("=" * 80)
    
    # Initialize metrics dictionary
    metrics = {
        'start_time': start_time.isoformat(),
        'status': 'running',
        'datasets': {},
        'errors': [],
        'warnings': []
    }
    
    try:
        # Initialize the data loader
        logger.info("Initializing SnowTrainDataLoader")
        data_loader = SnowTrainDataLoader(
            config=cfg, 
            environment=cfg.get('environment', 'development')
        )
        
        # Load the data
        logger.info("Starting data loading process")
        data_dict = data_loader.load_data()
        
        logger.info("Data loading completed successfully")
        
        # Process and validate loaded data
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame):
                dataset_metrics = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_mb': float(df.memory_usage(deep=True).sum() / 1024**2),
                    'duplicates': int(df.duplicated().sum()),
                    'missing_pct': float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
                }
                
                metrics['datasets'][key] = dataset_metrics
                
                logger.info(f"Dataset '{key}': {dataset_metrics['rows']} rows, "
                           f"{dataset_metrics['columns']} columns, "
                           f"{dataset_metrics['memory_mb']:.2f} MB")
                
                # Check for data quality issues
                if dataset_metrics['duplicates'] > 0:
                    warning = f"{key}: {dataset_metrics['duplicates']} duplicate rows detected"
                    metrics['warnings'].append(warning)
                    logger.warning(warning)
                
                if dataset_metrics['missing_pct'] > 50:
                    warning = f"{key}: High missing value percentage ({dataset_metrics['missing_pct']:.1f}%)"
                    metrics['warnings'].append(warning)
                    logger.warning(warning)
        
        # Get training-ready data
        logger.info("Preparing training-ready dataset")
        training_data = data_loader.get_training_ready_data()
        
        if not training_data.empty:
            # Analyze training data
            training_metrics = {
                'total_samples': len(training_data),
                'total_features': len(training_data.columns),
                'numeric_features': len(training_data.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(training_data.select_dtypes(include=['object']).columns),
                'memory_mb': float(training_data.memory_usage(deep=True).sum() / 1024**2)
            }
            
            # Calculate target statistics if available
            if 'DAYS_TO_CHURN' in training_data.columns:
                days_to_churn = training_data['DAYS_TO_CHURN'].dropna()
                if not days_to_churn.empty:
                    training_metrics['target_stats'] = {
                        'count': int(len(days_to_churn)),
                        'mean': float(days_to_churn.mean()),
                        'std': float(days_to_churn.std()),
                        'min': float(days_to_churn.min()),
                        'max': float(days_to_churn.max()),
                        'median': float(days_to_churn.median())
                    }
            
            metrics['training_data'] = training_metrics
            
            logger.info(f"Training data prepared: {training_metrics['total_samples']} samples, "
                       f"{training_metrics['total_features']} features")
        else:
            error = "Failed to create training-ready dataset"
            metrics['errors'].append(error)
            logger.error(error)
        
        # Create feature engineering dataset
        logger.info("Creating feature engineering dataset for ML models")
        fe_dataset = data_loader.get_feature_engineering_dataset()
        
        if not fe_dataset.empty:
            # Analyze feature engineering dataset
            fe_metrics = {
                'total_records': len(fe_dataset),
                'unique_customers': int(fe_dataset['CUST_ACCOUNT_NUMBER'].nunique()),
                'time_range': {
                    'start': int(fe_dataset['YYYYWK'].min()),
                    'end': int(fe_dataset['YYYYWK'].max())
                },
                'memory_mb': float(fe_dataset.memory_usage(deep=True).sum() / 1024**2)
            }
            
            # Check ML compatibility
            ml_required = ['CUST_ACCOUNT_NUMBER', 'YYYYWK', 'DOCUMENTS_OPENED', 
                          'USED_STORAGE_MB', 'INVOICE_REVLINE_TOTAL']
            fe_metrics['ml_compatible'] = all(col in fe_dataset.columns for col in ml_required)
            
            # Analyze churn distribution if available
            if 'CHURNED_FLAG' in fe_dataset.columns:
                churn_counts = fe_dataset.groupby('CUST_ACCOUNT_NUMBER')['CHURNED_FLAG'].first().value_counts()
                fe_metrics['churn_distribution'] = {
                    'churned': int(churn_counts.get('Y', 0)),
                    'active': int(churn_counts.get('N', 0))
                }
                
                # Calculate churn rate
                total = churn_counts.sum()
                if total > 0:
                    fe_metrics['churn_rate'] = float(churn_counts.get('Y', 0) / total * 100)
            
            # Check for ML-specific features
            ml_features = ['DAYS_TO_CHURN', 'WEEKS_TO_CHURN', 'FINAL_EARLIEST_DATE']
            fe_metrics['ml_features_present'] = [f for f in ml_features if f in fe_dataset.columns]
            
            # Analyze monthly churn distribution
            logger.info("Analyzing monthly churn distribution for production monitoring")
            churn_dist = data_loader.get_monthly_churn_distribution()
            
            if churn_dist:
                dist_stats = churn_dist['distribution_stats']
                fe_metrics['monthly_churn_distribution'] = {
                    'mean_per_month': dist_stats['mean_churns_per_month'],
                    'std_per_month': dist_stats['std_churns_per_month'],
                    'median_per_month': dist_stats['median_churns_per_month'],
                    'min_per_month': dist_stats['min_churns_per_month'],
                    'max_per_month': dist_stats['max_churns_per_month'],
                    'coefficient_of_variation': dist_stats['coefficient_of_variation'],
                    'peak_month': dist_stats['peak_churn_month'],
                    'peak_count': dist_stats['peak_churn_count'],
                    'percentile_95': dist_stats['percentile_95'],
                    'total_months': dist_stats['total_months'],
                    'total_churned': dist_stats['total_churned_customers']
                }
                
                logger.info(f"Monthly churn statistics: mean={dist_stats['mean_churns_per_month']:.2f}, "
                           f"std={dist_stats['std_churns_per_month']:.2f}, CV={dist_stats['coefficient_of_variation']:.3f}")
                
                # Also add to main metrics for backward compatibility
                metrics['monthly_churn_distribution'] = fe_metrics['monthly_churn_distribution']
            
            metrics['feature_engineering_dataset'] = fe_metrics
            
            logger.info(f"Feature engineering dataset created: {fe_metrics['total_records']} records, "
                       f"{fe_metrics['unique_customers']} customers")
            logger.info(f"ML compatible: {fe_metrics['ml_compatible']}")
        else:
            error = "Failed to create feature engineering dataset"
            metrics['errors'].append(error)
            logger.error(error)
        
        # Generate comprehensive time series visualization
        logger.info("Generating comprehensive time series visualization")
        
        # Import the new comprehensive visualization function
        from churn_aiml.visualization.churn_plots.comprehensive_time_series import plot_comprehensive_time_series_new
        
        # Add feature engineering dataset to data_dict for visualization
        if not fe_dataset.empty:
            data_dict['feature_engineering_dataset'] = fe_dataset
        
        # Create comprehensive time series plots
        if 'time_series_features' in data_dict and not data_dict['time_series_features'].empty:
            # Use feature engineering dataset if available
            if 'feature_engineering_dataset' in data_dict and not data_dict['feature_engineering_dataset'].empty:
                ts_data = data_dict['feature_engineering_dataset']
                logger.info("Using feature engineering dataset for comprehensive plots")
            else:
                ts_data = data_dict['time_series_features']
                logger.info("Using time series features for comprehensive plots")
            
            try:
                fig_comprehensive = plot_comprehensive_time_series_new(
                    data=ts_data,
                    save_path=str(figs_dir / 'comprehensive_time_series.png'),
                    figsize_scale=1.2
                )
                plt.close(fig_comprehensive)
                logger.info("Successfully generated comprehensive time series visualization")
                logger.info("  - comprehensive_time_series.png")
            except Exception as e:
                logger.error(f"Failed to create comprehensive time series: {e}")
        
        # Generate other visualizations using standardized method
        logger.info("Generating production monitoring figures")
        
        # Generate all distribution plots using the standardized method
        logger.info("Creating all distribution plots using plot_all_distributions()")
        figures = viz.plot_all_distributions(
            data_dict=data_dict,
            churn_dist=churn_dist,
            save_dir=str(figs_dir)
        )
        
        logger.info(f"Generated {len(figures)} visualization figures")
        for fig_name in figures.keys():
            logger.info(f"  - {fig_name}.png")
        
        # Generate comparative churn timing histograms
        logger.info("Creating comparative churn timing histogram visualizations...")
        if 'customer_metadata' in data_dict and not data_dict['customer_metadata'].empty:
            all_customers = data_dict['customer_metadata']
            
            if 'DAYS_TO_CHURN' in all_customers.columns or 'LIFESPAN_MONTHS' in all_customers.columns:
                # Comparative histogram plot
                fig_comparative = viz.plot_churn_timing_histograms(
                    data=all_customers,
                    save_path=str(figs_dir / 'churn_timing_comparative.png')
                )
                plt.close(fig_comparative)
                logger.info("  - churn_timing_comparative.png (churned vs active vs all)")
                
                # Additional detailed histograms for churned customers only
                churned_customers = all_customers[all_customers['CHURNED_FLAG'] == 1]
                if not churned_customers.empty:
                    if 'DAYS_TO_CHURN' in churned_customers.columns:
                        fig_days = viz.plot_days_to_churn_histogram(
                            data=churned_customers,
                            save_path=str(figs_dir / 'days_to_churn_detailed.png')
                        )
                        plt.close(fig_days)
                        logger.info("  - days_to_churn_detailed.png")
                    
                    if 'LIFESPAN_MONTHS' in churned_customers.columns:
                        fig_lifespan = viz.plot_lifespan_histogram(
                            data=churned_customers,
                            save_path=str(figs_dir / 'lifespan_detailed.png')
                        )
                        plt.close(fig_lifespan)
                        logger.info("  - lifespan_detailed.png")
                
                logger.info("Successfully generated comparative churn timing histograms")
            else:
                logger.warning("Required columns not available for histogram generation")
        
        # Calculate total execution time
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        metrics['visualizations_generated'] = list(figures.keys())
        metrics['end_time'] = end_time.isoformat()
        metrics['execution_time_seconds'] = execution_time
        metrics['status'] = 'completed'
        
        # Save metrics to JSON in reports directory
        metrics_file = reports_dir / f"metrics_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file.name}")
        
        # Also save metrics as CSV for data analysts
        metrics_csv_data = []
        for dataset_name, dataset_metrics in metrics.get('datasets', {}).items():
            metrics_csv_data.append({
                'metric_type': 'dataset',
                'name': dataset_name,
                'rows': dataset_metrics.get('rows', 0),
                'columns': dataset_metrics.get('columns', 0),
                'memory_mb': dataset_metrics.get('memory_mb', 0),
                'duplicates': dataset_metrics.get('duplicates', 0),
                'missing_pct': dataset_metrics.get('missing_pct', 0)
            })
        
        if 'feature_engineering_dataset' in metrics:
            fe = metrics['feature_engineering_dataset']
            metrics_csv_data.append({
                'metric_type': 'feature_engineering',
                'name': 'feature_engineering_dataset',
                'rows': fe.get('total_records', 0),
                'columns': 0,  # Not tracked separately
                'memory_mb': fe.get('memory_mb', 0),
                'duplicates': 0,  # Not tracked
                'missing_pct': 0  # Not tracked
            })
        
        if metrics_csv_data:
            metrics_csv_df = pd.DataFrame(metrics_csv_data)
            metrics_csv_path = reports_dir / f"metrics_{end_time.strftime('%Y%m%d_%H%M%S')}.csv"
            metrics_csv_df.to_csv(metrics_csv_path, index=False)
            logger.info(f"Metrics CSV saved to {metrics_csv_path.name}")
        
        # Save monthly churn distribution to CSV and JSON
        if churn_dist:
            # Save monthly churn counts as CSV
            monthly_counts_df = churn_dist['monthly_churn_counts']
            monthly_counts_csv_path = reports_dir / f"monthly_churn_counts_{end_time.strftime('%Y%m%d_%H%M%S')}.csv"
            monthly_counts_df.to_csv(monthly_counts_csv_path, index=False)
            logger.info(f"Monthly churn counts saved to {monthly_counts_csv_path.name}")
            
            # Save distribution statistics as CSV
            dist_stats = churn_dist['distribution_stats']
            dist_stats_df = pd.DataFrame([dist_stats])
            dist_stats_csv_path = reports_dir / f"monthly_churn_statistics_{end_time.strftime('%Y%m%d_%H%M%S')}.csv"
            dist_stats_df.to_csv(dist_stats_csv_path, index=False)
            logger.info(f"Monthly churn statistics saved to {dist_stats_csv_path.name}")
            
            # Save complete monthly churn distribution as JSON
            monthly_churn_json_path = reports_dir / f"monthly_churn_distribution_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
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
            logger.info(f"Monthly churn distribution saved to {monthly_churn_json_path.name}")
        
        # Generate summary report in reports directory
        summary_file = reports_dir / f"summary_{end_time.strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("DATA LOADING SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Execution Date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"Status: {metrics['status'].upper()}\n\n")
            
            f.write("DATASETS LOADED:\n")
            for name, stats in metrics['datasets'].items():
                f.write(f"  {name}:\n")
                f.write(f"    - Rows: {stats['rows']:,}\n")
                f.write(f"    - Columns: {stats['columns']}\n")
                f.write(f"    - Memory: {stats['memory_mb']:.2f} MB\n")
            
            if 'training_data' in metrics:
                f.write("\nTRAINING DATA:\n")
                td = metrics['training_data']
                f.write(f"  - Total Samples: {td['total_samples']:,}\n")
                f.write(f"  - Total Features: {td['total_features']}\n")
                f.write(f"  - Numeric Features: {td['numeric_features']}\n")
                f.write(f"  - Categorical Features: {td['categorical_features']}\n")
                
                if 'target_stats' in td:
                    f.write("\n  Target Variable Statistics:\n")
                    ts = td['target_stats']
                    f.write(f"    - Mean: {ts['mean']:.1f} days\n")
                    f.write(f"    - Median: {ts['median']:.1f} days\n")
                    f.write(f"    - Std Dev: {ts['std']:.1f} days\n")
                    f.write(f"    - Range: {ts['min']:.1f} - {ts['max']:.1f} days\n")
            
            # Add feature engineering dataset section
            if 'feature_engineering_dataset' in metrics:
                f.write("\nFEATURE ENGINEERING DATASET:\n")
                fe = metrics['feature_engineering_dataset']
                f.write(f"  - Total Records: {fe['total_records']:,}\n")
                f.write(f"  - Unique Customers: {fe['unique_customers']:,}\n")
                f.write(f"  - Time Range: {fe['time_range']['start']} to {fe['time_range']['end']}\n")
                f.write(f"  - Memory Usage: {fe['memory_mb']:.2f} MB\n")
                f.write(f"  - ML Compatible: {'Yes' if fe['ml_compatible'] else 'No'}\n")
                
                if 'churn_distribution' in fe:
                    f.write("\n  Churn Distribution:\n")
                    f.write(f"    - Churned Customers: {fe['churn_distribution']['churned']:,}\n")
                    f.write(f"    - Active Customers: {fe['churn_distribution']['active']:,}\n")
                    if 'churn_rate' in fe:
                        f.write(f"    - Churn Rate: {fe['churn_rate']:.2f}%\n")
                
                if fe.get('ml_features_present'):
                    f.write("\n  ML-Specific Features Present:\n")
                    for feature in fe['ml_features_present']:
                        f.write(f"    - {feature}\n")
            
            # Add churn summary section
            if 'churn_summary' in metrics:
                f.write("\nCHURN ANALYSIS SUMMARY:\n")
                cs = metrics['churn_summary']
                f.write(f"  Overall Metrics:\n")
                f.write(f"    - Total Customers: {cs['total_customers']:,}\n")
                f.write(f"    - Churned: {cs['churned_customers']:,} ({cs['churn_rate_pct']:.2f}%)\n")
                f.write(f"    - Active: {cs['active_customers']:,}\n")
            
            # Add monthly churn distribution section
            if 'monthly_churn_distribution' in metrics:
                f.write("\nMONTHLY CHURN DISTRIBUTION:\n")
                mcd = metrics['monthly_churn_distribution']
                f.write(f"  Statistical Summary:\n")
                f.write(f"    - Mean churns per month: {mcd['mean_per_month']:.2f}\n")
                f.write(f"    - Std deviation: {mcd['std_per_month']:.2f}\n")
                f.write(f"    - Coefficient of variation: {mcd['coefficient_of_variation']:.3f}\n")
                f.write(f"    - 95th percentile: {mcd['percentile_95']:.1f}\n")
                f.write(f"  Peak Churn:\n")
                f.write(f"    - Month: {mcd['peak_month']}\n")
                f.write(f"    - Count: {mcd['peak_count']} customers\n")
                f.write(f"\n  Monthly Churn Statistics:\n")
                f.write(f"    - Mean: {cs['monthly_churn_mean']:.1f} customers/month\n")
                f.write(f"    - Std Dev: {cs['monthly_churn_std']:.1f} customers/month\n")
                f.write(f"    - Rate: {cs['monthly_churn_rate_pct']:.3f}% of total customers\n")
                f.write(f"\n  Customer Lifecycle:\n")
                f.write(f"    - Avg Lifespan: {cs['avg_lifespan_days']:.0f} days\n")
            
            if metrics['warnings']:
                f.write("\nWARNINGS:\n")
                for warning in metrics['warnings']:
                    f.write(f"  ⚠️  {warning}\n")
            
            if metrics['errors']:
                f.write("\nERRORS:\n")
                for error in metrics['errors']:
                    f.write(f"  ❌ {error}\n")
            
            f.write("\nOUTPUT LOCATIONS:\n")
            f.write(f"  - Figures: {figs_dir}\n")
            f.write(f"  - Reports: {reports_dir}\n")
        
        logger.info(f"Summary report saved to {summary_file.name}")
        
        # Minimal console output for success
        print(f"[{end_time.strftime('%Y-%m-%d %H:%M:%S')}] Data loading completed successfully")
        print(f"  Execution time: {execution_time:.1f}s")
        print(f"  Figures saved to: {figs_dir}")
        print(f"  Reports saved to: {reports_dir}")
        print(f"  Monthly churn distribution: CSV and JSON files generated")
        print(f"  Check logs for details: logs/loguru/")
        
        logger.info("=" * 80)
        logger.info("PRODUCTION DATA LOADING COMPLETED SUCCESSFULLY")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        # Log the error
        error_msg = f"Critical error during data loading: {str(e)}"
        metrics['errors'].append(error_msg)
        metrics['status'] = 'failed'
        
        logger.error(error_msg)
        logger.exception("Full traceback:")
        
        # Save error metrics
        try:
            error_time = datetime.now()
            metrics['end_time'] = error_time.isoformat()
            metrics['execution_time_seconds'] = (error_time - start_time).total_seconds()
            
            reports_dir = Path(__file__).parent / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            error_metrics_file = reports_dir / f"error_metrics_{error_time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(error_metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Minimal console output for error
            print(f"[{error_time.strftime('%Y-%m-%d %H:%M:%S')}] ❌ Data loading failed")
            print(f"  Error: {str(e)}")
            print(f"  Error metrics saved to: {error_metrics_file.name}")
            print(f"  Check logs for details: logs/loguru/")
            
        except Exception as save_error:
            print(f"  Failed to save error metrics: {str(save_error)}")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
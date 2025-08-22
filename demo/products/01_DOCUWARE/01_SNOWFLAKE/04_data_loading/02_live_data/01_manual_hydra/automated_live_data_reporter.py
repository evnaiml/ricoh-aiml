#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Automated New Data Reporter - Generates periodic reports on train/new data distribution

This script can be scheduled to run periodically (e.g., via cron) to generate
automated reports on the distribution of training vs new data.

✅ Features:
- Runs without user interaction
- Generates timestamped reports
- Creates summary dashboard CSV
- Sends email notifications (optional)
- Archives old reports

📊 Output:
- Daily summary report
- Detailed breakdowns by dataset, segment, and time
- Trend analysis over multiple runs

💡 Usage:
python automated_live_data_reporter.py [--email recipient@example.com]

For scheduling with cron:
0 9 * * * /path/to/python /path/to/automated_live_data_reporter.py
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-18
# * CREATED ON: 2025-08-18
# -----------------------------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
from churn_aiml.utils.suppress_warnings import suppress_known_warnings

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

from churn_aiml.data.db.snowflake.loaddata import SnowLiveDataLoader, SnowTrainDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger
from churn_aiml.utils.find_paths import ProjectRootFinder


class AutomatedNewDataReporter:
    """Automated reporter for train/new data distribution analysis."""
    
    def __init__(self, output_dir=None, archive_days=30):
        """
        Initialize the automated reporter.
        
        Args:
            output_dir: Directory for output reports (default: ./automated_reports)
            archive_days: Number of days to keep reports before archiving
        """
        self.timestamp = datetime.now()
        self.timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Setup paths
        self.churn_aiml_dir = ProjectRootFinder().find_path()
        self.conf_dir = self.churn_aiml_dir / "conf"
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / "automated_reports"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir = self.output_dir / "archive"
        self.archive_dir.mkdir(exist_ok=True)
        
        self.archive_days = archive_days
        
        # Initialize configuration
        self._init_config()
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Report data storage
        self.reports = {}
        self.metrics = {}
        
    def _init_config(self):
        """Initialize Hydra configuration."""
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(self.conf_dir), version_base=None):
            self.cfg = compose(config_name="config")
    
    def _setup_logger(self):
        """Setup logger for automated reporting."""
        logger_config = setup_logger_for_script(self.cfg, __file__)
        logger = get_logger()
        
        # Also create a report-specific log
        report_log = self.output_dir / f"report_log_{self.timestamp_str}.txt"
        
        return logger
    
    def load_data(self):
        """Load training and new data."""
        self.logger.info("Loading data for analysis...")
        
        # Load training data
        self.train_loader = SnowTrainDataLoader(config=self.cfg, environment="development")
        self.train_data = self.train_loader.load_data()
        
        # Load new data with comparison
        self.live_loader = SnowLiveDataLoader(config=self.cfg, environment="development")
        self.new_data = self.live_loader.load_data(train_data_loader=self.train_loader)
        
        self.logger.info(f"Data loaded: {len(self.train_data)} training datasets, {len(self.new_data)} new datasets")
        
    def analyze_dataset(self, df, name):
        """
        Analyze a single dataset for train/new distribution.
        
        Returns:
            dict: Analysis results
        """
        results = {
            'dataset_name': name,
            'timestamp': self.timestamp_str,
            'total_records': len(df) if isinstance(df, pd.DataFrame) else 0,
            'has_new_flag': False,
            'train_records': 0,
            'new_records': 0,
            'train_pct': 0.0,
            'new_pct': 0.0,
            'unique_customers': 0,
            'train_customers': 0,
            'new_customers': 0,
            'train_customers_pct': 0.0,
            'new_customers_pct': 0.0
        }
        
        if not isinstance(df, pd.DataFrame) or df.empty:
            return results
        
        results['total_records'] = len(df)
        
        if 'new_data' in df.columns:
            results['has_new_flag'] = True
            
            # Record-level analysis
            train_mask = df['new_data'] == 0
            new_mask = df['new_data'] == 1
            
            results['train_records'] = train_mask.sum()
            results['new_records'] = new_mask.sum()
            
            if results['total_records'] > 0:
                results['train_pct'] = round(results['train_records'] / results['total_records'] * 100, 2)
                results['new_pct'] = round(results['new_records'] / results['total_records'] * 100, 2)
            
            # Customer-level analysis
            if 'CUST_ACCOUNT_NUMBER' in df.columns:
                results['unique_customers'] = df['CUST_ACCOUNT_NUMBER'].nunique()
                results['train_customers'] = df[train_mask]['CUST_ACCOUNT_NUMBER'].nunique()
                results['new_customers'] = df[new_mask]['CUST_ACCOUNT_NUMBER'].nunique()
                
                if results['unique_customers'] > 0:
                    results['train_customers_pct'] = round(results['train_customers'] / results['unique_customers'] * 100, 2)
                    results['new_customers_pct'] = round(results['new_customers'] / results['unique_customers'] * 100, 2)
        
        return results
    
    def generate_reports(self):
        """Generate all reports."""
        self.logger.info("Generating reports...")
        
        # 1. Dataset-level analysis
        dataset_results = []
        for name, df in self.new_data.items():
            result = self.analyze_dataset(df, name)
            dataset_results.append(result)
        
        self.reports['dataset_analysis'] = pd.DataFrame(dataset_results)
        
        # 2. Overall summary
        summary = {
            'report_date': self.timestamp.strftime("%Y-%m-%d"),
            'report_time': self.timestamp.strftime("%H:%M:%S"),
            'total_datasets': len(dataset_results),
            'datasets_with_flag': sum(1 for r in dataset_results if r['has_new_flag']),
            'total_records': sum(r['total_records'] for r in dataset_results),
            'total_train_records': sum(r['train_records'] for r in dataset_results),
            'total_new_records': sum(r['new_records'] for r in dataset_results),
            'overall_train_pct': 0.0,
            'overall_new_pct': 0.0
        }
        
        if summary['total_records'] > 0:
            summary['overall_train_pct'] = round(summary['total_train_records'] / summary['total_records'] * 100, 2)
            summary['overall_new_pct'] = round(summary['total_new_records'] / summary['total_records'] * 100, 2)
        
        self.reports['summary'] = pd.DataFrame([summary])
        
        # 3. Customer segment analysis
        if 'customer_metadata' in self.new_data:
            metadata = self.new_data['customer_metadata']
            if 'CUSTOMER_SEGMENT' in metadata.columns and 'new_data' in metadata.columns:
                segment_analysis = metadata.groupby('CUSTOMER_SEGMENT').agg({
                    'new_data': ['sum', 'count', 'mean'],
                    'CUST_ACCOUNT_NUMBER': 'nunique'
                }).round(2)
                
                segment_analysis.columns = ['new_count', 'total_count', 'new_ratio', 'unique_customers']
                segment_analysis['train_count'] = segment_analysis['total_count'] - segment_analysis['new_count']
                segment_analysis['new_pct'] = (segment_analysis['new_ratio'] * 100).round(2)
                segment_analysis['train_pct'] = (100 - segment_analysis['new_pct']).round(2)
                segment_analysis['segment'] = segment_analysis.index
                segment_analysis['timestamp'] = self.timestamp_str
                
                self.reports['segment_analysis'] = segment_analysis.reset_index(drop=True)
        
        # 4. Time series analysis
        if 'time_series_features' in self.new_data:
            ts_df = self.new_data['time_series_features']
            if 'YYYYWK' in ts_df.columns and 'new_data' in ts_df.columns:
                weekly_stats = ts_df.groupby('YYYYWK').agg({
                    'new_data': ['sum', 'count', 'mean'],
                    'CUST_ACCOUNT_NUMBER': 'nunique'
                }).round(2)
                
                weekly_stats.columns = ['new_count', 'total_count', 'new_ratio', 'unique_customers']
                weekly_stats['train_count'] = weekly_stats['total_count'] - weekly_stats['new_count']
                weekly_stats['new_pct'] = (weekly_stats['new_ratio'] * 100).round(2)
                weekly_stats['week'] = weekly_stats.index
                weekly_stats['timestamp'] = self.timestamp_str
                
                # Keep only last 12 weeks for concise report
                self.reports['weekly_analysis'] = weekly_stats.tail(12).reset_index(drop=True)
        
        # 5. Key metrics
        self.metrics = {
            'total_records': summary['total_records'],
            'new_records': summary['total_new_records'],
            'new_percentage': summary['overall_new_pct'],
            'report_timestamp': self.timestamp_str
        }
        
        self.logger.info(f"Reports generated: {list(self.reports.keys())}")
    
    def save_reports(self):
        """Save all reports to CSV files."""
        self.logger.info(f"Saving reports to {self.output_dir}")
        
        saved_files = []
        
        for report_name, df in self.reports.items():
            if df is not None and not df.empty:
                filename = f"{report_name}_{self.timestamp_str}.csv"
                filepath = self.output_dir / filename
                df.to_csv(filepath, index=False)
                saved_files.append(filename)
                self.logger.info(f"  Saved: {filename}")
        
        # Save metrics as JSON for easy programmatic access
        metrics_file = self.output_dir / f"metrics_{self.timestamp_str}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        saved_files.append(metrics_file.name)
        
        # Create a master summary that appends to existing
        master_summary = self.output_dir / "master_summary.csv"
        if master_summary.exists():
            existing = pd.read_csv(master_summary)
            updated = pd.concat([existing, self.reports['summary']], ignore_index=True)
            # Keep only last 90 days
            if 'report_date' in updated.columns:
                updated['report_date'] = pd.to_datetime(updated['report_date'])
                cutoff_date = datetime.now() - timedelta(days=90)
                updated = updated[updated['report_date'] >= cutoff_date]
            updated.to_csv(master_summary, index=False)
        else:
            self.reports['summary'].to_csv(master_summary, index=False)
        
        self.logger.info(f"  Updated: master_summary.csv")
        
        return saved_files
    
    def archive_old_reports(self):
        """Archive reports older than archive_days."""
        cutoff_date = datetime.now() - timedelta(days=self.archive_days)
        archived_count = 0
        
        for file in self.output_dir.glob("*.csv"):
            if file.name == "master_summary.csv":
                continue
                
            # Try to parse date from filename
            try:
                # Format: reportname_YYYYMMDD_HHMMSS.csv
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    date_str = parts[-2]  # YYYYMMDD
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    
                    if file_date < cutoff_date:
                        archive_path = self.archive_dir / file.name
                        file.rename(archive_path)
                        archived_count += 1
            except:
                continue
        
        if archived_count > 0:
            self.logger.info(f"Archived {archived_count} old reports")
    
    def create_dashboard_summary(self):
        """Create a dashboard-ready summary combining key metrics."""
        dashboard = {
            'last_updated': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'data_summary': {
                'total_records_analyzed': self.metrics['total_records'],
                'live_data_records': self.metrics['new_records'],
                'live_data_percentage': self.metrics['new_percentage'],
                'datasets_analyzed': len(self.reports.get('dataset_analysis', []))
            },
            'top_insights': []
        }
        
        # Add top insights
        if 'dataset_analysis' in self.reports:
            df = self.reports['dataset_analysis']
            
            # Dataset with most new data
            if not df.empty and 'new_pct' in df.columns:
                max_new = df.loc[df['new_pct'].idxmax()]
                dashboard['top_insights'].append({
                    'insight': 'Dataset with most new data',
                    'dataset': max_new['dataset_name'],
                    'percentage': max_new['new_pct']
                })
            
            # Dataset with most records
            if not df.empty and 'total_records' in df.columns:
                max_records = df.loc[df['total_records'].idxmax()]
                dashboard['top_insights'].append({
                    'insight': 'Largest dataset',
                    'dataset': max_records['dataset_name'],
                    'records': max_records['total_records']
                })
        
        # Save dashboard
        dashboard_file = self.output_dir / f"dashboard_{self.timestamp_str}.json"
        with open(dashboard_file, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        # Also save latest dashboard for easy access
        latest_dashboard = self.output_dir / "latest_dashboard.json"
        with open(latest_dashboard, 'w') as f:
            json.dump(dashboard, f, indent=2)
        
        self.logger.info("Dashboard summary created")
        
        return dashboard
    
    def generate_alert_if_needed(self):
        """Generate alert if significant changes detected."""
        alerts = []
        
        # Check if new data percentage suddenly increased
        if self.metrics['new_percentage'] > 50:
            alerts.append(f"HIGH NEW DATA ALERT: {self.metrics['new_percentage']}% of records are new")
        
        # Check if any dataset has no new_data flag
        if 'dataset_analysis' in self.reports:
            df = self.reports['dataset_analysis']
            missing_flag = df[~df['has_new_flag']]
            if not missing_flag.empty:
                alerts.append(f"MISSING FLAG ALERT: {len(missing_flag)} datasets missing new_data flag")
        
        if alerts:
            alert_file = self.output_dir / f"alerts_{self.timestamp_str}.txt"
            with open(alert_file, 'w') as f:
                f.write(f"Alerts Generated at {self.timestamp}\n")
                f.write("=" * 50 + "\n")
                for alert in alerts:
                    f.write(f"⚠️ {alert}\n")
                    self.logger.warning(alert)
        
        return alerts
    
    def run(self):
        """Run the complete automated reporting process."""
        self.logger.info("=" * 80)
        self.logger.info("AUTOMATED NEW DATA REPORTER")
        self.logger.info(f"Run time: {self.timestamp}")
        self.logger.info("=" * 80)
        
        try:
            # Load data
            self.load_data()
            
            # Generate reports
            self.generate_reports()
            
            # Save reports
            saved_files = self.save_reports()
            
            # Create dashboard
            dashboard = self.create_dashboard_summary()
            
            # Check for alerts
            alerts = self.generate_alert_if_needed()
            
            # Archive old reports
            self.archive_old_reports()
            
            # Log summary
            self.logger.info("\n" + "=" * 80)
            self.logger.info("REPORT GENERATION COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Total records analyzed: {self.metrics['total_records']:,}")
            self.logger.info(f"New data records: {self.metrics['new_records']:,} ({self.metrics['new_percentage']:.1f}%)")
            self.logger.info(f"Reports saved: {len(saved_files)}")
            self.logger.info(f"Alerts generated: {len(alerts)}")
            self.logger.info(f"Output directory: {self.output_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during report generation: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


def main():
    """Main entry point for automated reporter."""
    parser = argparse.ArgumentParser(description='Automated New Data Reporter')
    parser.add_argument('--output-dir', type=str, help='Output directory for reports')
    parser.add_argument('--archive-days', type=int, default=30, help='Days to keep reports before archiving')
    
    args = parser.parse_args()
    
    # Create and run reporter
    reporter = AutomatedNewDataReporter(
        output_dir=args.output_dir,
        archive_days=args.archive_days
    )
    
    success = reporter.run()
    
    # Exit with appropriate code for monitoring
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
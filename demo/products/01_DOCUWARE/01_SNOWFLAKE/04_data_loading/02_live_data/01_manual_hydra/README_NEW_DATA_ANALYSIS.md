# Train/New Data Flag Analysis and Reporting

## Overview
This directory contains comprehensive tools for analyzing and reporting on the distribution of training data (flag=0) vs new data (flag=1) across all datasets.

## Key Features

### 1. New Data Flag (0/1)
- **0**: Training data - customers/records that were present in the training dataset
- **1**: New data - customers/records that are NOT in the training dataset

### 2. Analysis Scripts

#### `test_new_manual.py`
Interactive exploration script with enhanced new_data reporting:
- Displays train/new distribution for each dataset
- Shows customer-level and record-level statistics
- Provides sample new customer IDs
- Calculates overall percentages

#### `test_new_data_analysis.py`
Comprehensive analysis and CSV reporting:
- Generates detailed reports in `reports/` directory
- Creates segment-wise analysis
- Time series distribution analysis
- Data quality checks
- Customer lists with train/new flags

#### `automated_new_data_reporter.py`
Automated reporting for scheduled runs:
- Can be run via cron for periodic reports
- Generates timestamped reports in `automated_reports/`
- Creates master summary for trend tracking
- Archives old reports automatically
- Generates alerts for anomalies

#### `view_reports.py`
Simple viewer for generated reports:
- Shows latest summary statistics
- Lists all available reports
- Displays trends from master summary

## Generated Reports

### Reports Directory Structure
```
reports/
├── new_data_summary_YYYYMMDD_HHMMSS.csv       # Overall summary
├── new_data_by_dataset_YYYYMMDD_HHMMSS.csv    # Dataset breakdown
├── new_data_by_segment_YYYYMMDD_HHMMSS.csv    # Customer segment analysis
├── new_data_time_series_YYYYMMDD_HHMMSS.csv   # Weekly distribution
└── customer_list_YYYYMMDD_HHMMSS.csv          # Detailed customer list

automated_reports/
├── summary_YYYYMMDD_HHMMSS.csv                # Run summary
├── dataset_analysis_YYYYMMDD_HHMMSS.csv       # Dataset analysis
├── segment_analysis_YYYYMMDD_HHMMSS.csv       # Segment breakdown
├── weekly_analysis_YYYYMMDD_HHMMSS.csv        # Time series
├── metrics_YYYYMMDD_HHMMSS.json               # JSON metrics
├── dashboard_YYYYMMDD_HHMMSS.json             # Dashboard data
├── latest_dashboard.json                       # Latest dashboard
├── master_summary.csv                          # Historical trends
└── archive/                                    # Old reports
```

## Report Contents

### Summary Report
- Total records analyzed
- Train vs new record counts and percentages
- Unique customer counts for train and new
- Overlapping customers between datasets

### Dataset Analysis
- Per-dataset breakdown of train/new distribution
- Record-level and customer-level statistics
- Identification of datasets missing new_data flag

### Segment Analysis
- Customer segment distribution
- Train vs new percentages by segment
- Identification of segments with high new data

### Time Series Analysis
- Weekly distribution of train vs new data
- Identification of when new data starts appearing
- Trends over time

## Usage Examples

### 1. Interactive Analysis
```bash
python test_new_manual.py
```
This provides real-time exploration with detailed logging.

### 2. Generate Comprehensive Reports
```bash
python test_new_data_analysis.py
```
Creates detailed CSV reports in the `reports/` directory.

### 3. Automated Reporting
```bash
python automated_new_data_reporter.py
```
For scheduled runs via cron:
```bash
0 9 * * * /path/to/python /path/to/automated_new_data_reporter.py
```

### 4. View Reports
```bash
python view_reports.py
```
Quick summary of latest reports.

## Key Metrics Tracked

### Record-Level Metrics
- Total records across all datasets
- Training records (new_data=0)
- New records (new_data=1)
- Percentage distribution

### Customer-Level Metrics
- Unique training customers
- Unique new customers
- Overlapping customers
- Customer distribution by segment

### Quality Metrics
- Datasets with new_data flag
- Flag consistency across datasets
- Invalid flag values detection

## Logging

All scripts provide comprehensive logging including:
- Dataset-by-dataset analysis
- Train/new distribution statistics
- Sample customer IDs for verification
- Data quality issues
- Performance metrics

Logs include special formatting:
- 📊 Dataset information
- 📈 Distribution statistics
- ✅ Successful operations
- ⚠️ Warnings and issues
- 📁 File operations

## Data Quality Checks

The analysis includes automatic quality checks:
1. Verification that new_data values are only 0 or 1
2. Consistency checks across datasets for same customers
3. Detection of missing new_data flags
4. Validation of percentages and counts

## Notes

- The new_data flag is added during data loading by comparing with training data
- All active customers are analyzed (CHURNED_FLAG=0)
- Reports are timestamped for tracking changes over time
- CSV files can be easily imported into Excel or other tools for further analysis

## Troubleshooting

If no new data is found (all records show as training data):
1. This is expected if the current dataset is a subset of training data
2. New customers will appear as new_data=1 when they are added
3. Check the data loading logs for confirmation

## Contact
For questions or issues, contact: evgeni.nikolaev@ricoh-usa.com
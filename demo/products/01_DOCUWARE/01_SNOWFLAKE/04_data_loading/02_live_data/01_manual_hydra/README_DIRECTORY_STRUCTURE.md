# Directory Structure and Script Responsibilities

## Overview
This document explains the directory structure and responsibilities of each script in the new data loading framework.

## 📁 Directory Structure

```
01_manual_hydra/
├── logs/                     # Created by ALL scripts (interactive & automated)
│   └── loguru/              # Loguru log files organized by date
│
├── reports/                  # Created by AUTOMATED analysis scripts only
│   ├── new_data_summary_*.csv
│   ├── new_data_by_dataset_*.csv
│   ├── new_data_by_segment_*.csv
│   ├── new_data_time_series_*.csv
│   └── customer_list_*.csv
│
├── automated_reports/        # Created by AUTOMATED reporter script
│   ├── summary_*.csv
│   ├── dataset_analysis_*.csv
│   ├── segment_analysis_*.csv
│   ├── weekly_analysis_*.csv
│   ├── metrics_*.json
│   ├── dashboard_*.json
│   ├── master_summary.csv
│   └── archive/            # Old reports archived here
│
└── figs/                    # Created by VISUALIZATION script only
    ├── overall_distribution_pie_*.png
    ├── time_series_train_new_*.png
    ├── dataset_distribution_bar_*.png
    ├── segment_distribution_*.png
    ├── weekly_trend_*.png
    └── figure_summary_*.txt
```

## 🔧 Script Responsibilities

### 1. **test_new_manual.py** (INTERACTIVE)
- **Purpose**: Interactive exploration and data viewing
- **Creates**: Only `logs/` directory
- **Does NOT create**: No reports, no figures
- **Output**: Console output only via `print(df.head().to_string())`
- **Use case**: Manual exploration, debugging, understanding data

### 2. **test_new_data_analysis.py** (AUTOMATED ANALYSIS)
- **Purpose**: Comprehensive train/new data analysis
- **Creates**: 
  - `logs/` directory for logging
  - `reports/` directory for CSV reports
- **Output**: Detailed CSV reports with statistics
- **Use case**: Scheduled analysis, detailed reporting

### 3. **automated_new_data_reporter.py** (AUTOMATED REPORTER)
- **Purpose**: Periodic automated reporting with archiving
- **Creates**:
  - `logs/` directory for logging
  - `automated_reports/` directory for reports
  - Archives old reports automatically
- **Output**: Timestamped reports, master summary, JSON metrics
- **Use case**: Cron jobs, scheduled monitoring

### 4. **test_new_data_visualizations.py** (VISUALIZATION)
- **Purpose**: Generate visualization figures
- **Creates**:
  - `logs/` directory for logging
  - `figs/` directory for PNG visualizations
- **Output**: Charts and graphs showing train/new distribution
- **Features**:
  - Vertical dashed lines at transition points
  - Diamond symbols for new data points
  - Time series plots with markers
- **Use case**: Visual reporting, presentations

### 5. **test_complete_data_loading.py** (VALIDATION)
- **Purpose**: Complete end-to-end data loading validation
- **Creates**: Only `logs/` directory
- **Output**: Console validation messages
- **Use case**: Testing, validation of data pipeline

## 📊 New Data Flag Convention

All scripts work with the `new_data` column:
- **0**: Training data - customers/records present in training dataset
- **1**: New data - customers/records NOT in training dataset

## 🔄 Workflow Examples

### Interactive Exploration
```bash
# For manual exploration - no files created except logs
python test_new_manual.py
```

### Generate Reports
```bash
# For CSV reports
python test_new_data_analysis.py

# For visualizations
python test_new_data_visualizations.py
```

### Automated Monitoring
```bash
# For scheduled runs (e.g., via cron)
python automated_new_data_reporter.py
```

### Cron Setup Example
```bash
# Daily report at 9 AM
0 9 * * * cd /path/to/01_manual_hydra && python automated_new_data_reporter.py

# Weekly visualization generation (Mondays at 8 AM)
0 8 * * 1 cd /path/to/01_manual_hydra && python test_new_data_visualizations.py
```

## 🔍 Key Differences

| Script | Interactive | Creates Reports | Creates Figures | For Cron |
|--------|------------|-----------------|-----------------|----------|
| test_new_manual.py | ✅ | ❌ | ❌ | ❌ |
| test_new_data_analysis.py | ❌ | ✅ | ❌ | ✅ |
| automated_new_data_reporter.py | ❌ | ✅ | ❌ | ✅ |
| test_new_data_visualizations.py | ❌ | ❌ | ✅ | ✅ |

## 📝 Notes

1. **Logs are universal**: All scripts create logs in the `logs/` directory
2. **Interactive means no output files**: Interactive scripts only display in console
3. **Automated scripts create files**: For tracking, archiving, and reporting
4. **Separation of concerns**: Each script has a specific purpose and output location
5. **No overlap**: Scripts don't interfere with each other's output directories

## 🚀 Best Practices

1. Use `test_new_manual.py` for exploration and debugging
2. Use automated scripts for production reporting
3. Schedule automated scripts via cron for regular monitoring
4. Check `figs/` directory for the latest visualizations
5. Review `master_summary.csv` for historical trends
6. Archive old reports periodically (automated_new_data_reporter.py does this)

## 📧 Contact

For questions or issues: evgeni.nikolaev@ricoh-usa.com
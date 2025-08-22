# Data Loading Scripts - File Structure Summary

## Overview
This document summarizes the complete file structure for training and new/active customer data loading scripts.

## 📁 Directory Structure

```
04_data_loading/
├── 01_train_data/                     # Training data scripts
│   ├── 01_manual_hydra/
│   │   └── test_train_manual.py       # Interactive training data exploration
│   ├── 02_script/
│   │   └── test_train_script.py       # Automated training data loading with reports
│   └── 03_production/
│       └── test_train_production_script.py  # Production training data pipeline
│
└── 02_new_data/                        # New/Active customer data scripts
    ├── 01_manual_hydra/
    │   └── test_new_manual.py         # Interactive new data exploration
    ├── 02_script/
    │   └── test_new_script.py         # Automated new data loading with reports
    └── 03_production/
        └── test_new_production_script.py    # Production new data pipeline
```

## 🔧 Script Naming Convention

### Training Data Scripts (01_train_data/)
- `test_train_manual.py` - Interactive exploration of training data
- `test_train_script.py` - Development-grade automated training data loading
- `test_train_production_script.py` - Production-grade training data pipeline

### New/Active Data Scripts (02_new_data/)
- `test_new_manual.py` - Interactive exploration of new/active customer data
- `test_new_script.py` - Development-grade automated new data loading
- `test_new_production_script.py` - Production-grade new data pipeline

## 📊 Key Differences

| Aspect | Training Scripts | New Data Scripts |
|--------|-----------------|------------------|
| **Data Source** | All customers (churned & active) | Active customers only |
| **Imputation** | With 2023 H1 imputation | WITHOUT imputation |
| **Churn Analysis** | Full churn metrics | Active customer metrics |
| **Visualizations** | Standard plots | Train/new indicators (shapes, colors, lines) |
| **Comparison** | N/A | Compares with training data |
| **new_data Flag** | Not present | 0=training, 1=new |

## 🎨 Visualization Conventions for New Data

### Color Scheme
- **Training Data**: Blue (#3498db)
- **New Data**: Red (#e74c3c)
- **Transition Line**: Green (#2ecc71)

### Shape Markers
- **Training Data**: Circles ('o')
- **New Data**: Diamonds ('D')

### Special Indicators
- **Vertical Dashed Line**: Marks first appearance of new data in time series
- **Separate Bars**: Side-by-side bars for train vs new in bar plots
- **Overlapping Histograms**: Semi-transparent overlays with different colors

## 📁 Output Directory Structure

Each script creates the following directories:

### Manual Scripts (test_*_manual.py)
```
logs/           # Logging only
```

### Development Scripts (test_*_script.py)
```
logs/           # Logging
reports/        # CSV and JSON reports
figs/           # Visualizations (300 DPI)
```

### Production Scripts (test_*_production_script.py)
```
logs/           # Logging
reports/        # Metrics and monitoring
figs/           # Dashboard (150 DPI optimized)
```

## 💡 Usage Examples

### Training Data
```bash
# Interactive exploration
python test_train_manual.py

# Development with reports
python test_train_script.py

# Production pipeline
python test_train_production_script.py
```

### New/Active Data
```bash
# Interactive exploration
python test_new_manual.py

# Development with reports
python test_new_script.py

# Production pipeline
python test_new_production_script.py
```

## 📝 Key Features by Script Type

### Manual Scripts
- Interactive console output
- No file outputs (except logs)
- Real-time data exploration
- Suitable for debugging

### Development Scripts
- Comprehensive reports and visualizations
- Detailed logging and metrics
- Sample data extraction
- Quality checks and validation

### Production Scripts
- Minimal console output
- Optimized performance
- Error handling and recovery
- Monitoring dashboards
- Ready for scheduling (cron/airflow)

## 🔄 Data Processing Consistency

All scripts follow the same processing pattern:
1. Load data from Snowflake
2. Process usage data (with/without imputation)
3. Merge datasets
4. Prepare time series features
5. Prepare static features
6. Calculate derived features
7. For new data: Compare with training and add new_data flag

## 📊 Reporting Consistency

All automated scripts generate:
- JSON reports for systems integration
- CSV reports for data analysts
- Sample data files for validation
- Performance metrics
- Memory usage statistics

## 🚀 Deployment Notes

- Training scripts: Run during model training phase
- New data scripts: Run for inference on active customers
- Production scripts: Suitable for scheduled execution
- All scripts: Support Hydra configuration overrides

## 📧 Contact

For questions or issues: evgeni.nikolaev@ricoh-usa.com

---
Updated: 2025-08-18
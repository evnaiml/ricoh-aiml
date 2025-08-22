# Ricoh AI/ML - Churn Prediction Framework

<div align="center">

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Active%20Development-orange.svg)
![ML Framework](https://img.shields.io/badge/ML-CatBoost%20%2B%20TSFresh-purple.svg)
![Data Platform](https://img.shields.io/badge/Data-Snowflake-blue.svg)

*A production-ready machine learning framework for customer churn prediction with advanced feature engineering and type-safe data pipelines*

</div>

## 📋 Table of Contents

- [About](#-about)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Documentation](#-documentation)
- [Architecture](#-architecture)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#-authors)
- [Acknowledgments](#-acknowledgments)

## 📖 About

The Ricoh AI/ML Churn Prediction Framework is an enterprise-grade solution designed to predict customer churn with high accuracy using advanced machine learning techniques. Built specifically for DOCUWARE product analytics, this framework combines state-of-the-art time series feature engineering with gradient boosting models to deliver actionable insights about customer retention.

### Problem Statement
Current churn prediction models are overestimating churn rates by 4-8x (predicting ~50% annual churn vs. actual 10-15%), leading to inefficient resource allocation and missed opportunities for customer retention.

### Solution
Our framework addresses this through:
- **TSFresh Feature Engineering**: Automatic extraction of 794+ time series features with intelligent selection to the top 150 most predictive features
- **CatBoost with GPU Acceleration**: Native handling of high-cardinality categorical features using 4 GPUs (24GB each) for rapid training
- **Type-Safe Data Pipeline**: Pydantic schema validation ensuring data consistency from Snowflake to model training
- **Production-Ready Infrastructure**: Comprehensive logging, monitoring, and visualization dashboards

## ✨ Key Features

- 🚀 **High-Performance Computing**: Leverages 48 CPUs and 4 GPUs for distributed processing
- 📊 **Advanced Feature Engineering**: TSFresh-based time series feature extraction with multi-method selection
- 🔒 **Type-Safe Data Pipeline**: Pydantic validation with automatic type conversion and error handling
- 📈 **Comprehensive Visualization**: Centralized dashboard for churn analysis and model monitoring
- 🔄 **Boolean Standardization**: Consistent 0/1 encoding for all ML operations
- 🏭 **Production-Grade Logging**: Structured logging with Loguru for debugging and monitoring
- 🎯 **Merge Validation Protocol**: Automated checks preventing silent pipeline failures
- 📝 **Extensive Documentation**: 21+ example scripts demonstrating best practices
- 🔀 **Dual Data Loading Pipelines**: Separate pipelines for training (with imputation) and inferencing (without imputation)
- 🏷️ **New Customer Detection**: Automatic identification and flagging of customers not in training data

## 📁 Project Structure

```
ricoh_aiml/                                                 ✅ Project root directory
├── .vscode/                                                ✅ VSCode IDE configuration settings
├── conf/                                                   ✅ Hydra configuration files for all components
│   ├── loggers/                                            ✅ Logging configuration settings
│   │   └── loguru/                                         ✅ Loguru logger configurations
│   ├── paths_to_folders/                                   ✅ Path configuration for project directories
│   ├── products/                                           ✅ Product-specific configurations
│   │   └── DOCUWARE/                                       ✅ DOCUWARE product configurations
│   │       └── db/                                         ✅ Database connection configurations
│   │           └── snowflake/                              ✅ Snowflake-specific settings
│   │               ├── csv_config/                         ✅ CSV export/import configurations
│   │               └── sequel_rules/                       ✅ SQL join query definitions
│   └── validation/                                         ✅ Data validation configurations
│       └── pydantic/                                       ✅ Pydantic schema settings
├── data/                                                   ✅ Output directory for generated data
│   ├── feature_store/                                      ✅ Processed features for ML models
│   └── products/                                           ✅ Product-specific data outputs
│       └── DOCUWARE/                                       ✅ DOCUWARE data outputs
│           └── DB/                                         ✅ Database-related outputs
│               └── snowflake/                              ✅ Snowflake query results
│                   └── csv/                                ✅ CSV export files
│                       ├── join_rules/                     ✅ Joined table schemas
│                       └── raw/                            ✅ Raw table schemas
├── demo/                                                   ✅ Demo examples and notebooks
│   └── products/                                           ✅ Product-specific demos
│       └── 01_DOCUWARE/                                    ✅ DOCUWARE demonstrations
│           └── 01_SNOWFLAKE/                               ✅ Snowflake data fetching examples
│               ├── 01_fetch_examples_no_validation/        ✅ Basic fetching without validation
│               ├── 02_fetch_examples_validation_created/   ✅ Schema generation from data
│               ├── 03_fetch_examples_validation_enforced/  ✅ Type enforcement with schemas
│               └── 04_data_loading/                        ✅ Comprehensive data loading framework
│                   ├── 01_train_data/                      ✅ Training data loading scripts
│                   │   ├── 01_manual_hydra/                ✅ Interactive exploration
│                   │   │   └── test_train_manual.py        ✅ Manual training data exploration
│                   │   ├── 02_script/                      ✅ Automated with reports
│                   │   │   └── test_train_script.py        ✅ Development training script
│                   │   └── 03_production/                  ✅ Production pipeline
│                   │       └── test_train_production_script.py ✅ Production training pipeline
│                   └── 02_live_data/                        ✅ Live/Active customer data scripts
│                       ├── 01_manual_hydra/                ✅ Interactive exploration
│                       │   └── test_new_manual.py          ✅ Manual new data exploration
│                       ├── 02_script/                      ✅ Automated with reports
│                       │   └── test_new_script.py          ✅ Development new data script
│                       └── 03_production/                  ✅ Production pipeline
│                           └── test_new_production_script.py ✅ Production new data pipeline
├── examples/                                               ✅ Additional code examples
│   └── 01_manual_hydra/                                    ✅ Manual Hydra configuration examples
├── logs/                                                   ✅ Application log files
├── outputs/                                                ✅ Generated outputs and results
├── src/                                                    ✅ Source code root
│   └── churn_aiml/                                         ✅ Main application package
│       ├── data/                                           ✅ Data processing modules
│       │   ├── csv/                                        ✅ CSV handling utilities
│       │   ├── db/                                         ✅ Database connectors
│       │   │   └── snowflake/                              ✅ Snowflake connector
│       │   │       ├── fetchdata.py                        ✅ Enhanced data fetcher with analyzers
│       │   │       └── loaddata.py                         ✅ SnowTrainDataLoader for ML pipelines
│       │   ├── dtype_preservers/                           ✅ Data type preservation utilities
│       │   ├── pandas_ml/                                  ✅ Pandas ML extensions
│       │   ├── parquet_format/                             ✅ Parquet file handling
│       │   └── validation/                                 ✅ Data validation modules
│       ├── loggers/                                        ✅ Logging configurations
│       │   └── loguru/                                     ✅ Loguru logger setup
│       ├── ml/                                             ✅ Machine learning modules
│       │   ├── datetime/                                   ✅ Datetime processing
│       │   └── models/                                     ✅ ML model utilities
│       ├── utils/                                          ✅ General utilities
│       ├── validation/                                     ✅ Validation utilities
│       │   └── pydantic/                                   ✅ Pydantic validators
│       ├── visualization/                                  ✅ Data visualization modules
│       │   └── churn_plots/                                ✅ Churn visualization framework
│       ├── xdata_processing/                               ✅ Extended data processing
│       ├── xeda/                                           ✅ Exploratory data analysis
│       ├── xfeatures/                                      ✅ Feature engineering
│       └── xprediction/                                    ✅ Prediction utilities
├── tests/                                                  ✅ Test suite
├── .env                                                    ✅ Environment variables
├── .gitignore                                              ✅ Git ignore patterns
├── README.md                                               ✅ This documentation file
├── requirements.txt                                        ✅ Python dependencies
└── setup.py                                                ✅ Package setup configuration
```

## 🚀 Installation

### Prerequisites

- Python 3.10.18
- CUDA 12.9 compatible GPUs (4x GPUs with 24GB memory recommended)
- Snowflake account with appropriate permissions
- 48+ CPU cores for optimal performance
- Conda/Mamba package manager

### Key Dependencies

The project uses a comprehensive ML stack including:
- **Deep Learning**: PyTorch 2.6.0+cu124, CUDA 12.9
- **ML Frameworks**: CatBoost 1.2.8, XGBoost 3.0.3, scikit-learn 1.7.1
- **Feature Engineering**: TSFresh 0.21.0, Feature-engine 1.8.3, SHAP 0.48.0
- **Data Processing**: Pandas 2.3.1, Dask 2025.7.0, RAPIDS cuDF 25.08.00
- **Database**: Snowflake-connector-python 3.16.0, Snowpark 1.35.0
- **Configuration**: Hydra-core 1.3.2, OmegaConf 2.3.0, Pydantic 2.11.7
- **Logging**: Loguru 0.7.2
- **Visualization**: Matplotlib 3.10.5, Seaborn 0.13.2, Plotly 6.2.0
- **Experiment Tracking**: ClearML 2.0.2

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/ricoh/churn-aiml.git
   cd ricoh_aiml
   ```

2. **Create conda environment from requirements file**
   ```bash
   # Create environment with all dependencies
   conda env create -f requirements.txt
   conda activate churn-ml-env
   ```

   Or create manually:
   ```bash
   # Create base environment
   conda create -n churn-ml-env python=3.10.18
   conda activate churn-ml-env
   
   # Add channels
   conda config --add channels rapidsai
   conda config --add channels nvidia
   conda config --add channels conda-forge
   
   # Install key packages
   conda install -c rapidsai -c nvidia -c conda-forge \
     catboost=1.2.8 \
     dask=2025.7.0 \
     hydra-core=1.3.2 \
     loguru=0.7.2 \
     pandas=2.3.1 \
     pydantic=2.11.7 \
     snowflake-connector-python=3.16.0 \
     tsfresh=0.21.0
   ```

3. **Install the package in development mode**
   ```bash
   pip install -e .

   ```

4. **Configure Snowflake connection**
   ```bash
   # Edit the configuration file
   vim conf/products/DOCUWARE/db/snowflake/sessions/development.yaml
   ```

   Add your credentials:
   ```yaml
   account: your_account
   user: your_username
   password: your_password
   warehouse: your_warehouse
   database: your_database
   schema: your_schema
   ```

5. **Set environment variables**
   ```bash
   # Create .env file if needed
   touch .env
   # Add any required environment variables
   echo "NUMBA_DISABLE_CUDA=1" >> .env  # Optional: disable CUDA for Numba if needed
   ```

### Note on requirements.txt

The `requirements.txt` file is a conda environment specification file, not a pip requirements file. It includes:
- Conda channels configuration (rapidsai, nvidia, conda-forge)
- Comprehensive list of conda dependencies with exact versions
- CUDA 12.9 toolkit and related libraries
- RAPIDS ecosystem for GPU-accelerated data processing
- All necessary ML, data processing, and visualization libraries

## 🎯 Quick Start

### Basic Data Fetching

```python
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Initialize fetcher with type enforcement
    with SnowFetch(config=cfg, environment="development") as fetcher:
        # Fetch data with automatic type conversion
        df = fetcher.fetch_data_validation_enforced(
            table_name="PS_DOCUWARE_L1_CUST",
            context="raw"
        )
        print(f"Fetched {len(df)} records with validated types")

if __name__ == "__main__":
    main()
```

### Loading Training Data

```python
from churn_aiml.data.db.snowflake.loaddata import SnowTrainDataLoader
from churn_aiml.visualization.churn_plots import ChurnLifecycleVizSnowflake

# Initialize data loader
data_loader = SnowTrainDataLoader(config=cfg)
data_dict = data_loader.load_data()

# Create visualizations
viz = ChurnLifecycleVizSnowflake()
viz.plot_churn_distribution(data_dict['customer_metadata'])
viz.plot_usage_trends(data_dict['time_series_features'])
```

## 📚 Usage

### Demo Examples

The `demo/` directory contains comprehensive examples progressing from basic to production-ready implementations:

#### 1. Basic Fetching (No Validation)
```bash
# Single table fetching
cd demo/products/01_DOCUWARE/01_SNOWFLAKE/01_fetch_examples_no_validation/
python no_rules/01_manual_no_rules/manual.py

# With SQL join rules
python with_sequel_rules/01_manual_with_join_rules/manual.py
```

#### 2. Schema Generation
```bash
# Generate schema from single table
cd demo/products/01_DOCUWARE/01_SNOWFLAKE/02_fetch_examples_validation_created/
python no_rules/01_manual_no_rules/test_manual.py

# Script-based generation
python no_rules/02_script_no_rules/test_script.py

# Production-grade schema generation
python no_rules/03_production_no_rules/production_script.py
```

#### 3. Type Enforcement with Validation
```bash
# Enforce types using schemas
cd demo/products/01_DOCUWARE/01_SNOWFLAKE/03_fetch_examples_validation_enforced/
python no_rules/01_manual_no_rules/test_manual.py

# With join rules and type enforcement
python with_rules/03_production_with_join_rules/production_script.py
```

#### 4. Comprehensive Data Loading

##### Training Data Scripts
```bash
# Interactive exploration of training data
cd demo/products/01_DOCUWARE/01_SNOWFLAKE/04_data_loading/01_train_data/
python 01_manual_hydra/test_train_manual.py

# Automated training data script with visualizations
python 02_script/test_train_script.py

# Production training data pipeline
python 03_production/test_train_production_script.py
```

##### New/Active Customer Data Scripts
```bash
# Interactive exploration of new/active customers
cd demo/products/01_DOCUWARE/01_SNOWFLAKE/04_data_loading/02_live_data/
python 01_manual_hydra/test_new_manual.py

# Automated new data script with train/new comparison
python 02_script/test_new_script.py

# Production new data pipeline for inferencing
python 03_production/test_new_production_script.py
```

### Key Components

#### Enhanced SnowFetch
```python
from churn_aiml.data.db.snowflake.fetchdata import (
    SnowFetch, 
    DtypeTransformationAnalyzer,
    JoinRuleAnalyzer
)

# Analyze data transformations
analyzer = DtypeTransformationAnalyzer(logger)
report = analyzer.analyze_table(fetcher, df_before, df_after, table_name)
```

#### Data Loading Framework

##### Training Data Loading
```python
from churn_aiml.data.db.snowflake.loaddata import SnowTrainDataLoader

# Load training data with imputation
loader = SnowTrainDataLoader(config=cfg, environment="development")
data = loader.load_data()  # Returns dict with multiple DataFrames
feature_df = loader.get_feature_engineering_dataset()  # TSFresh-ready
training_df = loader.get_training_ready_data()  # Combined features
churn_summary = loader.get_churn_summary()  # Churn statistics
```

##### New/Active Customer Data Loading
```python
from churn_aiml.data.db.snowflake.loaddata import SnowLiveDataLoader, SnowTrainDataLoader

# Load training data for comparison
train_loader = SnowTrainDataLoader(config=cfg)
train_data = train_loader.load_data()

# Load new/active customer data WITHOUT imputation
live_loader = SnowLiveDataLoader(config=cfg, environment="development")
new_data = new_loader.load_data(train_data_loader=train_loader)
# Data includes 'new_data' column: 0=training, 1=new

# Get feature engineering dataset for inferencing
fe_dataset = new_loader.get_feature_engineering_dataset()
active_summary = new_loader.get_active_summary()  # Active customer statistics
```

#### Visualization Framework
```python
from churn_aiml.visualization.churn_plots import ChurnLifecycleVizSnowflake

viz = ChurnLifecycleVizSnowflake()

# Standard visualizations for training data
viz.plot_all_distributions(data_dict, save_path="./figures")
viz.create_monitoring_dashboard(metrics, data_dict)

# Enhanced visualizations for new data with train/new indicators:
# - Blue (#3498db) for training data, Red (#e74c3c) for new data
# - Circles ('o') for training, Diamonds ('D') for new in scatter plots
# - Vertical dashed lines at transition points in time series
# - Separate bars for train vs new in bar plots
```

## 📖 Documentation

### Module Documentation

Each module follows the project's documentation pattern:

```python
"""
Comprehensive module description with production-ready features.

This module provides [functionality description] suitable for enterprise environments.

## Module Organization:

### Core Classes:
1. **ClassName** (Type)
   - Description of class purpose
   - Key responsibilities

### Utility Functions:
- function_name() - Brief description

## Key Features:
- ✅ Feature 1 with benefit
- ✅ Feature 2 with benefit

## Usage:
    Basic usage example

## Author:
    Evgeni Nikolaev
    Email: evgeni.nikolaev@ricoh-usa.com
    Ricoh AI/ML Team
    
## Version:
    1.0.0
    
## Created:
    2025-07-29
    
## Last Updated:
    2025-08-16
"""
```

### Configuration Management

All configurations are managed through Hydra:
- **Database**: `conf/products/DOCUWARE/db/snowflake/`
- **Logging**: `conf/loggers/loguru/`
- **Validation**: `conf/validation/pydantic/`
- **Paths**: `conf/paths_to_folders/`

### Data Validation

Pydantic schemas ensure type safety:
- Automatic generation from Snowflake data
- Type enforcement with error handling
- Boolean standardization (0/1 encoding)
- NaN/NaT handling for failed conversions

## 🏗️ Architecture

### Data Pipeline Architecture

```
Snowflake → Pydantic Validation → Type Enforcement → Feature Engineering → Model Training
     ↓              ↓                    ↓                   ↓                  ↓
Raw Data    Schema Validation    Clean Types        TSFresh Features    CatBoost Model
```

### Data Loading Framework Architecture

#### Training Data Pipeline (SnowTrainDataLoader)
```
Raw Data → Usage Processing → 2023 H1 Imputation → Merge Datasets → Feature Engineering
    ↓            ↓                    ↓                 ↓                    ↓
All Customers  Weekly Agg    Fill Missing Values  Combined Data    TSFresh Features
```

#### Live/Active Data Pipeline (SnowLiveDataLoader)
```
Raw Data → Filter Active → Usage Processing → NO Imputation → Compare with Training → Add Flag
    ↓           ↓                ↓                  ↓                ↓                    ↓
All Data   CHURNED_FLAG=0   Weekly Agg      Keep Missing     Identify New      new_data: 0/1
```

#### Key Differences
| Aspect | Training Pipeline | New Data Pipeline |
|--------|------------------|-------------------|
| **Customers** | All (churned & active) | Active only (CHURNED_FLAG=0) |
| **Imputation** | Yes (2023 H1 method) | No (keep missing values) |
| **Purpose** | Model training | Inferencing |
| **Output** | Training features | Features with new_data flag |
| **Comparison** | N/A | Compares with training data |

### Feature Engineering Pipeline

```python
Pipeline([
    ('tsfresh', TSFreshTransformer()),
    ('selection', TSFreshFeatureSelector(
        methods=['tsfresh_fdr', 'boruta', 'catboost', 'shap'],
        max_features=150
    ))
])
```

### Model Architecture

```python
TwoStageCatBoostChurnModel:
    - Stage 1: Survival regression (when will they churn?)
    - Stage 2: Calibrated classification (will they churn at time t?)
```

### Technology Stack

- **Data Platform**: Snowflake
- **ML Framework**: CatBoost (GPU-accelerated)
- **Feature Engineering**: TSFresh
- **Data Validation**: Pydantic
- **Configuration**: Hydra
- **Logging**: Loguru
- **Visualization**: Matplotlib/Seaborn
- **Parallelization**: Dask

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Follow the coding standards:
   - Use type hints for all functions
   - Include comprehensive docstrings
   - Add unit tests for new features
   - Update documentation as needed

### Coding Standards

- **Docstrings**: Follow the project's documentation pattern
- **Type Hints**: Required for all function parameters and returns
- **Logging**: Use structured logging with Loguru
- **Testing**: Maintain >80% code coverage
- **Commits**: Use descriptive commit messages

### Pull Request Process

1. Update the README.md with details of changes
2. Ensure all tests pass
3. Update documentation
4. Request review from maintainers

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

**Ricoh AI/ML Team**

- **Lead Data Scientist**: Evgeni Nikolaev (evgeni.nikolaev@ricoh-usa.com)
- **ML Engineers**: Contributing Team Members
- **Data Engineers**: Contributing Team Members

*For questions or support, please contact: evgeni.nikolaev@ricoh-usa.com*

## 📄 Copyright

```
COPYRIGHT © 2025 Ricoh. All rights reserved.
The information contained herein is copyright and proprietary to
Ricoh and may not be reproduced, disclosed, or used in
any manner without prior written permission from Ricoh.
```

## 🙏 Acknowledgments

- **CatBoost Team** - For the excellent gradient boosting framework
- **TSFresh Contributors** - For the time series feature extraction library
- **Pydantic Team** - For the data validation framework
- **Hydra Framework** - For configuration management
- **Snowflake** - For the data platform

## 📊 Recent Updates

### August 16, 2025
- Enhanced documentation with professional README structure
- Added comprehensive module docstrings following project patterns
- Maintained complete project tree structure
- Updated with accurate dependency versions from requirements.txt
- Added proper author attribution and copyright information

### August 15, 2025
- Visualization improvements with enhanced readability
- Business logic alignment for churn metrics
- Font size optimization and marker adjustments

### August 14, 2025
- Complete test_manual.py refactoring to production-ready code
- Boolean variable standardization (0/1 encoding)
- Schema-based type conversion infrastructure
- Comprehensive business variable documentation

### August 13, 2025
- Enhanced SnowFetch with dtype transformation analysis
- Centralized analysis utilities in fetchdata.py
- Created 18+ example scripts for data operations

### August 11-12, 2025
- Initial framework design and architecture
- TSFresh + CatBoost pipeline implementation
- Diagnostic tools for prediction analysis

---

<div align="center">

**Built with ❤️ by Ricoh AI/ML Team**

*Project Lead: Evgeni Nikolaev (evgeni.nikolaev@ricoh-usa.com)*

*Empowering data-driven decisions through advanced machine learning*

**©2025 Ricoh. All rights reserved.**

</div>
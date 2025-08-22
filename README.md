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
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Documentation](#-documentation)
- [Architecture](#-architecture)
- [Contributing](#-contributing)
- [License](#-license)
- [Authors](#-authors)
- [Acknowledgments](#-acknowledgments)

## 📖 About

The Ricoh AI/ML Churn Prediction Framework is an enterprise-grade solution designed to predict customer churn with high accuracy using advanced machine learning techniques.

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

- **Lead Data Scientist**: Evgeni Nikolaev, PhD (evgeni.nikolaev@ricoh-usa.com)

*For questions or support, please contact: evgeni.nikolaev@ricoh-usa.com*

## 📄 Copyright

```
COPYRIGHT © 2025 Ricoh. All rights reserved.
The information contained herein is copyright and proprietary to
Ricoh and may not be reproduced, disclosed, or used in
any manner without prior written permission from Ricoh.
```

---

<div align="center">

**Built with ❤️ by Ricoh AI/ML Team**

*Project Lead: Evgeni Nikolaev (evgeni.nikolaev@ricoh-usa.com)*

*Empowering data-driven decisions through advanced machine learning*

**©2025 Ricoh. All rights reserved.**

</div>
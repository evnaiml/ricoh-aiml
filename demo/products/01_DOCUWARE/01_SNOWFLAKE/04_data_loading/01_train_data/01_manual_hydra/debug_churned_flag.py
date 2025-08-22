#!/usr/bin/env python
"""Debug script to check CHURNED_FLAG values and customer counting logic"""

import sys
import os
sys.path.append('/home/applaimlgen/ricoh_aiml')
os.chdir('/home/applaimlgen/ricoh_aiml/demo/products/01_DOCUWARE/01_SNOWFLAKE/04_data_loading/01_manual_hydra')

import pandas as pd
import hydra
from omegaconf import DictConfig
from pathlib import Path
from churn_aiml.data.db.snowflake.loaddata import SnowTrainDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger

@hydra.main(config_path="../../../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup logging
    logger_config = setup_logger_for_script(cfg, __file__)
    logger = get_logger()
    
    logger.info("Debugging CHURNED_FLAG values and customer counting")
    
    # Initialize data loader
    data_loader = SnowTrainDataLoader(config=cfg, environment="development")
    
    # Load data
    logger.info("Loading data...")
    data_dict = data_loader.load_data()
    
    # Check feature engineering dataset
    if 'feature_engineering_dataset' in data_dict:
        fe_dataset = data_dict['feature_engineering_dataset']
        
        print("\n" + "=" * 80)
        print("CHURNED_FLAG ANALYSIS IN FEATURE ENGINEERING DATASET")
        print("=" * 80)
        
        if 'CHURNED_FLAG' in fe_dataset.columns:
            print(f"\nCHURNED_FLAG dtype: {fe_dataset['CHURNED_FLAG'].dtype}")
            print(f"Total rows: {len(fe_dataset)}")
            
            # Show unique values
            unique_vals = fe_dataset['CHURNED_FLAG'].unique()
            print(f"\nUnique values in CHURNED_FLAG: {unique_vals}")
            
            # Show value counts
            print("\nValue counts:")
            print(fe_dataset['CHURNED_FLAG'].value_counts())
            
            # Check for different encodings
            print("\n" + "-" * 40)
            print("Testing different encodings:")
            print("-" * 40)
            
            # Test Int64 encoding (0/1)
            if fe_dataset['CHURNED_FLAG'].dtype == 'Int64' or fe_dataset['CHURNED_FLAG'].dtype == 'int64':
                print("\nTesting integer encoding (0/1):")
                churned_1 = fe_dataset[fe_dataset['CHURNED_FLAG'] == 1]
                active_0 = fe_dataset[fe_dataset['CHURNED_FLAG'] == 0]
                print(f"  Rows with CHURNED_FLAG == 1: {len(churned_1)}")
                print(f"  Unique customers with CHURNED_FLAG == 1: {churned_1['CUST_ACCOUNT_NUMBER'].nunique()}")
                print(f"  Rows with CHURNED_FLAG == 0: {len(active_0)}")
                print(f"  Unique customers with CHURNED_FLAG == 0: {active_0['CUST_ACCOUNT_NUMBER'].nunique()}")
            
            # Test string encoding (Y/N)
            print("\nTesting string encoding (Y/N):")
            churned_y = fe_dataset[fe_dataset['CHURNED_FLAG'] == 'Y']
            active_n = fe_dataset[fe_dataset['CHURNED_FLAG'] == 'N']
            print(f"  Rows with CHURNED_FLAG == 'Y': {len(churned_y)}")
            print(f"  Unique customers with CHURNED_FLAG == 'Y': {churned_y['CUST_ACCOUNT_NUMBER'].nunique() if not churned_y.empty else 0}")
            print(f"  Rows with CHURNED_FLAG == 'N': {len(active_n)}")
            print(f"  Unique customers with CHURNED_FLAG == 'N': {active_n['CUST_ACCOUNT_NUMBER'].nunique() if not active_n.empty else 0}")
            
            # Test boolean encoding
            print("\nTesting boolean encoding (True/False):")
            churned_true = fe_dataset[fe_dataset['CHURNED_FLAG'] == True]
            active_false = fe_dataset[fe_dataset['CHURNED_FLAG'] == False]
            print(f"  Rows with CHURNED_FLAG == True: {len(churned_true)}")
            print(f"  Unique customers with CHURNED_FLAG == True: {churned_true['CUST_ACCOUNT_NUMBER'].nunique() if not churned_true.empty else 0}")
            print(f"  Rows with CHURNED_FLAG == False: {len(active_false)}")
            print(f"  Unique customers with CHURNED_FLAG == False: {active_false['CUST_ACCOUNT_NUMBER'].nunique() if not active_false.empty else 0}")
            
            # Show sample data
            print("\n" + "-" * 40)
            print("Sample data (first 10 rows):")
            print("-" * 40)
            print(fe_dataset[['CUST_ACCOUNT_NUMBER', 'CHURNED_FLAG', 'DAYS_TO_CHURN']].head(10))
            
            # Check if all customers are churned
            print("\n" + "-" * 40)
            print("Customer distribution summary:")
            print("-" * 40)
            
            # Get unique customers by churned flag value
            customer_groups = fe_dataset.groupby('CHURNED_FLAG')['CUST_ACCOUNT_NUMBER'].nunique()
            print("\nUnique customers by CHURNED_FLAG value:")
            for flag_val, count in customer_groups.items():
                print(f"  CHURNED_FLAG = {repr(flag_val)}: {count} customers")
            
            # Check if there's a CHURN_DATE to determine active vs churned
            if 'CHURN_DATE' in fe_dataset.columns:
                print("\n" + "-" * 40)
                print("CHURN_DATE analysis:")
                print("-" * 40)
                has_churn_date = fe_dataset['CHURN_DATE'].notna()
                print(f"  Rows with CHURN_DATE: {has_churn_date.sum()}")
                print(f"  Rows without CHURN_DATE: {(~has_churn_date).sum()}")
                print(f"  Unique customers with CHURN_DATE: {fe_dataset[has_churn_date]['CUST_ACCOUNT_NUMBER'].nunique()}")
                print(f"  Unique customers without CHURN_DATE: {fe_dataset[~has_churn_date]['CUST_ACCOUNT_NUMBER'].nunique()}")
                
        else:
            print("CHURNED_FLAG column not found in feature engineering dataset!")
    else:
        print("Feature engineering dataset not found in data_dict!")
    
    print("\n" + "=" * 80)
    print("Debug analysis complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
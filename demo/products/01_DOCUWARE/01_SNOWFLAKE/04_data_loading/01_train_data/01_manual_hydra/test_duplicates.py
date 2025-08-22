#!/usr/bin/env python
"""Test script to verify deduplication is working properly"""

import sys
import os
sys.path.append('/home/applaimlgen/ricoh_aiml')
os.chdir('/home/applaimlgen/ricoh_aiml/demo/products/01_DOCUWARE/01_SNOWFLAKE/04_data_loading/01_manual_hydra')

import pandas as pd
import hydra
from omegaconf import DictConfig
from churn_aiml.data.db.snowflake.loaddata import SnowTrainDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger

@hydra.main(config_path="../../../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Setup logging
    logger_config = setup_logger_for_script(cfg, __file__)
    logger = get_logger()
    
    logger.info("Testing deduplication in data loading")
    
    # Initialize data loader
    data_loader = SnowTrainDataLoader(config=cfg, environment="development")
    
    # Load data
    logger.info("Loading data...")
    data_dict = data_loader.load_data()
    
    print("\n" + "=" * 80)
    print("DEDUPLICATION VERIFICATION REPORT")
    print("=" * 80)
    
    # Check each dataset for duplicates
    for dataset_name, df in data_dict.items():
        if df is not None and not df.empty:
            print(f"\n{dataset_name}:")
            print(f"  Total rows: {len(df):,}")
            
            # Check for complete row duplicates
            duplicates = df.duplicated().sum()
            print(f"  Complete duplicate rows: {duplicates}")
            
            # Check for duplicates on key columns depending on dataset
            if 'CUST_ACCOUNT_NUMBER' in df.columns and 'YYYYWK' in df.columns:
                # For time series data
                dup_key = df.duplicated(subset=['CUST_ACCOUNT_NUMBER', 'YYYYWK']).sum()
                print(f"  Duplicates on (CUST_ACCOUNT_NUMBER, YYYYWK): {dup_key}")
                
                if dup_key > 0:
                    print("    ⚠️ WARNING: Found duplicate customer-week combinations!")
                    # Show sample duplicates
                    dup_mask = df.duplicated(subset=['CUST_ACCOUNT_NUMBER', 'YYYYWK'], keep=False)
                    sample = df[dup_mask].head(10)
                    print(f"    Sample duplicates:\n{sample[['CUST_ACCOUNT_NUMBER', 'YYYYWK']].to_string()}")
            
            elif 'CUST_ACCOUNT_NUMBER' in df.columns:
                # For customer-level data
                dup_key = df.duplicated(subset=['CUST_ACCOUNT_NUMBER']).sum()
                print(f"  Duplicates on CUST_ACCOUNT_NUMBER: {dup_key}")
                
                if dup_key > 0:
                    print("    ⚠️ WARNING: Found duplicate customers!")
    
    # Specifically check feature engineering dataset
    if 'feature_engineering_dataset' in data_dict:
        fe_dataset = data_dict['feature_engineering_dataset']
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING DATASET DETAILED CHECK")
        print("=" * 80)
        
        print(f"\nTotal rows: {len(fe_dataset):,}")
        print(f"Unique customers: {fe_dataset['CUST_ACCOUNT_NUMBER'].nunique()}")
        print(f"Unique weeks: {fe_dataset['YYYYWK'].nunique()}")
        
        # Calculate expected vs actual rows
        expected_max = fe_dataset['CUST_ACCOUNT_NUMBER'].nunique() * fe_dataset['YYYYWK'].nunique()
        print(f"Max possible rows (customers × weeks): {expected_max:,}")
        print(f"Actual density: {len(fe_dataset) / expected_max * 100:.1f}%")
        
        # Check for any customer-week duplicates
        dup_check = fe_dataset.groupby(['CUST_ACCOUNT_NUMBER', 'YYYYWK']).size()
        duplicates = dup_check[dup_check > 1]
        
        if len(duplicates) > 0:
            print(f"\n⚠️ Found {len(duplicates)} customer-week combinations with duplicates!")
            print("Sample duplicate combinations:")
            print(duplicates.head(10))
        else:
            print("\n✅ No duplicate customer-week combinations found!")
    
    print("\n" + "=" * 80)
    print("Deduplication verification complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
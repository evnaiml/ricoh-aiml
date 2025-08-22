#!/usr/bin/env python
"""Debug script to check why CHURNED_FLAG has only 1 unique value"""

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
    
    logger.info("Debugging why CHURNED_FLAG has only 1 unique value")
    
    # Initialize data loader
    data_loader = SnowTrainDataLoader(config=cfg, environment="development")
    
    # Load data
    logger.info("Loading data...")
    data_dict = data_loader.load_data()
    
    # Check all datasets for CHURNED_FLAG
    print("\n" + "=" * 80)
    print("CHURNED_FLAG ANALYSIS ACROSS ALL DATASETS")
    print("=" * 80)
    
    for dataset_name, df in data_dict.items():
        if df is not None and not df.empty:
            print(f"\n{dataset_name}:")
            print(f"  Shape: {df.shape}")
            
            if 'CHURNED_FLAG' in df.columns:
                print(f"  CHURNED_FLAG found!")
                print(f"    dtype: {df['CHURNED_FLAG'].dtype}")
                unique_vals = df['CHURNED_FLAG'].unique()
                print(f"    Unique values: {unique_vals}")
                print(f"    Number of unique values: {len(unique_vals)}")
                print(f"    Value counts:")
                value_counts = df['CHURNED_FLAG'].value_counts()
                for val, count in value_counts.items():
                    print(f"      {repr(val)}: {count} rows")
                
                # Check null values
                null_count = df['CHURNED_FLAG'].isna().sum()
                print(f"    Null values: {null_count}")
                
                # Show sample
                print(f"    Sample values (first 10):")
                print(f"      {df['CHURNED_FLAG'].head(10).tolist()}")
                
            else:
                print(f"  CHURNED_FLAG not found in this dataset")
    
    # Specifically check the feature engineering dataset creation
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING DATASET CREATION ANALYSIS")
    print("=" * 80)
    
    # Get the source datasets
    if 'time_series_features' in data_dict:
        tsf = data_dict['time_series_features']
        print(f"\ntime_series_features:")
        print(f"  Shape: {tsf.shape}")
        if 'CHURNED_FLAG' in tsf.columns:
            print(f"  CHURNED_FLAG unique values: {tsf['CHURNED_FLAG'].unique()}")
            print(f"  CHURNED_FLAG value counts: {tsf['CHURNED_FLAG'].value_counts().to_dict()}")
    
    if 'static_features' in data_dict:
        sf = data_dict['static_features']
        print(f"\nstatic_features:")
        print(f"  Shape: {sf.shape}")
        if 'CHURNED_FLAG' in sf.columns:
            print(f"  CHURNED_FLAG unique values: {sf['CHURNED_FLAG'].unique()}")
            print(f"  CHURNED_FLAG value counts: {sf['CHURNED_FLAG'].value_counts().to_dict()}")
    
    # Check the merge process
    print("\n" + "=" * 80)
    print("CHECKING DATA MERGE PROCESS")
    print("=" * 80)
    
    # Manually recreate the feature engineering dataset to debug
    if 'time_series_features' in data_dict and 'static_features' in data_dict:
        tsf = data_dict['time_series_features']
        sf = data_dict['static_features']
        
        print(f"\nBefore merge:")
        print(f"  time_series_features unique customers: {tsf['CUST_ACCOUNT_NUMBER'].nunique()}")
        print(f"  static_features unique customers: {sf['CUST_ACCOUNT_NUMBER'].nunique()}")
        
        # Check overlap
        tsf_customers = set(tsf['CUST_ACCOUNT_NUMBER'].unique())
        sf_customers = set(sf['CUST_ACCOUNT_NUMBER'].unique())
        common_customers = tsf_customers & sf_customers
        
        print(f"\nCustomer overlap:")
        print(f"  Common customers: {len(common_customers)}")
        print(f"  Only in time_series: {len(tsf_customers - sf_customers)}")
        print(f"  Only in static: {len(sf_customers - tsf_customers)}")
        
        # Check CHURNED_FLAG for common customers
        if common_customers:
            sample_customer = list(common_customers)[0]
            print(f"\nSample customer {sample_customer}:")
            
            if 'CHURNED_FLAG' in tsf.columns:
                tsf_churned = tsf[tsf['CUST_ACCOUNT_NUMBER'] == sample_customer]['CHURNED_FLAG'].iloc[0]
                print(f"  CHURNED_FLAG in time_series: {repr(tsf_churned)}")
            
            if 'CHURNED_FLAG' in sf.columns:
                sf_churned = sf[sf['CUST_ACCOUNT_NUMBER'] == sample_customer]['CHURNED_FLAG'].iloc[0]
                print(f"  CHURNED_FLAG in static: {repr(sf_churned)}")
    
    # Check the raw data source
    print("\n" + "=" * 80)
    print("CHECKING RAW DATA SOURCE")
    print("=" * 80)
    
    if 'customer_metadata' in data_dict:
        cm = data_dict['customer_metadata']
        print(f"\ncustomer_metadata (source of CHURNED_FLAG):")
        print(f"  Shape: {cm.shape}")
        if 'CHURNED_FLAG' in cm.columns:
            print(f"  CHURNED_FLAG unique values: {cm['CHURNED_FLAG'].unique()}")
            print(f"  CHURNED_FLAG value counts:")
            value_counts = cm['CHURNED_FLAG'].value_counts()
            for val, count in value_counts.items():
                print(f"    {repr(val)}: {count} rows")
            
            # Check if we're filtering to only churned customers somewhere
            print(f"\n  Checking for potential filtering...")
            if 'CHURN_DATE' in cm.columns:
                has_churn_date = cm['CHURN_DATE'].notna()
                print(f"  Rows with CHURN_DATE: {has_churn_date.sum()}")
                print(f"  Rows without CHURN_DATE: {(~has_churn_date).sum()}")
                
                # Cross-check CHURNED_FLAG with CHURN_DATE
                if cm['CHURNED_FLAG'].dtype in ['Int64', 'int64']:
                    churned_with_date = cm[(cm['CHURNED_FLAG'] == 1) & has_churn_date]
                    churned_no_date = cm[(cm['CHURNED_FLAG'] == 1) & ~has_churn_date]
                    active_with_date = cm[(cm['CHURNED_FLAG'] == 0) & has_churn_date]
                    active_no_date = cm[(cm['CHURNED_FLAG'] == 0) & ~has_churn_date]
                    
                    print(f"\n  Cross-check with CHURN_DATE:")
                    print(f"    CHURNED_FLAG=1 with CHURN_DATE: {len(churned_with_date)}")
                    print(f"    CHURNED_FLAG=1 without CHURN_DATE: {len(churned_no_date)}")
                    print(f"    CHURNED_FLAG=0 with CHURN_DATE: {len(active_with_date)}")
                    print(f"    CHURNED_FLAG=0 without CHURN_DATE: {len(active_no_date)}")
    
    print("\n" + "=" * 80)
    print("Debug analysis complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
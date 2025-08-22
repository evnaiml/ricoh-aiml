#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple script to view generated reports for train/new data analysis
"""

import pandas as pd
from pathlib import Path
import sys

def view_latest_reports():
    """View the latest generated reports."""
    
    # Check both report directories
    report_dirs = [
        Path("reports"),
        Path("automated_reports")
    ]
    
    for report_dir in report_dirs:
        if not report_dir.exists():
            print(f"❌ Directory {report_dir} does not exist")
            continue
            
        print(f"\n📁 Reports in {report_dir}:")
        print("=" * 60)
        
        # Find all CSV files
        csv_files = sorted(report_dir.glob("*.csv"))
        
        if not csv_files:
            print("  No CSV files found")
            continue
        
        # Show summary report if exists
        summary_files = [f for f in csv_files if "summary" in f.name]
        
        if summary_files:
            latest_summary = summary_files[-1]  # Get most recent
            print(f"\n📊 Latest Summary Report: {latest_summary.name}")
            print("-" * 40)
            
            df = pd.read_csv(latest_summary)
            
            # Display key metrics
            if 'total_records' in df.columns:
                print(f"Total records analyzed: {df['total_records'].iloc[-1]:,}")
            
            if 'train_records' in df.columns and 'new_records' in df.columns:
                train = df['train_records'].iloc[-1]
                new = df['new_records'].iloc[-1]
                total = train + new
                
                if total > 0:
                    print(f"Training data (0): {train:,} ({train/total*100:.1f}%)")
                    print(f"New data (1): {new:,} ({new/total*100:.1f}%)")
            
            if 'unique_train_customers' in df.columns and 'unique_new_customers' in df.columns:
                train_cust = df['unique_train_customers'].iloc[-1]
                new_cust = df['unique_new_customers'].iloc[-1]
                print(f"\nUnique customers:")
                print(f"  Training: {train_cust:,}")
                print(f"  New: {new_cust:,}")
        
        # List all available reports
        print(f"\n📋 All available reports ({len(csv_files)} files):")
        for f in csv_files[-10:]:  # Show last 10
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.name} ({size_kb:.1f} KB)")
        
        if len(csv_files) > 10:
            print(f"  ... and {len(csv_files) - 10} more files")
    
    # Check for master summary
    master_summary = Path("automated_reports/master_summary.csv")
    if master_summary.exists():
        print("\n📈 Master Summary Trends:")
        print("=" * 60)
        
        df = pd.read_csv(master_summary)
        if not df.empty:
            # Show last 5 entries
            print("\nLast 5 report runs:")
            cols_to_show = ['report_date', 'total_records', 'total_new_records', 'overall_new_pct']
            available_cols = [c for c in cols_to_show if c in df.columns]
            
            if available_cols:
                print(df[available_cols].tail(5).to_string(index=False))

if __name__ == "__main__":
    print("=" * 60)
    print("TRAIN/NEW DATA REPORTS VIEWER")
    print("=" * 60)
    
    view_latest_reports()
    
    print("\n✅ Report viewing complete")
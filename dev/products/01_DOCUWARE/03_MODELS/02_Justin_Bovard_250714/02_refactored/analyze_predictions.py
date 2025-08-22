"""
Analyze final_predicted_df using the LifecyclePredictionVisualizer
This script will help diagnose if your churn predictions are reasonable
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the src directory to path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.append(str(project_root / 'src'))

from churn_aiml.visualization.lifecycle.lifecycle_prediction_panels import LifecyclePredictionVisualizer
from churn_aiml.visualization.lifecycle.example_usage import diagnose_prediction_issues, visualize_predictions

def analyze_churn_predictions(final_predicted_df):
    """
    Complete analysis of your churn predictions
    """
    
    print("="*70)
    print("CHURN PREDICTION ANALYSIS")
    print("="*70)
    
    # 1. Basic Statistics
    print("\n📊 BASIC STATISTICS:")
    print("-"*50)
    print(f"Total customers: {len(final_predicted_df):,}")
    print(f"Analysis date: {final_predicted_df['CREATION_DATE'].iloc[0]}")
    print(f"\nMONTHS_REMAINING Statistics:")
    print(f"  Mean: {final_predicted_df['MONTHS_REMAINING'].mean():.2f} months")
    print(f"  Median: {final_predicted_df['MONTHS_REMAINING'].median():.2f} months")
    print(f"  Std Dev: {final_predicted_df['MONTHS_REMAINING'].std():.2f} months")
    print(f"  Min: {final_predicted_df['MONTHS_REMAINING'].min():.2f} months")
    print(f"  Max: {final_predicted_df['MONTHS_REMAINING'].max():.2f} months")
    
    # 2. Risk Assessment
    print("\n⚠️ RISK ASSESSMENT:")
    print("-"*50)
    
    # Calculate risk buckets
    total = len(final_predicted_df)
    immediate_risk = (final_predicted_df['MONTHS_REMAINING'] < 1).sum()
    high_risk = ((final_predicted_df['MONTHS_REMAINING'] >= 1) & 
                 (final_predicted_df['MONTHS_REMAINING'] < 3)).sum()
    medium_risk = ((final_predicted_df['MONTHS_REMAINING'] >= 3) & 
                   (final_predicted_df['MONTHS_REMAINING'] < 6)).sum()
    low_risk = ((final_predicted_df['MONTHS_REMAINING'] >= 6) & 
                (final_predicted_df['MONTHS_REMAINING'] < 12)).sum()
    very_low_risk = (final_predicted_df['MONTHS_REMAINING'] >= 12).sum()
    
    print(f"Immediate (<1 month):  {immediate_risk:,} ({immediate_risk/total*100:.1f}%)")
    print(f"High Risk (1-3 months): {high_risk:,} ({high_risk/total*100:.1f}%)")
    print(f"Medium Risk (3-6 months): {medium_risk:,} ({medium_risk/total*100:.1f}%)")
    print(f"Low Risk (6-12 months): {low_risk:,} ({low_risk/total*100:.1f}%)")
    print(f"Very Low Risk (>12 months): {very_low_risk:,} ({very_low_risk/total*100:.1f}%)")
    
    # 3. Diagnosis
    print("\n🔍 DIAGNOSIS:")
    print("-"*50)
    
    # Check for problems
    risk_3m = (immediate_risk + high_risk) / total * 100
    risk_6m = (immediate_risk + high_risk + medium_risk) / total * 100
    
    if risk_3m > 30:
        print(f"🔴 CRITICAL: {risk_3m:.1f}% predicted to churn within 3 months")
        print("   This is unusually high and suggests model calibration issues:")
        print("   • Check if training data had class imbalance")
        print("   • Verify no data leakage in features")
        print("   • Consider adjusting prediction threshold")
        print("   • Review if model was trained on crisis period data")
    elif risk_3m > 20:
        print(f"🟠 WARNING: {risk_3m:.1f}% predicted to churn within 3 months")
        print("   This is higher than typical (~10-15%)")
        print("   • Validate against historical churn rates")
        print("   • Check feature importance for anomalies")
    elif risk_3m > 10:
        print(f"🟡 MODERATE: {risk_3m:.1f}% predicted to churn within 3 months")
        print("   This is within reasonable range but monitor closely")
    else:
        print(f"🟢 HEALTHY: {risk_3m:.1f}% predicted to churn within 3 months")
        print("   Predictions appear reasonable")
    
    # Annual churn projection
    annual_churn = (final_predicted_df['MONTHS_REMAINING'] <= 12).sum() / total * 100
    print(f"\n📅 Projected annual churn rate: {annual_churn:.1f}%")
    
    if annual_churn > 35:
        print("   ⚠️ This seems high for B2B SaaS (typical: 10-20%)")
    elif annual_churn < 5:
        print("   ⚠️ This seems low - model might be too conservative")
    else:
        print("   ✓ Within reasonable range")
    
    # 4. Customer Tenure Analysis
    print("\n📈 CUSTOMER TENURE INSIGHTS:")
    print("-"*50)
    
    # Analyze relationship between tenure and predicted churn
    final_predicted_df['TENURE_YEARS'] = final_predicted_df['MONTHS_ELAPSED'] / 12
    
    # Group by tenure buckets
    tenure_buckets = pd.cut(final_predicted_df['TENURE_YEARS'], 
                            bins=[0, 0.5, 1, 2, 3, 5, 100],
                            labels=['<6mo', '6mo-1yr', '1-2yr', '2-3yr', '3-5yr', '>5yr'])
    
    tenure_analysis = final_predicted_df.groupby(tenure_buckets)['MONTHS_REMAINING'].agg([
        'count', 'mean', 'median'
    ]).round(2)
    
    print("\nAverage Months Remaining by Customer Tenure:")
    print(tenure_analysis.to_string())
    
    # 5. Check for concerning patterns
    print("\n🚨 PATTERN CHECKS:")
    print("-"*50)
    
    # Check variance
    cv = final_predicted_df['MONTHS_REMAINING'].std() / final_predicted_df['MONTHS_REMAINING'].mean()
    if cv < 0.3:
        print("⚠️ Low variance in predictions - model may be too certain")
    else:
        print(f"✓ Coefficient of variation: {cv:.2f} (reasonable spread)")
    
    # Check for clustering
    from scipy import stats
    _, p_value = stats.normaltest(final_predicted_df['MONTHS_REMAINING'])
    if p_value < 0.05:
        print("⚠️ Predictions don't follow normal distribution (p={:.4f})".format(p_value))
        print("   Consider checking for multiple customer segments")
    
    # Check for outliers
    Q1 = final_predicted_df['MONTHS_REMAINING'].quantile(0.25)
    Q3 = final_predicted_df['MONTHS_REMAINING'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((final_predicted_df['MONTHS_REMAINING'] < (Q1 - 1.5 * IQR)) | 
                (final_predicted_df['MONTHS_REMAINING'] > (Q3 + 1.5 * IQR))).sum()
    
    print(f"\nOutliers detected: {outliers} ({outliers/total*100:.1f}%)")
    
    return {
        'total_customers': total,
        'risk_3m_pct': risk_3m,
        'risk_6m_pct': risk_6m,
        'annual_churn_pct': annual_churn,
        'mean_months': final_predicted_df['MONTHS_REMAINING'].mean(),
        'median_months': final_predicted_df['MONTHS_REMAINING'].median()
    }


def create_visualizations(final_predicted_df, output_dir='outputs/churn_analysis'):
    """
    Create all visualizations for the predictions
    """
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Initialize visualizer
    viz = LifecyclePredictionVisualizer(figsize=(14, 8))
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Monthly Churn Predictions
    print("\n1. Creating monthly churn forecast...")
    viz.plot_monthly_churn_predictions(
        final_predicted_df,
        months_col='MONTHS_REMAINING',
        save_path=output_path / 'monthly_churn_forecast.png',
        show=True
    )
    
    # 2. Lifecycle Distribution
    print("\n2. Creating lifecycle distribution analysis...")
    viz.plot_lifecycle_distribution(
        final_predicted_df,
        months_col='MONTHS_REMAINING',
        save_path=output_path / 'lifecycle_distribution.png',
        show=True
    )
    
    # 3. Cohort Analysis by Tenure
    print("\n3. Creating cohort analysis...")
    
    # Add tenure cohorts
    final_predicted_df['TENURE_COHORT'] = pd.cut(
        final_predicted_df['MONTHS_ELAPSED'],
        bins=[0, 6, 12, 24, 36, 1000],
        labels=['New (<6mo)', 'Growing (6-12mo)', 'Established (1-2yr)', 
                'Mature (2-3yr)', 'Loyal (>3yr)']
    )
    
    viz.plot_cohort_analysis(
        final_predicted_df,
        months_col='MONTHS_REMAINING',
        segment_col='TENURE_COHORT',
        save_path=output_path / 'tenure_cohort_analysis.png',
        show=True
    )
    
    # 4. Generate Complete Summary Report
    print("\n4. Generating comprehensive report...")
    stats = viz.create_summary_report(
        final_predicted_df,
        months_col='MONTHS_REMAINING',
        save_dir=output_path / 'full_report',
        show=False  # Don't duplicate plots
    )
    
    print(f"\n✅ All visualizations saved to: {output_path}")
    
    return stats


def compare_with_benchmarks(stats):
    """
    Compare your predictions with industry benchmarks
    """
    
    print("\n" + "="*70)
    print("BENCHMARK COMPARISON")
    print("="*70)
    
    benchmarks = {
        'B2B SaaS': {
            'annual_churn': (5, 15),
            '3m_risk': (3, 7),
            'typical_lifecycle_months': 36
        },
        'Enterprise Software': {
            'annual_churn': (5, 10),
            '3m_risk': (2, 5),
            'typical_lifecycle_months': 48
        },
        'SMB Software': {
            'annual_churn': (10, 20),
            '3m_risk': (5, 10),
            'typical_lifecycle_months': 24
        }
    }
    
    your_stats = {
        'annual_churn': stats['annual_churn_pct'],
        '3m_risk': stats['risk_3m_pct'],
        'avg_lifecycle': stats['mean_months']
    }
    
    print(f"\nYour Statistics:")
    print(f"  Annual Churn: {your_stats['annual_churn']:.1f}%")
    print(f"  3-Month Risk: {your_stats['3m_risk']:.1f}%")
    print(f"  Avg Lifecycle: {your_stats['avg_lifecycle']:.1f} months")
    
    print("\nIndustry Benchmarks:")
    for industry, bench in benchmarks.items():
        print(f"\n{industry}:")
        
        # Annual churn comparison
        if bench['annual_churn'][0] <= your_stats['annual_churn'] <= bench['annual_churn'][1]:
            status = "✓"
        else:
            status = "✗"
        print(f"  Annual Churn: {bench['annual_churn'][0]}-{bench['annual_churn'][1]}% {status}")
        
        # 3-month risk comparison
        if bench['3m_risk'][0] <= your_stats['3m_risk'] <= bench['3m_risk'][1]:
            status = "✓"
        else:
            status = "✗"
        print(f"  3-Month Risk: {bench['3m_risk'][0]}-{bench['3m_risk'][1]}% {status}")
        
        # Lifecycle comparison
        diff = abs(your_stats['avg_lifecycle'] - bench['typical_lifecycle_months'])
        if diff < 12:
            status = "✓"
        else:
            status = "✗"
        print(f"  Typical Lifecycle: {bench['typical_lifecycle_months']} months {status}")


# Main execution
if __name__ == "__main__":
    
    # Load your data (replace this with your actual data loading)
    # final_predicted_df = pd.read_csv('your_predictions.csv')
    # OR if you're running this in the same session:
    # final_predicted_df is already in memory
    
    # For demonstration, I'll create sample data similar to yours
    print("Note: Using sample data. Replace with your actual final_predicted_df")
    
    sample_data = pd.DataFrame({
        'CUST_ACCOUNT_NUMBER': range(1000, 2000),
        'MONTHS_REMAINING': np.random.gamma(4, 2, 1000),  # Similar distribution
        'FINAL_EARLIEST_DATE': pd.date_range('2020-01-01', periods=1000, freq='D'),
        'CREATION_DATE': pd.Timestamp('2025-08-11'),
        'DAYS_ELAPSED': np.random.randint(500, 2000, 1000),
        'MONTHS_ELAPSED': np.random.randint(20, 60, 1000),
        'LIFESPAN_MONTHS': np.random.randint(30, 80, 1000)
    })
    
    # Ensure positive values
    sample_data['MONTHS_REMAINING'] = np.maximum(sample_data['MONTHS_REMAINING'], 0.5)
    
    # Run analysis
    print("\n🔍 Running Comprehensive Analysis...")
    stats = analyze_churn_predictions(sample_data)  # Replace with final_predicted_df
    
    # Create visualizations
    create_visualizations(sample_data)  # Replace with final_predicted_df
    
    # Compare with benchmarks
    compare_with_benchmarks(stats)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nNext Steps:")
    print("1. Review the risk assessment above")
    print("2. Check visualizations in outputs/churn_analysis/")
    print("3. If >30% churn in 3 months, model needs recalibration")
    print("4. Consider implementing the two-stage model we designed")
"""
Quick analysis script to run in your current environment
Just copy-paste this into your notebook or add after line 2605
"""

# Add this right after your final_predicted_df is created (line 2605)

# Quick Statistical Analysis
print("="*60)
print("QUICK CHURN PREDICTION ANALYSIS")
print("="*60)

# Basic stats
total = len(final_predicted_df)
mean_months = final_predicted_df['MONTHS_REMAINING'].mean()
median_months = final_predicted_df['MONTHS_REMAINING'].median()

print(f"\nTotal customers: {total:,}")
print(f"Average months until churn: {mean_months:.2f}")
print(f"Median months until churn: {median_months:.2f}")

# Risk buckets
at_risk_1m = (final_predicted_df['MONTHS_REMAINING'] < 1).sum()
at_risk_3m = (final_predicted_df['MONTHS_REMAINING'] < 3).sum()
at_risk_6m = (final_predicted_df['MONTHS_REMAINING'] < 6).sum()
at_risk_12m = (final_predicted_df['MONTHS_REMAINING'] < 12).sum()

print(f"\n⚠️ RISK ASSESSMENT:")
print(f"Churning within 1 month:  {at_risk_1m:,} ({at_risk_1m/total*100:.1f}%)")
print(f"Churning within 3 months: {at_risk_3m:,} ({at_risk_3m/total*100:.1f}%)")
print(f"Churning within 6 months: {at_risk_6m:,} ({at_risk_6m/total*100:.1f}%)")
print(f"Churning within 12 months: {at_risk_12m:,} ({at_risk_12m/total*100:.1f}%)")

# Diagnosis
risk_3m_pct = at_risk_3m/total*100
if risk_3m_pct > 30:
    print(f"\n🔴 CRITICAL: {risk_3m_pct:.1f}% predicted to churn in 3 months is TOO HIGH!")
    print("   Model likely needs recalibration")
elif risk_3m_pct > 20:
    print(f"\n🟠 WARNING: {risk_3m_pct:.1f}% predicted to churn in 3 months is high")
else:
    print(f"\n🟢 OK: {risk_3m_pct:.1f}% predicted to churn in 3 months")

# Distribution check
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram
axes[0].hist(final_predicted_df['MONTHS_REMAINING'], bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(mean_months, color='red', linestyle='--', label=f'Mean: {mean_months:.1f}')
axes[0].axvline(median_months, color='green', linestyle='--', label=f'Median: {median_months:.1f}')
axes[0].set_xlabel('Months Until Churn')
axes[0].set_ylabel('Number of Customers')
axes[0].set_title('Distribution of Predicted Churn Timeline')
axes[0].legend()

# Cumulative distribution
sorted_months = np.sort(final_predicted_df['MONTHS_REMAINING'])
cumulative = np.arange(1, len(sorted_months) + 1) / len(sorted_months) * 100

axes[1].plot(sorted_months, cumulative, linewidth=2)
axes[1].fill_between(sorted_months, cumulative, alpha=0.3)
axes[1].axhline(50, color='gray', linestyle=':', alpha=0.5)
axes[1].axvline(3, color='red', linestyle=':', alpha=0.5, label='3 months')
axes[1].axvline(6, color='orange', linestyle=':', alpha=0.5, label='6 months')
axes[1].axvline(12, color='yellow', linestyle=':', alpha=0.5, label='12 months')
axes[1].set_xlabel('Months Until Churn')
axes[1].set_ylabel('Cumulative % of Customers')
axes[1].set_title('Cumulative Churn Timeline')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tenure vs Churn Analysis
if 'MONTHS_ELAPSED' in final_predicted_df.columns:
    plt.figure(figsize=(10, 6))
    plt.scatter(final_predicted_df['MONTHS_ELAPSED'], 
                final_predicted_df['MONTHS_REMAINING'],
                alpha=0.5, s=20)
    plt.xlabel('Customer Tenure (Months Elapsed)')
    plt.ylabel('Predicted Months Remaining')
    plt.title('Customer Tenure vs Predicted Remaining Lifetime')
    
    # Add trend line
    z = np.polyfit(final_predicted_df['MONTHS_ELAPSED'], 
                   final_predicted_df['MONTHS_REMAINING'], 1)
    p = np.poly1d(z)
    plt.plot(final_predicted_df['MONTHS_ELAPSED'], 
             p(final_predicted_df['MONTHS_ELAPSED']), 
             "r--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Check correlation
    correlation = final_predicted_df['MONTHS_ELAPSED'].corr(final_predicted_df['MONTHS_REMAINING'])
    print(f"\nCorrelation between tenure and remaining lifetime: {correlation:.3f}")
    if abs(correlation) < 0.1:
        print("  ⚠️ Weak correlation - model might not be considering tenure properly")
    
print("\n" + "="*60)
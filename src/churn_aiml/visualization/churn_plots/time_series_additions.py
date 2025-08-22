"""
Additional time series visualization methods for ChurnLifecycleVizSnowflake class.
These methods should be added to the main class.
"""

def plot_comprehensive_time_series(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
    """
    Plot all available time series metrics in a comprehensive dashboard.
    
    Args:
        data: DataFrame with time series data including YYYYWK and multiple metrics
        save_path: Optional path to save the figure
        **kwargs: Additional plotting parameters
        
    Returns:
        matplotlib.figure.Figure: The created figure with all time series plots
    """
    # Define all potential time series columns
    time_series_metrics = {
        'DOCUMENTS_OPENED': {'color': 'blue', 'label': 'Documents Opened', 'ylabel': 'Count'},
        'USED_STORAGE_MB': {'color': 'green', 'label': 'Storage Used (MB)', 'ylabel': 'MB'},
        'INVOICE_REVLINE_TOTAL': {'color': 'purple', 'label': 'Invoice Revenue', 'ylabel': 'Amount ($)'},
        'ORIGINAL_AMOUNT_DUE': {'color': 'orange', 'label': 'Original Amount Due', 'ylabel': 'Amount ($)'},
        'FUNCTIONAL_AMOUNT': {'color': 'red', 'label': 'Functional Amount', 'ylabel': 'Amount ($)'},
        'DAYS_TO_CHURN': {'color': 'darkred', 'label': 'Days to Churn', 'ylabel': 'Days'},
        'WEEKS_TO_CHURN': {'color': 'brown', 'label': 'Weeks to Churn', 'ylabel': 'Weeks'}
    }
    
    # Filter to only available columns
    available_metrics = {k: v for k, v in time_series_metrics.items() if k in data.columns}
    
    if not available_metrics or 'YYYYWK' not in data.columns:
        print("Warning: No time series metrics found in data")
        return plt.figure()
    
    # Calculate subplot layout
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16 * self.figsize_scale, 5 * n_rows * self.figsize_scale))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Hide extra subplots if odd number of metrics
    if n_metrics % 2 == 1 and n_metrics > 1:
        axes[-1].set_visible(False)
    
    # Process each metric
    for idx, (metric_name, metric_info) in enumerate(available_metrics.items()):
        ax = axes[idx] if n_metrics > 1 else axes[0]
        
        # Aggregate by week
        weekly_data = data.groupby('YYYYWK')[metric_name].agg(['mean', 'sum', 'count']).reset_index()
        weekly_data['YYYYWK_date'] = pd.to_datetime(weekly_data['YYYYWK'].astype(str) + '1', format='%Y%W%w')
        
        # Plot the sum (total) as main line
        ax.plot(weekly_data['YYYYWK_date'], weekly_data['sum'], 
               color=metric_info['color'], linewidth=2, alpha=0.8, label='Weekly Total')
        ax.scatter(weekly_data['YYYYWK_date'], weekly_data['sum'],
                  color=metric_info['color'], s=30, alpha=0.6)
        
        # Add trend line
        if len(weekly_data) > 1:
            z = np.polyfit(range(len(weekly_data)), weekly_data['sum'], 1)
            p = np.poly1d(z)
            trend_y = p(range(len(weekly_data)))
            ax.plot(weekly_data['YYYYWK_date'], trend_y, 
                   color=metric_info['color'], linestyle='--', alpha=0.5, linewidth=1.5,
                   label=f'Trend: {z[0]:.2f}')
        
        # Add mean line
        mean_val = weekly_data['sum'].mean()
        ax.axhline(mean_val, color='gray', linestyle=':', alpha=0.5,
                  label=f'Mean: {mean_val:.2f}')
        
        # Formatting
        ax.set_xlabel('Week', fontsize=11)
        ax.set_ylabel(metric_info['ylabel'], fontsize=11)
        ax.set_title(metric_info['label'], fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis for currency if applicable
        if 'Amount' in metric_info['ylabel']:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Add statistics text
        stats_text = f'Total: {weekly_data["sum"].sum():,.0f}\n' \
                    f'Avg/Week: {weekly_data["sum"].mean():,.1f}\n' \
                    f'Weeks: {len(weekly_data)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Overall title
    fig.suptitle('Comprehensive Time Series Analysis', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig

def plot_time_series_comparison(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
    """
    Plot time series comparison between churned and active customers.
    
    Args:
        data: DataFrame with time series data and CHURNED_FLAG
        save_path: Optional path to save the figure
        **kwargs: Additional plotting parameters
        
    Returns:
        matplotlib.figure.Figure: The created figure comparing churned vs active
    """
    if 'CHURNED_FLAG' not in data.columns or 'YYYYWK' not in data.columns:
        print("Warning: Required columns not found for comparison")
        return plt.figure()
    
    # Key metrics to compare
    comparison_metrics = ['DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'INVOICE_REVLINE_TOTAL', 'FUNCTIONAL_AMOUNT']
    available_metrics = [m for m in comparison_metrics if m in data.columns]
    
    if not available_metrics:
        return plt.figure()
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16 * self.figsize_scale, 12 * self.figsize_scale))
    axes = axes.flatten()
    
    # Separate churned and active
    churned_data = data[data['CHURNED_FLAG'] == 1]
    active_data = data[data['CHURNED_FLAG'] == 0]
    
    for idx, metric in enumerate(available_metrics[:4]):
        ax = axes[idx]
        
        # Aggregate by week for each group
        if not churned_data.empty:
            churned_weekly = churned_data.groupby('YYYYWK')[metric].mean().reset_index()
            churned_weekly['YYYYWK_date'] = pd.to_datetime(churned_weekly['YYYYWK'].astype(str) + '1', format='%Y%W%w')
            ax.plot(churned_weekly['YYYYWK_date'], churned_weekly[metric],
                   color='red', linewidth=2, alpha=0.7, label='Churned Customers')
        
        if not active_data.empty:
            active_weekly = active_data.groupby('YYYYWK')[metric].mean().reset_index()
            active_weekly['YYYYWK_date'] = pd.to_datetime(active_weekly['YYYYWK'].astype(str) + '1', format='%Y%W%w')
            ax.plot(active_weekly['YYYYWK_date'], active_weekly[metric],
                   color='green', linewidth=2, alpha=0.7, label='Active Customers')
        
        # Overall average
        overall_weekly = data.groupby('YYYYWK')[metric].mean().reset_index()
        overall_weekly['YYYYWK_date'] = pd.to_datetime(overall_weekly['YYYYWK'].astype(str) + '1', format='%Y%W%w')
        ax.plot(overall_weekly['YYYYWK_date'], overall_weekly[metric],
               color='blue', linewidth=1, alpha=0.5, linestyle='--', label='Overall Average')
        
        # Formatting
        ax.set_xlabel('Week', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # Format y-axis for currency if applicable
        if 'AMOUNT' in metric or 'REVENUE' in metric or 'TOTAL' in metric:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Overall title
    fig.suptitle('Time Series Comparison: Churned vs Active Customers', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        
    return fig
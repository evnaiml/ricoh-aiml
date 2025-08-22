# Additional methods to add to ChurnLifecycleVizSnowflake class

def plot_time_series_scatter(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
    """
    Plot time series data as scatter plots for multiple metrics.
    
    Args:
        data: DataFrame with time series data (must have YYYYWK column)
        save_path: Optional path to save the figure
        **kwargs: Additional parameters including:
            - metrics: List of column names to plot (default: ['DOCUMENTS_OPENED', 'USED_STORAGE_MB'])
            - sample_customers: Number of customers to sample for plotting
            - figsize: Figure size tuple
            - title: Overall title for the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    metrics = kwargs.get('metrics', ['DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'INVOICE_REVLINE_TOTAL'])
    sample_size = kwargs.get('sample_customers', 100)
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in data.columns]
    if not available_metrics:
        warnings.warn("No valid metrics found in data")
        return plt.figure()
    
    # Sample customers if needed
    if 'CUST_ACCOUNT_NUMBER' in data.columns:
        unique_customers = data['CUST_ACCOUNT_NUMBER'].unique()
        if len(unique_customers) > sample_size:
            sampled = np.random.choice(unique_customers, sample_size, replace=False)
            plot_data = data[data['CUST_ACCOUNT_NUMBER'].isin(sampled)].copy()
        else:
            plot_data = data.copy()
    else:
        plot_data = data.copy()
    
    # Create subplots
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=kwargs.get('figsize', (14, 4*n_metrics)))
    if n_metrics == 1:
        axes = [axes]
    
    # Color mapping for customers
    if 'CUST_ACCOUNT_NUMBER' in plot_data.columns:
        customers = plot_data['CUST_ACCOUNT_NUMBER'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, min(20, len(customers))))
        color_map = {cust: colors[i % len(colors)] for i, cust in enumerate(customers)}
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        
        if 'CUST_ACCOUNT_NUMBER' in plot_data.columns:
            # Plot each customer separately with different colors
            for cust in customers[:20]:  # Limit to 20 customers for clarity
                cust_data = plot_data[plot_data['CUST_ACCOUNT_NUMBER'] == cust]
                if not cust_data.empty:
                    ax.scatter(cust_data['YYYYWK'], cust_data[metric], 
                             alpha=0.6, s=20, color=color_map[cust])
        else:
            # Simple scatter plot
            ax.scatter(plot_data['YYYYWK'], plot_data[metric], alpha=0.6, s=20)
        
        ax.set_xlabel('Week (YYYYWK)', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle(kwargs.get('title', 'Time Series Scatter Plots'), 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
    return fig

def plot_monthly_churn_time_series(self, churn_dist: Dict, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
    """
    Plot monthly churn counts as a time series with trend line.
    
    Args:
        churn_dist: Dictionary from get_monthly_churn_distribution() method
        save_path: Optional path to save the figure
        **kwargs: Additional parameters
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if not churn_dist or 'monthly_churn_counts' not in churn_dist:
        warnings.warn("No monthly churn data available")
        return plt.figure()
    
    monthly_counts = churn_dist['monthly_churn_counts']
    dist_stats = churn_dist.get('distribution_stats', {})
    
    fig, axes = plt.subplots(2, 2, figsize=kwargs.get('figsize', (15, 10)))
    
    # 1. Time series plot with scatter and line
    ax1 = axes[0, 0]
    x = np.arange(len(monthly_counts))
    y = monthly_counts['customers_churned'].values
    
    ax1.scatter(x, y, alpha=0.6, s=50, color='darkblue', label='Monthly Churns')
    ax1.plot(x, y, alpha=0.3, color='blue')
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax1.plot(x, p(x), "r--", alpha=0.7, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Add mean line
    mean_val = dist_stats.get('mean_churns_per_month', y.mean())
    ax1.axhline(y=mean_val, color='green', linestyle='--', alpha=0.5, 
               label=f'Mean: {mean_val:.2f}')
    
    ax1.set_xlabel('Month Index')
    ax1.set_ylabel('Number of Churned Customers')
    ax1.set_title('Monthly Churn Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(y, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    ax2.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax2.axvline(x=dist_stats.get('median_churns_per_month', np.median(y)), 
               color='green', linestyle='--', 
               label=f"Median: {dist_stats.get('median_churns_per_month', np.median(y)):.2f}")
    ax2.set_xlabel('Churns per Month')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Monthly Churns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot by year
    ax3 = axes[1, 0]
    if 'yearly_stats' in churn_dist:
        yearly = churn_dist['yearly_stats']
        years = sorted(yearly.get('customers_churned', {}).keys())
        
        # Create box plot data by year
        monthly_counts['year'] = monthly_counts['CHURN_YEAR_MONTH'].astype(str).str[:4].astype(int)
        box_data = [monthly_counts[monthly_counts['year'] == year]['customers_churned'].values 
                   for year in years if year in monthly_counts['year'].values]
        
        if box_data:
            bp = ax3.boxplot(box_data, labels=[str(y) for y in years if y in monthly_counts['year'].values])
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Monthly Churns')
            ax3.set_title('Yearly Distribution of Monthly Churns')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    
    # 4. Seasonal pattern
    ax4 = axes[1, 1]
    if 'quarterly_averages' in churn_dist:
        quarterly = churn_dist['quarterly_averages']
        quarters = sorted(quarterly.get('total_churns', {}).keys())
        values = [quarterly['total_churns'][q] for q in quarters]
        
        ax4.bar(quarters, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax4.set_xlabel('Quarter')
        ax4.set_ylabel('Total Churns')
        ax4.set_title('Seasonal Pattern (Total by Quarter)')
        ax4.set_xticks(quarters)
        ax4.set_xticklabels(['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)'])
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(kwargs.get('title', 'Monthly Churn Distribution Analysis'), 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
    return fig

def plot_all_distributions(self, data_dict: Dict[str, pd.DataFrame], 
                         churn_dist: Dict = None,
                         save_dir: Optional[str] = None, **kwargs) -> Dict[str, plt.Figure]:
    """
    Generate all distribution plots in one call.
    
    Args:
        data_dict: Dictionary of DataFrames from load_data()
        churn_dist: Monthly churn distribution from get_monthly_churn_distribution()
        save_dir: Directory to save all plots
        **kwargs: Additional parameters
    
    Returns:
        Dictionary of figure names to Figure objects
    """
    figures = {}
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall churn distribution
    if 'customer_metadata' in data_dict:
        save_path = str(save_dir / 'churn_distribution.png') if save_dir else None
        fig = self.plot_churn_distribution(data_dict['customer_metadata'], save_path=save_path)
        figures['churn_distribution'] = fig
    
    # 2. Monthly churn time series
    if churn_dist:
        save_path = str(save_dir / 'monthly_churn_analysis.png') if save_dir else None
        fig = self.plot_monthly_churn_time_series(churn_dist, save_path=save_path)
        figures['monthly_churn_analysis'] = fig
    
    # 3. Time series scatter plots
    if 'time_series_features' in data_dict:
        save_path = str(save_dir / 'time_series_scatter.png') if save_dir else None
        fig = self.plot_time_series_scatter(data_dict['time_series_features'], save_path=save_path)
        figures['time_series_scatter'] = fig
    
    # 4. Customer lifecycle analysis
    if 'feature_engineering_dataset' in data_dict:
        save_path = str(save_dir / 'customer_lifecycle.png') if save_dir else None
        fig = self.plot_customer_lifecycle_analysis(data_dict['feature_engineering_dataset'], save_path=save_path)
        figures['customer_lifecycle'] = fig
    
    # 5. Segment comparison
    if 'customer_metadata' in data_dict:
        save_path = str(save_dir / 'segment_comparison.png') if save_dir else None
        fig = self.plot_segment_churn_comparison(data_dict['customer_metadata'], save_path=save_path)
        figures['segment_comparison'] = fig
    
    return figures
"""
Churn Lifecycle Visualization Classes

This module provides comprehensive visualization capabilities for churn analysis,
centralizing all plotting functions in one place for maintainability and reusability.

✅ Abstract Base Class:
- ChurnLifecycleVizBase: Defines interface for all churn visualizations

📊 Concrete Implementation:
- ChurnLifecycleVizSnowflake: Snowflake-specific churn visualizations

🔍 Visualization Categories:
1. Distribution plots (churn rates, segments)
2. Time series plots (monthly trends, usage patterns)
3. Lifecycle analysis (customer tenure, survival)
4. Comparison plots (churned vs active)
5. Monitoring dashboards (production metrics)

💡 Usage:
```python
from churn_aiml.visualization.churn_plots import ChurnLifecycleVizSnowflake

viz = ChurnLifecycleVizSnowflake()
viz.plot_monthly_churn_distribution(data, save_path='figs/monthly_churn.png')
viz.create_monitoring_dashboard(metrics, save_path='figs/dashboard.png')
```

Author: Evgeni Nikolaev
Email: evgeni.nikolaev@ricoh-usa.com
Updated: 2025-08-14
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
import yaml
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression

# Set style globally for consistent appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_dates_config():
    """Load dates configuration from YAML file."""
    config_path = Path('/home/applaimlgen/ricoh_aiml/conf/products/DOCUWARE/db/snowflake/data_config/dates_config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

def add_data_update_markers(ax, dates_config=None, date_format='datetime'):
    """
    Add vertical dashed red lines for data update dates to a plot.
    
    Args:
        ax: Matplotlib axis to add markers to
        dates_config: Optional configuration dictionary
        date_format: Format of dates on x-axis ('datetime', 'YYYYWK', or 'YYYY-MM')
    """
    if dates_config is None:
        dates_config = load_dates_config()
    
    if dates_config and 'data_update_dates' in dates_config:
        markers = dates_config.get('update_markers', {})
        
        for update in dates_config['data_update_dates']:
            update_date = pd.to_datetime(update['date'])
            
            # Convert date to appropriate format based on plot type
            if date_format == 'YYYYWK':
                # Convert to YYYYWK format (year + week number)
                year = update_date.year
                week = update_date.isocalendar()[1]  # ISO week number
                marker_x = year * 100 + week  # e.g., 202527 for week 27 of 2025
            elif date_format == 'YYYY-MM':
                # Convert to year-month string
                marker_x = update_date.strftime('%Y-%m')
            else:
                # Default datetime format
                marker_x = update_date
            
            # Add vertical dashed gray line (no text to avoid crowding)
            try:
                ax.axvline(marker_x, 
                          color=markers.get('line_color', 'gray'),
                          linestyle=markers.get('line_style', '--'),
                          linewidth=markers.get('line_width', 1.5),
                          alpha=markers.get('alpha', 0.7),
                          zorder=10)
            except:
                # Skip if date is outside plot range
                pass


class ChurnLifecycleVizBase(ABC):
    """
    Abstract base class for churn lifecycle visualizations.

    This class defines the interface that all churn visualization implementations
    must follow, ensuring consistency across different data sources and platforms.
    """

    @abstractmethod
    def plot_churn_distribution(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot the overall churn distribution (churned vs active customers).

        Args:
            data: DataFrame with customer data including churn flags
            save_path: Optional path to save the figure
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass

    @abstractmethod
    def plot_monthly_churn_distribution(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot the distribution of churns per month.

        Args:
            data: DataFrame with churn dates
            save_path: Optional path to save the figure
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass

    @abstractmethod
    def plot_usage_trends(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot time series usage trends.

        Args:
            data: DataFrame with time series usage data
            save_path: Optional path to save the figure
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass

    @abstractmethod
    def plot_customer_lifecycle_analysis(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot comprehensive customer lifecycle analysis.

        Args:
            data: DataFrame with customer lifecycle data
            save_path: Optional path to save the figure
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass

    @abstractmethod
    def create_monitoring_dashboard(self, metrics: Dict, data: Dict[str, pd.DataFrame],
                                  save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a comprehensive monitoring dashboard.

        Args:
            metrics: Dictionary of computed metrics
            data: Dictionary of DataFrames with various data
            save_path: Optional path to save the figure
            **kwargs: Additional plotting parameters

        Returns:
            matplotlib.figure.Figure: The created figure
        """
        pass


class ChurnLifecycleVizSnowflake(ChurnLifecycleVizBase):
    """
    Concrete implementation of churn lifecycle visualizations for Snowflake data.

    This class contains all plotting methods extracted from the example scripts,
    centralized in one location for better maintainability and reusability.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', palette: str = 'husl', figsize_scale: float = 1.0):
        """
        Initialize the visualization class with style settings.

        Args:
            style: Matplotlib style to use
            palette: Seaborn color palette
            figsize_scale: Scale factor for figure sizes
        """
        self.style = style
        self.palette = palette
        self.figsize_scale = figsize_scale

        # Apply style settings
        plt.style.use(self.style)
        sns.set_palette(self.palette)

    def _save_figure(self, fig: plt.Figure, save_path: Optional[str], dpi: int = 300, bbox_inches: str = 'tight'):
        """Helper method to save figures consistently."""
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    
    def _calculate_loess_trend(self, x_data: np.ndarray, y_data: np.ndarray, frac: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate LOESS (locally weighted regression) trend line.
        
        Args:
            x_data: X-axis data points
            y_data: Y-axis data points
            frac: Fraction of data to use for each local regression (0-1)
            
        Returns:
            Tuple of (x_smooth, y_smooth) arrays for the trend line
        """
        # Remove NaN values
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        if len(x_clean) < 3:
            return x_clean, y_clean
        
        try:
            # Sort data by x-values for proper LOWESS calculation
            sort_indices = np.argsort(x_clean)
            x_sorted = x_clean[sort_indices]
            y_sorted = y_clean[sort_indices]
            
            # Apply LOWESS smoothing
            lowess_result = lowess(y_sorted, x_sorted, frac=frac, it=3, return_sorted=True)
            
            # Extract smoothed values
            x_smooth = lowess_result[:, 0]
            y_smooth = lowess_result[:, 1]
            
            return x_smooth, y_smooth
            
        except Exception as e:
            # Fallback to simple linear trend if LOWESS fails
            warnings.warn(f"LOWESS calculation failed, using linear trend: {e}")
            return self._calculate_linear_trend(x_clean, y_clean)
    
    def _calculate_linear_trend(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate simple linear trend line as fallback.
        
        Args:
            x_data: X-axis data points
            y_data: Y-axis data points
            
        Returns:
            Tuple of (x_line, y_line) arrays for the trend line
        """
        if len(x_data) < 2:
            return x_data, y_data
            
        model = LinearRegression()
        model.fit(x_data.reshape(-1, 1), y_data)
        
        x_line = np.array([x_data.min(), x_data.max()])
        y_line = model.predict(x_line.reshape(-1, 1))
        
        return x_line, y_line
    
    def _get_trend_color_and_label(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[str, str]:
        """
        Determine trend color and label based on slope.
        
        Args:
            x_data: X-axis data points
            y_data: Y-axis data points
            
        Returns:
            Tuple of (color, label) for the trend
        """
        if len(x_data) < 2:
            return 'gray', 'Insufficient Data'
        
        # Calculate overall slope
        slope = np.polyfit(x_data, y_data, 1)[0]
        
        # Determine color and label based on slope
        if slope < -0.01:
            return 'red', f'Declining Trend ({slope:.2f})'
        elif slope > 0.01:
            return 'green', f'Growing Trend ({slope:.2f})'
        else:
            return 'orange', f'Stable Trend ({slope:.2f})'

    def plot_churn_distribution(self, data: pd.DataFrame, save_path: Optional[str] = None,
                               show_counts: bool = True, **kwargs) -> plt.Figure:
        """
        Plot the overall churn distribution with pie and bar charts.

        Args:
            data: DataFrame with 'CHURNED_FLAG' column
            save_path: Optional path to save the figure
            show_counts: Whether to show actual counts in labels
            **kwargs: Additional parameters (title, colors, etc.)
        """
        if 'CHURNED_FLAG' not in data.columns:
            raise ValueError("Data must contain 'CHURNED_FLAG' column")

        fig, axes = plt.subplots(1, 2, figsize=(12 * self.figsize_scale, 5 * self.figsize_scale))

        # Get churn counts
        churn_counts = data['CHURNED_FLAG'].value_counts()

        # Prepare labels
        if show_counts:
            labels = [f'Active ({churn_counts.get("N", 0):,})',
                     f'Churned ({churn_counts.get("Y", 0):,})']
        else:
            labels = ['Active', 'Churned']

        # Pie chart
        colors = kwargs.get('colors', ['#2ecc71', '#e74c3c'])
        axes[0].pie(churn_counts.values, labels=labels, autopct='%1.1f%%',
                   startangle=90, colors=colors)
        axes[0].set_title(kwargs.get('pie_title', 'Customer Churn Distribution'))

        # Bar chart
        churn_counts.plot(kind='bar', ax=axes[1], color=colors)
        axes[1].set_title(kwargs.get('bar_title', 'Customer Status Count'))
        axes[1].set_xlabel('Status')
        axes[1].set_ylabel('Number of Customers')
        axes[1].set_xticklabels(['Active', 'Churned'], rotation=0)

        # Add value labels on bars
        for i, v in enumerate(churn_counts.values):
            axes[1].text(i, v + max(churn_counts.values) * 0.01, f'{v:,}',
                        ha='center', va='bottom')

        plt.suptitle(kwargs.get('main_title', 'Churn Analysis Overview'),
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig

    def plot_monthly_churn_distribution(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot the distribution of churns per month with statistics.

        Args:
            data: DataFrame with 'CHURN_DATE' column
            save_path: Optional path to save the figure
            **kwargs: Additional parameters
        """
        if 'CHURN_DATE' not in data.columns:
            raise ValueError("Data must contain 'CHURN_DATE' column")

        # Filter churned customers only
        churned_data = data[data.get('CHURNED_FLAG', pd.Series(['Y']*len(data))) == 'Y'].copy()

        if churned_data.empty:
            warnings.warn("No churned customers found in data")
            return plt.figure()

        # Convert to monthly periods
        churned_data['CHURN_MONTH'] = pd.to_datetime(churned_data['CHURN_DATE']).dt.to_period('M')
        monthly_churn = churned_data.groupby('CHURN_MONTH').size()

        # Create complete month range
        if not monthly_churn.empty:
            all_months = pd.period_range(start=monthly_churn.index.min(),
                                        end=monthly_churn.index.max(), freq='M')
            monthly_churn = monthly_churn.reindex(all_months, fill_value=0)

        # Calculate statistics
        mean_churn = monthly_churn.mean()
        std_churn = monthly_churn.std()
        median_churn = monthly_churn.median()

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14 * self.figsize_scale, 10 * self.figsize_scale))

        # 1. Time series plot
        ax1 = axes[0, 0]
        monthly_churn.plot(kind='line', ax=ax1, marker='o', linewidth=2,
                          color=kwargs.get('line_color', '#3498db'))
        ax1.axhline(y=mean_churn, color='r', linestyle='--', label=f'Mean: {mean_churn:.1f}')
        ax1.fill_between(range(len(monthly_churn)), mean_churn - std_churn, mean_churn + std_churn,
                        alpha=0.2, color='red', label=f'±1 Std: {std_churn:.1f}')
        ax1.set_title('Monthly Churn Trend')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Number of Churns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Distribution histogram
        ax2 = axes[0, 1]
        ax2.hist(monthly_churn.values, bins=20, color=kwargs.get('hist_color', '#9b59b6'),
                alpha=0.7, edgecolor='black')
        ax2.axvline(x=mean_churn, color='r', linestyle='--', label=f'Mean: {mean_churn:.1f}')
        ax2.axvline(x=median_churn, color='g', linestyle='--', label=f'Median: {median_churn:.1f}')
        ax2.set_title('Distribution of Monthly Churns')
        ax2.set_xlabel('Number of Churns per Month')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # 3. Box plot by year
        ax3 = axes[1, 0]
        if len(monthly_churn) > 12:
            yearly_data = pd.DataFrame({
                'month': monthly_churn.index.month,
                'year': monthly_churn.index.year,
                'churns': monthly_churn.values
            })
            yearly_pivot = yearly_data.pivot(index='month', columns='year', values='churns')
            yearly_pivot.plot(kind='box', ax=ax3)
            ax3.set_title('Churn Distribution by Year')
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Monthly Churns')
        else:
            # If less than a year of data, show monthly box plot
            monthly_churn.plot(kind='box', ax=ax3, vert=True)
            ax3.set_title('Overall Monthly Churn Distribution')
            ax3.set_ylabel('Number of Churns')

        # 4. Cumulative churn
        ax4 = axes[1, 1]
        cumulative_churn = monthly_churn.cumsum()
        cumulative_churn.plot(kind='area', ax=ax4, alpha=0.5,
                             color=kwargs.get('area_color', '#e74c3c'))
        ax4.set_title('Cumulative Churns Over Time')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Cumulative Churns')
        ax4.grid(True, alpha=0.3)
        
        # Add data update markers to all subplots (monthly plots use datetime format)
        for ax in [ax1, ax2, ax3, ax4]:
            add_data_update_markers(ax, date_format='datetime')

        # Add statistics text box
        stats_text = f'Statistics:\nMean: {mean_churn:.1f}\nStd Dev: {std_churn:.1f}\n'
        stats_text += f'Median: {median_churn:.1f}\nMin: {monthly_churn.min():.0f}\n'
        stats_text += f'Max: {monthly_churn.max():.0f}\nTotal: {monthly_churn.sum():.0f}'

        fig.text(0.02, 0.02, stats_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(kwargs.get('title', 'Monthly Churn Distribution Analysis'),
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig

    def plot_usage_trends(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot time series usage trends for documents and storage.

        Args:
            data: DataFrame with 'YYYYWK', 'DOCUMENTS_OPENED', 'USED_STORAGE_MB' columns
            save_path: Optional path to save the figure
            **kwargs: Additional parameters
        """
        required_cols = ['YYYYWK']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Aggregate weekly usage
        weekly_usage = data.groupby('YYYYWK').agg({
            col: 'sum' for col in data.columns
            if col in ['DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'INVOICE_REVLINE_TOTAL']
        }).reset_index()

        # Determine number of subplots based on available columns
        plot_cols = [col for col in ['DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'INVOICE_REVLINE_TOTAL']
                    if col in weekly_usage.columns]
        n_plots = len(plot_cols)

        if n_plots == 0:
            warnings.warn("No usage columns found in data")
            return plt.figure()

        fig, axes = plt.subplots(n_plots, 1, figsize=(14 * self.figsize_scale, 4 * n_plots * self.figsize_scale))
        if n_plots == 1:
            axes = [axes]

        colors = kwargs.get('colors', ['#3498db', '#9b59b6', '#e67e22'])
        titles = kwargs.get('titles', {
            'DOCUMENTS_OPENED': 'Weekly Document Usage Trend',
            'USED_STORAGE_MB': 'Weekly Storage Usage Trend',
            'INVOICE_REVLINE_TOTAL': 'Weekly Revenue Trend'
        })

        for i, col in enumerate(plot_cols):
            ax = axes[i]
            x = weekly_usage['YYYYWK'].values
            y = weekly_usage[col].values
            
            # Scatter plot only (no connecting lines)
            ax.scatter(x, y, alpha=0.6, s=30, color=colors[i % len(colors)], 
                      label=f'{col.replace("_", " ").title()} (Weekly Total)')
            
            # Add trend line with proper color
            if len(x) > 1:
                # Calculate trend line
                x_numeric = np.arange(len(x))
                z = np.polyfit(x_numeric, y, 1)
                p = np.poly1d(z)
                
                # Calculate trend line values ensuring no negative values
                trend_y = p(x_numeric)
                trend_y = np.maximum(trend_y, 0)  # Ensure no negative values
                
                # Color based on trend direction
                trend_slope = z[0]
                trend_color = 'green' if trend_slope >= 0 else 'red'
                trend_style = 'g--' if trend_slope >= 0 else 'r--'
                
                ax.plot(x, trend_y, trend_style, alpha=0.7, linewidth=2,
                       label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
            
            # Better Y-axis labels with units
            ylabel_map = {
                'DOCUMENTS_OPENED': 'Number of Documents (Absolute)',
                'USED_STORAGE_MB': 'Storage Used (MB, Absolute)',
                'INVOICE_REVLINE_TOTAL': 'Invoice Total ($, Absolute)'
            }
            
            # Set title with trend indicator
            base_title = titles.get(col, f'{col} Trend')
            if trend_slope < 0:
                title_text = f"⚠️ ALARM: Negative Trend - {base_title}"
                title_color = 'red'
            else:
                title_text = base_title
                title_color = 'black'
            ax.set_title(title_text, fontsize=12, fontweight='bold', color=title_color)
            ax.set_xlabel('Year-Week', fontsize=11, fontweight='bold')
            ax.set_ylabel(ylabel_map.get(col, col.replace('_', ' ').title()), fontsize=11)
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Improve X-axis formatting with more ticks
            unique_weeks = sorted(ts_data['YYYYWK'].unique())
            if len(unique_weeks) > 0:
                # Show more ticks for better granularity
                step = max(1, len(unique_weeks) // 15)  # Show ~15 ticks
                all_ticks = unique_weeks[::step]
                
                ax.set_xticks(all_ticks)
                
                # Format labels with clear year-week format
                labels = []
                for w in all_ticks:
                    year = str(w)[:4]
                    week_num = int(str(w)[-2:])
                    # Always show year and week for clarity
                    labels.append(f'{year}-{week_num:02d}')
                
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            
            # Add data update markers to each subplot (using YYYYWK format)
            add_data_update_markers(ax, date_format='YYYYWK')

        plt.suptitle(kwargs.get('title', 'Usage Trends Analysis'),
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig

    def plot_customer_lifecycle_analysis(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot comprehensive customer lifecycle analysis.

        Args:
            data: DataFrame with customer lifecycle data
            save_path: Optional path to save the figure
            **kwargs: Additional parameters
        """
        # Prepare customer statistics
        customer_stats = data.groupby('CUST_ACCOUNT_NUMBER').agg({
            'YYYYWK': ['min', 'max', 'count'],
            'DOCUMENTS_OPENED': 'sum' if 'DOCUMENTS_OPENED' in data.columns else lambda x: 0,
            'CHURNED_FLAG': 'first' if 'CHURNED_FLAG' in data.columns else lambda x: 0
        })
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
        customer_stats['weeks_active'] = customer_stats['YYYYWK_max'] - customer_stats['YYYYWK_min']

        # Handle different CHURNED_FLAG encodings (0/1, Y/N, True/False)
        if 'CHURNED_FLAG_first' in customer_stats.columns:
            churn_col = customer_stats['CHURNED_FLAG_first']
            # Convert to consistent 0/1 encoding
            if churn_col.dtype == 'object':
                customer_stats['is_churned'] = churn_col.map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0, True: 1, False: 0})
            elif churn_col.dtype == 'bool':
                customer_stats['is_churned'] = churn_col.astype(int)
            else:
                customer_stats['is_churned'] = churn_col
        else:
            customer_stats['is_churned'] = 0

        fig, axes = plt.subplots(2, 2, figsize=(14 * self.figsize_scale, 10 * self.figsize_scale))

        # 1. Weeks active distribution by churn status
        if 'is_churned' in customer_stats.columns:
            churned = customer_stats[customer_stats['is_churned'] == 1]['weeks_active']
            active = customer_stats[customer_stats['is_churned'] == 0]['weeks_active']

            axes[0, 0].hist([active.values, churned.values], bins=30, label=['Active', 'Churned'],
                          color=['#2ecc71', '#e74c3c'], alpha=0.7)
            axes[0, 0].set_title('Customer Lifecycle Duration Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Weeks Active', fontsize=12)
            axes[0, 0].set_ylabel('Number of Customers', fontsize=12)
            axes[0, 0].legend(fontsize=11)
            axes[0, 0].tick_params(labelsize=11)

        # 2. Document usage by churn status
        if 'DOCUMENTS_OPENED_sum' in customer_stats.columns and 'is_churned' in customer_stats.columns:
            data_active = customer_stats[customer_stats['is_churned'] == 0]['DOCUMENTS_OPENED_sum']
            data_churned = customer_stats[customer_stats['is_churned'] == 1]['DOCUMENTS_OPENED_sum']

            # Filter out zeros and extreme outliers for better visualization
            data_active = data_active[data_active > 0]
            data_churned = data_churned[data_churned > 0]

            if not data_active.empty or not data_churned.empty:
                bp = axes[0, 1].boxplot([data_active.values, data_churned.values],
                                      labels=['Active', 'Churned'], 
                                      patch_artist=True)
                # Set colors: green for active, red for churned
                bp['boxes'][0].set_facecolor('#2ecc71')  # Green for active
                bp['boxes'][1].set_facecolor('#e74c3c')  # Red for churned
                axes[0, 1].set_title('Document Usage by Customer Status', fontsize=14, fontweight='bold')
                axes[0, 1].set_ylabel('Total Documents Opened', fontsize=12)
                axes[0, 1].set_yscale('log')
                axes[0, 1].tick_params(labelsize=11)

        # 3. Weekly activity distribution
        axes[1, 0].hist(customer_stats['YYYYWK_count'], bins=50, color='#3498db', alpha=0.7)
        axes[1, 0].set_title('Customer Activity Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Number of Active Weeks', fontsize=12)
        axes[1, 0].set_ylabel('Number of Customers', fontsize=12)
        axes[1, 0].tick_params(labelsize=11)

        # 4. Engagement pattern scatter plot
        if 'DOCUMENTS_OPENED_sum' in customer_stats.columns and 'is_churned' in customer_stats.columns:
            # Color by documents opened - higher values = green, lower values = red
            # Normalize the values for color mapping
            docs_opened = customer_stats['DOCUMENTS_OPENED_sum'].values
            # Use log scale for normalization to handle wide range of values
            docs_log = np.log1p(docs_opened)  # log1p to handle zeros
            
            # Create custom colormap from red (low) to yellow (medium) to green (high)
            from matplotlib.colors import LinearSegmentedColormap
            colors = ['darkred', 'red', 'orange', 'yellow', 'lightgreen', 'green', 'darkgreen']
            n_bins = 100  # Smooth gradient
            cmap = LinearSegmentedColormap.from_list('engagement', colors, N=n_bins)
            
            scatter = axes[1, 1].scatter(customer_stats['weeks_active'],
                                        customer_stats['DOCUMENTS_OPENED_sum'],
                                        c=docs_log,
                                        cmap=cmap, alpha=0.6, s=30,
                                        edgecolors='black', linewidth=0.5)
            axes[1, 1].set_title('Customer Engagement Pattern', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Weeks Active', fontsize=12)
            axes[1, 1].set_ylabel('Total Documents Opened', fontsize=12)
            axes[1, 1].set_yscale('log')
            axes[1, 1].tick_params(labelsize=11)
            
            # Add colorbar with proper label
            cbar = plt.colorbar(scatter, ax=axes[1, 1], label='Engagement Level')
            # Set colorbar ticks to show actual values (not log)
            cbar_ticks = cbar.get_ticks()
            cbar_labels = [f'{int(np.expm1(t)):,}' if t > 0 else '0' for t in cbar_ticks]
            cbar.set_ticklabels(cbar_labels)

        plt.suptitle(kwargs.get('title', 'Customer Lifecycle Analysis'),
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig

    def create_monitoring_dashboard(self, metrics: Dict, data: Dict[str, pd.DataFrame],
                                  save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a comprehensive monitoring dashboard for production.

        Args:
            metrics: Dictionary of computed metrics
            data: Dictionary of DataFrames with various data types
            save_path: Optional path to save the figure
            **kwargs: Additional parameters (dpi, title, etc.)
        """
        fig = plt.figure(figsize=(16 * self.figsize_scale, 10 * self.figsize_scale))

        # 1. Churn distribution pie chart (enhanced with metrics)
        ax1 = plt.subplot(2, 3, 1)
        if 'churn_summary' in metrics:
            churn_data = [
                metrics['churn_summary']['active_customers'],
                metrics['churn_summary']['churned_customers']
            ]
            labels = [
                f"Active ({metrics['churn_summary']['active_customers']:,})",
                f"Churned ({metrics['churn_summary']['churned_customers']:,})"
            ]
            ax1.pie(churn_data, labels=labels, autopct='%1.2f%%',
                   startangle=90, colors=['#2ecc71', '#e74c3c'])
            ax1.set_title(f"Churn Distribution (Rate: {metrics['churn_summary']['churn_rate_pct']:.2f}%)")
        elif 'customer_metadata' in data:
            metadata = data['customer_metadata']
            churn_counts = metadata['CHURNED_FLAG'].value_counts()
            ax1.pie(churn_counts.values, labels=['Active', 'Churned'],
                   autopct='%1.1f%%', startangle=90, colors=['#2ecc71', '#e74c3c'])
            ax1.set_title('Customer Churn Distribution')

        # 2. Customer segments bar chart
        ax2 = plt.subplot(2, 3, 2)
        if 'customer_metadata' in data and 'CUSTOMER_SEGMENT' in data['customer_metadata'].columns:
            segment_counts = data['customer_metadata']['CUSTOMER_SEGMENT'].value_counts().head(10)
            segment_counts.plot(kind='barh', ax=ax2, color='#3498db')
            ax2.set_title('Top 10 Customer Segments')
            ax2.set_xlabel('Count')

        # 3. Documents trend (scatter plot with trend line)
        ax3 = plt.subplot(2, 3, 3)
        if 'time_series_features' in data:
            ts_data = data['time_series_features']
            # Aggregate with proper NaN handling
            weekly_docs = ts_data.groupby('YYYYWK')['DOCUMENTS_OPENED'].agg(
                lambda x: x.sum() if x.notna().any() else np.nan
            ).reset_index()
            weekly_docs = weekly_docs[weekly_docs['DOCUMENTS_OPENED'].notna()].copy()
            x = weekly_docs['YYYYWK'].values
            y = weekly_docs['DOCUMENTS_OPENED'].values
            
            # Scatter plot
            ax3.scatter(x, y, alpha=0.6, s=20, color='#9b59b6')
            
            # Add trend line with proper color
            if len(x) > 1:
                x_numeric = np.arange(len(x))
                z = np.polyfit(x_numeric, y, 1)
                p = np.poly1d(z)
                trend_y = np.maximum(p(x_numeric), 0)  # Ensure no negative values
                trend_style = 'g--' if z[0] >= 0 else 'r--'
                ax3.plot(x, trend_y, trend_style, alpha=0.7, linewidth=2)
            
            ax3.set_title('Weekly Document Usage Trend', fontsize=11, fontweight='bold')
            ax3.set_xlabel('Year-Week', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Number of Documents (Absolute)', fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Better X-axis formatting
            if len(x) > 0:
                step = max(1, len(x) // 10)  # Show ~10 ticks
                tick_indices = list(range(0, len(x), step))
                ax3.set_xticks(x[tick_indices])
                labels = []
                for w in x[tick_indices]:
                    week_num = int(str(w)[-2:])
                    if week_num <= 4:
                        labels.append(f'{str(w)[:4]}\n{week_num:02d}')
                    else:
                        labels.append(f'{week_num:02d}')
                ax3.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

        # 4. Storage trend (scatter plot with trend line)
        ax4 = plt.subplot(2, 3, 4)
        if 'time_series_features' in data and 'USED_STORAGE_MB' in data['time_series_features'].columns:
            # Aggregate with proper NaN handling
            weekly_storage = ts_data.groupby('YYYYWK')['USED_STORAGE_MB'].agg(
                lambda x: x.sum() if x.notna().any() else np.nan
            ).reset_index()
            weekly_storage = weekly_storage[weekly_storage['USED_STORAGE_MB'].notna()].copy()
            x = weekly_storage['YYYYWK'].values
            y = weekly_storage['USED_STORAGE_MB'].values
            
            # Scatter plot
            ax4.scatter(x, y, alpha=0.6, s=20, color='#e67e22')
            
            # Add trend line with proper color
            if len(x) > 1:
                x_numeric = np.arange(len(x))
                z = np.polyfit(x_numeric, y, 1)
                p = np.poly1d(z)
                trend_y = np.maximum(p(x_numeric), 0)  # Ensure no negative values
                trend_style = 'g--' if z[0] >= 0 else 'r--'
                ax4.plot(x, trend_y, trend_style, alpha=0.7, linewidth=2)
            
            ax4.set_title('Weekly Storage Usage Trend', fontsize=11, fontweight='bold')
            ax4.set_xlabel('Year-Week', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Storage Used (MB, Absolute)', fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            # Better X-axis formatting
            if len(x) > 0:
                step = max(1, len(x) // 10)  # Show ~10 ticks
                tick_indices = list(range(0, len(x), step))
                ax4.set_xticks(x[tick_indices])
                labels = []
                for w in x[tick_indices]:
                    week_num = int(str(w)[-2:])
                    if week_num <= 4:
                        labels.append(f'{str(w)[:4]}\n{week_num:02d}')
                    else:
                        labels.append(f'{week_num:02d}')
                ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

        # 5. Customer activity distribution
        ax5 = plt.subplot(2, 3, 5)
        if 'feature_engineering_dataset' in data:
            fe_data = data['feature_engineering_dataset']
            customer_activity = fe_data.groupby('CUST_ACCOUNT_NUMBER')['YYYYWK'].count()
            ax5.hist(customer_activity, bins=50, color='#34495e', alpha=0.7)
            ax5.set_title('Customer Activity Distribution')
            ax5.set_xlabel('Number of Active Weeks')
            ax5.set_ylabel('Number of Customers')

        # 6. Days to churn distribution
        ax6 = plt.subplot(2, 3, 6)
        if 'feature_engineering_dataset' in data and 'DAYS_TO_CHURN' in data['feature_engineering_dataset'].columns:
            churned_days = data['feature_engineering_dataset'][
                data['feature_engineering_dataset']['DAYS_TO_CHURN'] < 9999
            ]['DAYS_TO_CHURN']
            if not churned_days.empty:
                ax6.hist(churned_days, bins=30, color='#c0392b', alpha=0.7)
                ax6.set_title('Days to Churn Distribution')
                ax6.set_xlabel('Days to Churn')
                ax6.set_ylabel('Frequency')

        # Add timestamp
        timestamp = kwargs.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M"))
        plt.suptitle(f'Production Data Loading Monitoring Dashboard - {timestamp}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 150))
        return fig

    def plot_segment_churn_comparison(self, data: pd.DataFrame, save_path: Optional[str] = None,
                                     top_n: int = 10, **kwargs) -> plt.Figure:
        """
        Plot churn rates comparison across customer segments.

        Args:
            data: DataFrame with 'CUSTOMER_SEGMENT' and 'CHURNED_FLAG' columns
            save_path: Optional path to save the figure
            top_n: Number of top segments to display
            **kwargs: Additional parameters
        """
        if not all(col in data.columns for col in ['CUSTOMER_SEGMENT', 'CHURNED_FLAG']):
            raise ValueError("Data must contain 'CUSTOMER_SEGMENT' and 'CHURNED_FLAG' columns")

        # Handle different CHURNED_FLAG encodings
        data_copy = data.copy()
        if data_copy['CHURNED_FLAG'].dtype == 'object':
            # String encoding (Y/N)
            data_copy['is_churned'] = data_copy['CHURNED_FLAG'].map({'Y': 1, 'N': 0, 'Yes': 1, 'No': 0}).fillna(0)
        elif data_copy['CHURNED_FLAG'].dtype == 'bool':
            # Boolean encoding
            data_copy['is_churned'] = data_copy['CHURNED_FLAG'].astype(int)
        else:
            # Numeric encoding (0/1)
            data_copy['is_churned'] = data_copy['CHURNED_FLAG']
        
        # Calculate churn rates by segment
        segment_stats = data_copy.groupby('CUSTOMER_SEGMENT').agg({
            'is_churned': 'sum',
            'CUST_ACCOUNT_NUMBER': 'count'
        })
        segment_stats.columns = ['churned_count', 'total_count']
        segment_stats['churn_rate'] = (segment_stats['churned_count'] / segment_stats['total_count']) * 100

        # Get top segments by total count
        top_segments = segment_stats.nlargest(top_n, 'total_count')

        # Adjust figure size for better layout - make it wider and shorter
        fig, axes = plt.subplots(1, 2, figsize=(16 * self.figsize_scale, 5 * self.figsize_scale))

        # 1. Churn rate by segment
        ax1 = axes[0]
        top_segments['churn_rate'].plot(kind='bar', ax=ax1, color='#e74c3c')
        ax1.set_title(f'Churn Rate by Top {top_n} Segments', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Customer Segment', fontsize=13)
        ax1.set_ylabel('Churn Rate (%)', fontsize=13)
        ax1.axhline(y=segment_stats['churn_rate'].mean(), color='b', linestyle='--',
                   label=f'Overall Avg: {segment_stats["churn_rate"].mean():.1f}%')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Improve X-axis labels
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        plt.setp(ax1.xaxis.get_majorticklabels(), ha='right')

        # Add value labels on bars
        for i, (idx, row) in enumerate(top_segments.iterrows()):
            ax1.text(i, row['churn_rate'] + 0.5, f'{row["churn_rate"]:.1f}%',
                    ha='center', va='bottom')

        # 2. Customer count by segment (stacked bar)
        ax2 = axes[1]
        segment_data = pd.DataFrame({
            'Active': top_segments['total_count'] - top_segments['churned_count'],
            'Churned': top_segments['churned_count']
        })
        segment_data.plot(kind='bar', stacked=True, ax=ax2,
                         color=['#2ecc71', '#e74c3c'])
        ax2.set_title(f'Customer Distribution in Top {top_n} Segments', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Customer Segment', fontsize=13)
        ax2.set_ylabel('Number of Customers', fontsize=13)
        ax2.legend(title='Status', fontsize=12)
        
        # Improve X-axis labels
        ax2.tick_params(axis='x', rotation=45, labelsize=12)
        ax2.tick_params(axis='y', labelsize=12)
        plt.setp(ax2.xaxis.get_majorticklabels(), ha='right')

        plt.suptitle(kwargs.get('title', 'Segment-Wise Churn Analysis'),
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig

    def plot_risk_score_analysis(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot risk score distributions for churned vs active customers.

        Args:
            data: DataFrame with 'CUST_RISK_SCORE' and 'CHURNED_FLAG' columns
            save_path: Optional path to save the figure
            **kwargs: Additional parameters
        """
        if not all(col in data.columns for col in ['CUST_RISK_SCORE', 'CHURNED_FLAG']):
            raise ValueError("Data must contain 'CUST_RISK_SCORE' and 'CHURNED_FLAG' columns")

        # Separate churned and active customers
        churned_risk = data[data['CHURNED_FLAG'] == 'Y']['CUST_RISK_SCORE'].dropna()
        active_risk = data[data['CHURNED_FLAG'] == 'N']['CUST_RISK_SCORE'].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(12 * self.figsize_scale, 5 * self.figsize_scale))

        # 1. Distribution comparison
        ax1 = axes[0]
        ax1.hist([active_risk, churned_risk], bins=30, label=['Active', 'Churned'],
                color=['#2ecc71', '#e74c3c'], alpha=0.6)
        ax1.axvline(x=active_risk.mean(), color='#2ecc71', linestyle='--',
                   label=f'Active Mean: {active_risk.mean():.1f}')
        ax1.axvline(x=churned_risk.mean(), color='#e74c3c', linestyle='--',
                   label=f'Churned Mean: {churned_risk.mean():.1f}')
        ax1.set_title('Risk Score Distribution Comparison')
        ax1.set_xlabel('Risk Score')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # 2. Box plot comparison
        ax2 = axes[1]
        ax2.boxplot([active_risk, churned_risk], labels=['Active', 'Churned'])
        ax2.set_title('Risk Score Statistics')
        ax2.set_ylabel('Risk Score')
        ax2.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'Active: Mean={active_risk.mean():.1f}, Std={active_risk.std():.1f}\n'
        stats_text += f'Churned: Mean={churned_risk.mean():.1f}, Std={churned_risk.std():.1f}'
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(kwargs.get('title', 'Risk Score Analysis'),
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig
    
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
                customers_to_plot = customers[:20]  # Limit to 20 customers for clarity
                for i, cust in enumerate(customers_to_plot):
                    cust_data = plot_data[plot_data['CUST_ACCOUNT_NUMBER'] == cust]
                    if not cust_data.empty:
                        # Add label only for first few customers to avoid cluttered legend
                        if i < 5:
                            label = f'Customer {cust}'
                        else:
                            label = None
                        ax.scatter(cust_data['YYYYWK'], cust_data[metric], 
                                 alpha=0.6, s=20, color=color_map[cust], label=label)
                
                # Add note about number of customers shown
                if len(customers_to_plot) > 5:
                    ax.scatter([], [], c='gray', alpha=0.6, s=20, 
                             label=f'... and {len(customers_to_plot) - 5} more customers')
            else:
                # Simple scatter plot
                ax.scatter(plot_data['YYYYWK'], plot_data[metric], alpha=0.6, s=20, 
                          label=f'{metric.replace("_", " ").title()} Data')
            
            # Add trend line with color based on trend direction
            # Calculate weekly aggregates for trend line with proper NaN handling
            weekly_agg = plot_data.groupby('YYYYWK')[metric].agg([
                ('sum', lambda x: x.sum() if x.notna().any() else np.nan),
                ('mean', lambda x: x.mean() if x.notna().any() else np.nan),
                ('count', lambda x: x.notna().sum())
            ]).reset_index()
            # Filter out weeks where all values are NaN
            weekly_agg = weekly_agg[weekly_agg['sum'].notna()].copy()
            
            trend_slope = 0  # Initialize trend_slope
            if len(weekly_agg) > 1:
                unique_x = weekly_agg['YYYYWK'].values
                # Use sum for a more accurate overall trend that reflects total activity
                y_totals = weekly_agg['sum'].values
                
                # Create numeric x values for regression
                x_numeric = np.arange(len(unique_x))
                
                # Fit trend line to the totals
                z = np.polyfit(x_numeric, y_totals, 1)
                p = np.poly1d(z)
                trend_slope = z[0]
                
                # Calculate trend line values ensuring no negative values
                trend_y = p(x_numeric)
                trend_y = np.maximum(trend_y, 0)  # Ensure no negative values
                
                # Scale trend line back to mean level for visualization
                # This shows the trend direction while staying within the data range
                scale_factor = weekly_agg['mean'].mean() / y_totals.mean() if y_totals.mean() > 0 else 1
                trend_y_scaled = trend_y * scale_factor
                
                # Color based on trend direction
                trend_color = 'green' if trend_slope >= 0 else 'red'
                trend_style = 'g--' if trend_slope >= 0 else 'r--'
                
                ax.plot(unique_x, trend_y_scaled, trend_style, alpha=0.7, linewidth=2,
                       label=f'Overall Trend: {"↑" if trend_slope >= 0 else "↓"} {abs(z[0]*scale_factor):.2e}x')
            
            # Add legend with increased font size
            ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=1)
            
            ax.set_xlabel('Year-Week', fontsize=12, fontweight='bold')
            
            # Better Y-axis labels with units (Absolute)
            ylabel_map = {
                'DOCUMENTS_OPENED': 'Number of Documents (Absolute)',
                'USED_STORAGE_MB': 'Storage Used (MB, Absolute)',
                'INVOICE_REVLINE_TOTAL': 'Invoice Total ($, Absolute)',
                'ORIGINAL_AMOUNT_DUE': 'Amount Due ($, Absolute)',
                'FUNCTIONAL_AMOUNT': 'Amount ($, Absolute)'
            }
            ylabel = ylabel_map.get(metric, metric.replace('_', ' ').title() + ' (Absolute)')
            ax.set_ylabel(ylabel, fontsize=10)
            
            # Add data range info and trend indicator to title
            data_min = plot_data[metric].min()
            data_max = plot_data[metric].max()
            
            # Add trend indicator to title with color
            if trend_slope != 0:
                trend_text = 'Positive Trend' if trend_slope >= 0 else 'Negative Trend'
                trend_color = 'green' if trend_slope >= 0 else 'red'
                title = f'{metric.replace("_", " ").title()} Over Time ({trend_text})'
                ax.set_title(title, fontsize=11, fontweight='bold', color=trend_color)
                # Add range info as subtitle
                ax.text(0.5, 0.95, f'Range: {data_min:.1f} - {data_max:.1f}', 
                       transform=ax.transAxes, fontsize=9, ha='center', va='top')
            else:
                ax.set_title(f'{metric.replace("_", " ").title()} Over Time\n(Range: {data_min:.1f} - {data_max:.1f})', 
                            fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Improve X-axis formatting with more ticks
            unique_weeks = sorted(plot_data['YYYYWK'].unique())
            if len(unique_weeks) > 0:
                # Show more ticks for better granularity
                step = max(1, len(unique_weeks) // 20)  # Show ~20 ticks total
                all_ticks = unique_weeks[::step]
                
                ax.set_xticks(all_ticks)
                
                # Format labels with clear year-week format
                labels = []
                for w in all_ticks:
                    year = str(w)[:4]
                    week_num = int(str(w)[-2:])
                    # Always show year and week for clarity
                    labels.append(f'{year}-{week_num:02d}')
                
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
            
            # Add data update markers (vertical dashed red lines) using YYYYWK format
            add_data_update_markers(ax, date_format='YYYYWK')
        
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
        
        # 1. Time series plot with scatter (no line connecting points)
        ax1 = axes[0, 0]
        
        # Convert CHURN_YEAR_MONTH to datetime for proper x-axis formatting
        # Handle different formats: YYYY-MM or YYYYMM
        churn_month_str = monthly_counts['CHURN_YEAR_MONTH'].astype(str).iloc[0]
        if '-' in churn_month_str:
            # Format is already YYYY-MM
            monthly_counts['date'] = pd.to_datetime(monthly_counts['CHURN_YEAR_MONTH'].astype(str), format='%Y-%m')
        else:
            # Format is YYYYMM
            monthly_counts['date'] = pd.to_datetime(monthly_counts['CHURN_YEAR_MONTH'].astype(str), format='%Y%m')
        
        x_dates = monthly_counts['date']
        y = monthly_counts['customers_churned'].values
        
        # Scatter plot only (no connecting line as requested)
        ax1.scatter(x_dates, y, alpha=0.7, s=30, color='darkblue', label='Monthly Churns', edgecolors='navy')
        
        # Add trend line - for churn, always use red dashed line
        x_numeric = np.arange(len(monthly_counts))
        z = np.polyfit(x_numeric, y, 1)
        p = np.poly1d(z)
        trend_slope = z[0]
        
        # Calculate trend line values ensuring no negative values
        trend_y = p(x_numeric)
        trend_y = np.maximum(trend_y, 0)  # Ensure no negative values
        
        # For churn, always use red dashed line
        ax1.plot(x_dates, trend_y, 'r--', alpha=0.7, linewidth=2,
                label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
        
        # Add mean line
        mean_val = dist_stats.get('mean_churns_per_month', y.mean())
        ax1.axhline(y=mean_val, color='green', linestyle='--', alpha=0.5, linewidth=2,
                   label=f'Mean: {mean_val:.2f}')
        
        # Format x-axis with better date labels
        import matplotlib.dates as mdates
        
        # Determine appropriate tick frequency based on date range
        date_range = (x_dates.max() - x_dates.min()).days
        
        if date_range > 365 * 5:  # More than 5 years - show years only
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax1.tick_params(axis='x', which='major', labelsize=11, width=2, length=7)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontweight='bold')
        elif date_range > 365 * 2:  # 2-5 years - show years and quarters
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax1.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
            # Don't show minor labels to avoid overlap
            ax1.xaxis.set_minor_formatter(mdates.DateFormatter(''))
            ax1.tick_params(axis='x', which='major', labelsize=11, width=2, length=7)
            ax1.tick_params(axis='x', which='minor', labelsize=0, width=1, length=4)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0, ha='center', fontweight='bold')
        else:  # Less than 2 years - show months
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax1.tick_params(axis='x', which='major', labelsize=10, width=2, length=7)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax1.set_xlabel('Year-Month', fontsize=10)
        ax1.set_ylabel('Number of Churned Customers', fontsize=10)
        # Add trend indicator to title
        # For churn from business perspective: increasing churns = negative business trend
        # Use "Negative Trend" when churns are increasing (bad for business)
        if trend_slope >= 0:
            trend_text = '⚠️ ALARM: Negative Trend'
            title_color = 'red'
        else:
            trend_text = 'Positive Trend'
            title_color = 'green'
        ax1.set_title(f'Monthly Churn Time Series ({trend_text})', 
                     fontsize=12, fontweight='bold', color=title_color)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add data update markers (vertical dashed red lines)
        add_data_update_markers(ax1, date_format='datetime')
        
        # 2. Distribution histogram
        ax2 = axes[0, 1]
        ax2.hist(y, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
        ax2.axvline(x=mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        ax2.axvline(x=dist_stats.get('median_churns_per_month', np.median(y)), 
                   color='green', linestyle='--', 
                   label=f"Median: {dist_stats.get('median_churns_per_month', np.median(y)):.2f}")
        ax2.set_xlabel('Churns per Month', fontsize=10)
        ax2.set_ylabel('Frequency', fontsize=10)
        ax2.set_title('Distribution of Monthly Churns', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=10)
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
                ax3.set_xlabel('Year', fontsize=10)
                ax3.set_ylabel('Monthly Churns', fontsize=10)
                ax3.set_title('Yearly Distribution of Monthly Churns', fontsize=11, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Seasonal pattern
        ax4 = axes[1, 1]
        if 'quarterly_averages' in churn_dist:
            quarterly = churn_dist['quarterly_averages']
            quarters = sorted(quarterly.get('total_churns', {}).keys())
            values = [quarterly['total_churns'][q] for q in quarters]
            
            # Calculate the time span for the title
            years_span = len(churn_dist.get('yearly_stats', {}).get('customers_churned', {}))
            total_churns = sum(values)
            
            ax4.bar(quarters, values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
            ax4.set_xlabel('Quarter of Year')
            ax4.set_ylabel('Total Churns Across All Years')
            ax4.set_title(f'Seasonal Pattern: Total Churns by Quarter\n(Aggregated over {years_span} years, Total: {total_churns} churns)')
            ax4.set_xticks(quarters)
            ax4.set_xticklabels(['Q1\n(Jan-Mar)', 'Q2\n(Apr-Jun)', 'Q3\n(Jul-Sep)', 'Q4\n(Oct-Dec)'])
            
            # Add value labels on top of bars
            for i, (q, v) in enumerate(zip(quarters, values)):
                ax4.text(q, v + 1, str(int(v)), ha='center', va='bottom', fontweight='bold')
            
            ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(kwargs.get('title', 'Monthly Churn Distribution Analysis'), 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig
    
    def plot_active_customers_time_series(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot time series of active customers count with trend analysis.
        
        Args:
            data: DataFrame with customer data including CHURNED_FLAG and time columns
            save_path: Optional path to save the figure
            **kwargs: Additional parameters
        
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(1, 1, figsize=kwargs.get('figsize', (12, 6)))
        
        # Determine time column (YYYYWK or month-based)
        time_col = None
        if 'YYYYWK' in data.columns:
            time_col = 'YYYYWK'
        elif 'CHURN_YEAR_MONTH' in data.columns:
            time_col = 'CHURN_YEAR_MONTH'
        elif 'date' in data.columns:
            time_col = 'date'
        
        if not time_col:
            warnings.warn("No time column found in data")
            return fig
        
        # Handle different CHURNED_FLAG encodings (0/1, Y/N, True/False)
        if 'CHURNED_FLAG' in data.columns:
            if data['CHURNED_FLAG'].dtype == 'Int64' or data['CHURNED_FLAG'].dtype == 'int64':
                data['is_active'] = (data['CHURNED_FLAG'] == 0).astype(int)
            elif data['CHURNED_FLAG'].dtype == 'bool':
                data['is_active'] = (~data['CHURNED_FLAG']).astype(int)
            else:  # String type
                data['is_active'] = (data['CHURNED_FLAG'] == 'N').astype(int)
        else:
            warnings.warn("CHURNED_FLAG column not found")
            return fig
        
        # Group by time period and count active customers
        if 'CUST_ACCOUNT_NUMBER' in data.columns:
            # For customer-level data, count unique active customers per time period
            active_counts = data.groupby(time_col).apply(
                lambda x: x[x['is_active'] == 1]['CUST_ACCOUNT_NUMBER'].nunique()
            ).reset_index(name='active_customers')
        else:
            # For aggregated data, sum active flags
            active_counts = data.groupby(time_col)['is_active'].sum().reset_index(name='active_customers')
        
        # Sort by time
        active_counts = active_counts.sort_values(time_col)
        
        # Plot scatter points
        x = active_counts[time_col].values
        y = active_counts['active_customers'].values
        
        ax.scatter(x, y, alpha=0.7, s=30, color='darkgreen', label='Active Customers', edgecolors='green')
        
        # Add trend line with color based on direction
        if len(x) > 1:
            x_numeric = np.arange(len(x))
            z = np.polyfit(x_numeric, y, 1)
            p = np.poly1d(z)
            trend_slope = z[0]
            
            # Calculate trend line values ensuring no negative values
            trend_y = p(x_numeric)
            trend_y = np.maximum(trend_y, 0)  # Ensure no negative values
            
            # Color based on trend direction
            trend_color = 'green' if trend_slope >= 0 else 'red'
            trend_style = 'g--' if trend_slope >= 0 else 'r--'
            
            # Plot straight trend line
            ax.plot(x, trend_y, trend_style, alpha=0.7, linewidth=2,
                   label=f'Trend: {z[0]:.2f}x + {z[1]:.2f}')
            
            # Add trend indicator to title
            # For active customers: decreasing = negative trend (bad)
            if trend_slope >= 0:
                trend_text = 'Positive Trend'
                title_color = 'green'
            else:
                trend_text = '⚠️ ALARM: Negative Trend'
                title_color = 'red'
            ax.set_title(f'Active Customers Over Time ({trend_text})', 
                        fontsize=12, fontweight='bold', color=title_color)
        else:
            ax.set_title('Active Customers Over Time', fontsize=12, fontweight='bold')
        
        # Add mean line
        mean_val = y.mean()
        ax.axhline(y=mean_val, color='blue', linestyle='--', alpha=0.5, linewidth=2,
                  label=f'Mean: {mean_val:.0f}')
        
        # Format axes
        ax.set_xlabel('Year-Week', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Active Customers', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Improve X-axis formatting based on time column type
        if time_col == 'YYYYWK':
            # For weekly data - show clear year-week format
            unique_weeks = sorted(active_counts[time_col].unique())
            if len(unique_weeks) > 0:
                # Show more ticks for better granularity
                step = max(1, len(unique_weeks) // 15)  # Show ~15 ticks for better readability
                all_ticks = unique_weeks[::step]
                
                ax.set_xticks(all_ticks)
                
                # Format labels with intuitive year-week format
                labels = []
                for w in all_ticks:
                    year = str(w)[:4]
                    week_num = int(str(w)[-2:])
                    # Always show year and week for clarity
                    labels.append(f'{year}-{week_num:02d}')
                
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
                
                # Make tick labels larger and clearer
                ax.tick_params(axis='y', labelsize=11)
        else:
            # For date-based data, make labels larger
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
        
        # Add data range info
        data_min = y.min()
        data_max = y.max()
        ax.text(0.02, 0.98, f'Range: {data_min:.0f} - {data_max:.0f}', 
               transform=ax.transAxes, fontsize=10, va='top')
        
        # Add data update markers (vertical dashed red lines)
        add_data_update_markers(ax, date_format='YYYYWK')
        
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
        
        # 3. Active customers time series (moved to appear right after monthly churn)
        if 'time_series_features' in data_dict or 'feature_engineering_dataset' in data_dict:
            # Use feature_engineering_dataset if available, otherwise use time_series_features
            data_to_use = data_dict.get('feature_engineering_dataset', data_dict.get('time_series_features'))
            if data_to_use is not None and not data_to_use.empty:
                save_path = str(save_dir / 'active_customers_time_series.png') if save_dir else None
                fig = self.plot_active_customers_time_series(data_to_use, save_path=save_path)
                figures['active_customers_time_series'] = fig
        
        # 4. Time series scatter plots
        if 'time_series_features' in data_dict:
            save_path = str(save_dir / 'time_series_scatter.png') if save_dir else None
            fig = self.plot_time_series_scatter(data_dict['time_series_features'], save_path=save_path)
            figures['time_series_scatter'] = fig
        
        # 5. Customer lifecycle analysis
        if 'feature_engineering_dataset' in data_dict:
            save_path = str(save_dir / 'customer_lifecycle.png') if save_dir else None
            fig = self.plot_customer_lifecycle_analysis(data_dict['feature_engineering_dataset'], save_path=save_path)
            figures['customer_lifecycle'] = fig
        
        # 6. Segment comparison
        if 'customer_metadata' in data_dict:
            save_path = str(save_dir / 'segment_comparison.png') if save_dir else None
            fig = self.plot_segment_churn_comparison(data_dict['customer_metadata'], save_path=save_path)
            figures['segment_comparison'] = fig
        
        return figures
    
    def plot_churn_timing_histograms(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot comparative histograms for days to churn and customer lifespan.
        Shows churned vs active vs all customers overlaid.
        
        Args:
            data: DataFrame with DAYS_TO_CHURN, LIFESPAN_MONTHS, and CHURNED_FLAG columns
            save_path: Optional path to save the figure
            **kwargs: Additional plotting parameters
            
        Returns:
            matplotlib.figure.Figure: The created figure with dual comparative histograms
        """
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16 * self.figsize_scale, 7 * self.figsize_scale))
        
        # Separate churned and active customers
        if 'CHURNED_FLAG' in data.columns:
            churned_data = data[data['CHURNED_FLAG'] == 1]
            active_data = data[data['CHURNED_FLAG'] == 0]
        else:
            # If no CHURNED_FLAG, assume all are churned
            churned_data = data
            active_data = pd.DataFrame()
        
        # Days to Churn Histogram
        if 'DAYS_TO_CHURN' in data.columns:
            ax = axes[0]
            
            # Get data for each group
            all_days = data['DAYS_TO_CHURN'].dropna()
            churned_days = churned_data['DAYS_TO_CHURN'].dropna() if not churned_data.empty else pd.Series()
            active_days = active_data['DAYS_TO_CHURN'].dropna() if not active_data.empty else pd.Series()
            
            # Determine common bins for all histograms
            if not all_days.empty:
                n_bins = min(30, int(np.sqrt(len(all_days))))
                bin_edges = np.histogram_bin_edges(all_days, bins=n_bins)
                
                # Plot all customers first (background)
                ax.hist(all_days, bins=bin_edges, alpha=0.4, color='blue', 
                       edgecolor='darkblue', linewidth=0.5, label=f'All ({len(all_days)} customers)')
                
                # Plot churned customers
                if not churned_days.empty:
                    ax.hist(churned_days, bins=bin_edges, alpha=0.5, color='red',
                           edgecolor='darkred', linewidth=0.5, label=f'Churned ({len(churned_days)} customers)')
                
                # Plot active customers
                if not active_days.empty:
                    ax.hist(active_days, bins=bin_edges, alpha=0.5, color='green',
                           edgecolor='darkgreen', linewidth=0.5, label=f'Active ({len(active_days)} customers)')
                
                # Add mean lines for each group
                ax.axvline(all_days.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.7,
                          label=f'All Mean: {all_days.mean():.1f}d')
                if not churned_days.empty:
                    ax.axvline(churned_days.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7,
                              label=f'Churned Mean: {churned_days.mean():.1f}d')
                if not active_days.empty:
                    ax.axvline(active_days.mean(), color='green', linestyle='--', linewidth=2, alpha=0.7,
                              label=f'Active Mean: {active_days.mean():.1f}d')
            
            # Formatting
            ax.set_xlabel('Days to Churn', fontsize=12)
            ax.set_ylabel('Number of Customers', fontsize=12)
            ax.set_title('Days to Churn: Churned vs Active Customers', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Customer Lifespan Histogram
        if 'LIFESPAN_MONTHS' in data.columns:
            ax = axes[1]
            
            # Get data for each group
            all_months = data['LIFESPAN_MONTHS'].dropna()
            churned_months = churned_data['LIFESPAN_MONTHS'].dropna() if not churned_data.empty else pd.Series()
            active_months = active_data['LIFESPAN_MONTHS'].dropna() if not active_data.empty else pd.Series()
            
            # Determine common bins
            if not all_months.empty:
                n_bins = min(30, int(np.sqrt(len(all_months))))
                bin_edges = np.histogram_bin_edges(all_months, bins=n_bins)
                
                # Plot all customers first (background)
                ax.hist(all_months, bins=bin_edges, alpha=0.4, color='blue',
                       edgecolor='darkblue', linewidth=0.5, label=f'All ({len(all_months)} customers)')
                
                # Plot churned customers
                if not churned_months.empty:
                    ax.hist(churned_months, bins=bin_edges, alpha=0.5, color='red',
                           edgecolor='darkred', linewidth=0.5, label=f'Churned ({len(churned_months)} customers)')
                
                # Plot active customers
                if not active_months.empty:
                    ax.hist(active_months, bins=bin_edges, alpha=0.5, color='green',
                           edgecolor='darkgreen', linewidth=0.5, label=f'Active ({len(active_months)} customers)')
                
                # Add mean lines for each group
                ax.axvline(all_months.mean(), color='blue', linestyle='--', linewidth=2, alpha=0.7,
                          label=f'All Mean: {all_months.mean():.1f}m')
                if not churned_months.empty:
                    ax.axvline(churned_months.mean(), color='red', linestyle='--', linewidth=2, alpha=0.7,
                              label=f'Churned Mean: {churned_months.mean():.1f}m')
                if not active_months.empty:
                    ax.axvline(active_months.mean(), color='green', linestyle='--', linewidth=2, alpha=0.7,
                              label=f'Active Mean: {active_months.mean():.1f}m')
                
                # Add year markers
                for year in range(1, int(all_months.max()/12) + 1):
                    ax.axvline(year * 12, color='gray', linestyle=':', linewidth=1, alpha=0.3)
                    ax.text(year * 12, ax.get_ylim()[1] * 0.95, f'{year}yr',
                           ha='center', fontsize=9, color='gray')
            
            # Formatting
            ax.set_xlabel('Customer Lifespan (Months)', fontsize=12)
            ax.set_ylabel('Number of Customers', fontsize=12)
            ax.set_title('Customer Lifespan: Churned vs Active Customers', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle('Customer Timing Analysis: Comparative View', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            
        return fig
    
    def plot_days_to_churn_histogram(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot histogram for days to churn only.
        
        Args:
            data: DataFrame with DAYS_TO_CHURN column
            save_path: Optional path to save the figure
            **kwargs: Additional plotting parameters
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10 * self.figsize_scale, 6 * self.figsize_scale))
        
        if 'DAYS_TO_CHURN' in data.columns:
            days_data = data['DAYS_TO_CHURN'].dropna()
            
            # Create histogram
            n_bins = kwargs.get('bins', min(30, int(np.sqrt(len(days_data)))))
            counts, bins, patches = ax.hist(days_data, bins=n_bins,
                                           color='green', alpha=0.7, edgecolor='darkgreen')
            
            # Color code the bars by quartiles using shades of green
            q1, q2, q3 = days_data.quantile([0.25, 0.5, 0.75])
            for patch, left_edge in zip(patches, bins[:-1]):
                if left_edge < q1:
                    patch.set_facecolor('darkgreen')
                    patch.set_alpha(0.8)
                elif left_edge < q2:
                    patch.set_facecolor('forestgreen')
                    patch.set_alpha(0.7)
                elif left_edge < q3:
                    patch.set_facecolor('limegreen')
                    patch.set_alpha(0.7)
                else:
                    patch.set_facecolor('lightgreen')
                    patch.set_alpha(0.7)
            
            # Add statistics lines
            mean_days = days_data.mean()
            median_days = days_data.median()
            ax.axvline(mean_days, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_days:.1f} days', alpha=0.8)
            ax.axvline(median_days, color='blue', linestyle='--', linewidth=2,
                      label=f'Median: {median_days:.0f} days', alpha=0.8)
            
            # Formatting
            ax.set_xlabel('Days to Churn', fontsize=13)
            ax.set_ylabel('Number of Customers', fontsize=13)
            ax.set_title('Distribution of Days to Churn', fontsize=15, fontweight='bold')
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle=':')
            
            # Add detailed statistics text
            stats_text = f'Churn Timing Statistics:\n' \
                        f'────────────────────\n' \
                        f'Mean: {mean_days:.1f} days\n' \
                        f'Median: {median_days:.0f} days\n' \
                        f'Std Dev: {days_data.std():.1f} days\n' \
                        f'Min: {days_data.min():.0f} days\n' \
                        f'Max: {days_data.max():.0f} days\n' \
                        f'────────────────────\n' \
                        f'Q1 (25%): {q1:.0f} days\n' \
                        f'Q2 (50%): {q2:.0f} days\n' \
                        f'Q3 (75%): {q3:.0f} days\n' \
                        f'────────────────────\n' \
                        f'Total Customers: {len(days_data):,}'
            
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                   family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            
        return fig
    
    def plot_lifespan_histogram(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot histogram for customer lifespan in months.
        
        Args:
            data: DataFrame with LIFESPAN_MONTHS column
            save_path: Optional path to save the figure
            **kwargs: Additional plotting parameters
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, ax = plt.subplots(figsize=(10 * self.figsize_scale, 6 * self.figsize_scale))
        
        if 'LIFESPAN_MONTHS' in data.columns:
            months_data = data['LIFESPAN_MONTHS'].dropna()
            
            # Create histogram
            n_bins = kwargs.get('bins', min(25, int(np.sqrt(len(months_data)))))
            counts, bins, patches = ax.hist(months_data, bins=n_bins,
                                           color='coral', alpha=0.7, edgecolor='black')
            
            # Color code the bars by quartiles
            q1, q2, q3 = months_data.quantile([0.25, 0.5, 0.75])
            for patch, left_edge in zip(patches, bins[:-1]):
                if left_edge < q1:
                    patch.set_facecolor('darkred')
                    patch.set_alpha(0.7)
                elif left_edge < q2:
                    patch.set_facecolor('orange')
                    patch.set_alpha(0.7)
                elif left_edge < q3:
                    patch.set_facecolor('gold')  
                    patch.set_alpha(0.7)
                else:
                    patch.set_facecolor('green')
                    patch.set_alpha(0.7)
            
            # Add statistics lines
            mean_months = months_data.mean()
            median_months = months_data.median()
            ax.axvline(mean_months, color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {mean_months:.1f} months', alpha=0.8)
            ax.axvline(median_months, color='blue', linestyle='--', linewidth=2,
                      label=f'Median: {median_months:.1f} months', alpha=0.8)
            
            # Add year markers
            for year in range(1, int(months_data.max()/12) + 1):
                ax.axvline(year * 12, color='gray', linestyle=':', linewidth=1, alpha=0.5)
                ax.text(year * 12, ax.get_ylim()[1] * 0.95, f'{year}yr',
                       ha='center', fontsize=9, color='gray')
            
            # Formatting
            ax.set_xlabel('Customer Lifespan (Months)', fontsize=13)
            ax.set_ylabel('Number of Customers', fontsize=13)
            ax.set_title('Distribution of Customer Lifespan', fontsize=15, fontweight='bold')
            ax.legend(loc='upper right', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle=':')
            
            # Add detailed statistics text
            stats_text = f'Lifespan Statistics:\n' \
                        f'────────────────────\n' \
                        f'Mean: {mean_months:.1f} months\n' \
                        f'Median: {median_months:.1f} months\n' \
                        f'Std Dev: {months_data.std():.1f} months\n' \
                        f'Min: {months_data.min():.1f} months\n' \
                        f'Max: {months_data.max():.1f} months\n' \
                        f'────────────────────\n' \
                        f'Q1 (25%): {q1:.1f} months\n' \
                        f'Q2 (50%): {q2:.1f} months\n' \
                        f'Q3 (75%): {q3:.1f} months\n' \
                        f'────────────────────\n' \
                        f'Total Customers: {len(months_data):,}'
            
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'),
                   family='monospace')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            
        return fig
    
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
        # Define all potential time series columns (using colors that don't conflict with trend indicators)
        time_series_metrics = {
            'DOCUMENTS_OPENED': {'color': 'steelblue', 'label': 'Documents Opened', 'ylabel': 'Count'},
            'USED_STORAGE_MB': {'color': 'mediumpurple', 'label': 'Storage Used (MB)', 'ylabel': 'MB'},
            'INVOICE_REVLINE_TOTAL': {'color': 'darkorchid', 'label': 'Invoice Revenue', 'ylabel': 'Amount ($)'},
            'ORIGINAL_AMOUNT_DUE': {'color': 'cornflowerblue', 'label': 'Original Amount Due', 'ylabel': 'Amount ($)'},
            'FUNCTIONAL_AMOUNT': {'color': 'slateblue', 'label': 'Functional Amount', 'ylabel': 'Amount ($)'},
            'DAYS_TO_CHURN': {'color': 'darkviolet', 'label': 'Days to Churn', 'ylabel': 'Days'},
            'WEEKS_TO_CHURN': {'color': 'mediumblue', 'label': 'Weeks to Churn', 'ylabel': 'Weeks'}
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
            axes = list(axes)
        else:
            axes = list(axes.flatten())
        
        # Hide extra subplots if odd number of metrics
        if n_metrics % 2 == 1 and n_metrics > 1:
            axes[-1].set_visible(False)
        
        # Process each metric
        for idx, (metric_name, metric_info) in enumerate(available_metrics.items()):
            ax = axes[idx]
            
            # Filter negative values for revenue/amount metrics
            plot_data = data.copy()
            if metric_name in ['INVOICE_REVLINE_TOTAL', 'ORIGINAL_AMOUNT_DUE', 'FUNCTIONAL_AMOUNT']:
                # Drop negative values for financial metrics
                plot_data = plot_data[plot_data[metric_name] >= 0].copy()
            
            # Filter out sentinel values (9999) for DAYS_TO_CHURN and WEEKS_TO_CHURN
            if metric_name == 'DAYS_TO_CHURN':
                plot_data = plot_data[plot_data[metric_name] < 9999].copy()
            elif metric_name == 'WEEKS_TO_CHURN':
                plot_data = plot_data[plot_data[metric_name] < 1000].copy()
            
            # Aggregate by week (handle NaN properly)
            # Use skipna=False to preserve NaN when all values are NaN
            weekly_agg = plot_data.groupby('YYYYWK')[metric_name].agg([
                ('mean', lambda x: x.mean() if x.notna().any() else np.nan),
                ('sum', lambda x: x.sum() if x.notna().any() else np.nan),
                ('count', lambda x: x.notna().sum())
            ]).reset_index()
            weekly_data = weekly_agg.copy()
            weekly_data['YYYYWK_date'] = pd.to_datetime(weekly_data['YYYYWK'].astype(str) + '1', format='%Y%W%w')
            
            # Filter out weeks where all values are NaN (don't plot as zero)
            valid_data = weekly_data[weekly_data['sum'].notna()].copy()
            
            # Plot the sum (total) as scatter plot only (no line) - excluding NaN weeks
            if len(valid_data) > 0:
                # Only add label for non-financial metrics to reduce legend clutter
                scatter_label = 'Weekly Total' if 'Amount' not in metric_info['ylabel'] else None
                ax.scatter(valid_data['YYYYWK_date'], valid_data['sum'],
                          color=metric_info['color'], s=50, alpha=0.7, 
                          edgecolors='darkgray', linewidth=0.5, label=scatter_label)
            
            # Add LOESS trend line with color-coded segments
            if len(valid_data) > 1:
                # Convert dates to numeric for LOESS calculation
                x_numeric = np.arange(len(valid_data))
                y_values = valid_data['sum'].values
                
                # Calculate LOESS trend
                x_smooth, y_smooth = self._calculate_loess_trend(x_numeric, y_values)
                
                # Get trend color and label
                trend_color, trend_label = self._get_trend_color_and_label(x_numeric, y_values)
                
                # Map numeric x back to dates for plotting
                if len(x_smooth) > 0:
                    x_dates_smooth = []
                    for x_val in x_smooth:
                        idx = int(np.clip(x_val, 0, len(valid_data) - 1))
                        x_dates_smooth.append(valid_data['YYYYWK_date'].iloc[idx])
                    
                    # Plot LOESS trend with segments colored by local slope
                    window_size = min(5, len(x_smooth) // 4)
                    for i in range(len(x_smooth) - 1):
                        # Calculate local slope for color
                        start_idx = max(0, i - window_size)
                        end_idx = min(len(x_smooth), i + window_size + 1)
                        
                        if end_idx > start_idx + 1:
                            local_x = x_smooth[start_idx:end_idx]
                            local_y = y_smooth[start_idx:end_idx]
                            
                            if len(local_x) >= 2:
                                local_slope = np.polyfit(local_x, local_y, 1)[0]
                                
                                # Determine segment color based on slope
                                if local_slope < -0.01:
                                    segment_color = 'red'
                                elif local_slope > 0.01:
                                    segment_color = 'green'
                                else:
                                    segment_color = 'orange'
                            else:
                                segment_color = 'gray'
                        else:
                            segment_color = 'gray'
                        
                        # Plot segment
                        ax.plot(x_dates_smooth[i:i+2], y_smooth[i:i+2],
                               color=segment_color, linewidth=2.5, alpha=0.8)
                    
                    # Add legend entry for overall trend
                    ax.plot([], [], color=trend_color, linewidth=2.5, alpha=0.8, label=trend_label)
            
            # Add mean line (only for valid data)
            if len(valid_data) > 0:
                mean_val = valid_data['sum'].mean()
                # Only add mean line label for non-financial metrics
                mean_label = f'Mean: {mean_val:.2f}' if 'Amount' not in metric_info['ylabel'] else None
                ax.axhline(mean_val, color='gray', linestyle=':', alpha=0.5,
                          label=mean_label)
            
            # Formatting
            ax.set_xlabel('Week', fontsize=11)
            ax.set_ylabel(metric_info['ylabel'], fontsize=11)
            
            # Set title with color based on weighted trend (recent data has more weight)
            if len(valid_data) > 1:
                # Calculate weighted trend giving more importance to recent data
                x_values = np.arange(len(valid_data))
                y_values = valid_data['sum'].values
                
                # Focus on last 30% of data for trend direction
                recent_portion = max(3, int(len(valid_data) * 0.3))
                recent_x = x_values[-recent_portion:]
                recent_y = y_values[-recent_portion:]
                
                if len(recent_x) >= 2:
                    # Calculate recent trend slope
                    recent_slope = np.polyfit(recent_x, recent_y, 1)[0]
                    
                    # Normalize slope by the mean value to get percentage change
                    mean_value = np.mean(recent_y)
                    if mean_value > 0:
                        normalized_slope = recent_slope / mean_value
                    else:
                        normalized_slope = 0
                    
                    # Also check recent momentum (comparing last few points to previous few)
                    if len(recent_y) >= 6:
                        first_half_mean = np.mean(recent_y[:len(recent_y)//2])
                        second_half_mean = np.mean(recent_y[len(recent_y)//2:])
                        momentum = (second_half_mean - first_half_mean) / first_half_mean if first_half_mean > 0 else 0
                    else:
                        momentum = normalized_slope
                    
                    # Determine alarm based on normalized recent trend and momentum
                    # Use percentage thresholds for better consistency across metrics
                    if normalized_slope < -0.001 or momentum < -0.05:  # Declining >0.1% or momentum down >5%
                        title_color = 'red'
                        alarm_indicator = '⚠️ '
                    elif normalized_slope > 0.001 or momentum > 0.05:  # Growing >0.1% or momentum up >5%
                        title_color = 'green'
                        alarm_indicator = '✅ '
                    else:
                        title_color = 'orange'
                        alarm_indicator = '➖ '
                else:
                    # Fallback to overall trend if not enough recent data
                    overall_slope = np.polyfit(x_values, y_values, 1)[0]
                    mean_value = np.mean(y_values)
                    normalized_slope = overall_slope / mean_value if mean_value > 0 else 0
                    
                    if normalized_slope < -0.001:
                        title_color = 'red'
                        alarm_indicator = '⚠️ '
                    elif normalized_slope > 0.001:
                        title_color = 'green'
                        alarm_indicator = '✅ '
                    else:
                        title_color = 'orange'
                        alarm_indicator = '➖ '
            else:
                title_color = 'black'
                alarm_indicator = ''
            
            ax.set_title(f'{alarm_indicator}{metric_info["label"]}', 
                        fontsize=13, fontweight='bold', color=title_color)
            
            # Only show legend if there are labeled items, and use better positioning
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                # For financial metrics, position legend in upper left to avoid overlap with stats
                if 'Amount' in metric_info['ylabel']:
                    ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=1)
                else:
                    ax.legend(loc='best', fontsize=9, framealpha=0.9)
            
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
            
            # Add data update markers (vertical dashed red lines)
            add_data_update_markers(ax)
        
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
            
            # Aggregate by week for each group with proper NaN handling
            # Use colors that don't conflict with trend indicators (red=declining, green=growing)
            if not churned_data.empty:
                # Custom aggregation that preserves NaN when all values are NaN
                churned_weekly = churned_data.groupby('YYYYWK')[metric].agg(
                    lambda x: x.mean() if x.notna().any() else np.nan
                ).reset_index()
                # Filter out weeks where all values are NaN
                churned_weekly = churned_weekly[churned_weekly[metric].notna()].copy()
                
                if not churned_weekly.empty:
                    churned_weekly['YYYYWK_date'] = pd.to_datetime(churned_weekly['YYYYWK'].astype(str) + '1', format='%Y%W%w')
                    ax.scatter(churned_weekly['YYYYWK_date'], churned_weekly[metric],
                              color='darkviolet', s=40, alpha=0.7, edgecolors='purple', linewidth=0.5, label='Churned Customers')
                
                    # Add LOESS trend for churned customers
                    if len(churned_weekly) > 2:
                        x_numeric = np.arange(len(churned_weekly))
                        y_values = churned_weekly[metric].values
                        x_smooth, y_smooth = self._calculate_loess_trend(x_numeric, y_values)
                        
                        if len(x_smooth) > 0:
                            x_dates_smooth = []
                            for x_val in x_smooth:
                                idx = int(np.clip(x_val, 0, len(churned_weekly) - 1))
                                x_dates_smooth.append(churned_weekly['YYYYWK_date'].iloc[idx])
                            
                            # Plot LOESS trend with color segments
                            window_size = min(5, len(x_smooth) // 4)
                            for i in range(len(x_smooth) - 1):
                                start_idx = max(0, i - window_size)
                                end_idx = min(len(x_smooth), i + window_size + 1)
                                
                                if end_idx > start_idx + 1:
                                    local_x = x_smooth[start_idx:end_idx]
                                    local_y = y_smooth[start_idx:end_idx]
                                    
                                    if len(local_x) >= 2:
                                        local_slope = np.polyfit(local_x, local_y, 1)[0]
                                        segment_color = 'red' if local_slope < -0.01 else 'green' if local_slope > 0.01 else 'orange'
                                    else:
                                        segment_color = 'gray'
                                else:
                                    segment_color = 'gray'
                                
                                ax.plot(x_dates_smooth[i:i+2], y_smooth[i:i+2],
                                       color=segment_color, linewidth=2, alpha=0.7, linestyle='-')
            
            if not active_data.empty:
                # Custom aggregation that preserves NaN when all values are NaN
                active_weekly = active_data.groupby('YYYYWK')[metric].agg(
                    lambda x: x.mean() if x.notna().any() else np.nan
                ).reset_index()
                # Filter out weeks where all values are NaN
                active_weekly = active_weekly[active_weekly[metric].notna()].copy()
                
                if not active_weekly.empty:
                    active_weekly['YYYYWK_date'] = pd.to_datetime(active_weekly['YYYYWK'].astype(str) + '1', format='%Y%W%w')
                    ax.scatter(active_weekly['YYYYWK_date'], active_weekly[metric],
                              color='steelblue', s=40, alpha=0.7, edgecolors='navy', linewidth=0.5, label='Active Customers')
                
                    # Add LOESS trend for active customers
                    if len(active_weekly) > 2:
                        x_numeric = np.arange(len(active_weekly))
                        y_values = active_weekly[metric].values
                        x_smooth, y_smooth = self._calculate_loess_trend(x_numeric, y_values)
                        
                        if len(x_smooth) > 0:
                            x_dates_smooth = []
                            for x_val in x_smooth:
                                idx = int(np.clip(x_val, 0, len(active_weekly) - 1))
                                x_dates_smooth.append(active_weekly['YYYYWK_date'].iloc[idx])
                            
                            # Plot LOESS trend with color segments
                            window_size = min(5, len(x_smooth) // 4)
                            for i in range(len(x_smooth) - 1):
                                start_idx = max(0, i - window_size)
                                end_idx = min(len(x_smooth), i + window_size + 1)
                                
                                if end_idx > start_idx + 1:
                                    local_x = x_smooth[start_idx:end_idx]
                                    local_y = y_smooth[start_idx:end_idx]
                                    
                                    if len(local_x) >= 2:
                                        local_slope = np.polyfit(local_x, local_y, 1)[0]
                                        segment_color = 'red' if local_slope < -0.01 else 'green' if local_slope > 0.01 else 'orange'
                                    else:
                                        segment_color = 'gray'
                                else:
                                    segment_color = 'gray'
                                
                                ax.plot(x_dates_smooth[i:i+2], y_smooth[i:i+2],
                                       color=segment_color, linewidth=2, alpha=0.7, linestyle='--')
            
            # Overall average with proper NaN handling - keep as dotted line for reference
            overall_weekly = data.groupby('YYYYWK')[metric].agg(
                lambda x: x.mean() if x.notna().any() else np.nan
            ).reset_index()
            # Filter out weeks where all values are NaN
            overall_weekly = overall_weekly[overall_weekly[metric].notna()].copy()
            
            if not overall_weekly.empty:
                overall_weekly['YYYYWK_date'] = pd.to_datetime(overall_weekly['YYYYWK'].astype(str) + '1', format='%Y%W%w')
                ax.plot(overall_weekly['YYYYWK_date'], overall_weekly[metric],
                       color='gray', linewidth=1, alpha=0.5, linestyle=':', label='Overall Average')
            
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
            
            # Add data update markers (vertical dashed red lines)
            add_data_update_markers(ax)
        
        # Overall title
        fig.suptitle('Time Series Comparison: Churned vs Active Customers', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches='tight')
            
        return fig

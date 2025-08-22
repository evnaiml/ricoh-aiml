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

# Set style globally for consistent appearance
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


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
            ax.plot(weekly_usage['YYYYWK'], weekly_usage[col],
                   marker='o', markersize=3, linewidth=1, color=colors[i % len(colors)])
            ax.set_title(titles.get(col, f'{col} Trend'))
            ax.set_xlabel('Week (YYYYWK)')
            ax.set_ylabel(col.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)

            # Add trend line
            if len(weekly_usage) > 10:
                z = np.polyfit(range(len(weekly_usage)), weekly_usage[col].values, 1)
                p = np.poly1d(z)
                ax.plot(weekly_usage['YYYYWK'], p(range(len(weekly_usage))),
                       "r--", alpha=0.5, label='Trend')
                ax.legend()

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
            'CHURNED_FLAG': 'first' if 'CHURNED_FLAG' in data.columns else lambda x: 'N'
        })
        customer_stats.columns = ['_'.join(col).strip() for col in customer_stats.columns]
        customer_stats['weeks_active'] = customer_stats['YYYYWK_max'] - customer_stats['YYYYWK_min']

        fig, axes = plt.subplots(2, 2, figsize=(14 * self.figsize_scale, 10 * self.figsize_scale))

        # 1. Weeks active distribution by churn status
        if 'CHURNED_FLAG_first' in customer_stats.columns:
            churned = customer_stats[customer_stats['CHURNED_FLAG_first'] == 'Y']['weeks_active']
            active = customer_stats[customer_stats['CHURNED_FLAG_first'] == 'N']['weeks_active']

            axes[0, 0].hist([active.values, churned.values], bins=30, label=['Active', 'Churned'],
                          color=['#2ecc71', '#e74c3c'], alpha=0.7)
            axes[0, 0].set_title('Customer Lifecycle Duration Distribution')
            axes[0, 0].set_xlabel('Weeks Active')
            axes[0, 0].set_ylabel('Number of Customers')
            axes[0, 0].legend()

        # 2. Document usage by churn status
        if 'DOCUMENTS_OPENED_sum' in customer_stats.columns and 'CHURNED_FLAG_first' in customer_stats.columns:
            data_active = customer_stats[customer_stats['CHURNED_FLAG_first'] == 'N']['DOCUMENTS_OPENED_sum']
            data_churned = customer_stats[customer_stats['CHURNED_FLAG_first'] == 'Y']['DOCUMENTS_OPENED_sum']

            # Filter out zeros and extreme outliers for better visualization
            data_active = data_active[data_active > 0]
            data_churned = data_churned[data_churned > 0]

            if not data_active.empty or not data_churned.empty:
                axes[0, 1].boxplot([data_active.values, data_churned.values],
                                 labels=['Active', 'Churned'])
                axes[0, 1].set_title('Document Usage by Customer Status')
                axes[0, 1].set_ylabel('Total Documents Opened')
                axes[0, 1].set_yscale('log')

        # 3. Weekly activity distribution
        axes[1, 0].hist(customer_stats['YYYYWK_count'], bins=50, color='#3498db', alpha=0.7)
        axes[1, 0].set_title('Customer Activity Distribution')
        axes[1, 0].set_xlabel('Number of Active Weeks')
        axes[1, 0].set_ylabel('Number of Customers')

        # 4. Engagement pattern scatter plot
        if 'DOCUMENTS_OPENED_sum' in customer_stats.columns and 'CHURNED_FLAG_first' in customer_stats.columns:
            scatter = axes[1, 1].scatter(customer_stats['weeks_active'],
                                        customer_stats['DOCUMENTS_OPENED_sum'],
                                        c=customer_stats['CHURNED_FLAG_first'].map({'Y': 1, 'N': 0}),
                                        cmap='RdYlGn_r', alpha=0.5, s=20)
            axes[1, 1].set_title('Customer Engagement Pattern')
            axes[1, 1].set_xlabel('Weeks Active')
            axes[1, 1].set_ylabel('Total Documents Opened')
            axes[1, 1].set_yscale('log')
            plt.colorbar(scatter, ax=axes[1, 1], label='Churned')

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

        # 3. Documents trend
        ax3 = plt.subplot(2, 3, 3)
        if 'time_series_features' in data:
            ts_data = data['time_series_features']
            weekly_docs = ts_data.groupby('YYYYWK')['DOCUMENTS_OPENED'].sum().reset_index()
            ax3.plot(weekly_docs['YYYYWK'], weekly_docs['DOCUMENTS_OPENED'],
                    linewidth=1, color='#9b59b6')
            ax3.set_title('Weekly Document Usage Trend')
            ax3.set_xlabel('Week')
            ax3.set_ylabel('Documents')
            ax3.tick_params(axis='x', rotation=45)

        # 4. Storage trend
        ax4 = plt.subplot(2, 3, 4)
        if 'time_series_features' in data and 'USED_STORAGE_MB' in data['time_series_features'].columns:
            weekly_storage = ts_data.groupby('YYYYWK')['USED_STORAGE_MB'].sum().reset_index()
            ax4.plot(weekly_storage['YYYYWK'], weekly_storage['USED_STORAGE_MB'],
                    linewidth=1, color='#e67e22')
            ax4.set_title('Weekly Storage Usage Trend')
            ax4.set_xlabel('Week')
            ax4.set_ylabel('Storage (MB)')
            ax4.tick_params(axis='x', rotation=45)

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

        # Calculate churn rates by segment
        segment_stats = data.groupby('CUSTOMER_SEGMENT').agg({
            'CHURNED_FLAG': lambda x: (x == 'Y').sum(),
            'CUST_ACCOUNT_NUMBER': 'count'
        })
        segment_stats.columns = ['churned_count', 'total_count']
        segment_stats['churn_rate'] = (segment_stats['churned_count'] / segment_stats['total_count']) * 100

        # Get top segments by total count
        top_segments = segment_stats.nlargest(top_n, 'total_count')

        fig, axes = plt.subplots(1, 2, figsize=(14 * self.figsize_scale, 6 * self.figsize_scale))

        # 1. Churn rate by segment
        ax1 = axes[0]
        top_segments['churn_rate'].plot(kind='bar', ax=ax1, color='#e74c3c')
        ax1.set_title(f'Churn Rate by Top {top_n} Segments')
        ax1.set_xlabel('Customer Segment')
        ax1.set_ylabel('Churn Rate (%)')
        ax1.axhline(y=segment_stats['churn_rate'].mean(), color='b', linestyle='--',
                   label=f'Overall Avg: {segment_stats["churn_rate"].mean():.1f}%')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

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
        ax2.set_title(f'Customer Distribution in Top {top_n} Segments')
        ax2.set_xlabel('Customer Segment')
        ax2.set_ylabel('Number of Customers')
        ax2.legend(title='Status')

        plt.suptitle(kwargs.get('title', 'Segment-wise Churn Analysis'),
                    fontsize=14, fontweight='bold')
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
        return fig# Additional methods to add to ChurnLifecycleVizSnowflake class

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
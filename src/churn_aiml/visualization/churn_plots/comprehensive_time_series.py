"""
Comprehensive Time Series Visualization for Training Data
Creates visualizations for all time series columns with LOESS trends and health indicators

Author: Assistant
Date: 2025-08-19
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import warnings
from datetime import datetime
from statsmodels.nonparametric.smoothers_lowess import lowess
import yaml


def load_data_update_dates() -> List[str]:
    """Load data update dates from configuration"""
    try:
        config_path = Path('/home/applaimlgen/ricoh_aiml/conf/products/DOCUWARE/db/snowflake/data_config/dates_config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return [update['date'] for update in config.get('data_update_dates', [])]
    except:
        pass
    return []


def plot_comprehensive_time_series_new(
    data: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize_scale: float = 1.0,
    **kwargs
) -> plt.Figure:
    """
    Create comprehensive time series visualization for all numeric columns
    
    Features:
    - Scatter plots with neutral colors (no red/green/orange shapes)
    - LOESS trend lines with green segments for increasing, red for decreasing
    - Vertical lines for data update dates
    - Title color coding: green for healthy trend, red for unhealthy
    - Warning indicators for declining recent trends
    - Drops all negative values
    
    Args:
        data: DataFrame with time series data including YYYYWK column
        save_path: Optional path to save the figure
        figsize_scale: Scale factor for figure size
        **kwargs: Additional parameters
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    
    # Check for required column
    if 'YYYYWK' not in data.columns:
        warnings.warn("YYYYWK column not found in data")
        return plt.figure()
    
    # Filter data from 2020 onwards
    start_year = kwargs.get('start_year', 2020)
    start_week = start_year * 100 + 1
    filtered_data = data[data['YYYYWK'] >= start_week].copy()
    
    if filtered_data.empty:
        warnings.warn(f"No data available from {start_year} onwards")
        return plt.figure()
    
    # Identify all numeric columns except YYYYWK
    numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
    if 'YYYYWK' in numeric_cols:
        numeric_cols.remove('YYYYWK')
    
    # Remove columns that should not be plotted
    # IS_ACTIVE should be included if present (it's a valid time series metric)
    exclude_cols = ['CUST_ACCOUNT_NUMBER', 'CHURNED_FLAG', 'WEEKS_TO_CHURN']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Common time series columns to prioritize
    priority_cols = [
        'DOCUMENTS_OPENED', 'USED_STORAGE_MB', 'INVOICE_REVLINE_TOTAL',
        'ORIGINAL_AMOUNT_DUE', 'FUNCTIONAL_AMOUNT', 'DAYS_TO_CHURN', 
        'IS_ACTIVE', 'PAYMENT_AMOUNT', 'TOTAL_REVENUE'
    ]
    
    # Order columns: priority first, then others
    ordered_cols = []
    for col in priority_cols:
        if col in numeric_cols:
            ordered_cols.append(col)
    for col in numeric_cols:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    # Limit to first 9 columns for 3x3 grid
    plot_cols = ordered_cols[:9]
    
    if not plot_cols:
        warnings.warn("No numeric columns found for plotting")
        return plt.figure()
    
    # Calculate subplot layout
    n_plots = len(plot_cols)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(18 * figsize_scale, 6 * n_rows * figsize_scale))
    
    # Flatten axes for easier iteration
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Hide extra subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Load data update dates
    update_dates = load_data_update_dates()
    
    # Plot each time series
    for idx, col in enumerate(plot_cols):
        ax = axes[idx]
        
        # Prepare data for this column
        plot_data = filtered_data[['YYYYWK', col]].copy()
        
        # Drop negative values
        plot_data = plot_data[plot_data[col] >= 0].copy()
        
        # Drop NaN values
        plot_data = plot_data.dropna()
        
        if plot_data.empty:
            ax.text(0.5, 0.5, f'No valid data for {col}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col)
            continue
        
        # Aggregate by week (handle duplicates)
        weekly_data = plot_data.groupby('YYYYWK')[col].sum().reset_index()
        
        # Convert YYYYWK to datetime for better x-axis
        # Handle YYYYWK format properly (e.g., 202001 = 2020 week 01)
        def yyyywk_to_date(yyyywk):
            """Convert YYYYWK to date, handling edge cases"""
            year = yyyywk // 100
            week = yyyywk % 100
            # Clamp week to valid range (1-52/53)
            week = min(week, 52)
            week = max(week, 1)
            # Use ISO week date format
            try:
                return pd.Timestamp.fromisocalendar(year, week, 1)  # Monday of that week
            except:
                # Fallback for invalid weeks
                return pd.Timestamp(year, 1, 1) + pd.Timedelta(weeks=week-1)
        
        weekly_data['date'] = weekly_data['YYYYWK'].apply(yyyywk_to_date)
        
        # Sort by date
        weekly_data = weekly_data.sort_values('date')
        
        # Plot scatter points with neutral color (blue)
        ax.scatter(weekly_data['date'], weekly_data[col],
                  color='steelblue', s=30, alpha=0.6, 
                  edgecolors='navy', linewidth=0.5, zorder=3)
        
        # Add LOESS trend line with color-coded segments
        if len(weekly_data) >= 3:
            # Special handling for metrics with inverse logic
            # For these metrics: decreasing is good (green), increasing is bad (red)
            inverse_metrics = ['DAYS_TO_CHURN', 'ORIGINAL_AMOUNT_DUE']
            
            if col in inverse_metrics:
                title_color = add_loess_trend_with_segments(
                    ax, weekly_data['date'], weekly_data[col], col, inverse_trend=True
                )
            else:
                title_color = add_loess_trend_with_segments(
                    ax, weekly_data['date'], weekly_data[col], col
                )
        else:
            title_color = 'orange'
        
        # Add vertical lines for data update dates (gray color)
        for update_date in update_dates:
            try:
                # Convert YYYY-MM-DD to datetime
                update_dt = pd.to_datetime(update_date)
                ax.axvline(update_dt, color='gray', linestyle='--', 
                          alpha=0.5, linewidth=1, zorder=1)
            except:
                pass
        
        # Format the plot
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(format_column_name(col), fontsize=10)
        
        # Set title with color based on trend
        base_title = format_column_name(col)
        
        # Determine the title text and color based on the metric and trend
        if col == 'DAYS_TO_CHURN':
            # DAYS_TO_CHURN: decreasing recent trend = good (green), increasing = bad (red)
            if title_color == 'green':
                # Recent trend is decreasing (good for Days to Churn)
                title_text = f"✓ {base_title}"
            elif title_color == 'red':
                # Recent trend is increasing (bad for Days to Churn)
                title_text = f"⚠️ ALARM: {base_title}"
            elif title_color == 'orange':
                # Flat trend
                title_text = f"➡️ STABLE: {base_title}"
            else:
                title_text = base_title
                
        elif col == 'ORIGINAL_AMOUNT_DUE':
            # ORIGINAL_AMOUNT_DUE: decreasing recent trend = good (green), increasing = bad (red)
            if title_color == 'green':
                # Recent trend is decreasing (good for Amount Due)
                title_text = f"✓ {base_title}"
            elif title_color == 'red':
                # Recent trend is increasing (bad for Amount Due)
                title_text = f"⚠️ ALARM: {base_title}"
            elif title_color == 'orange':
                # Flat trend
                title_text = f"➡️ STABLE: {base_title}"
            else:
                title_text = base_title
                
        elif col == 'IS_ACTIVE' or 'active' in col.lower():
            # IS_ACTIVE / Active Customers: increasing recent trend = good (green), decreasing = bad (red)
            if title_color == 'green':
                # Recent trend is increasing (good for Active Customers)
                title_text = f"✓ {base_title}"
            elif title_color == 'red':
                # Recent trend is decreasing (bad for Active Customers)
                title_text = f"⚠️ ALARM: {base_title}"
            elif title_color == 'orange':
                # Flat trend
                title_text = f"➡️ STABLE: {base_title}"
            else:
                title_text = base_title
                
        else:
            # All other normal metrics: increasing = good, decreasing = bad
            if title_color == 'green':
                title_text = f"✓ {base_title}"
            elif title_color == 'red':
                title_text = f"⚠️ ALARM: {base_title}"
            elif title_color == 'orange':
                # Flat trend
                title_text = f"➡️ STABLE: {base_title}"
            else:
                title_text = base_title
        
        ax.set_title(title_text, fontsize=12, fontweight='bold', color=title_color)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add statistics box
        add_statistics_box(ax, weekly_data[col])
    
    # Overall title
    fig.suptitle('Comprehensive Time Series Analysis - All Metrics', 
                fontsize=16 * figsize_scale, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def add_loess_trend_with_segments(ax, x_dates, y_values, col_name, inverse_trend=False) -> str:
    """
    Add LOESS trend line with color-coded segments
    Green segments for increasing trend, red for decreasing
    
    Args:
        ax: Matplotlib axis
        x_dates: Date values for x-axis
        y_values: Y-axis values
        col_name: Name of the column being plotted
        inverse_trend: If True, decreasing is good (green) and increasing is bad (red)
                      Used for metrics like DAYS_TO_CHURN
    
    Returns:
        str: Color for the title based on recent trend ('green', 'red', or 'orange')
    """
    # Convert dates to numeric for LOESS
    x_numeric = np.arange(len(x_dates))
    
    # Remove NaN values
    valid_mask = ~np.isnan(y_values)
    if not valid_mask.any():
        return 'orange'
    
    x_clean = x_numeric[valid_mask]
    y_clean = y_values.values[valid_mask] if hasattr(y_values, 'values') else y_values[valid_mask]
    
    if len(x_clean) < 3:
        return 'orange'
    
    try:
        # Calculate LOESS smoothing
        smoothed = lowess(y_clean, x_clean, frac=0.3, it=3, return_sorted=True)
        x_smooth = smoothed[:, 0]
        y_smooth = smoothed[:, 1]
        
        # Map back to dates
        x_dates_smooth = []
        for x_val in x_smooth:
            idx = int(np.clip(x_val, 0, len(x_dates) - 1))
            x_dates_smooth.append(x_dates.iloc[idx])
        
        # Plot trend line with color-coded segments
        for i in range(len(x_smooth) - 1):
            # Calculate local slope
            if i > 0 and i < len(x_smooth) - 1:
                # Use points before and after for slope
                dx = x_smooth[i+1] - x_smooth[i-1]
                dy = y_smooth[i+1] - y_smooth[i-1]
                slope = dy / dx if dx != 0 else 0
            else:
                # Use adjacent points
                dx = x_smooth[min(i+1, len(x_smooth)-1)] - x_smooth[max(i-1, 0)]
                dy = y_smooth[min(i+1, len(y_smooth)-1)] - y_smooth[max(i-1, 0)]
                slope = dy / dx if dx != 0 else 0
            
            # Determine color based on slope (with inverse logic for certain metrics)
            if inverse_trend:
                # For DAYS_TO_CHURN: decreasing is good (green), increasing is bad (red)
                if slope < -0.01:  # Decreasing
                    segment_color = 'green'
                elif slope > 0.01:  # Increasing
                    segment_color = 'red'
                else:  # Flat
                    segment_color = 'gray'
            else:
                # Normal logic: increasing is good (green), decreasing is bad (red)
                if slope > 0.01:  # Increasing
                    segment_color = 'green'
                elif slope < -0.01:  # Decreasing
                    segment_color = 'red'
                else:  # Flat
                    segment_color = 'gray'
            
            # Plot segment
            ax.plot(x_dates_smooth[i:i+2], y_smooth[i:i+2],
                   color=segment_color, linewidth=2.5, alpha=0.7, zorder=2)
        
        # Determine title color based on recent trend (last 20% of data)
        recent_portion = max(3, int(len(x_smooth) * 0.2))
        recent_x = x_smooth[-recent_portion:]
        recent_y = y_smooth[-recent_portion:]
        
        if len(recent_x) >= 2:
            # Calculate recent trend
            recent_slope = np.polyfit(recent_x, recent_y, 1)[0]
            
            # Normalize by mean to get percentage change
            mean_value = np.mean(recent_y)
            if mean_value > 0:
                normalized_slope = recent_slope / mean_value
            else:
                normalized_slope = 0
            
            # Determine title color (with inverse logic for certain metrics)
            if inverse_trend:
                # For DAYS_TO_CHURN: declining is good (green), growing is bad (red)
                if normalized_slope < -0.001:  # Declining
                    return 'green'
                elif normalized_slope > 0.001:  # Growing
                    return 'red'
                else:
                    return 'orange'
            else:
                # Normal logic: declining is bad (red), growing is good (green)
                if normalized_slope < -0.001:  # Declining
                    return 'red'
                elif normalized_slope > 0.001:  # Growing
                    return 'green'
                else:
                    return 'orange'
        else:
            return 'orange'
            
    except Exception as e:
        warnings.warn(f"Error calculating LOESS trend for {col_name}: {e}")
        return 'orange'


def add_statistics_box(ax, values):
    """Add statistics box to the plot"""
    if len(values) == 0:
        return
    
    # Calculate statistics
    stats_text = (
        f'Mean: {values.mean():.2f}\n'
        f'Median: {values.median():.2f}\n'
        f'Std: {values.std():.2f}\n'
        f'Count: {len(values)}'
    )
    
    # Add text box
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                    alpha=0.8, edgecolor='gray'))


def format_column_name(col: str) -> str:
    """Format column name for display"""
    # Replace underscores with spaces and title case
    formatted = col.replace('_', ' ').title()
    
    # Special cases
    replacements = {
        'Mb': 'MB',
        'Db': 'DB',
        'Id': 'ID',
        'Cust': 'Customer',
        'Docs': 'Documents',
        'Avg': 'Average',
        'Pct': 'Percent',
        'Is Active': 'Active Customers'  # Better label for IS_ACTIVE
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted
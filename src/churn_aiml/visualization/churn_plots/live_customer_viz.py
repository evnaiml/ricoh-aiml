"""
Live Customer Visualization Extension for ChurnLifecycleVizSnowflake

This module extends the existing ChurnLifecycleVizSnowflake class with methods
specifically designed for visualizing live/active customer data that doesn't
contain churn flags or churn-related information.

Key differences from churn visualizations:
- No churn flag dependencies
- Focus on engagement, usage patterns, and tenure
- Active customer segmentation and trends
- Predictive readiness analysis

Author: Evgeni Nikolaev
Email: evgeni.nikolaev@ricoh-usa.com
Created: 2025-08-18
"""

from typing import Dict, Optional, Union, List, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression

# Optional scipy imports for advanced signal processing
try:
    from scipy.ndimage import uniform_filter1d
    from scipy import interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .churn_lifecycle import ChurnLifecycleVizSnowflake, add_data_update_markers


class LiveCustomerVizMixin:
    """
    Mixin class that adds live customer visualization methods to ChurnLifecycleVizSnowflake.
    This approach allows us to extend functionality without modifying the base class.
    """
    
    def _calculate_loess_trend(self, x_data: np.ndarray, y_data: np.ndarray, frac: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate LOESS (locally weighted regression) trend line using statsmodels.
        
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
            # Use statsmodels LOWESS (locally weighted scatterplot smoothing)
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
    
    def _detect_inflection_points(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[List[int], List[str]]:
        """
        Detect inflection points where trend changes from growth to decline.
        
        Args:
            x_data: X-axis data points (sorted)
            y_data: Y-axis data points corresponding to x_data
            
        Returns:
            Tuple of (inflection_indices, trend_changes)
            - inflection_indices: List of indices where trend changes
            - trend_changes: List of trend change descriptions
        """
        if len(x_data) < 5:  # Need at least 5 points for meaningful analysis
            return [], []
        
        # Calculate first derivative (rate of change)
        dy = np.diff(y_data)
        dx = np.diff(x_data)
        derivative = dy / dx
        
        # Smooth the derivative to reduce noise
        if len(derivative) > 3:
            if SCIPY_AVAILABLE:
                try:
                    smoothed_derivative = uniform_filter1d(derivative, size=3)
                except Exception:
                    # Fallback: simple moving average
                    smoothed_derivative = np.convolve(derivative, np.ones(3)/3, mode='same')
            else:
                # Fallback: simple moving average
                smoothed_derivative = np.convolve(derivative, np.ones(3)/3, mode='same')
        else:
            smoothed_derivative = derivative
        
        # Find where derivative changes sign (inflection points)
        inflection_indices = []
        trend_changes = []
        
        for i in range(1, len(smoothed_derivative)):
            prev_trend = 'increasing' if smoothed_derivative[i-1] > 0 else 'decreasing'
            curr_trend = 'increasing' if smoothed_derivative[i] > 0 else 'decreasing'
            
            # Detect growth-to-decline inflection (most critical)
            if prev_trend == 'increasing' and curr_trend == 'decreasing':
                inflection_indices.append(i + 1)  # +1 because derivative is one element shorter
                trend_changes.append('growth_to_decline')
            # Also detect decline-to-growth (recovery)
            elif prev_trend == 'decreasing' and curr_trend == 'increasing':
                inflection_indices.append(i + 1)
                trend_changes.append('decline_to_growth')
        
        return inflection_indices, trend_changes
    
    def _analyze_trend_direction(self, x_data: np.ndarray, y_data: np.ndarray) -> Tuple[str, str, str, List[int], List[str]]:
        """
        Analyze trend direction, detect inflection points, and determine health status.
        
        Args:
            x_data: X-axis data points
            y_data: Y-axis data points
            
        Returns:
            Tuple of (trend_direction, color, status_text, inflection_indices, trend_changes)
            - trend_direction: 'increasing', 'decreasing', or 'stable'
            - color: 'darkgreen', 'red', or 'gray'
            - status_text: Status description with warnings
            - inflection_indices: List of inflection point indices
            - trend_changes: List of trend change types
        """
        # Remove NaN values and sort by x
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        if len(x_clean) < 2:
            return 'stable', 'gray', 'Insufficient Data', [], []
        
        # Sort data by x-values
        sort_indices = np.argsort(x_clean)
        x_sorted = x_clean[sort_indices]
        y_sorted = y_clean[sort_indices]
        
        # Detect inflection points
        inflection_indices, trend_changes = self._detect_inflection_points(x_sorted, y_sorted)
        
        # Calculate overall trend (linear regression slope)
        model = LinearRegression()
        model.fit(x_sorted.reshape(-1, 1), y_sorted)
        slope = model.coef_[0]
        
        # Calculate relative slope (normalized by data range)
        y_range = y_sorted.max() - y_sorted.min()
        x_range = x_sorted.max() - x_sorted.min()
        
        if y_range == 0:
            return 'stable', 'gray', 'Stable', inflection_indices, trend_changes
        
        # Normalize slope by data ranges to get relative trend strength
        relative_slope = slope * x_range / y_range
        
        # Define thresholds for trend classification
        increase_threshold = 0.1  # 10% relative increase over time range
        decrease_threshold = -0.1  # 10% relative decrease over time range
        
        # Check recent trend (last 30% of data)
        recent_cutoff = int(len(x_sorted) * 0.7)
        if recent_cutoff < len(x_sorted) - 1:
            recent_x = x_sorted[recent_cutoff:]
            recent_y = y_sorted[recent_cutoff:]
            if len(recent_x) >= 2:
                recent_model = LinearRegression()
                recent_model.fit(recent_x.reshape(-1, 1), recent_y)
                recent_slope = recent_model.coef_[0]
                recent_relative_slope = recent_slope * (recent_x.max() - recent_x.min()) / (recent_y.max() - recent_y.min()) if (recent_y.max() - recent_y.min()) != 0 else 0
            else:
                recent_relative_slope = relative_slope
        else:
            recent_relative_slope = relative_slope
        
        # Create status message with inflection point warnings
        base_status = ""
        warning_suffix = ""
        
        # Check for growth-to-decline inflections
        growth_to_decline_count = sum(1 for change in trend_changes if change == 'growth_to_decline')
        if growth_to_decline_count > 0:
            warning_suffix = f" ⚠️ {growth_to_decline_count} Inflection Point(s)"
        
        # Determine overall status based on recent trend and inflection points
        if recent_relative_slope < decrease_threshold or (growth_to_decline_count > 0 and recent_relative_slope <= 0):
            return 'decreasing', 'red', f'ALARM: Declining{warning_suffix}', inflection_indices, trend_changes
        elif relative_slope > increase_threshold and recent_relative_slope >= 0:
            if growth_to_decline_count > 0:
                return 'mixed', '#FFA500', f'Mixed Trend{warning_suffix}', inflection_indices, trend_changes
            else:
                return 'increasing', 'darkgreen', 'Healthy Trend ✓', inflection_indices, trend_changes
        else:
            return 'stable', '#FFA500', f'Stable Trend{warning_suffix}', inflection_indices, trend_changes
    
    def _add_trend_to_plot(self, ax, x_data: np.ndarray, y_data: np.ndarray, metric_name: str):
        """
        Add continuous LOESS trend line with color-coded segments for growth/decline.
        The trend line spans the entire time interval without gaps.
        
        Args:
            ax: Matplotlib axis object
            x_data: X-axis data points
            y_data: Y-axis data points
            metric_name: Name of the metric for title updates
        """
        if len(x_data) < 2 or np.all(np.isnan(y_data)):
            return
        
        # Remove NaN values and sort
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        if len(x_clean) < 2:
            return
        
        # Sort data by x-values
        sort_indices = np.argsort(x_clean)
        x_sorted = x_clean[sort_indices]
        y_sorted = y_clean[sort_indices]
        
        try:
            # Calculate smooth trend line
            x_smooth, y_smooth = self._calculate_loess_trend(x_sorted, y_sorted)
            
            # Ensure we have enough points for smooth color transitions
            if len(x_smooth) < 50:
                # Interpolate to get more points for smoother color transitions
                from scipy import interpolate
                try:
                    f = interpolate.interp1d(x_smooth, y_smooth, kind='linear')
                    x_smooth_dense = np.linspace(x_smooth.min(), x_smooth.max(), 100)
                    y_smooth_dense = f(x_smooth_dense)
                    x_smooth = x_smooth_dense
                    y_smooth = y_smooth_dense
                except:
                    pass  # Use original if interpolation fails
            
            # Calculate point-by-point slope for color determination
            if len(x_smooth) > 1:
                # Calculate local slopes using a sliding window
                window_size = min(5, len(x_smooth) // 4)  # Adaptive window size
                colors = []
                
                for i in range(len(x_smooth)):
                    # Calculate local slope around point i
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(x_smooth), i + window_size + 1)
                    
                    if end_idx > start_idx + 1:
                        local_x = x_smooth[start_idx:end_idx]
                        local_y = y_smooth[start_idx:end_idx]
                        
                        # Calculate slope using linear regression on local window
                        if len(local_x) >= 2:
                            slope = np.polyfit(local_x, local_y, 1)[0]
                            
                            # Determine color based on slope
                            if slope < -0.01:  # Declining
                                colors.append('red')
                            elif slope > 0.01:  # Growing
                                colors.append('darkgreen')
                            else:  # Stable
                                colors.append('#FFA500')  # Orange
                        else:
                            colors.append('#FFA500')
                    else:
                        colors.append('#FFA500')
                
                # Plot continuous line segments with appropriate colors
                for i in range(len(x_smooth) - 1):
                    ax.plot(x_smooth[i:i+2], y_smooth[i:i+2], 
                           color=colors[i], linewidth=2.5, alpha=0.8, linestyle='-')
                
                # Add single label for the trend
                trend_direction, _, status_text, _, _ = self._analyze_trend_direction(x_sorted, y_sorted)
                # Create a dummy line for the legend
                if trend_direction == 'decreasing':
                    ax.plot([], [], color='red', linewidth=2.5, alpha=0.8, label=f'Trend: {status_text}')
                elif trend_direction == 'increasing':
                    ax.plot([], [], color='darkgreen', linewidth=2.5, alpha=0.8, label=f'Trend: {status_text}')
                else:
                    ax.plot([], [], color='#FFA500', linewidth=2.5, alpha=0.8, label=f'Trend: {status_text}')
            else:
                # Fallback for single point
                ax.plot(x_smooth, y_smooth, color='blue', linewidth=2.5, alpha=0.8, 
                       linestyle='-', label='Trend')
                
            # Optionally mark significant inflection points
            # Analyze to find inflection points
            _, _, _, inflection_indices, trend_changes = self._analyze_trend_direction(x_sorted, y_sorted)
            
            # Mark only the most significant inflection points
            if len(inflection_indices) > 0 and len(inflection_indices) <= 5:  # Only show if not too many
                for inf_idx, change_type in zip(inflection_indices, trend_changes):
                    if inf_idx < len(x_sorted):
                        # Find corresponding point on smooth curve
                        closest_smooth_idx = np.argmin(np.abs(x_smooth - x_sorted[inf_idx]))
                        if closest_smooth_idx < len(x_smooth):
                            if change_type == 'growth_to_decline':
                                # Small red marker for inflection
                                ax.scatter(x_smooth[closest_smooth_idx], y_smooth[closest_smooth_idx], 
                                         s=50, color='red', marker='v', edgecolors='darkred', 
                                         linewidth=1, alpha=0.7, zorder=5)
                            elif change_type == 'decline_to_growth':
                                # Small green marker for recovery
                                ax.scatter(x_smooth[closest_smooth_idx], y_smooth[closest_smooth_idx], 
                                         s=50, color='green', marker='^', edgecolors='darkgreen', 
                                         linewidth=1, alpha=0.7, zorder=5)
            
            # Determine trend direction at the END (most recent time points)
            # This is more relevant for understanding current state
            recent_trend_direction = 'stable'
            if len(x_smooth) >= 5:
                # Look at last 20% of data or at least last 5 points
                lookback = max(5, len(x_smooth) // 5)
                recent_x = x_smooth[-lookback:]
                recent_y = y_smooth[-lookback:]
                
                # Calculate slope of recent trend
                if len(recent_x) >= 2:
                    recent_slope = np.polyfit(recent_x, recent_y, 1)[0]
                    
                    # Normalize by data range for relative comparison
                    y_range = y_smooth.max() - y_smooth.min() if (y_smooth.max() - y_smooth.min()) > 0 else 1
                    x_range = x_smooth.max() - x_smooth.min() if (x_smooth.max() - x_smooth.min()) > 0 else 1
                    relative_recent_slope = recent_slope * x_range / y_range
                    
                    # Determine recent trend direction
                    if relative_recent_slope > 0.05:  # Increasing threshold
                        recent_trend_direction = 'increasing'
                    elif relative_recent_slope < -0.05:  # Decreasing threshold
                        recent_trend_direction = 'decreasing'
                    else:
                        recent_trend_direction = 'stable'
            
            # Update plot title with color based on RECENT trend (towards the end)
            current_title = ax.get_title()
            if current_title:  # Only update if title exists
                # Get current font size from axes (may have been scaled)
                current_fontsize = ax.title.get_fontsize() if hasattr(ax, 'title') and ax.title else 14
                
                # Set title color based on recent trend direction
                if recent_trend_direction == 'increasing':
                    # Green title for increasing trend at the end
                    new_title = f"{current_title}\n✓ {status_text}"
                    ax.set_title(new_title, fontsize=current_fontsize, fontweight='bold', color='darkgreen')
                elif recent_trend_direction == 'decreasing':
                    # Red title for decreasing trend at the end
                    new_title = f"{current_title}\n⚠ {status_text}"
                    ax.set_title(new_title, fontsize=current_fontsize, fontweight='bold', color='red')
                else:
                    # Orange for stable trend at the end
                    new_title = f"{current_title}\n- {status_text}"
                    ax.set_title(new_title, fontsize=current_fontsize, fontweight='bold', color='darkorange')
            
            # Add trend line to legend with larger font
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) <= 3:  # Only add legend if not too crowded
                ax.legend(loc='upper left', fontsize=11, framealpha=0.8, frameon=True, 
                         edgecolor='gray', borderpad=0.5)
                
        except Exception as e:
            warnings.warn(f"Could not add enhanced trend line for {metric_name}: {e}")
            # Fallback to simple trend line
            try:
                x_smooth, y_smooth = self._calculate_loess_trend(x_sorted, y_sorted)
                ax.plot(x_smooth, y_smooth, color='blue', linewidth=2, alpha=0.7, 
                       linestyle='-', label='Trend (simplified)')
            except:
                pass
    
    def plot_active_customer_distribution(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot distribution of active customers by segments, tenure, and engagement.
        Uses green color scheme for live customer data. Filters data from 2020 onwards.
        
        Args:
            data: DataFrame with customer metadata (no churn flags needed)
            save_path: Optional path to save the figure
            **kwargs: Additional parameters (start_year defaults to 2020)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Filter data from start_year onwards if date columns are available
        start_year = kwargs.get('start_year', 2020)
        filtered_data = data.copy()
        
        # Try to filter by contract start date or other date fields
        for date_col in ['CONTRACT_START', 'FINAL_EARLIEST_DATE']:
            if date_col in data.columns:
                date_filter = pd.to_datetime(filtered_data[date_col], errors='coerce') >= f'{start_year}-01-01'
                filtered_data = filtered_data[date_filter]
                break
        
        fig, axes = plt.subplots(2, 2, figsize=kwargs.get('figsize', (15, 12)))
        
        # Define green color scheme for live customers
        live_green = '#27AE60'
        light_green = '#58D68D'
        
        # 1. Customer Segment Distribution
        if 'CUSTOMER_SEGMENT' in filtered_data.columns:
            segment_counts = filtered_data['CUSTOMER_SEGMENT'].value_counts().head(10)
            axes[0, 0].bar(range(len(segment_counts)), segment_counts.values, color=live_green, alpha=0.8, edgecolor='darkgreen')
            axes[0, 0].set_xticks(range(len(segment_counts)))
            axes[0, 0].set_xticklabels(segment_counts.index, rotation=45, ha='right')
            axes[0, 0].set_title('Active Customer Segment Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].set_ylabel('Number of Customers')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Customer Segment Pie Chart
        if 'CUSTOMER_SEGMENT' in filtered_data.columns:
            segment_pie_data = filtered_data['CUSTOMER_SEGMENT'].value_counts().head(5)
            colors_pie = ['#27AE60', '#58D68D', '#82E0AA', '#ABEBC6', '#D5F4E6']
            axes[0, 1].pie(segment_pie_data.values, labels=segment_pie_data.index, colors=colors_pie,
                          autopct='%1.1f%%', startangle=90)
            axes[0, 1].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        else:
            axes[0, 1].axis('off')
        
        # 3. Customer Tenure Distribution or alternative visualization
        if 'MONTHS_ELAPSED' in filtered_data.columns:
            tenure = filtered_data['MONTHS_ELAPSED'].dropna()
            if len(tenure) > 0:
                axes[1, 0].hist(tenure, bins=30, color=live_green, alpha=0.7, edgecolor='darkgreen')
                axes[1, 0].set_xlabel('Months Since First Activity')
                axes[1, 0].set_ylabel('Number of Customers')
                axes[1, 0].set_title('Customer Tenure Distribution', fontsize=14, fontweight='bold')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                # Show message if no tenure data
                axes[1, 0].text(0.5, 0.5, 'No Tenure Data\nAvailable', ha='center', va='center',
                               fontsize=12, transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Customer Tenure', fontsize=14, fontweight='bold')
                axes[1, 0].axis('off')
        else:
            # Alternative: Show customer registration timeline
            if 'CONTRACT_START' in filtered_data.columns:
                contract_dates = pd.to_datetime(filtered_data['CONTRACT_START'], errors='coerce').dropna()
                if len(contract_dates) > 0:
                    # Group by month-year and count registrations
                    monthly_reg = contract_dates.dt.to_period('M').value_counts().sort_index()
                    if len(monthly_reg) > 0:
                        axes[1, 0].bar(range(len(monthly_reg)), monthly_reg.values, 
                                      color=live_green, alpha=0.8, edgecolor='darkgreen')
                        axes[1, 0].set_xlabel('Registration Period')
                        axes[1, 0].set_ylabel('New Customers')
                        axes[1, 0].set_title('Customer Registration Timeline', fontsize=14, fontweight='bold')
                        axes[1, 0].grid(True, alpha=0.3)
                        # Rotate x labels to prevent overlap
                        axes[1, 0].tick_params(axis='x', rotation=45)
                    else:
                        axes[1, 0].text(0.5, 0.5, 'No Registration\nData Available', ha='center', va='center',
                                       fontsize=12, transform=axes[1, 0].transAxes)
                        axes[1, 0].axis('off')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No Registration\nData Available', ha='center', va='center',
                                   fontsize=12, transform=axes[1, 0].transAxes)
                    axes[1, 0].axis('off')
            else:
                # Final fallback: show empty
                axes[1, 0].axis('off')
        
        # 4. Top Segments by Customer Count
        if 'CUSTOMER_SEGMENT' in filtered_data.columns:
            segment_counts = filtered_data['CUSTOMER_SEGMENT'].value_counts().head(10)
            x = np.arange(len(segment_counts))
            axes[1, 1].bar(x, segment_counts.values, color=light_green, alpha=0.8, edgecolor='darkgreen')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(segment_counts.index, rotation=45, ha='right')
            axes[1, 1].set_ylabel('Number of Customers')
            axes[1, 1].set_title('Top 10 Segments by Customer Count', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Active Customer Analysis Dashboard - From {start_year}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig
    
    def plot_live_usage_trends(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot all 7 time series trends for live customers using scatter plots with green color.
        Filters data from 2020 onwards by default.
        
        The 7 time series plotted are:
        1. DOCUMENTS_OPENED - Weekly documents opened
        2. USED_STORAGE_MB - Weekly storage usage in MB
        3. INVOICE_REVLINE_TOTAL - Weekly invoice revenue total
        4. ORIGINAL_AMOUNT_DUE - Weekly original amount due
        5. FUNCTIONAL_AMOUNT - Weekly functional amount
        6. CUST_ACCOUNT_NUMBER - Weekly active customer count (unique)
        7. AVG_DOCS_PER_CUSTOMER - Weekly average documents per customer (calculated)
        
        Each time series includes:
        - Scatter plot points in green color (always green)
        - LOESS trend line with health indicators:
          * Green trend segments = Growing/healthy portions
          * Red trend segments = Declining portions only
          * Orange trend = Stable Trend (no significant change)
        
        Args:
            data: Time series DataFrame with usage data
            save_path: Optional path to save the figure
            **kwargs: Additional parameters (start_year defaults to 2020, font_scale defaults to 1.3)
            
        Returns:
            matplotlib.figure.Figure: The created figure with 3x3 subplot grid
        """
        # Increase font sizes globally for this figure
        font_scale = kwargs.get('font_scale', 1.3)
        plt.rcParams.update({
            'font.size': 10 * font_scale,
            'axes.titlesize': 14 * font_scale,
            'axes.labelsize': 11 * font_scale,
            'xtick.labelsize': 10 * font_scale,
            'ytick.labelsize': 10 * font_scale,
            'legend.fontsize': 10 * font_scale
        })
        
        fig, axes = plt.subplots(3, 3, figsize=kwargs.get('figsize', (24, 18)))
        
        if 'YYYYWK' not in data.columns:
            warnings.warn("YYYYWK column not found in data")
            return fig
        
        # Filter data from start_year onwards (default 2020)
        start_year = kwargs.get('start_year', 2020)
        start_week = start_year * 100 + 1  # e.g., 202001 for 2020
        filtered_data = data[data['YYYYWK'] >= start_week].copy()
        
        if filtered_data.empty:
            warnings.warn(f"No data available from {start_year} onwards")
            return fig
        
        # Define green color for ALL scatter points (never changes)
        live_green = '#27AE60'  # Professional green color for all points
        
        # Define all 7 time series to plot
        time_series_metrics = [
            ('DOCUMENTS_OPENED', 'Documents Opened', 'Weekly Documents Opened'),
            ('USED_STORAGE_MB', 'Storage Used (MB)', 'Weekly Storage Usage (MB)'),
            ('INVOICE_REVLINE_TOTAL', 'Invoice Revenue', 'Weekly Invoice Revenue Total'),
            ('ORIGINAL_AMOUNT_DUE', 'Amount Due', 'Weekly Original Amount Due'),
            ('FUNCTIONAL_AMOUNT', 'Functional Amount', 'Weekly Functional Amount'),
            ('CUST_ACCOUNT_NUMBER', 'Active Customers', 'Weekly Active Customer Count'),
            ('DOCUMENTS_OPENED', 'Avg Docs/Customer', 'Weekly Avg Documents per Customer')
        ]
        
        # Aggregate by week for all metrics with proper NaN handling
        agg_dict = {}
        for metric, _, _ in time_series_metrics[:5]:  # First 5 are direct aggregations
            if metric in filtered_data.columns:
                # Use simple 'sum' aggregation - pandas handles NaN properly
                agg_dict[metric] = 'sum'
        
        # Add customer count (nunique doesn't have NaN issues)
        agg_dict['CUST_ACCOUNT_NUMBER'] = 'nunique'
        
        weekly_data = filtered_data.groupby('YYYYWK').agg(agg_dict).reset_index()
        
        # Remove weeks where all metrics are NaN
        metrics_cols = [m for m, _, _ in time_series_metrics[:5] if m in weekly_data.columns]
        if metrics_cols:
            # Keep rows where at least one metric is not NaN
            valid_mask = weekly_data[metrics_cols].notna().any(axis=1)
            weekly_data = weekly_data[valid_mask].copy()
        
        # Calculate average documents per customer
        if 'DOCUMENTS_OPENED' in weekly_data.columns and 'CUST_ACCOUNT_NUMBER' in weekly_data.columns:
            weekly_data['AVG_DOCS_PER_CUSTOMER'] = (
                weekly_data['DOCUMENTS_OPENED'] / weekly_data['CUST_ACCOUNT_NUMBER']
            ).fillna(0)
        
        # Plot all 7 time series as scatter plots with meaningful titles
        plot_configs = [
            ('DOCUMENTS_OPENED', 'Document Usage Trend', axes[0, 0]),
            ('USED_STORAGE_MB', 'Storage Usage Trend', axes[0, 1]),
            ('INVOICE_REVLINE_TOTAL', 'Revenue Performance Trend', axes[0, 2]),
            ('ORIGINAL_AMOUNT_DUE', 'Payment Due Trend', axes[1, 0]),
            ('FUNCTIONAL_AMOUNT', 'Functional Revenue Trend', axes[1, 1]),
            ('CUST_ACCOUNT_NUMBER', 'Customer Engagement Trend', axes[1, 2]),
            ('AVG_DOCS_PER_CUSTOMER', 'Usage Efficiency Trend', axes[2, 0])
        ]
        
        for metric, title, ax in plot_configs:
            if metric in weekly_data.columns:
                # ALWAYS use green for scatter points (never change color)
                ax.scatter(weekly_data['YYYYWK'], weekly_data[metric], 
                          s=40, alpha=0.8, color=live_green, edgecolors='darkgreen', 
                          linewidth=0.8, label='Data Points', zorder=3)
                ax.set_xlabel('Week (YYYYWK)', fontsize=12 * font_scale, fontweight='bold')
                ax.set_ylabel(title.split('Weekly ')[-1], fontsize=12 * font_scale, fontweight='bold')
                # Don't set title here - let _add_trend_to_plot handle it with proper color
                ax.set_title(title)  # Set basic title, color will be updated by trend analysis
                ax.grid(True, alpha=0.3)
                
                # Add LOESS trend line with health indicator (will also color the title)
                self._add_trend_to_plot(ax, weekly_data['YYYYWK'].values, weekly_data[metric].values, metric)
                
                # Add data update markers (vertical dashed red lines)
                add_data_update_markers(ax, date_format='YYYYWK')
                
                # Format x-axis for better readability with larger font
                if len(weekly_data) > 20:
                    ax.tick_params(axis='x', rotation=45, labelsize=10 * font_scale)
        
        # Use remaining subplots for additional insights
        # Plot 8: Recent 12 weeks trend (bar chart in green)
        if len(weekly_data) >= 12:
            recent_data = weekly_data.tail(12)
            ax = axes[2, 1]
            ax.bar(range(len(recent_data)), recent_data['CUST_ACCOUNT_NUMBER'], 
                   color=live_green, alpha=0.8, edgecolor='darkgreen')
            ax.set_xlabel('Recent Weeks')
            ax.set_ylabel('Active Customers')
            ax.set_title('Last 12 Weeks: Active Customers', fontsize=12, fontweight='bold')
            ax.set_xticks(range(0, len(recent_data), 2))
            ax.set_xticklabels([str(week) for week in recent_data['YYYYWK'].iloc[::2]], rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Plot 9: Growth Rate Analysis
        ax = axes[2, 2]
        if len(weekly_data) > 4:
            # Calculate week-over-week growth rates
            growth_metrics = []
            for col in ['DOCUMENTS_OPENED', 'CUST_ACCOUNT_NUMBER', 'USED_STORAGE_MB']:
                if col in weekly_data.columns:
                    weekly_values = weekly_data[col].values
                    growth_rate = np.diff(weekly_values) / weekly_values[:-1] * 100
                    growth_rate = growth_rate[~np.isnan(growth_rate) & ~np.isinf(growth_rate)]
                    if len(growth_rate) > 0:
                        avg_growth = np.mean(growth_rate)
                        growth_metrics.append((col.replace('_', ' ').title(), avg_growth))
            
            if growth_metrics:
                names, rates = zip(*growth_metrics)
                colors_growth = ['green' if r > 0 else 'red' for r in rates]
                ax.barh(range(len(names)), rates, color=colors_growth, alpha=0.7)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels(names)
                ax.set_xlabel('Avg Weekly Growth Rate (%)')
                ax.set_title('Average Weekly Growth Rates', fontsize=12, fontweight='bold')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.axis('off')
        else:
            ax.axis('off')
        
        plt.suptitle(f'Live Customer Time Series Analysis (7 Metrics) - From {start_year}', 
                    fontsize=18 * font_scale, fontweight='bold')
        plt.tight_layout()
        
        # Reset font sizes to defaults after figure creation
        plt.rcParams.update(plt.rcParamsDefault)
        
        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig
    
    def plot_live_customer_engagement(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot customer engagement metrics for live customers using green color scheme.
        Filters data from 2020 onwards and uses scatter plots with LOESS trends for time series.
        
        Features LOESS trend analysis for weekly engagement patterns:
        - Green trend segments = Growing/healthy portions
        - Red trend segments = Declining portions only
        - Orange trend line = "Stable Trend" (stable engagement)
        
        Args:
            data: DataFrame with usage and engagement data
            save_path: Optional path to save the figure
            **kwargs: Additional parameters (start_year defaults to 2020, font_scale defaults to 1.3)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        # Increase font sizes
        font_scale = kwargs.get('font_scale', 1.3)
        plt.rcParams.update({
            'font.size': 10 * font_scale,
            'axes.titlesize': 14 * font_scale,
            'axes.labelsize': 11 * font_scale,
            'xtick.labelsize': 10 * font_scale,
            'ytick.labelsize': 10 * font_scale,
            'legend.fontsize': 10 * font_scale
        })
        
        # Filter data from start_year onwards
        start_year = kwargs.get('start_year', 2020)
        start_week = start_year * 100 + 1  # e.g., 202001 for 2020
        
        filtered_data = data.copy()
        if 'YYYYWK' in data.columns:
            filtered_data = data[data['YYYYWK'] >= start_week].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=kwargs.get('figsize', (18, 12)))
        
        # Define green color scheme for live customers
        live_green = '#27AE60'
        light_green = '#58D68D'
        dark_green = '#1E7E34'
        
        # 1. Usage Distribution (Documents)
        if 'DOCUMENTS_OPENED' in filtered_data.columns:
            usage = filtered_data['DOCUMENTS_OPENED'][filtered_data['DOCUMENTS_OPENED'] > 0]
            axes[0, 0].hist(usage, bins=50, color=live_green, alpha=0.7, edgecolor=dark_green)
            axes[0, 0].set_xlabel('Documents Opened')
            axes[0, 0].set_ylabel('Number of Records')
            axes[0, 0].set_title('Document Usage Distribution', fontsize=12, fontweight='bold')
            axes[0, 0].set_yscale('log')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Storage Usage Distribution
        if 'USED_STORAGE_MB' in filtered_data.columns:
            storage = filtered_data['USED_STORAGE_MB'][filtered_data['USED_STORAGE_MB'] > 0]
            axes[0, 1].hist(storage, bins=50, color=light_green, alpha=0.7, edgecolor=dark_green)
            axes[0, 1].set_xlabel('Storage Used (MB)')
            axes[0, 1].set_ylabel('Number of Records')
            axes[0, 1].set_title('Storage Usage Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Customer Activity Level (Green color scheme)
        if 'CUST_ACCOUNT_NUMBER' in filtered_data.columns and 'DOCUMENTS_OPENED' in filtered_data.columns:
            # Aggregate with proper NaN handling
            customer_activity = filtered_data.groupby('CUST_ACCOUNT_NUMBER')['DOCUMENTS_OPENED'].agg(
                lambda x: x.sum() if x.notna().any() else np.nan
            )
            # Remove customers with all NaN values
            customer_activity = customer_activity.dropna()
            
            # Categorize activity levels
            low_activity = (customer_activity <= customer_activity.quantile(0.33)).sum()
            med_activity = ((customer_activity > customer_activity.quantile(0.33)) & 
                           (customer_activity <= customer_activity.quantile(0.67))).sum()
            high_activity = (customer_activity > customer_activity.quantile(0.67)).sum()
            
            labels = ['Low Activity', 'Medium Activity', 'High Activity']
            sizes = [low_activity, med_activity, high_activity]
            # Use different shades of green instead of mixed colors
            colors = [light_green, live_green, dark_green]
            
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            axes[1, 0].set_title('Customer Activity Levels', fontsize=12, fontweight='bold')
        
        # 4. Engagement Trend (Weekly Activity) - Use scatter plot with LOESS trend
        if 'YYYYWK' in filtered_data.columns and 'CUST_ACCOUNT_NUMBER' in filtered_data.columns:
            weekly_engagement = filtered_data.groupby('YYYYWK')['CUST_ACCOUNT_NUMBER'].nunique()
            
            # ALWAYS use green for scatter points (never change color)
            axes[1, 1].scatter(weekly_engagement.index, weekly_engagement.values, 
                             s=40, alpha=0.8, color=live_green, edgecolors=dark_green, 
                             linewidth=0.8, label='Data Points', zorder=3)
            axes[1, 1].set_xlabel('Week (YYYYWK)', fontsize=12 * font_scale, fontweight='bold')
            axes[1, 1].set_ylabel('Engaged Customers', fontsize=12 * font_scale, fontweight='bold')
            # Don't set title formatting here - let _add_trend_to_plot handle it with proper color
            axes[1, 1].set_title('Weekly Customer Engagement Trend')  # Basic title
            axes[1, 1].grid(True, alpha=0.3)
            
            # Add LOESS trend line with health indicator (will also color the title)
            self._add_trend_to_plot(axes[1, 1], weekly_engagement.index.values, weekly_engagement.values, 'WEEKLY_ENGAGEMENT')
            
            # Add data update markers (vertical dashed red lines)
            add_data_update_markers(axes[1, 1], date_format='YYYYWK')
            
            # Format x-axis for better readability with larger font
            if len(weekly_engagement) > 20:
                axes[1, 1].tick_params(axis='x', rotation=45, labelsize=10 * font_scale)
        
        plt.suptitle(f'Live Customer Engagement Analysis - From {start_year}', 
                    fontsize=18 * font_scale, fontweight='bold')
        plt.tight_layout()
        
        # Reset font sizes to defaults
        plt.rcParams.update(plt.rcParamsDefault)
        
        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig
    
    def plot_customer_activity_duration(self, data: pd.DataFrame, save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Plot comprehensive statistics and histograms of customer activity duration.
        Shows how long each customer has been active in the system.
        
        Args:
            data: DataFrame with time series data containing CUST_ACCOUNT_NUMBER and YYYYWK
            save_path: Optional path to save the figure
            **kwargs: Additional parameters (font_scale defaults to 1.3)
            
        Returns:
            matplotlib.figure.Figure: The created figure with activity duration analysis
        """
        # Increase font sizes
        font_scale = kwargs.get('font_scale', 1.3)
        plt.rcParams.update({
            'font.size': 10 * font_scale,
            'axes.titlesize': 14 * font_scale,
            'axes.labelsize': 11 * font_scale,
            'xtick.labelsize': 10 * font_scale,
            'ytick.labelsize': 10 * font_scale,
            'legend.fontsize': 10 * font_scale
        })
        
        fig, axes = plt.subplots(2, 3, figsize=kwargs.get('figsize', (20, 12)))
        
        # Calculate activity duration for each customer
        if 'CUST_ACCOUNT_NUMBER' in data.columns and 'YYYYWK' in data.columns:
            # Group by customer and calculate duration
            customer_activity = data.groupby('CUST_ACCOUNT_NUMBER')['YYYYWK'].agg(['min', 'max', 'count'])
            customer_activity['duration_weeks'] = customer_activity['count']
            
            # Convert YYYYWK to actual weeks between min and max
            customer_activity['first_week'] = customer_activity['min']
            customer_activity['last_week'] = customer_activity['max']
            
            # Calculate actual duration in weeks (approximate)
            def weeks_between(start_yyyywk, end_yyyywk):
                start_year = start_yyyywk // 100
                start_week = start_yyyywk % 100
                end_year = end_yyyywk // 100
                end_week = end_yyyywk % 100
                return (end_year - start_year) * 52 + (end_week - start_week) + 1
            
            customer_activity['actual_weeks'] = customer_activity.apply(
                lambda row: weeks_between(row['min'], row['max']), axis=1
            )
            customer_activity['actual_months'] = customer_activity['actual_weeks'] / 4.33
            customer_activity['actual_years'] = customer_activity['actual_weeks'] / 52
            
            # Define green color scheme
            live_green = '#27AE60'
            light_green = '#58D68D'
            dark_green = '#1E7E34'
            
            # 1. Distribution of Activity Duration (Weeks)
            ax = axes[0, 0]
            ax.hist(customer_activity['actual_weeks'], bins=30, color=live_green, 
                   alpha=0.7, edgecolor=dark_green)
            ax.set_xlabel('Activity Duration (Weeks)', fontsize=12 * font_scale, fontweight='bold')
            ax.set_ylabel('Number of Customers', fontsize=12 * font_scale, fontweight='bold')
            ax.set_title('Customer Activity Duration Distribution (Weeks)', 
                        fontsize=14 * font_scale, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_weeks = customer_activity['actual_weeks'].mean()
            median_weeks = customer_activity['actual_weeks'].median()
            ax.text(0.7, 0.9, f'Mean: {mean_weeks:.1f} weeks', 
                   transform=ax.transAxes, fontsize=11 * font_scale, fontweight='bold')
            ax.text(0.7, 0.85, f'Median: {median_weeks:.1f} weeks', 
                   transform=ax.transAxes, fontsize=11 * font_scale, fontweight='bold')
            
            # 2. Distribution of Activity Duration (Months)
            ax = axes[0, 1]
            ax.hist(customer_activity['actual_months'], bins=30, color=light_green, 
                   alpha=0.7, edgecolor=dark_green)
            ax.set_xlabel('Activity Duration (Months)', fontsize=12 * font_scale, fontweight='bold')
            ax.set_ylabel('Number of Customers', fontsize=12 * font_scale, fontweight='bold')
            ax.set_title('Customer Activity Duration Distribution (Months)', 
                        fontsize=14 * font_scale, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text
            mean_months = customer_activity['actual_months'].mean()
            median_months = customer_activity['actual_months'].median()
            ax.text(0.7, 0.9, f'Mean: {mean_months:.1f} months', 
                   transform=ax.transAxes, fontsize=11 * font_scale, fontweight='bold')
            ax.text(0.7, 0.85, f'Median: {median_months:.1f} months', 
                   transform=ax.transAxes, fontsize=11 * font_scale, fontweight='bold')
            
            # 3. Activity Duration Categories
            ax = axes[0, 2]
            # Categorize customers by activity duration
            bins = [0, 13, 26, 52, 104, float('inf')]  # weeks
            labels = ['< 3 months', '3-6 months', '6-12 months', '1-2 years', '> 2 years']
            customer_activity['duration_category'] = pd.cut(customer_activity['actual_weeks'], 
                                                            bins=bins, labels=labels)
            category_counts = customer_activity['duration_category'].value_counts()
            
            colors_gradient = [light_green, '#48C376', live_green, '#20904A', dark_green]
            ax.pie(category_counts.values, labels=category_counts.index, colors=colors_gradient,
                  autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11 * font_scale})
            ax.set_title('Customer Activity Duration Categories', 
                        fontsize=14 * font_scale, fontweight='bold')
            
            # 4. Box Plot of Activity Duration
            ax = axes[1, 0]
            box_data = [customer_activity['actual_weeks'], 
                       customer_activity['actual_months'] * 4.33,  # Convert back to weeks for comparison
                       customer_activity['actual_years'] * 52]  # Convert back to weeks
            bp = ax.boxplot(box_data, labels=['Weeks', 'Months (as weeks)', 'Years (as weeks)'],
                           patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor(live_green)
                patch.set_alpha(0.7)
            ax.set_ylabel('Duration (Weeks)', fontsize=12 * font_scale, fontweight='bold')
            ax.set_title('Activity Duration Box Plots', fontsize=14 * font_scale, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # 5. Activity Statistics Summary Table
            ax = axes[1, 1]
            stats_data = [
                ['Metric', 'Value'],
                ['Total Customers', f'{len(customer_activity):,}'],
                ['Mean Duration', f'{mean_weeks:.1f} weeks ({mean_months:.1f} months)'],
                ['Median Duration', f'{median_weeks:.1f} weeks ({median_months:.1f} months)'],
                ['Min Duration', f'{customer_activity["actual_weeks"].min():.0f} weeks'],
                ['Max Duration', f'{customer_activity["actual_weeks"].max():.0f} weeks'],
                ['Std Deviation', f'{customer_activity["actual_weeks"].std():.1f} weeks']
            ]
            
            table = ax.table(cellText=stats_data[1:], colLabels=stats_data[0],
                           cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11 * font_scale)
            table.scale(1.3, 2.0)
            
            # Color header
            for i in range(2):
                table[(0, i)].set_facecolor('#d4edda')
                table[(0, i)].set_text_props(weight='bold')
            
            ax.axis('off')
            ax.set_title('Activity Duration Statistics', fontsize=14 * font_scale, fontweight='bold')
            
            # 6. Cumulative Distribution
            ax = axes[1, 2]
            sorted_weeks = np.sort(customer_activity['actual_weeks'])
            cumulative = np.arange(1, len(sorted_weeks) + 1) / len(sorted_weeks) * 100
            
            ax.plot(sorted_weeks, cumulative, color=dark_green, linewidth=2.5)
            ax.fill_between(sorted_weeks, 0, cumulative, color=live_green, alpha=0.3)
            ax.set_xlabel('Activity Duration (Weeks)', fontsize=12 * font_scale, fontweight='bold')
            ax.set_ylabel('Cumulative % of Customers', fontsize=12 * font_scale, fontweight='bold')
            ax.set_title('Cumulative Distribution of Activity Duration', 
                        fontsize=14 * font_scale, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add percentile markers
            percentiles = [25, 50, 75]
            for p in percentiles:
                val = np.percentile(sorted_weeks, p)
                ax.axvline(x=val, color='red', linestyle='--', alpha=0.5)
                ax.text(val, 5, f'{p}th%\n{val:.0f}w', fontsize=10 * font_scale, 
                       ha='center', color='red')
        
        plt.suptitle('Customer Activity Duration Analysis', 
                    fontsize=18 * font_scale, fontweight='bold')
        plt.tight_layout()
        
        # Reset font sizes to defaults
        plt.rcParams.update(plt.rcParamsDefault)
        
        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig
    
    def plot_live_readiness_dashboard(self, data_dict: Dict[str, pd.DataFrame], 
                                     save_path: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create a comprehensive dashboard showing live data readiness for ML inference.
        
        Args:
            data_dict: Dictionary of DataFrames from live data loading
            save_path: Optional path to save the figure
            **kwargs: Additional parameters
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        fig, axes = plt.subplots(2, 3, figsize=kwargs.get('figsize', (18, 12)))
        
        # 1. Data Quality Overview
        ax = axes[0, 0]
        quality_data = [
            ['Dataset', 'Records', 'Columns', 'Missing %'],
        ]
        
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                quality_data.append([key[:15], f"{len(df):,}", str(df.shape[1]), f"{missing_pct:.1f}%"])
        
        # Create table
        table = ax.table(cellText=quality_data[1:], colLabels=quality_data[0],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.axis('off')
        ax.set_title('Data Quality Overview', fontsize=12, fontweight='bold')
        
        # 2. Customer Count by Dataset
        ax = axes[0, 1]
        if data_dict:
            dataset_counts = []
            dataset_names = []
            for key, df in data_dict.items():
                if isinstance(df, pd.DataFrame) and 'CUST_ACCOUNT_NUMBER' in df.columns:
                    count = df['CUST_ACCOUNT_NUMBER'].nunique()
                    dataset_counts.append(count)
                    dataset_names.append(key[:10])
            
            if dataset_counts:
                # Use green color scheme
                bars = ax.bar(range(len(dataset_counts)), dataset_counts, 
                             color='#27AE60', alpha=0.8, edgecolor='darkgreen')
                ax.set_xticks(range(len(dataset_names)))
                ax.set_xticklabels(dataset_names, rotation=45, ha='right')
                ax.set_ylabel('Unique Customers')
                ax.set_title('Customer Count by Dataset', fontsize=12, fontweight='bold')
                
                # Add value labels on bars
                for bar, count in zip(bars, dataset_counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count:,}', ha='center', va='bottom')
        
        # 3. Feature Completeness
        ax = axes[0, 2]
        if 'feature_engineering_dataset' in data_dict:
            fe_data = data_dict['feature_engineering_dataset']
            
            # Filter from 2020 onwards if YYYYWK is available
            start_year = kwargs.get('start_year', 2020)
            if 'YYYYWK' in fe_data.columns:
                start_week = start_year * 100 + 1
                fe_data = fe_data[fe_data['YYYYWK'] >= start_week]
            
            completeness = (1 - fe_data.isnull().mean()) * 100
            top_features = completeness.nlargest(10)
            
            ax.barh(range(len(top_features)), top_features.values, 
                   color='#27AE60', alpha=0.8, edgecolor='darkgreen')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels([name[:15] for name in top_features.index])
            ax.set_xlabel('Completeness %')
            ax.set_title(f'Top 10 Feature Completeness (From {start_year})', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Memory Usage
        ax = axes[1, 0]
        memory_data = []
        labels = []
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame):
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                memory_data.append(memory_mb)
                labels.append(key[:10])
        
        if memory_data:
            # Use a horizontal bar chart instead of pie chart to avoid overlapping labels
            ax.barh(range(len(labels)), memory_data, color='#27AE60', alpha=0.8, edgecolor='darkgreen')
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel('Memory Usage (MB)')
            ax.set_title('Memory Usage by Dataset', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels to bars
            for i, (label, value) in enumerate(zip(labels, memory_data)):
                ax.text(value + max(memory_data) * 0.02, i, f'{value:.1f}MB', 
                       va='center', fontsize=9)
        
        # 5. Missing Data Analysis
        ax = axes[1, 1]
        missing_data = []
        
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                missing_data.append((key[:15], missing_pct))
        
        if missing_data:
            names, percentages = zip(*missing_data)
            # Color based on missing percentage
            colors_missing = ['#27AE60' if p < 10 else '#FFA500' if p < 30 else '#E74C3C' for p in percentages]
            
            ax.bar(range(len(names)), percentages, color=colors_missing, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha='right')
            ax.set_ylabel('Missing Data (%)')
            ax.set_title('Missing Data by Dataset', fontsize=12, fontweight='bold')
            ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Good (<10%)')
            ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='Moderate (<30%)')
            ax.legend(loc='upper left', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.axis('off')
        
        # 6. ML Readiness Checklist
        ax = axes[1, 2]
        
        # Calculate actual status values
        total_records = sum(len(df) for df in data_dict.values() if isinstance(df, pd.DataFrame))
        unique_customers = 0
        has_customer_ids = False
        
        if 'customer_metadata' in data_dict and not data_dict['customer_metadata'].empty:
            if 'CUST_ACCOUNT_NUMBER' in data_dict['customer_metadata'].columns:
                unique_customers = data_dict['customer_metadata']['CUST_ACCOUNT_NUMBER'].nunique()
                has_customer_ids = True
        
        # Check for time range coverage
        time_coverage_ok = False
        weeks_span = 0
        if 'time_series_features' in data_dict and 'YYYYWK' in data_dict['time_series_features'].columns:
            weeks_span = data_dict['time_series_features']['YYYYWK'].nunique()
            time_coverage_ok = weeks_span >= 12  # At least 12 weeks of data
        
        # Use clear text instead of symbols: READY/NOT READY instead of checkmark/X
        readiness_items = [
            ['Criterion', 'Status', 'Details'],
            ['Time series data', 'READY' if 'time_series_features' in data_dict else 'NOT READY', 
             f"{data_dict['time_series_features'].shape[0]:,} rows" if 'time_series_features' in data_dict else 'Missing'],
            ['Customer metadata', 'READY' if 'customer_metadata' in data_dict else 'NOT READY',
             f"{unique_customers:,} customers" if unique_customers > 0 else 'Missing'],
            ['Feature engineering data', 'READY' if 'feature_engineering_dataset' in data_dict else 'NOT READY',
             f"{data_dict['feature_engineering_dataset'].shape[0]:,} rows" if 'feature_engineering_dataset' in data_dict else 'Missing'],
            ['Sufficient time coverage', 'READY' if time_coverage_ok else 'NOT READY',
             f"{weeks_span} weeks" if weeks_span > 0 else 'Unknown'],
            ['Minimum data volume', 'READY' if total_records >= 1000 else 'NOT READY',
             f"{total_records:,} total records"]
        ]
        
        table = ax.table(cellText=readiness_items[1:], colLabels=readiness_items[0],
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.4, 1.3)
        
        # Color code the status cells based on READY/NOT READY text
        for i in range(1, len(readiness_items)):
            status = readiness_items[i][1]
            if status == 'READY':
                table[(i, 1)].set_facecolor('#d4edda')  # Light green for ready
                table[(i, 1)].set_text_props(weight='bold', color='darkgreen')
            elif status == 'NOT READY':
                table[(i, 1)].set_facecolor('#f8d7da')  # Light red for not ready
                table[(i, 1)].set_text_props(weight='bold', color='darkred')
        ax.axis('off')
        ax.set_title('ML Readiness Checklist', fontsize=12, fontweight='bold')
        
        start_year = kwargs.get('start_year', 2020)
        plt.suptitle(f'Live Data ML Readiness Dashboard - From {start_year}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        self._save_figure(fig, save_path, dpi=kwargs.get('dpi', 300))
        return fig


# Create the extended class by combining the base class with the mixin
class LiveCustomerVizSnowflake(ChurnLifecycleVizSnowflake, LiveCustomerVizMixin):
    """
    Extended visualization class that includes both churn analysis and live customer methods.
    This class can handle both training data (with churn flags) and live data (without churn flags).
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', palette: str = 'husl', figsize_scale: float = 1.0):
        """
        Initialize the extended visualization class.
        
        Args:
            style: Matplotlib style to use
            palette: Seaborn color palette  
            figsize_scale: Scale factor for figure sizes
        """
        super().__init__(style, palette, figsize_scale)
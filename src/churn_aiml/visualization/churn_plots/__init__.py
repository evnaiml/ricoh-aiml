"""
Churn Lifecycle Visualization Module

This module provides comprehensive visualization capabilities for churn analysis,
including distribution plots, trend analysis, segment comparisons, and lifecycle metrics.

✅ Key Components:
- ChurnLifecycleVizBase: Abstract base class for visualization
- ChurnLifecycleVizSnowflake: Concrete implementation with all plot methods
- Centralized location for all churn-related visualizations

📊 Available Visualizations:
- Churn distribution (pie charts, bar charts)
- Monthly churn trends and distributions
- Customer lifecycle analysis
- Segment-wise churn comparisons
- Risk score distributions
- Time series usage patterns
- Comprehensive monitoring dashboards

Author: Evgeni Nikolaev
Email: evgeni.nikolaev@ricoh-usa.com
"""

from .churn_lifecycle import ChurnLifecycleVizBase, ChurnLifecycleVizSnowflake
from .live_customer_viz import LiveCustomerVizSnowflake, LiveCustomerVizMixin

__all__ = ['ChurnLifecycleVizBase', 'ChurnLifecycleVizSnowflake', 'LiveCustomerVizSnowflake', 'LiveCustomerVizMixin']
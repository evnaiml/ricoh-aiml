"""
Example Usage of LiveCustomerVizSnowflake for Live Data Visualization

This script demonstrates how to use the new LiveCustomerVizSnowflake class
to create visualizations for live customer data without churn flags.

Author: Evgeni Nikolaev
Email: evgeni.nikolaev@ricoh-usa.com
Created: 2025-08-18
"""

from pathlib import Path
import pandas as pd
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

# Import the new live customer visualization class
from churn_aiml.visualization.churn_plots import LiveCustomerVizSnowflake
from churn_aiml.data.db.snowflake.loaddata import SnowLiveDataLoader
from churn_aiml.loggers.loguru.config import setup_logger_for_script, get_logger
from churn_aiml.utils.find_paths import ProjectRootFinder

def main():
    """Example usage of live customer visualizations."""
    
    # Setup paths and configuration
    churn_aiml_dir = ProjectRootFinder().find_path()
    conf_dir = churn_aiml_dir / "conf"
    
    # Clear and initialize Hydra configuration
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(conf_dir), version_base=None):
        cfg = compose(config_name="config")
    
    # Setup logger
    logger_config = setup_logger_for_script(cfg, __file__)
    logger = get_logger()
    
    # Initialize the live data loader
    logger.info("Loading live customer data...")
    live_loader = SnowLiveDataLoader(config=cfg, environment="development")
    live_data = live_loader.load_data()
    
    # Initialize the new live customer visualization class
    viz = LiveCustomerVizSnowflake(figsize_scale=1.2)
    
    # Create output directory
    script_dir = Path(__file__).parent
    figs_dir = script_dir / "live_viz_examples"
    figs_dir.mkdir(exist_ok=True)
    
    logger.info("Creating live customer visualizations...")
    
    # 1. Active Customer Distribution Dashboard
    if 'customer_metadata' in live_data:
        logger.info("Creating active customer distribution dashboard...")
        viz.plot_active_customer_distribution(
            data=live_data['customer_metadata'],
            save_path=str(figs_dir / "active_customer_distribution.png")
        )
    
    # 2. Live Usage Trends
    if 'time_series_features' in live_data:
        logger.info("Creating live usage trends...")
        viz.plot_live_usage_trends(
            data=live_data['time_series_features'],
            save_path=str(figs_dir / "live_usage_trends.png")
        )
    
    # 3. Customer Engagement Analysis
    if 'time_series_features' in live_data:
        logger.info("Creating customer engagement analysis...")
        viz.plot_live_customer_engagement(
            data=live_data['time_series_features'],
            save_path=str(figs_dir / "customer_engagement.png")
        )
    
    # 4. ML Readiness Dashboard
    logger.info("Creating ML readiness dashboard...")
    viz.plot_live_readiness_dashboard(
        data_dict=live_data,
        save_path=str(figs_dir / "ml_readiness_dashboard.png")
    )
    
    # 5. Data Preparation for Feature Engineering
    # The live_data contains data prepared for feature engineering (intact, no imputation)
    # Missing values are preserved as NaN for downstream feature engineering tools
    # Example: feature_dataset = live_loader.get_feature_engineering_dataset()
    
    # 6. The LiveCustomerVizSnowflake class is specifically designed for live data
    # It provides specialized visualizations for active customer analysis
    # without requiring churn flags or historical comparison data
    
    logger.info(f"All visualizations saved to: {figs_dir}")
    logger.info("Live customer visualization example completed!")

if __name__ == "__main__":
    main()
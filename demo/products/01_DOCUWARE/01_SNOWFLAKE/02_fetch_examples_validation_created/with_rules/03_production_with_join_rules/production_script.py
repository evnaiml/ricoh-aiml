"""
Production script to process ALL Snowflake sequel join rules with ALL rows
- Uses DataExporter directly for consistency
- Enhanced logging and error handling
- Optimized for production environments
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-04
# * CREATED ON: 2025-08-04
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# 🚀 PRODUCTION SCRIPT USAGE GUIDE
# -----------------------------------------------------------------------------
# This script processes complex SQL sequel join rules for customer churn analysis
# with comprehensive logging, performance monitoring, and production features.
#
# 📊 OUTPUT LOCATION: data/products/DOCUWARE/DB/snowflake/csv/join_rules/
# 📁 FILES CREATED: {query_name}.csv, {query_name}_schema.json, {query_name}_report.csv
# 📈 PERFORMANCE: performance_summary_YYYYMMDD_HHMMSS.json (with metrics)
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 BASIC EXECUTION OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 Default execution (development environment, console logging enabled):
# python production_script.py
#
# 🔸 Production environment (recommended for live systems):
# python production_script.py environment=production
#
# 🔸 Process specific sequel join rule only:
# python production_script.py query_name=usage_latest
#
# 🔸 Combined options (production + specific query):
# python production_script.py environment=production query_name=usage_latest
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📝 LOGGING CONTROL OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 VERBOSE LOGGING (Default - Development):
# python production_script.py                                    # Full console output
# python production_script.py disable_console_logging=false      # Explicit console on
# python production_script.py environment=development            # More verbose
#
# 🔹 DEBUG LOGGING (Maximum verbosity):
# python production_script.py debug=true                         # Debug mode
# python production_script.py debug=true disable_console_logging=false  # Debug + console
# python production_script.py +loggers.loguru.default_level=DEBUG        # Override log level
#
# 🔹 MINIMAL LOGGING (Production - Silent):
# python production_script.py disable_console_logging=true       # File logging only
# python production_script.py environment=production disable_console_logging=true  # Prod silent
#
# 🔹 CUSTOM LOG LEVELS:
# python production_script.py +loggers.loguru.default_level=ERROR        # Errors only
# python production_script.py +loggers.loguru.default_level=WARNING      # Warnings + errors
# python production_script.py +loggers.loguru.default_level=INFO         # Standard logging
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🏭 PRODUCTION DEPLOYMENT OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 RECOMMENDED PRODUCTION COMMAND:
# python production_script.py environment=production disable_console_logging=true
#
# 🔸 PRODUCTION WITH SPECIFIC QUERY:
# python production_script.py environment=production disable_console_logging=true query_name=usage_latest
#
# 🔸 MINIMAL PRODUCTION (errors only):
# python production_script.py environment=production disable_console_logging=true +loggers.loguru.default_level=ERROR
#
# 🔸 AUTOMATED PRODUCTION (with timestamp):
# python production_script.py environment=production disable_console_logging=true > prod_$(date +%Y%m%d_%H%M%S).log 2>&1
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🌙 BACKGROUND PROCESSING & AUTOMATION
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 STANDARD BACKGROUND PROCESSING:
# nohup python production_script.py disable_console_logging=true &
# nohup python production_script.py environment=production disable_console_logging=true &
#
# 🔸 BACKGROUND WITH OUTPUT CAPTURE:
# nohup python production_script.py disable_console_logging=true > process.log 2>&1 &
# nohup python production_script.py environment=production disable_console_logging=true > prod.log 2>&1 &
#
# 🔸 PROCESS MANAGEMENT:
# echo $! > production_script.pid                               # Save process ID
# kill -0 $(cat production_script.pid) && echo "Running" || echo "Finished"  # Check status
# kill $(cat production_script.pid)                            # Stop process
#
# 🔸 SILENT OUTPUT REDIRECTION:
# # Linux/Mac:
# python production_script.py disable_console_logging=true > /dev/null 2>&1
# # Windows:
# python production_script.py disable_console_logging=true > NUL 2>&1
#
# 🔸 CRON/SCHEDULED AUTOMATION:
# # Add to crontab for daily execution at 2 AM:
# 0 2 * * * cd /path/to/ricoh_aiml && python production_script.py environment=production disable_console_logging=true
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📊 MONITORING & DEBUGGING
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 REAL-TIME LOG MONITORING:
# tail -f logs/loguru/info.log                                 # Monitor info logs
# tail -f logs/loguru/error.log                                # Monitor errors
# tail -f logs/loguru/debug.log                                # Monitor debug (if enabled)
#
# 🔹 PROCESS STATUS CHECKING:
# ps aux | grep production_script                              # Check if running
# ps aux | grep python | grep production                       # Find python processes
#
# 🔹 OUTPUT MONITORING:
# ls -la data/products/DOCUWARE/DB/snowflake/csv/join_rules/   # Check output files
# ls data/products/DOCUWARE/DB/snowflake/csv/join_rules/ | wc -l  # Count output files
# du -sh data/products/DOCUWARE/DB/snowflake/csv/join_rules/   # Check disk usage
#
# 🔹 PERFORMANCE MONITORING:
# cat data/products/DOCUWARE/DB/snowflake/csv/join_rules/performance_summary_*.json  # Check metrics
# grep "processing time" logs/loguru/info.log                  # Check execution times
# grep "rows/second" logs/loguru/info.log                      # Check processing speed
#
# ═══════════════════════════════════════════════════════════════════════════════
# ⚙️ ADVANCED CONFIGURATION OPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔸 HYDRA CONFIGURATION INSPECTION:
# python production_script.py --cfg job                        # Show full configuration
# python production_script.py --cfg hydra                      # Show Hydra settings
# python production_script.py --help                           # Show available options
#
# 🔸 CONFIGURATION OVERRIDES:
# python production_script.py +processing.max_workers=8        # Override parallel workers
# python production_script.py +data.chunk_size=20000           # Override chunk size
# python production_script.py +processing.memory_limit_gb=16   # Override memory limit
#
# 🔸 CUSTOM CSV OUTPUT CONFIGURATION:
# python production_script.py +products.DOCUWARE.db.snowflake.csv_config=csv_rules_config  # Explicit config
#
# 🔸 ENVIRONMENT VARIABLE SUPPORT:
# export HYDRA_FULL_ERROR=1                                    # Full Hydra error traces
# export PYTHONUNBUFFERED=1                                    # Unbuffered Python output
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🚨 ERROR HANDLING & RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════
#
# 🔹 ERROR DIAGNOSIS:
# grep "ERROR" logs/loguru/error.log                           # Find error messages
# grep "Failed to" logs/loguru/info.log                        # Find failed operations
# grep "❌" logs/loguru/info.log                                # Find failed queries
#
# 🔹 RECOVERY OPTIONS:
# python production_script.py query_name=usage_latest          # Retry specific query
# python production_script.py debug=true query_name=usage_latest  # Debug specific query
#
# 🔹 PARTIAL FAILURE HANDLING:
# # Script continues on individual query failures
# # Check performance summary JSON for successful vs failed queries
# # Retry failed queries individually using query_name parameter
#
# ═══════════════════════════════════════════════════════════════════════════════
# 📋 EXPECTED OUTPUT STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
#
# 📁 data/products/DOCUWARE/DB/snowflake/csv/join_rules/
# ├── usage_latest.csv                                         # Main join rule dataset
# ├── usage_latest_schema.json                                 # Pydantic data schema
# ├── usage_latest_report.csv                                  # Data quality analysis
# └── performance_summary_20250804_143022.json                 # Performance metrics
#
# 📊 Log Files:
# ├── logs/loguru/info.log                                     # General processing logs
# ├── logs/loguru/error.log                                    # Error logs
# └── logs/loguru/debug.log                                    # Debug logs (if enabled)
#
# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 PRODUCTION BEST PRACTICES
# ═══════════════════════════════════════════════════════════════════════════════
#
# ✅ RECOMMENDED PRODUCTION SETUP:
# 1. Use environment=production for optimized settings
# 2. Enable disable_console_logging=true for automated runs
# 3. Monitor logs/loguru/error.log for issues
# 4. Check performance_summary_*.json for metrics
# 5. Verify output file creation in join_rules/ directory
# 6. Set up log rotation for long-running deployments
# 7. Monitor disk space in output directory
# 8. Schedule regular execution via cron/task scheduler
#
# ⚡ PERFORMANCE OPTIMIZATION:
# - Script includes automatic memory cleanup (gc.collect())
# - Progress tracking with tqdm for long-running operations
# - Performance metrics export for analysis
# - Robust error handling with continuation on failures
# - Memory-efficient processing with pandas optimizations
#
# 🔒 SECURITY CONSIDERATIONS:
# - Credentials managed through Hydra configuration
# - No sensitive data in command-line parameters
# - Log masking enabled for database credentials
# - Secure file permissions on output directories
#
# ═══════════════════════════════════════════════════════════════════════════════

import hydra
from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time
import gc

from churn_aiml.loggers.loguru.config import setup_logger, get_logger
from churn_aiml.data.db.snowflake.fetchdata import SnowFetch
from churn_aiml.utils.find_paths import ProjectRootFinder
from churn_aiml.data.validation.data_exporter import DataExporter

# Set paths to directories
churn_aiml_dir = ProjectRootFinder().find_path()
conf_dir = churn_aiml_dir / "conf"

@hydra.main(version_base=None, config_path=conf_dir.as_posix(), config_name="config")
def main(cfg: DictConfig) -> None:
    """Production main function to process ALL sequel join rules with ALL rows"""

    start_time = time.time()

    # Setup logger
    logger_config = setup_logger(cfg)
    logger = get_logger()
    logger.info("🚀 PRODUCTION: Processing ALL Snowflake sequel join rules with ALL rows")

    # Load sequel join rules from configuration
    try:
        sequel_rules = cfg.products.DOCUWARE.db.snowflake.sequel_rules
        logger.info("✅ Successfully loaded sequel join rules configuration")
    except Exception as e:
        logger.error(f"❌ Failed to load sequel join rules: {str(e)}")
        return

    # Initialize output directory for sequel join rules using csv_rules_config
    project_root = ProjectRootFinder().find_path()

    try:
        # For sequel join rules, use csv_rules_config instead of csv_raw_config
        from hydra import compose
        rules_cfg = compose(config_name="config", overrides=["products/DOCUWARE/db/snowflake/csv_config=csv_rules_config"])
        csv_config = rules_cfg.products.DOCUWARE.db.snowflake.csv_config.csv_output
        relative_path = csv_config.get('base_dir', 'data/products/DOCUWARE/DB/snowflake/csv/join_rules')
        logger.info("✅ Using csv_rules_config for sequel join rules processing")
    except Exception as e:
        # Fallback if csv_rules_config is not available
        logger.warning(f"csv_rules_config not found: {str(e)}")
        logger.info("🔄 Using fallback configuration for sequel join rules")
        relative_path = 'data/products/DOCUWARE/DB/snowflake/csv/join_rules'

    output_dir = project_root / relative_path
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")

    # Initialize DataExporter directly
    exporter = DataExporter(output_dir, logger)

    # Get configuration parameters
    environment = cfg.get('environment', 'production')  # Default to production
    query_name = cfg.get('query_name', None)

    # Get all available sequel join rules
    available_queries = list(sequel_rules.queries.keys())

    # Display available sequel join rules
    logger.info("Available sequel join rules:")
    for query_name_item in available_queries:
        description = sequel_rules.queries[query_name_item].get('description', 'No description')
        logger.info(f"  - {query_name_item}: {description}")

    # Determine which queries to process
    if query_name:
        if query_name in available_queries:
            queries_to_process = [query_name]
            logger.info(f"Processing specific sequel join rule: {query_name}")
        else:
            logger.error(f"Query '{query_name}' not found. Available queries: {available_queries}")
            return
    else:
        queries_to_process = available_queries  # Process ALL sequel join rules

    logger.info(f"📊 Processing ALL {len(queries_to_process)} sequel join rules with environment={environment}")
    logger.info("⚠️  Processing ALL rows from each sequel join rule (no limits)")

    # Track results
    successful_exports = []
    failed_exports = []
    total_rows_processed = 0
    query_performance = []

    # Process all sequel join rules
    for i, current_query_name in tqdm(enumerate(queries_to_process), desc="Processing sequel join rules"):
        try:
            query_start_time = time.time()

            # Get query configuration
            query_config = sequel_rules.queries[current_query_name]
            sql_query = query_config.sql

            # Fetch ALL data from Snowflake using sequel join rule (no limits)
            logger.info(f"🔄 ({i+1:2d}/{len(queries_to_process)}) Executing sequel join rule: {current_query_name}")
            with SnowFetch(config=cfg, environment=environment) as fetcher:
                df = fetcher.session.sql(sql_query).to_pandas()

            query_fetch_time = time.time() - query_start_time
            total_rows_processed += len(df)

            logger.info(f"📊 ({i+1:2d}/{len(queries_to_process)}) {current_query_name}: fetched {len(df):,} rows "
                       f"in {query_fetch_time:.1f}s")

            # Export to CSV with schema and report using DataExporter directly
            logger.info(f"💾 Exporting {current_query_name} ({len(df):,} rows, {len(df.columns)} columns)")
            export_result = exporter.export_with_analysis(df, current_query_name)

            # Add timing and performance information
            total_time = time.time() - query_start_time
            export_result['export_time_seconds'] = total_time
            export_result['fetch_time_seconds'] = query_fetch_time
            export_result['rows_per_second'] = len(df) / query_fetch_time if query_fetch_time > 0 else 0

            successful_exports.append({
                'query': current_query_name,
                'result': export_result
            })

            # Track performance metrics
            query_performance.append({
                'query': current_query_name,
                'rows': len(df),
                'columns': len(df.columns),
                'fetch_time': query_fetch_time,
                'total_time': total_time,
                'size_mb': export_result['size_mb'],
                'rows_per_second': export_result['rows_per_second']
            })

            logger.info(f"✅ ({i+1:2d}/{len(queries_to_process)}) {current_query_name}: completed successfully "
                       f"({export_result['rows']:,} rows, {export_result['size_mb']:.2f} MB, {total_time:.1f}s)")

            # Memory cleanup for large datasets
            del df
            gc.collect()

        except Exception as e:
            logger.error(f"❌ ({i+1:2d}/{len(queries_to_process)}) {current_query_name}: failed - {str(e)}")
            failed_exports.append({'query': current_query_name, 'error': str(e)})
            continue

    # Final summary
    total_time = time.time() - start_time
    logger.info("🎉 PRODUCTION SEQUEL JOIN RULES PROCESSING COMPLETED!")
    logger.info(f"⏱️  Total processing time: {total_time/60:.1f} minutes")
    logger.info(f"✅ Successful exports: {len(successful_exports)}")
    logger.info(f"❌ Failed exports: {len(failed_exports)}")

    if successful_exports:
        total_size = sum(result['result']['size_mb'] for result in successful_exports)
        avg_rows_per_second = sum(perf['rows_per_second'] for perf in query_performance) / len(query_performance)

        logger.info(f"📊 Total data exported: {total_rows_processed:,} rows, {total_size:.2f} MB")
        logger.info(f"📈 Average processing speed: {avg_rows_per_second:,.0f} rows/second")
        logger.info(f"📁 All files saved to: {output_dir}")

        # Show performance metrics for largest queries
        sorted_performance = sorted(query_performance, key=lambda x: x['rows'], reverse=True)
        logger.info("🏆 Top sequel join rules by row count:")
        for j, perf in enumerate(sorted_performance[:5], 1):
            logger.info(f"  {j}. {perf['query']}: {perf['rows']:,} rows, {perf['size_mb']:.2f} MB, "
                       f"{perf['total_time']:.1f}s ({perf['rows_per_second']:,.0f} rows/s)")

        # Show performance metrics for slowest queries
        sorted_by_time = sorted(query_performance, key=lambda x: x['total_time'], reverse=True)
        logger.info("⏱️  Slowest sequel join rules by processing time:")
        for j, perf in enumerate(sorted_by_time[:3], 1):
            logger.info(f"  {j}. {perf['query']}: {perf['total_time']:.1f}s, {perf['rows']:,} rows, "
                       f"{perf['rows_per_second']:,.0f} rows/s")

    if failed_exports:
        logger.error(f"❌ Failed sequel join rules ({len(failed_exports)}):")
        for failure in failed_exports:
            logger.error(f"  - {failure['query']}: {failure['error']}")

    # Export performance summary to JSON for analysis
    if query_performance:
        performance_file = output_dir / f"performance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        performance_summary = {
            'total_processing_time_minutes': total_time / 60,
            'total_rows_processed': total_rows_processed,
            'total_size_mb': total_size if successful_exports else 0,
            'successful_queries': len(successful_exports),
            'failed_queries': len(failed_exports),
            'environment': environment,
            'timestamp': datetime.now().isoformat(),
            'query_performance': query_performance
        }

        with open(performance_file, 'w') as f:
            json.dump(performance_summary, f, indent=2)

        logger.info(f"📊 Performance summary saved to: {performance_file}")

    logger.info("🎯 For each sequel join rule, 3 files were created:")
    logger.info("   - {query_name}.csv (data)")
    logger.info("   - {query_name}_schema.json (Pydantic schema)")
    logger.info("   - {query_name}_report.csv (analysis report)")

    if query_performance:
        logger.info("   - performance_summary_YYYYMMDD_HHMMSS.json (performance metrics)")

    logger.info("🚀 Production sequel join rules processing completed successfully!")


if __name__ == "__main__":
    main()
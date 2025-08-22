"""
Loguru configuration module for the ML project with automatic script-local logging.

This module provides intelligent logging configuration that:
- Automatically creates logs in the main script's directory by default
- Detects the executing script location using __main__.__file__
- Falls back to current working directory for interactive environments
- Supports manual log directory override when needed
- Maintains backward compatibility with existing scripts

Key Features:
- Auto-detection of main script location for log placement
- Creates /logs/loguru/ in the same folder as the executing script
- Date-based subdirectories (YYYYMMDD) for log organization
- Time-stamped log files (am/pm-HH-MM-SS.log)
- Console and file logging with configurable levels
- Error-only log files for critical issue tracking
- No JSON/YAML file outputs (simplified version)

Usage:
    # Default behavior - logs created in script's directory
    logger_config = setup_logger(cfg)
    logger = get_logger()
    
    # With manual override
    logger_config = setup_logger(cfg, log_dir_override=Path('/custom/path'))
    logger = get_logger()
"""
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-08-13
# * CREATED ON: 2025-07-29
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from loguru import logger
from omegaconf import DictConfig, OmegaConf


class LoggerConfig:
    """Configure and manage Loguru logging for the entire project."""

    def __init__(self, config: DictConfig, log_dir_override: Optional[Path] = None):
        """
        Initialize logger configuration.
        
        Args:
            config: Hydra configuration
            log_dir_override: Optional path to override the log directory location.
                            If provided, logs will be created relative to this path.
        """
        self.config = config
        self.logger = logger
        self.log_dir_override = log_dir_override

        # Auto-detect where loguru config is located
        self.loguru_config = self._find_loguru_config()

        self._setup_logging()

    def _find_loguru_config(self):
        """Automatically find loguru configuration in the config structure."""
        # Default configuration structure
        default_config = {
            'default_level': 'INFO',
            'console_enabled': True,
            'file_enabled': True,
            'file': {
                'base_dir': 'logs/loguru',
                'format': '{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}',
                'rotation': '100 MB',
                'retention': '30 days',
                'compression': 'gz'
            },
            'console': {
                'format': '<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> | <level>{message}</level>',
                'colorize': True
            }
        }

        # Try different possible paths
        possible_paths = [
            'loggers.loguru',     # Standard nested path
            'loguru',             # Direct path
            'logging.loguru',     # Alternative nested path
            'logger.loguru',      # Alternative nested path
        ]

        found_config = None
        found_path = None

        for path in possible_paths:
            try:
                # Navigate through the config using the dot notation
                config_part = self.config
                for key in path.split('.'):
                    config_part = getattr(config_part, key)

                # If we get here without exception, we found it
                found_config = config_part
                found_path = path
                break

            except (AttributeError, KeyError):
                continue

        if found_config is not None:
            print(f"✅ Found loguru config at: {found_path}")

            # Merge found config with defaults to ensure all required keys exist
            merged_config = OmegaConf.merge(
                OmegaConf.create(default_config),
                found_config
            )
            return merged_config
        else:
            # If no config found, create a minimal default
            print("⚠️  No loguru config found, using minimal defaults")
            return OmegaConf.create(default_config)

    def _get_log_base_dir(self) -> Path:
        """
        Get the base directory for logs.
        By default, creates logs in the same directory as the main script being run.
        
        If log_dir_override is provided, use it as the parent directory.
        Otherwise, try to detect the main script location.
        """
        base_dir_str = self.loguru_config.file.base_dir
        
        if self.log_dir_override:
            # Use the override directory as the parent
            base_dir = self.log_dir_override / base_dir_str
        else:
            # Try to find the main script being executed
            import __main__
            
            if hasattr(__main__, '__file__') and __main__.__file__:
                # Get the directory of the main script
                main_script_dir = Path(__main__.__file__).parent.resolve()
                base_dir = main_script_dir / base_dir_str
            else:
                # Fallback to current working directory if we can't find the main script
                # This happens in interactive environments like Jupyter
                base_dir = Path.cwd() / base_dir_str
        
        return base_dir

    def _setup_logging(self):
        """Set up Loguru logging with file and console handlers."""
        # Remove default handler
        self.logger.remove()

        # Create base log directory
        base_dir = self._get_log_base_dir()
        base_dir.mkdir(parents=True, exist_ok=True)

        # Setup console logging if enabled
        if self.loguru_config.console_enabled:
            self._setup_console_logging()

        # Setup file logging if enabled
        if self.loguru_config.file_enabled:
            self._setup_file_logging()

        # Setup additional sinks if configured (simplified - no JSON/YAML outputs)
        if hasattr(self.loguru_config, 'additional_sinks'):
            self._setup_additional_sinks()

    def _setup_console_logging(self):
        """Setup console logging handler."""
        console_config = self.loguru_config.console

        self.logger.add(
            sys.stdout,
            format=console_config.format,
            level=self.loguru_config.default_level,
            colorize=console_config.colorize,
            enqueue=True
        )

    def _setup_file_logging(self):
        """Setup file logging handler with date/time directory structure."""
        file_config = self.loguru_config.file

        # Generate current date and time for directory structure
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")

        # Determine AM/PM and format time
        am_pm = "am" if now.hour < 12 else "pm"
        hour_12 = now.hour if now.hour <= 12 else now.hour - 12
        if hour_12 == 0:
            hour_12 = 12
        time_str = f"{am_pm}-{hour_12:02d}-{now.minute:02d}-{now.second:02d}"

        # Create log file path
        base_dir = self._get_log_base_dir()
        log_dir = base_dir / date_str
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{time_str}.log"

        self.logger.add(
            str(log_file),
            format=file_config.format,
            level=self.loguru_config.default_level,
            rotation=file_config.rotation,
            retention=file_config.retention,
            compression=file_config.compression,
            enqueue=True
        )

        # Store the log file path for reference
        self.current_log_file = log_file
        
        # Log where the file is being created
        print(f"📝 Log file created at: {log_file}")

    def _setup_additional_sinks(self):
        """Setup additional logging sinks (simplified - only error files)."""
        additional_sinks = self.loguru_config.additional_sinks

        # Generate current date and time for file naming
        now = datetime.now()
        date_str = now.strftime("%Y%m%d")
        am_pm = "am" if now.hour < 12 else "pm"
        hour_12 = now.hour if now.hour <= 12 else now.hour - 12
        if hour_12 == 0:
            hour_12 = 12
        time_str = f"{am_pm}-{hour_12:02d}-{now.minute:02d}-{now.second:02d}"

        base_dir = self._get_log_base_dir() / date_str
        base_dir.mkdir(parents=True, exist_ok=True)

        # Error file sink only
        if hasattr(additional_sinks, 'error_file') and additional_sinks.error_file.enabled:
            error_file = base_dir / f"errors-{time_str}.log"

            self.logger.add(
                str(error_file),
                format=additional_sinks.error_file.format,
                level=additional_sinks.error_file.level,
                enqueue=True
            )

    # ===============================================================
    # SIMPLIFIED LOGGING METHODS - NO FILE OUTPUTS
    # ===============================================================

    def log_with_context(self, level: str, message: str, **kwargs):
        """Log a message with additional context."""
        if kwargs:
            context_str = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
            enhanced_message = f"{message} | Context: {context_str}"
        else:
            enhanced_message = message

        # Use the logger
        log_func = getattr(self.logger, level.lower())
        log_func(enhanced_message)

    def log_performance(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()]) if metrics else ""
        message = f"Performance: {operation} took {duration:.3f}s"
        if metrics_str:
            message += f" | {metrics_str}"
        
        self.logger.info(message)

    def log_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log application events."""
        data_str = " | ".join([f"{k}={v}" for k, v in event_data.items()]) if event_data else ""
        message = f"Event: {event_type}"
        if data_str:
            message += f" | {data_str}"
            
        self.logger.success(message)

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log errors with additional context."""
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()]) if context else ""
        message = f"Error: {type(error).__name__}: {error}"
        if context_str:
            message += f" | Context: {context_str}"
            
        self.logger.error(message)

    def auto_log_snowflake_fetch(self, table_name: str, df, duration: float, limit: int = None, **fetch_info):
        """Log Snowflake fetch operation with performance info."""
        
        # Log performance
        self.log_performance(
            operation=f"snowflake_fetch_{table_name}",
            duration=duration,
            table_name=table_name,
            rows_fetched=len(df),
            columns_count=len(df.columns),
            memory_usage_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            rows_per_second=round(len(df) / duration, 2) if duration > 0 else 0,
            limit_applied=limit,
            **fetch_info
        )
        
        # Log table analysis
        self.logger.info(f"Table analysis: {table_name} | Rows: {len(df):,} | Columns: {len(df.columns)} | "
                        f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f}MB | "
                        f"Duration: {duration:.3f}s | Rate: {len(df) / duration:.2f} rows/s")
        
        # Log event
        self.log_event("table_fetch_completed", {
            "table_name": table_name,
            "success": True,
            "duration": duration,
            "rows": len(df),
            "limited": limit is not None
        })

    def log_batch_summary(self, successful_tables: list, failed_tables: list, total_duration: float, **batch_info):
        """Log batch processing summary."""
        total_rows = sum(count for _, count in successful_tables) if successful_tables else 0
        total_tables = len(successful_tables) + len(failed_tables)
        success_rate = (len(successful_tables) / total_tables) * 100 if total_tables > 0 else 0
        
        summary_msg = (f"Batch completed: {len(successful_tables)}/{total_tables} tables successful | "
                      f"Success rate: {success_rate:.1f}% | Total rows: {total_rows:,} | "
                      f"Duration: {total_duration:.3f}s | Rate: {total_rows / total_duration:.2f} rows/s")
        
        if successful_tables:
            largest_table = max(successful_tables, key=lambda x: x[1])
            smallest_table = min(successful_tables, key=lambda x: x[1])
            avg_size = total_rows / len(successful_tables)
            summary_msg += (f" | Largest: {largest_table[0]} ({largest_table[1]:,} rows) | "
                           f"Smallest: {smallest_table[0]} ({smallest_table[1]:,} rows) | "
                           f"Avg size: {avg_size:.0f} rows")
        
        # Add any additional batch info
        if batch_info:
            batch_str = " | ".join([f"{k}={v}" for k, v in batch_info.items()])
            summary_msg += f" | {batch_str}"
        
        self.logger.success(summary_msg)

    # ===============================================================
    # UTILITY METHODS
    # ===============================================================

    def create_child_logger(self, name: str, **extra_context):
        """Create a child logger with additional context."""
        return self.logger.bind(logger_name=name, **extra_context)

    def set_level(self, level: str):
        """Change the logging level dynamically."""
        # Remove existing handlers and re-setup with new level
        self.logger.remove()

        # Update the config
        original_level = self.loguru_config.default_level
        self.loguru_config.default_level = level

        # Re-setup logging
        self._setup_logging()

        self.logger.info(f"Logging level changed from {original_level} to {level}")

    def disable_console(self):
        """Disable console logging."""
        self.loguru_config.console_enabled = False
        self.logger.remove()
        self._setup_logging()
        self.logger.info("Console logging disabled")

    def enable_console(self):
        """Enable console logging."""
        self.loguru_config.console_enabled = True
        self.logger.remove()
        self._setup_logging()
        self.logger.info("Console logging enabled")

    def add_custom_sink(self, sink_path: str, level: str = "INFO",
                       format_str: Optional[str] = None):
        """Add a custom logging sink."""
        if format_str is None:
            format_str = self.loguru_config.file.format

        # Create directory if it doesn't exist
        Path(sink_path).parent.mkdir(parents=True, exist_ok=True)

        self.logger.add(
            sink_path,
            format=format_str,
            level=level,
            enqueue=True
        )

        self.logger.info(f"Added custom sink: {sink_path} with level {level}")


# Global logger instance - will be initialized when setup_logger is called
project_logger = None

def setup_logger(config: DictConfig, log_dir_override: Optional[Path] = None) -> LoggerConfig:
    """
    Setup the global logger for the project.
    By default, creates logs in the same directory as the main script being executed.
    
    Args:
        config (DictConfig): Hydra configuration containing logging settings
        log_dir_override (Optional[Path]): Optional path to override log directory location.
                                          If not provided, logs will be created in the
                                          main script's directory.

    Returns:
        LoggerConfig: Configured logger instance

    Example:
        setup_logger(cfg)  # Creates logs in script's directory by default
        logger = get_logger()
        logger.info("Hello world!")
    """
    global project_logger
    project_logger = LoggerConfig(config, log_dir_override)
    return project_logger


def get_logger():
    """
    Get the configured logger instance.
    Use this in your modules to get the same logger.

    Returns:
        Logger: Configured Loguru logger instance

    Raises:
        RuntimeError: If setup_logger() hasn't been called yet

    Example:
        logger = get_logger()
        logger.info("This message will be logged")
    """
    if project_logger is None:
        raise RuntimeError("Logger not initialized. Call setup_logger() first.")
    return project_logger.logger


def setup_logger_for_script(config: DictConfig, script_file: str = None) -> LoggerConfig:
    """
    Convenience function to setup logger for scripts with local log directory.
    
    Args:
        config: Hydra configuration
        script_file: The __file__ variable from the calling script.
                    If provided, logs will be created in the script's directory.
    
    Returns:
        LoggerConfig: Configured logger instance
        
    Example:
        # In your script:
        logger_config = setup_logger_for_script(cfg, __file__)
        logger = get_logger()
    """
    if script_file:
        log_dir_override = Path(script_file).parent
    else:
        log_dir_override = None
    
    return setup_logger(config, log_dir_override)
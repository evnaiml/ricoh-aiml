#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility module to suppress known deprecation warnings.

This module should be imported at the very beginning of scripts to suppress
warnings that come from third-party packages that we cannot directly fix.
"""

import warnings
import os

def suppress_known_warnings():
    """
    Suppress known deprecation warnings from third-party packages.

    Current suppressions:
    - pkg_resources deprecation from snowflake-snowpark-python
    """
    # Suppress pkg_resources deprecation warning
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*pkg_resources is deprecated.*"
    )

    # Suppress the warning about pkg_resources being slated for removal
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=".*pkg_resources package is slated for removal.*"
    )

    # Also set environment variable to suppress at import time
    os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:snowflake.snowpark.session"

# Auto-suppress when module is imported
suppress_known_warnings()
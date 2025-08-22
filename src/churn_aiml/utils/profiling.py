"""Functions to profile computations"""
#%% [markdown]
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com
# -----------------------------------------------------------------------------
#%%
# * UPDATED ON: 2025-06-12
# * CREATED ON: 2024-06-12
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# %% [markdown]
# -----------------------------------------------------------------------------
# * Load python modules
from typing import Any, Generator
import contextlib
import time
from datetime import timedelta
# %% [markdown]
# -----------------------------------------------------------------------------
@contextlib.contextmanager
# * Function timer(...)
def timer(msg: str = "") -> Generator[Any, Any, Any]:
    """This function evaluates and prints elapsed time for the context block.

    Args:
        logging_dir (str): The path to the logging directory.
        logging_fname (str): The name of the logging file.
        msg (str, optional): Special message to log. Defaults to "".

    Yields:
        Generator[Any, Any, Any]:
    """
    start = time.time()
    yield None
    end = time.time()
    seconds = end - start
    td = timedelta(seconds=seconds)
    if len(msg) > 0:
        msg += " "
    print(f"{msg}Elapsed: hh:mm:ss {td}")
# %%
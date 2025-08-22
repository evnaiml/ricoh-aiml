"""Set GPU / MPS / CPU device automatically"""
#%% [markdown]
# -----------------------------------------------------------------------------
# * Summary:
# - Functions to to automatically check and set gpu/mps/cpu devices if
# available on both Linux (gpu) and MAC (mps).
# - If gpu/mps is not available, the functions select cpu by default.
# - The functions also print out some hardware specs.
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@scipher.com, evgeni.v.nikolaev.ai@gmail.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-06-12
# * CREATED ON: 2024-06-12
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
#%% [markdown]
# -----------------------------------------------------------------------------
# * Load python modules
from typing import Optional
import os
import platform
import torch
# %% [markdown]
# -----------------------------------------------------------------------------
# Function: check_torch_device
def check_torch_device(is_getcwd: Optional[bool] = True) -> torch.device:
    """The function checks and sets GPU/MPS/CPU device for torch if used.

        Args:
            is_getcwd (Optional[bool], optional): A flag to log the current working directory. Defaults to True.

        Returns:
            torch.device: the torch device to use: cuda/mps or cpu
    """
    if "macOS" not in platform.platform():
        print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

        try:
            print(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
            print(f'torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}')
            print(f"torch.cuda GPU total memory: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")
            print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
        except Exception as e:
            print('torch.cuda.device_count(): NA')
            print('torch.cuda.get_device_name(0): NA')
            print("torch.cuda GPU total memory: NA")
            print('torch.cuda.current_device(): NA')

        # setting device on GPU if available, else CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        print(f"torch.has_mps: {torch.has_mps}")
        print(f"torch.backends.mps.is_available(): {torch.backends.mps.is_available()}")
        print(f"torch.backends.mps.is_built(): {torch.backends.mps.is_built()}")

        # setting device on GPU if available, else CPU
        device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
        print(f'USING DEVICE: {device}')

    if is_getcwd:
        print(f"Current directory: {os.getcwd()}")

    return device
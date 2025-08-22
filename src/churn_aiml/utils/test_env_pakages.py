"""Test GPU availability on Linux (Cuda) or Mac (MPS)"""
#%%
# -----------------------------------------------------------------------------
# * Author: Evgeni Nikolaev
# * Emails: evgeni.nikolaev@ricoh-usa.com, evgeni.v.nikolaev.ai@gmail.com
# -----------------------------------------------------------------------------
# * UPDATED ON: 2025-06-12
# * CREATED ON: 2025-06-12
# -----------------------------------------------------------------------------
# COPYRIGHT @ 2025 Ricoh. All rights reserved.
# The information contained herein is copyright and proprietary to
# Ricoh and may not be reproduced, disclosed, or used in
# any manner without prior written permission from Ricoh.
# -----------------------------------------------------------------------------
# %%
%load_ext autoreload
%autoreload 2
%load_ext watermark
%watermark
%matplotlib inline
#%%
import numpy
import pandas
import pydantic
import sklearn
import scipy
# import statsmodels
import category_encoders
# import feature_engine
import xgboost
# import lightgbm
import catboost

import tsfresh

import cudf
import cupy as cp
import dask

# import pycaret
# from pycaret.datasets import get_data
# from pycaret.regression import *
# import h2o
# import pymc
# import torch
# from torchviz import make_dot
# import jax
# import spacy
# import langchain
# from langchain_community.agent_toolkits import create_sql_agent
# from langchain_openai import ChatOpenAI
# import rdkit
from churn_aiml.utils.profiling import timer
from churn_aiml.utils.device import check_torch_device
import warnings; warnings.filterwarnings("ignore")
#%% Print platform parameters
print('Versions control:')
%watermark --iversions
#%%
with timer():
    device = check_torch_device()
# %%
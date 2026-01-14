#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:34:31 2024

@author: johbay
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
os.chdir('/project_cephfs/3022017.06/projects/lifespan_hbr/johbay/Velocity/CODE')
import pickle
from utils import SC_idp_cols, DES_idp_cols, DK_idp_cols
from numpy import matlib as mb
from sklearn.linear_model import LinearRegression
from pcntoolkit.util.utils import scaler
from pcntoolkit.normative_model.norm_hbr import quantile
import pcntoolkit as pkt
import math
from scipy import special as spp
from utilities_thrive import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
import importlib
import utilities_thrive
importlib.reload(utilities_thrive)
from utilities_thrive_new_version import *

#%%
measures = SC_idp_cols()
measures = DK_idp_cols()
measures = DES_idp_cols()
clean_df, data, data_all, data_demented = process_data(atlas = "DES", full_dataset=False)

#%%
for idp_nr, measure in enumerate(measures):
    if  idp_nr == 200:
        break
    measure = measures[idp_nr]
    make_velocity_plots(idp_nr=idp_nr, measure=measure,data=clean_df, selected_sex='male', model_type='SHASHb_1', atlas="DES",re_estimate = True, full_dataset=False)

#


#%%

   
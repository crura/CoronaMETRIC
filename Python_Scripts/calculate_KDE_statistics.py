#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:40:38 2023

@author: crura
"""

import os
from scipy.io import readsav
import matplotlib as mpl
import matplotlib.pyplot as plt
import git
from matplotlib.patches import Circle
from astropy.wcs import WCS
from astropy.io import fits
import sunpy
import sunpy.map
import matplotlib
import numpy as np
import scipy as sci
from tqdm import tqdm_notebook
import pandas as pd
import unittest
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib
from os.path import join, isfile
from os import listdir
from scipy.stats import gaussian_kde
matplotlib.use('TkAgg')
mpl.use('TkAgg')
from functions import KL_div, JS_Div, calculate_KDE, calculate_KDE_statistics


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
data_dir = os.path.join(repo_path,'Data/QRaFT/errors.sav')

datapath = join(repo_path, 'Data/QRaFT/COR-1_Errors_New')
datafiles = [join(datapath,f) for f in listdir(datapath) if isfile(join(datapath,f)) and f !='.DS_Store']

err_cor1_central_new = np.array([]) # idl_save_new['ERR_ARR_COR1']
err_cor1_los_new = np.array([])
err_forward_cor1_central_new = np.array([])
err_forward_cor1_los_new = np.array([])
err_random_new = np.array([])

for i in datafiles:
    idl_save = readsav(i)
    err_cor1_central_new = np.concatenate([err_cor1_central_new, idl_save['ERR_ARR_COR1']])
    err_cor1_los_new = np.concatenate([err_cor1_los_new,idl_save['ERR_ARR_LOS_COR1']])
    err_forward_cor1_central_new = np.concatenate([err_forward_cor1_central_new,idl_save['ERR_ARR_FORWARD']])
    err_forward_cor1_los_new = np.concatenate([err_forward_cor1_los_new,idl_save['ERR_ARR_LOS_FORWARD']])
    err_random_new = np.concatenate([err_random_new,idl_save['ERR_ARR_RND']])


# convert arrays from radians to degrees
err_cor1_central_deg_new = err_cor1_central_new[np.where(err_cor1_central_new > 0)]*180/np.pi
err_forward_cor1_central_deg_new = err_forward_cor1_central_new[np.where(err_forward_cor1_central_new > 0)]*180/np.pi
err_random_deg_new = err_random_new[np.where(err_random_new > 0)]*180/np.pi


KDE_cor1_central_deg_new = calculate_KDE(err_cor1_central_deg_new)
KDE_forward_cor1_central_deg_new = calculate_KDE(err_forward_cor1_central_deg_new)
KDE_random_deg_new = calculate_KDE(err_random_deg_new)

JSD_cor1_forward_central_new, KLD_cor1_forward_central_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_forward_cor1_central_deg_new)
JSD_cor1_central_random_new, KLD_cor1_central_random_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_random_deg_new)
JSD_COR1_Forward_Central_Random_new, KLDcor1_forward_central_random_new = calculate_KDE_statistics(KDE_forward_cor1_central_deg_new, KDE_random_deg_new)

combined_dict = dict(metric=['KL Divergence', 'JS Divergence'],
                    cor1_v_psi=[KLD_cor1_forward_central_new, JSD_cor1_forward_central_new],
                    cor1_v_random=[KLD_cor1_central_random_new, JSD_cor1_central_random_new],
                    psi_v_random=[KLDcor1_forward_central_random_new, JSD_COR1_Forward_Central_Random_new])

pd.set_option('display.float_format', '{:.3E}'.format)
stats_df = pd.DataFrame(combined_dict)
stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
print(stats_df.to_latex(index=False))

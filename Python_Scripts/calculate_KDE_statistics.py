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
matplotlib.use('TkAgg')
mpl.use('TkAgg')


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
    
    
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
matplotlib.use('TkAgg')
mpl.use('TkAgg')


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
data_dir = os.path.join(repo_path,'Data/QRaFT/errors.sav')

idl_save_qraft = readsav(data_dir)
err_mlso_central = idl_save_qraft['ERR_ARR_MLSO']
err_mlso_los = idl_save_qraft['ERR_ARR_LOS_MLSO']
err_forward_central = idl_save_qraft['ERR_ARR_FORWARD']
err_forward_los = idl_save_qraft['ERR_ARR_LOS_FORWARD']
err_random = idl_save_qraft['ERR_ARR_RND']

idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
date_obs =idl_save['DATE_OBS']
# crln_obs_print = idl_save['crln_obs_print']
# crlt_obs_print = idl_save['crlt_obs_print']
# date_print = str(idl_save['date_print'],'utf-8')
# fits_directory = str(idl_save['fits_directory'][0],'utf-8')
# occlt = idl_save['occlt']
# shape = idl_save['shape']
# detector = idl_save['detector']
outstring_list = idl_save['outstring_list']







params = date_print + str(detector,'utf-8') + '_PSI'

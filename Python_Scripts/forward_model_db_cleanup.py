# Copyright 2025 Christopher Rura

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import matplotlib
from tqdm import tqdm_notebook
import pandas as pd
import matplotlib as mpl
import git
from scipy.io import readsav
import unittest
from pathlib import Path
import sqlite3
from functions import get_files_from_pattern, determine_paths

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
crln_obs = idl_save['crln_obs']
crlt_obs = idl_save['crlt_obs']
crln_obs_print = idl_save['crln_obs_print']
crlt_obs_print = idl_save['crlt_obs_print']
occlt = idl_save['occlt']
r_sun_range = idl_save['range']
fits_directory = idl_save['fits_directory']

fits_path = str(fits_directory[0], 'utf-8')


idl_save_outstrings = readsav(os.path.join(repo_path,'Data/outstrings.sav'))
outstring_list = idl_save_outstrings['outstring_list']
directory_list_2 = idl_save_outstrings['directory_list']
directory_list_1 = idl_save_outstrings['directory_list_2']
occlt_list = idl_save_outstrings['occlt_list']
directory_list_combined = []


for i in range(len(directory_list_1)):
    directory_list_1[i] = os.path.join(repo_path, str(directory_list_1[i], 'utf-8'))
    directory_list_combined.append(directory_list_1[i])

for i in range(len(directory_list_2)):
    directory_list_2[i] = os.path.join(repo_path, str(directory_list_2[i], 'utf-8'))
    directory_list_combined.append(directory_list_2[i])

for path in directory_list_combined:
    if os.path.exists(path):
        hdul = fits.open(path)
        header = hdul[0].header
        try:
            header.remove('forward_input_data_id')
            # Save the changes to the FITS file
            hdul.writeto(path, overwrite=True)
            # Close the FITS file
            hdul.close()
        except KeyError:
            pass
    else:
        pass
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

con = sqlite3.connect("tutorial.db")

cur = con.cursor()

# cur.execute("CREATE TABLE stats(comparison, data_source, date, JSD, KLD)")

# cur.execute("CREATE TABLE stats2_new(data_type, data_source, date, mean, median, confidence interval, n)")

# cur.execute("DROP TABLE IF EXISTS forward_input_variables")

cur.execute("""CREATE TABLE IF NOT EXISTS forward_input_variables (
            forward_parameters_id INTEGER PRIMARY KEY,
            crln_obs,
            crlt_obs,
            occlt,
            r_sun_range,
            unique(crln_obs, crlt_obs, occlt, r_sun_range)
            )
            """
            )
forward_input_data = [(None, float(crln_obs), float(crlt_obs), float(occlt), float(r_sun_range))]

cur.executemany("""INSERT OR IGNORE INTO forward_input_variables VALUES(?, ?, ?, ?, ?)""", forward_input_data)
con.commit()

query = """SELECT * from forward_input_variables where 
            crln_obs={} and 
            crlt_obs={} and 
            occlt={} and 
            r_sun_range={};""".format(float(crln_obs), float(crlt_obs), float(occlt), float(r_sun_range))

cur.execute(query)
row = cur.fetchone()
forward_input_data_id, crln_obs_db, crlt_obs_db, occlt_db, r_sun_range_db = row


hdul = fits.open(fits_path)
header = hdul[0].header
# create a new keyword to header and set the value
header.set('forward_input_data_id', forward_input_data_id, 'unique identifier in forward_input_variables table')

try:
    date_obs = header['date-obs']
except:
    date_obs = header['date_obs']
    header.set('date-obs', date_obs, 'date of observation')

# Save the changes to the FITS file
hdul.writeto(fits_path, overwrite=True)
# Close the FITS file
hdul.close()
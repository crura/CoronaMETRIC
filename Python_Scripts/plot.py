#!'/Users/crura/.conda/envs/test_env/lib/python3.9'
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
import json
import sqlite3
import sunpy
import matplotlib.pyplot as plt
import sunpy.version
import sunpy.map
import matplotlib.colors
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import pandas as pd
import git
from scipy.io import readsav
import os
from matplotlib.patches import Circle
from functions import print_sql_query


import subprocess
subprocess.run(["mkdir","Output/Plots","-p"])

path = __file__
pathnew = Path(path)

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
date_obs =idl_save['DATE_OBS']
crln_obs_print = idl_save['crln_obs_print']
crlt_obs_print = idl_save['crlt_obs_print']
date_print = str(idl_save['date_print'],'utf-8')
fits_directory = str(idl_save['fits_directory'][0],'utf-8')
occlt = idl_save['occlt']
shape = idl_save['shape']
detector = idl_save['detector']
params = '__' + date_print + '__' + str(detector,'utf-8') + '__PSI'
#  File "/Users/crura/Desktop/Research/github/actions-runner/_work/Image-Coalignment/Image-Coalignment/Python_Scripts/plot.py", line 34, in <module>
#     params = date_print + detector + '_PSI'
# TypeError: can only concatenate str (not "bytes") to str

config_file = os.path.join(repo_path, 'config.json')
with open(config_file) as f:
    config = json.load(f)


fits_path = os.path.join(repo_path, 'Output/QRaFT_Results')
fits_input_path = os.path.join(repo_path, config['cor1_data_path'])



# fits_dir_cor1 = fits_input_path

for i in os.listdir(fits_input_path):
    fits_dir_cor1 = os.path.join(fits_input_path, i)
    # break

    data2 = fits.getdata(fits_dir_cor1)
    head2 = fits.getheader(fits_dir_cor1)
    # head2['detector'] = ('KCor')
    cor1map = sunpy.map.Map(data2, head2)

    dbName = "tutorial.db"
    obsDate = cor1map.date
    day_date = obsDate.strftime('%Y-%m-%d')
    query = "SELECT * from central_tendency_stats_cor1_new inner join forward_input_variables on central_tendency_stats_cor1_new.forward_input_data_id = forward_input_variables.forward_parameters_id;"
    query = "SELECT date, data_source, data_type from central_tendency_stats_cor1_new inner join forward_input_variables on central_tendency_stats_cor1_new.forward_input_data_id = forward_input_variables.forward_parameters_id where crlt_obs={} and data_type='{}';".format(round(cor1map.carrington_latitude.value,11), 'pB')
    query2 = "SELECT date from central_tendency_stats_cor1_new inner join forward_input_parameters on central_tendency_stats_cor1_new.forward_input_data_id = forward_input_parameters.id where date={}".format(day_date)
    con = sqlite3.connect(dbName)
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    date, data_source, data_type = rows[0]
    date_print = date.replace('-','_')
    params = '__' + date_print + '__' + data_source + '_' + data_type
    print(rows[0])




    fits_dir_psi = os.path.join(repo_path,'Output/fits_images/{}.fits'.format(params))
    data1 = fits.getdata(fits_dir_psi)
    head1 = fits.getheader(fits_dir_psi)
    head1['detector'] = ('Cor-1')
    psimap = sunpy.map.Map(data1, head1)


    str_strip = str(date_obs,'utf-8').split('T',1)[0]

    fig1 = plt.figure(figsize=(15, 8))
    ax1 = fig1.add_subplot(1, 2, 1, projection=cor1map)
    ax2 = fig1.add_subplot(1, 2, 2, projection=cor1map)
    cor1map.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']
    cor1map.plot(axes=ax2,title=False)

    R_SUN = occlt * (head2['rsun'] / head2['cdelt1'])
    ax2.add_patch(Circle((int(shape/2),int(shape/2)), R_SUN, color='black',zorder=100))



    psimap.plot_settings['norm'] = plt.Normalize(cor1map.min(), cor1map.max())

    psimap.plot(axes=ax1,title=False,norm=matplotlib.colors.LogNorm())

    query = "SELECT occlt from central_tendency_stats_cor1_new inner join forward_input_variables on central_tendency_stats_cor1_new.forward_input_data_id = forward_input_variables.forward_parameters_id where crlt_obs={} and data_type='{}';".format(round(cor1map.carrington_latitude.value,11), 'pB')
    cur.execute(query)
    rows = cur.fetchall()
    occlt = rows[0][0]

    R_SUN = occlt * (head1['rsun'] / head1['cdelt1'])
    ax1.add_patch(Circle((int(shape/2),int(shape/2)), R_SUN, color='black',zorder=100))
    ax2.add_patch(Circle((int(shape/2),int(shape/2)), R_SUN, color='black',zorder=100))
    ax2.set_xlabel('Helioprojective Longitude (Solar-X)',fontsize=18)
    ax1.set_xlabel('Helioprojective Longitude (Solar-X)',fontsize=18)
    ax2.set_ylabel('Helioprojective Latitude (Solar-Y)',fontsize=18)
    ax1.set_ylabel('Helioprojective Latitude (Solar-Y)',fontsize=18)
    ax2.set_title('COR-1 Observation {}'.format(date), fontsize=18)
    ax1.set_title('PSI MAS / FORWARD {} Model'.format(data_type), fontsize=18)

    string_print = str(date_obs,'utf-8').split('T')[0].replace('-','_') + 'cor1'

    plt.savefig(os.path.join(repo_path,'Output/Plots/Model_Comparison_{}_{}_{}.eps'.format(date, data_type, data_source.split('__')[0])), format='eps')
    # plt.savefig(os.path.join(repo_path,'Output/Plots/Model_Comparison_{}_{}.eps'.format(string_print, detector)), format='eps')
    # #plt.show()
    plt.close()

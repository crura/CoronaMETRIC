#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:44:57 2023

@author: crura
"""

# import subprocess
# subprocess.run(['source', ' ', 'env/bin/activate'])
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
from datetime import datetime, timedelta
from sunpy.sun import constants
import astropy.constants
import astropy.units as u
matplotlib.use('TkAgg')
mpl.use('TkAgg')
mpl.rcParams.update(mpl.rcParamsDefault)
from functions import create_six_fig_plot
from test_plot_qraft import plot_features
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl




def get_detector(fitsfilepath):
    fits_file = fitsfilepath
    data = fits.getdata(fits_file)
    head = fits.getheader(fits_file)
    telescope = head['telescop']
    instrument = head['instrume']
    if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
      head['detector'] = ('KCor')

    return head['detector']



repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir


idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
idl_save_outstring = readsav(os.path.join(repo_path,'Data/outstrings.sav'))
date_obs =idl_save['DATE_OBS']
# crln_obs_print = idl_save['crln_obs_print']
# crlt_obs_print = idl_save['crlt_obs_print']
# date_print = str(idl_save['date_print'],'utf-8')
# fits_directory = str(idl_save['fits_directory'][0],'utf-8')
# occlt = idl_save['occlt']
# shape = idl_save['shape']
# detector = idl_save['detector']
outstring_list = idl_save_outstring['outstring_list']
directory_list_2 = idl_save_outstring['directory_list']
directory_list_1 = idl_save_outstring['directory_list_2']
occlt_list = idl_save_outstring['occlt_list']

for i in range(len(directory_list_1)):
    directory_list_1[i] = os.path.join(repo_path, str(directory_list_1[i], 'utf-8'))

for i in range(len(directory_list_2)):
    directory_list_2[i] = os.path.join(repo_path, str(directory_list_2[i], 'utf-8'))


# # remove blank first element of list
# for i in range(len(outstring_list)):
#     if outstring_list[i] == '':
#         outstring_list_new = np.delete(outstring_list, i)
#     else:
#         pass

# Use list comprehension to filter files ending with "_pB.fits"
directory_path = os.path.join(repo_path, 'Output/fits_images')
directory_path_qraft = os.path.join(repo_path, 'Output/QRaFT_Results')
outstring_list_new = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path) if filename.endswith("_pB.fits")]
outstring_list_qraft = [os.path.join(directory_path_qraft, filename+'.sav') for filename in os.listdir(directory_path) if filename.endswith("_pB.fits")]
# # translate all utf-8 strings into normal strings
# for i in range(len(outstring_list_new)):
#     outstring_list_new[i] = str(outstring_list_new[i],'utf-8')
directory_list_1_qraft = []
directory_list_2_qraft = []
for i in range(len(directory_list_1)):
    directory_list_1_qraft.append(os.path.join(directory_path_qraft, directory_list_1[i].split('/')[-1]+'.sav'))

for i in range(len(directory_list_2)):
    directory_list_2_qraft.append(os.path.join(directory_path_qraft, directory_list_2[i].split('/')[-1]+'.sav'))

# remove blank first element of list
for i in range(len(occlt_list)):
    if occlt_list[i] == 0:
        occlt_list_new = np.delete(occlt_list, i)
    else:
        pass


# filter filenames into separate lists based on detector
keyword = outstring_list_new[0].split('__')[2]

indexes1, outstring_list_1 = zip(*[(index, item) for index, item in enumerate(outstring_list_new) if keyword in item])
indexes2, outstring_list_2 = zip(*[(index, item) for index, item in enumerate(outstring_list_new) if keyword not in item])

indexes1_qraft, outstring_list_1_qraft = zip(*[(index, item) for index, item in enumerate(outstring_list_qraft) if keyword in item])
indexes2_qraft, outstring_list_2_qraft = zip(*[(index, item) for index, item in enumerate(outstring_list_qraft) if keyword not in item])

# occlt_list_1 = [occlt_list_new[index] for index in indexes1]
# occlt_list_2 = [occlt_list_new[index] for index in indexes2]

print(directory_list_1, outstring_list_1, directory_list_2, outstring_list_2)
# print(occlt_list_1, occlt_list_2)


# match up all file paths by alphabetical order to match up time of observations
outstring_list_1 = sorted(outstring_list_1)
outstring_list_1_qraft = sorted(outstring_list_1_qraft)
outstring_list_2 = sorted(outstring_list_2)
outstring_list_2_qraft = sorted(outstring_list_2_qraft)
directory_list_1 = sorted(directory_list_1)
directory_list_1_qraft = sorted(directory_list_1_qraft)
directory_list_2 = sorted(directory_list_2)
directory_list_2_qraft = sorted(directory_list_2_qraft)

# os.path.join(repo_path,'Data/QRaFT/errors.sav')




idl_save_path = '/Users/crura/Desktop/Research/github/Test_Suite/Image-Coalignment/Output/QRaFT_Results/20170820_180657_kcor_l2_avg.fts.sav'
fits_path = '/Users/crura/Desktop/Research/github/Test_Suite/Image-Coalignment/Output/fits_images/20170820_180657_kcor_l2_avg.fts'
idl_save = readsav(idl_save_path)
IMG = idl_save['img_orig']
features = idl_save['features']
P = idl_save['P']

N = len(features)
N_nodes_max = len(features[0]['angles_xx_r'])
angle_err = np.zeros((N, N_nodes_max), dtype=float)
angle_err_signed = np.zeros((N, N_nodes_max), dtype=float)
angle_err_signed_test = np.zeros((N, N_nodes_max), dtype=float)

PSI = False
fits_file = fits_path
data = fits.getdata(fits_file)
head = fits.getheader(fits_file)
detector = get_detector(fits_path)
filename = fits_file.split('/')[-1]

if PSI:
    if detector == 'KCor':
        if 'KCor' in filename:
            keyword_By = 'KCor__PSI_By.fits'
            keyword_Bz = 'KCor__PSI_Bz.fits'
            file1_y = os.path.join(repo_path, 'Output/fits_images/' + filename.split('KCor')[0] + keyword_By)
            file1_z = os.path.join(repo_path, 'Output/fits_images/' + filename.split('KCor')[0] + keyword_Bz)

    elif detector == 'COR1':
        if 'COR1' in filename:
            keyword_By = 'COR1__PSI_By.fits'
            keyword_Bz = 'COR1__PSI_Bz.fits'
            file1_y = os.path.join(repo_path, 'Output/fits_images/' + filename.split('COR1')[0] + keyword_By)
            file1_z = os.path.join(repo_path, 'Output/fits_images/' + filename.split('COR1')[0] + keyword_Bz)
else:
    fits_dir = os.path.join(repo_path, 'Output/fits_images')
    if detector == 'KCor':
      for i in outstring_list_1:
          if head['date-obs'].replace('-','_').split('T')[0] in i:
              file1_y = i.replace('pB','By')
              file1_z = i.replace('pB', 'Bz')

    elif detector == 'COR1':
            if head['date-obs'].replace('-','_').split('T')[0] in filename:
                print('input file: {}'.format(fits_file))
                # print('qraft file: {}'.format(qraft_file))
                # print('corresponding B field file: {}'.format(i))
                file1_y = filename.replace('pB','By')
                file1_z = filename.replace('pB', 'Bz')
                print('By file: {}'.format(file1_y))
                print('Bz file: {}'.format(file1_z))


fits_dir_bz_los_coaligned = file1_z
data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
Bz = data_bz_los_coaligned
head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
head_bz_los_coaligned['detector'] = (detector)
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

wcs = WCS(head_bz_los_coaligned)

fits_dir_by_los_coaligned = file1_y
data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
head_by_los_coaligned['Observatory'] = ('PSI-MAS')
head_by_los_coaligned['detector'] = (detector)
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)
By = data_by_los_coaligned


# v1 = [features[i].xx_r[k+1]-features[i].xx_r[k], features[i].yy_r[k+1]-features[i].yy_r[k] ]
# v2 = [B1[xx[k], yy[k]], B2[xx[k], yy[k]]  ]
# v1_mag = sqrt(total(v1^2))
# v2_mag = sqrt(total(v2^2))

# d_angle = acos(total(v1*v2)/(v1_mag*v2_mag))
angles = []
angles_signed = []
angles_signed_test = []
angles_xx_positions = []
angles_yy_positions = []
for i in range(N):
    # this needs to be [:features[i]['n_nodes'] -1]
    xx = features[i]['angles_xx_r'][:features[i]['n_nodes']]
    yy = features[i]['angles_yy_r'][:features[i]['n_nodes']]
    for k in range(features[i]['n_nodes'] - 1):
        v1 = [features[i]['xx_r'][k+1] - features[i]['xx_r'][k], features[i]['yy_r'][k+1] - features[i]['yy_r'][k]]
        # Because IDL indexes backwards we index by y then x
        v2 = [By[int(yy[k]), int(xx[k])], Bz[int(yy[k]), int(xx[k])]]

        v1_mag = np.sqrt(np.sum(np.array(v1) ** 2))
        v2_mag = np.sqrt(np.sum(np.array(v2) ** 2))

        d_angle = np.arccos(np.sum(np.array(v1)*np.array(v2)) / (v1_mag * v2_mag) )
        if d_angle > math.pi/2:
            d_angle = math.pi - d_angle
        angle_err[i, k] = d_angle
        angles.append(d_angle)
        angles_xx_positions.append(int(xx[k]))
        angles_yy_positions.append(int(yy[k]))

        d_angle_signed = np.arcsin((v1[0] * v2[1] - v1[1] * v2[0]) / (v1_mag * v2_mag))
        d_angle_signed_test = np.arctan2(v1[0] * v2[1] - v1[1] * v2[0], v1[0] * v2[0] + v1[1] * v2[1])
        angles_signed.append(d_angle_signed)
        angles_signed_test.append(d_angle_signed_test)


fig = plt.figure(figsize=(10, 10))
map = sunpy.map.Map(data, head)

telescope = head['telescop']
instrument = head['instrume']
    
if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
  head['detector'] = ('KCor')

if head['detector'] == 'COR1':
    map.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']
    # rsun = (head['rsun'] / head['cdelt1']) * occlt_list # number of pixels in radius of sun
# else:
    # rsun = (head['rsun'] / head['cdelt1']) * occlt_list # number of pixels in radius of sun
axes = fig.add_subplot(1,1,1, projection=map)
if head['detector'] == 'PSI-MAS Forward Model' or head['telescop'] == 'PSI-MAS Forward Model':
    map.plot(axes=axes,title=False,norm=matplotlib.colors.LogNorm())
elif head['detector'] == 'COR1':
    map.plot(axes=axes,title=False,clip_interval=(1, 99.99)*u.percent)
else:
    map.plot(axes=axes,title=False)
# axes.add_patch(Circle((int(data.shape[0]/2),int(data.shape[1]/2)), rsun, color='black',zorder=100))

idl_save = readsav(idl_save_path)
IMG = idl_save['img_orig']
FEATURES = idl_save['features']
P = idl_save['P']
colors = plt.cm.jet(np.linspace(0, 1, len(FEATURES)))
# for i, feature in enumerate(FEATURES):
    # axes.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors[i], linewidth=3)
# Scatter plot for angle errors
sc = axes.scatter(angles_xx_positions, angles_yy_positions, c=np.degrees(angles), cmap='viridis', label=False)
divider = make_axes_locatable(axes)
cax = divider.append_axes('right', size='5%', pad=0.6)
# Add colorbar manually
# cb = mpl.colorbar.ColorbarBase(cax,orientation='vertical')

cax.yaxis.set_ticks_position('right')
cax.yaxis.set_label_position('right')
cbar = fig.colorbar(sc, cax=cax, label='Angle Error (degrees)', orientation='vertical')
# cax.set_xlabel(' ')
# cax.grid(axis='y')
lat = cax.coords[0]
# lat.set_ticks([20,20]*u.arcsec)
lat.set_ticks_visible(False)
lat.set_ticklabel_visible(False)
lat.set_axislabel('')
#plt.show()
plt.close()












fig = plt.figure(figsize=(10, 10))
map = sunpy.map.Map(data, head)

telescope = head['telescop']
instrument = head['instrume']

if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
  head['detector'] = ('KCor')

if head['detector'] == 'COR1':
    map.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']
    # rsun = (head['rsun'] / head['cdelt1']) * occlt_list # number of pixels in radius of sun
# else:
    # rsun = (head['rsun'] / head['cdelt1']) * occlt_list # number of pixels in radius of sun
axes = fig.add_subplot(1,1,1, projection=map)
if head['detector'] == 'PSI-MAS Forward Model' or head['telescop'] == 'PSI-MAS Forward Model':
    map.plot(axes=axes,title=False,norm=matplotlib.colors.LogNorm())
elif head['detector'] == 'COR1':
    map.plot(axes=axes,title=False,clip_interval=(1, 99.99)*u.percent)
else:
    map.plot(axes=axes,title=False)
# axes.add_patch(Circle((int(data.shape[0]/2),int(data.shape[1]/2)), rsun, color='black',zorder=100))

idl_save = readsav(idl_save_path)
IMG = idl_save['img_orig']
FEATURES = idl_save['features']
P = idl_save['P']
colors = plt.cm.jet(np.linspace(0, 1, len(FEATURES)))
# for i, feature in enumerate(FEATURES):
    # axes.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors[i], linewidth=3)
# Scatter plot for angle errors
sc = axes.scatter(angles_xx_positions, angles_yy_positions, c=np.degrees(angles_signed), cmap='Spectral', label=False)
divider = make_axes_locatable(axes)
cax = divider.append_axes('right', size='5%', pad=0.6)
# Add colorbar manually
# cb = mpl.colorbar.ColorbarBase(cax,orientation='vertical')

cax.yaxis.set_ticks_position('right')
cax.yaxis.set_label_position('right')
cbar = fig.colorbar(sc, cax=cax, label='Angle Error (degrees)', orientation='vertical')
# cax.set_xlabel(' ')
# cax.grid(axis='y')
lat = cax.coords[0]
# lat.set_ticks([20,20]*u.arcsec)
lat.set_ticks_visible(False)
lat.set_ticklabel_visible(False)
lat.set_axislabel('')
#plt.show()
plt.close()







# fig = plt.figure(figsize=(10, 10))
# map = sunpy.map.Map(data, head)

# map.plot(axes=axes,title=False,norm=matplotlib.colors.LogNorm())


# axes.scatter(angles_xx_positions, angles_yy_positions, c=np.degrees(angles), cmap='viridis')
# divider = make_axes_locatable(axes)
# cax = divider.append_axes('right', size='5%', pad=0.05)
# fig.colorbar(np.degrees(angles), cax, label='Angle Error (degrees)')
# #plt.show()
# plt.close()

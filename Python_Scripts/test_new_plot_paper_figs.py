#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 11:16:38 2023

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


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
"""
data_dir = os.path.join(repo_path,'Data/QRaFT/errors.sav')

idl_save_qraft = readsav(data_dir)
err_mlso_central = idl_save_qraft['ERR_ARR_MLSO']
err_mlso_los = idl_save_qraft['ERR_ARR_LOS_MLSO']
err_forward_central = idl_save_qraft['ERR_ARR_FORWARD']
err_forward_los = idl_save_qraft['ERR_ARR_LOS_FORWARD']
err_random = idl_save_qraft['ERR_ARR_RND']
"""

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


def display_fits_images(fits_files, qraft_files, outpath):
    # fig, axes = plt.subplots(nrows=int(n/2), ncols=2, figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))


    for i in range(len(fits_files)):

        data = fits.getdata(fits_files[i])
        head = fits.getheader(fits_files[i])

        idl_save_path = qraft_files[i]

        map = sunpy.map.Map(data, head)

        telescope = head['telescop']
        instrument = head['instrume']
        print(telescope)
        # print(head)
        if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
          head['detector'] = ('KCor')
          norm = map.plot_settings['norm']
          norm.vmin, norm.vmax = np.percentile(map.data, [1, 99.9])

        if head['detector'] == 'COR1':
            map.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']
            # rsun = (head['rsun'] / head['cdelt1']) * occlt_list[i] # number of pixels in radius of sun
        # else:
            # rsun = (head['rsun'] / head['cdelt1']) * occlt_list[i] # number of pixels in radius of sun
        axes = fig.add_subplot(int(len(fits_files)/2), 2, i+1, projection=map)
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
        for i, feature in enumerate(FEATURES):
            axes.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors[i], linewidth=3)

        # plot_features(idl_save_path, map=axes)
        # axes[i].imshow(data, cmap='gray')
        # axes[i].set_title(fits_file)

    plt.subplots_adjust(bottom=0.05, top=0.95)
    # plt.savefig(outpath)
    plt.show()
    plt.close()



def display_fits_image_with_features_and_B_field(fits_file, qraft_file, PSI=True):
    # fig, axes = plt.subplots(nrows=int(n/2), ncols=2, figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))


    filename = fits_file.split('/')[-1]

    data = fits.getdata(fits_file)
    head = fits.getheader(fits_file)

    idl_save_path = qraft_file

    map = sunpy.map.Map(data, head)

    telescope = head['telescop']
    instrument = head['instrume']
    print(telescope)
    # print(head)
    if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
      head['detector'] = ('KCor')
      norm = map.plot_settings['norm']
      norm.vmin, norm.vmax = np.percentile(map.data, [1, 99.9])

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
    for i, feature in enumerate(FEATURES):
        axes.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors[i], linewidth=3)

    detector = head['detector']
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
        if detector == 'KCor':
            for i in outstring_list_1:
                if head['date-obs'].replace('-','_').split('T')[0] in i:
                    file1_y = i.replace('pB','By')
                    file1_z = i.replace('pB', 'Bz')

        elif detector == 'COR1':
            for i in outstring_list_2:
                if head['date-obs'].replace('-','_').split('T')[0] in i:
                    print('input file: {}'.format(fits_file))
                    print('qraft file: {}'.format(qraft_file))
                    # print('corresponding B field file: {}'.format(i))
                    file1_y = i.replace('pB','By')
                    file1_z = i.replace('pB', 'Bz')
                    print('By file: {}'.format(file1_y))
                    print('Bz file: {}'.format(file1_z))






    fits_dir_bz_los_coaligned = file1_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)
    Bz = data_bz_los_coaligned

    wcs = WCS(head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file1_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)
    By = data_by_los_coaligned


    ny, nz = data_bz_los_coaligned.shape[0],data_bz_los_coaligned.shape[1]
    dy = np.linspace(0, int(ny), ny)
    dz = np.linspace(0, int(nz), nz)
    X, Y = np.meshgrid(np.linspace(0, 2 * np.pi, ny), np.linspace(0, 2 * np.pi, nz))
    # R_SUN = rsun
    # rsun = (head['rsun'] / head['cdelt1']) * occlt_list[i]
    widths = np.linspace(0,1024,by_los_coaligned_map.data.size)
    skip_val = int(by_los_coaligned_map.data.shape[0]/233.14285714285714) #73.14285714285714
    skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    skip1 = slice(None, None, skip_val)
    by = by_los_coaligned_map.data
    bz = bz_los_coaligned_map.data
    by_normalized = (by / np.sqrt(by**2 + bz**2))
    bz_normalized = (bz / np.sqrt(by**2 + bz**2))
    r = np.power(np.add(np.power(by,2), np.power(bz,2)),0.5) * 50000
    axes.quiver(dy[skip1],dz[skip1],by[skip],bz[skip],units='width',color='r')
    # qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
    #                coordinates='figure')



    # plot_features(idl_save_path, map=axes)
    # axes[i].imshow(data, cmap='gray')
    # axes[i].set_title(fits_file)

    plt.subplots_adjust(bottom=0.05, top=0.95)
    # plt.savefig(outpath)
    # plt.show()
    plt.close()




    features = FEATURES
    N = len(features)
    N_nodes_max = len(features[0]['angles_xx_r'])
    angle_err = np.zeros((N, N_nodes_max), dtype=float)
    angle_err_signed = np.zeros((N, N_nodes_max), dtype=float)
    angle_err_signed_test = np.zeros((N, N_nodes_max), dtype=float)
    angles = []
    angles_signed = []
    angles_signed_test = []
    angles_signed_test_2 = []
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
            d_angle_signed_test = np.arctan2((v1[0] * v2[1] - v1[1] * v2[0]),  (v1[0] * v2[0] + v1[1] * v2[1]))
            d_angle_signed_test_2 = np.arctan((v1[0] * v2[1] - v1[1] * v2[0]) / (v1[0] * v2[0] + v1[1] * v2[1]))
            angles_signed.append(d_angle_signed)
            angles_signed_test.append(d_angle_signed_test)
            angles_signed_test_2.append(d_angle_signed_test_2)


    fig = plt.figure(figsize=(10, 10))
    map = sunpy.map.Map(data, head)

    telescope = head['telescop']
    instrument = head['instrume']

    date_obs = head['date-obs']
    str_strip = date_obs.split('T',1)[0]
    string_print = date_obs.split('T')[0].replace('-','_')

    if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
      head['detector'] = ('KCor')
      norm = map.plot_settings['norm']
      norm.vmin, norm.vmax = np.percentile(map.data, [1, 99.9])

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
    norm = mpl.colors.Normalize(vmin=0, vmax=90)
    sc = axes.scatter(angles_xx_positions, angles_yy_positions, c=np.degrees(angles), cmap='viridis', label=False, norm=norm)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.6)
    # Add colorbar manually
    # cb = mpl.colorbar.ColorbarBase(cax,orientation='vertical')

    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    norm = mpl.colors.Normalize(vmin=0, vmax=90)
    cbar = fig.colorbar(sc, cax=cax, label='Angle Error (degrees)', orientation='vertical', norm=norm)
    # cax.set_xlabel(' ')
    # cax.grid(axis='y')
    lat = cax.coords[0]
    # lat.set_ticks([20,20]*u.arcsec)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)
    lat.set_axislabel('')
    if PSI:
        axes.set_title('Corresponding PSI/FORWARD pB Eclipse Model')
        plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Angle_Error_{}_{}_PSI.png'.format(string_print, detector)))
    else:
        axes.set_title('{} Observation {}'.format(detector, str_strip))
        plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Angle_Error_{}_{}.png'.format(string_print, detector)))
    # plt.show()
    plt.close()












    fig = plt.figure(figsize=(10, 10))
    map = sunpy.map.Map(data, head)

    telescope = head['telescop']
    instrument = head['instrume']
    date_obs = head['date-obs']
    str_strip = date_obs.split('T',1)[0]

    if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
      head['detector'] = ('KCor')
      norm = map.plot_settings['norm']
      norm.vmin, norm.vmax = np.percentile(map.data, [1, 99.9])

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
    norm = mpl.colors.Normalize(vmin=-90, vmax=90)
    sc = axes.scatter(angles_xx_positions, angles_yy_positions, c=np.degrees(angles_signed_test_2), cmap='coolwarm', label=False, norm=norm)
    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.6)
    # Add colorbar manually
    # cb = mpl.colorbar.ColorbarBase(cax,orientation='vertical')

    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')
    norm = mpl.colors.Normalize(vmin=-90, vmax=90)
    cbar = fig.colorbar(sc, cax=cax, label='Angle Error (degrees)', orientation='vertical', norm=norm)
    # cax.set_xlabel(' ')
    # cax.grid(axis='y')
    lat = cax.coords[0]
    # lat.set_ticks([20,20]*u.arcsec)
    lat.set_ticks_visible(False)
    lat.set_ticklabel_visible(False)
    lat.set_axislabel('')
    if PSI:
        axes.set_title('Corresponding PSI/FORWARD pB Eclipse Model')
        plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Angle_Error_Signed_{}_{}_PSI.png'.format(string_print, detector)))
    else:
        axes.set_title('{} Observation {}'.format(detector, str_strip))
        plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Angle_Error_Signed_{}_{}.png'.format(string_print, detector)))
    # plt.show()
    plt.close()



def plot_model_data_comparison_with_features(data_fits_file, data_qraft_file, model_fits_file, model_qraft_file, outpath, PSI=True):

    filename = data_fits_file.split('/')[-1]

    data = fits.getdata(data_fits_file)
    head = fits.getheader(data_fits_file)

    date_obs = head['date-obs']

    telescope = head['telescop']
    instrument = head['instrume']
    if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
      head['detector'] = ('KCor')

    detector = head['detector']

    str_strip = date_obs.split('T',1)[0]

    idl_save_path = data_qraft_file

    idl_save = readsav(idl_save_path)
    IMG = idl_save['img_orig']
    FEATURES = idl_save['features']
    P = idl_save['P']
    colors = plt.cm.jet(np.linspace(0, 1, len(FEATURES)))

    datamap = sunpy.map.Map(data, head)

    if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
      norm = datamap.plot_settings['norm']
      norm.vmin, norm.vmax = np.percentile(datamap.data, [1, 99.9])

    fits_dir_psi = model_fits_file
    data1 = fits.getdata(fits_dir_psi)
    head1 = fits.getheader(fits_dir_psi)

    idl_save_path_model = model_qraft_file

    idl_save_model = readsav(idl_save_path_model)
    IMG_model = idl_save_model['img_orig']
    FEATURES_model = idl_save_model['features']
    P_model = idl_save_model['P']
    colors_model = plt.cm.jet(np.linspace(0, 1, len(FEATURES_model)))

    # head1['detector'] = ('Cor-1')
    psimap = sunpy.map.Map(data1, head1)
    psimap.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']

    fig1 = plt.figure(figsize=(15, 8))
    ax1 = fig1.add_subplot(1, 2, 1, projection=datamap)
    datamap.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']
    datamap.plot(axes=ax1,title=False)
    for i, feature in enumerate(FEATURES):
        ax1.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors[i], linewidth=3)

    # R_SUN = occlt * (head2['rsun'] / head2['cdelt1'])
    # ax1.add_patch(Circle((int(shape/2),int(shape/2)), R_SUN, color='black',zorder=100))



    ax2 = fig1.add_subplot(1, 2, 2, projection=datamap)
    psimap.plot_settings['norm'] = plt.Normalize(datamap.min(), datamap.max())

    psimap.plot(axes=ax2,title=False,norm=matplotlib.colors.LogNorm())
    for i, feature in enumerate(FEATURES_model):
        ax2.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors_model[i], linewidth=3)
    # R_SUN = occlt * (head1['rsun'] / head1['cdelt1'])
    # ax2.add_patch(Circle((int(shape/2),int(shape/2)), R_SUN, color='black',zorder=100))
    # ax1.add_patch(Circle((int(shape/2),int(shape/2)), R_SUN, color='black',zorder=100))
    ax1.set_xlabel('Helioprojective Longitude (Solar-X)',fontsize=18)
    ax2.set_xlabel('Helioprojective Longitude (Solar-X)',fontsize=18)
    ax1.set_ylabel('Helioprojective Latitude (Solar-Y)',fontsize=18)
    ax2.set_ylabel('Helioprojective Latitude (Solar-Y)',fontsize=18)
    ax1.set_title('{} Observation {}'.format(detector, str_strip), fontsize=18)
    ax2.set_title('Corresponding PSI/FORWARD pB Eclipse Model', fontsize=18)

    string_print = date_obs.split('T')[0].replace('-','_')

    plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Model_Comparison_{}_{}.png'.format(string_print, detector)))
    # plt.show()
    plt.close()




# display_fits_image_with_features_and_B_field(outstring_list_1[0], outstring_list_1_qraft[0], PSI=True)
# display_fits_image_with_features_and_B_field(directory_list_1[0], directory_list_1_qraft[0], PSI=False)

# plot_model_data_comparison_with_features(directory_list_1[0], directory_list_1_qraft[0], outstring_list_1[0], outstring_list_1_qraft[0], os.path.join(repo_path,'Output/Plots/COR1_PSI_Plot_test.png'))
# display_fits_image_with_features_and_B_field(outstring_list_2[0], outstring_list_2_qraft[0], PSI=True)
# display_fits_image_with_features_and_B_field(directory_list_2[0], directory_list_2_qraft[0], PSI=False)

# display_fits_images(outstring_list_1, outstring_list_1_qraft,os.path.join(repo_path,'Output/Plots/COR1_PSI_Plots.png'))
# display_fits_images(directory_list_1, directory_list_1_qraft ,os.path.join(repo_path,'Output/Plots/COR1_Plots.png'))
# display_fits_images(outstring_list_2, outstring_list_2_qraft ,os.path.join(repo_path,'Output/Plots/MLSO_PSI_Plots.png'))
# display_fits_images(directory_list_2, directory_list_2_qraft ,os.path.join(repo_path,'Output/Plots/MLSO_Plots.png'))

for i in range(len(directory_list_1)):
    plot_model_data_comparison_with_features(directory_list_1[i], directory_list_1_qraft[i], outstring_list_1[i], outstring_list_1_qraft[i], os.path.join(repo_path,'Output/Plots/COR1_PSI_Plot_test.png'))
    display_fits_image_with_features_and_B_field(outstring_list_1[i], outstring_list_1_qraft[i], PSI=True)
    display_fits_image_with_features_and_B_field(directory_list_1[i], directory_list_1_qraft[i], PSI=False)

for i in range(len(directory_list_1)):
    plot_model_data_comparison_with_features(directory_list_2[i], directory_list_2_qraft[i], outstring_list_2[i], outstring_list_2_qraft[i], os.path.join(repo_path,'Output/Plots/COR1_PSI_Plot_test.png'))
    display_fits_image_with_features_and_B_field(outstring_list_2[i], outstring_list_2_qraft[i], PSI=True)
    display_fits_image_with_features_and_B_field(directory_list_2[i], directory_list_2_qraft[i], PSI=False)


# carrington lat/lon in degrees
files = directory_list_1
longitudes = []
latitudes = []
small_angle_const = (3600 * 360)/(2 * np.pi)
x_radius = []
y_radius = []
z_radius = []
for i in files:
    path = os.path.join(repo_path, i)
    head = fits.getheader(path)
    def has_seconds(a_string):
        return "." in a_string.split(":")[2]

    if has_seconds(head['DATE-OBS']):
        time = datetime.strptime(head['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f")
    else:
        time = datetime.strptime(head['DATE-OBS'], "%Y-%m-%dT%H:%M:%S")
    # time = datetime.strptime(head['DATE-OBS'],'%Y-%m-%dT%H:%M:%S')
    latitudes.append(head['CRLT_OBS'])
    longitudes.append(head['CRLN_OBS'])
    d_sun_obs = (constants.radius.to_value() * small_angle_const) / head['RSUN']
    x_radius.append(d_sun_obs)
    y_radius.append(d_sun_obs)
    z_radius.append(d_sun_obs)

files2 = directory_list_2
longitudes2 = []
latitudes2 = []
x2_radius = []
y2_radius = []
z2_radius = []
for i in files2:
    path2 = os.path.join(repo_path, i)
    head2 = fits.getheader(path2)
    # time2 = datetime.strptime(head2['DATE-OBS'],'%Y-%m-%dT%H:%M:%S')
    if has_seconds(head2['DATE-OBS']):
        time2 = datetime.strptime(head2['DATE-OBS'], "%Y-%m-%dT%H:%M:%S.%f")
    else:
        tim2e = datetime.strptime(head2['DATE-OBS'], "%Y-%m-%dT%H:%M:%S")
    latitudes2.append(head2['CRLT_OBS'])
    longitudes2.append(head2['CRLN_OBS'])
    x2_radius.append(head2['DSUN_OBS'])
    y2_radius.append(head2['DSUN_OBS'])
    z2_radius.append(head2['DSUN_OBS'])


theta = np.array(longitudes)
phi = np.array(latitudes)

theta2 = np.array(longitudes2)
phi2 = np.array(latitudes2)

x_radius = np.array(x_radius)
y_radius = np.array(y_radius)
z_radius = np.array(z_radius)

x2_radius = np.array(x2_radius)
y2_radius = np.array(y2_radius)
z2_radius = np.array(z2_radius)

# Calculate the x and y coordinates
x = x_radius * np.cos(theta) * np.cos(phi)
y = y_radius * np.sin(theta) * np.cos(phi)
z = z_radius * np.sin(phi)

x2 = x2_radius * np.cos(theta2) * np.cos(phi2)
y2 = y2_radius * np.sin(theta2) * np.cos(phi2)
z2 = z2_radius * np.sin(phi)


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'},figsize=(8, 8))
circle = plt.Circle((0.0, 0.0), (10*u.Rsun).to_value(u.AU),
                    transform=ax.transProjectionAffine + ax.transAxes, color="yellow",
                    alpha=1, label="Sun")
ax.add_artist(circle)
ax.scatter(np.deg2rad(longitudes), x_radius/astropy.constants.au.to_value(u.m), color='red',label='MLSO K-COR Observations')
ax.scatter(np.deg2rad(longitudes2), x2_radius/astropy.constants.au.to_value(u.m), color='blue',label='COR-1 Observations')
ax.set_theta_zero_location("S")
ax.legend(bbox_to_anchor=(1, 1.05), loc="upper right")
ax.set_title('Locations of Observations Chosen in CR 2194')
ax.set_rlim(0, 1.3)
plt.savefig(os.path.join(repo_path,'Output/Plots/Polar_Observations_Plot.png'))
# plt.show()

# params = date_print + str(detector,'utf-8') + '_PSI'

#
# path = os.path.join(repo_path,'Output/fits_images')
# directory = os.fsencode(path)
# directorylist = []
# for file in os.listdir(directory):
#     filename = os.fsdecode(file)
#     directorylist.append(os.path.join(path,filename))
# keyword_By_1 = 'COR1__PSI_By.fits'
# indexes_By_1, By_list_1 = zip(*[(index, item) for index, item in enumerate(directorylist) if keyword_By_1 in item])
#
# keyword_Bz_1 = 'COR1__PSI_Bz.fits'
# indexes_Bz_1, Bz_list_1 = zip(*[(index, item) for index, item in enumerate(directorylist) if keyword_Bz_1 in item])
#
#
# create_six_fig_plot(Bz_list_1, By_list_1, os.path.join(repo_path,'Output/Plots/Test_Vector_Plot.png'))

# Generate Vector Plot
"""
path = os.path.join(repo_path,'Output/fits_images')
directory = os.fsencode(path)
directorylist = []
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    directorylist.append(os.path.join(path,filename))
keyword_By_1 = 'COR1__PSI_By.fits'
indexes_By_1, By_list_1 = zip(*[(index, item) for index, item in enumerate(directorylist) if keyword_By_1 in item])

keyword_Bz_1 = 'COR1__PSI_Bz.fits'
indexes_Bz_1, Bz_list_1 = zip(*[(index, item) for index, item in enumerate(directorylist) if keyword_Bz_1 in item])

head = fits.getheader(Bz_list_1[0])
if head['detector'] == 'COR1':
    rsun = (head['rsun'] / head['cdelt1']) * 1.45 # number of pixels in radius of sun
    detector = 'COR1'
    create_six_fig_plot(Bz_list_1, By_list_1, os.path.join(repo_path,'Output/Plots/Test_Vector_Plot.png'), rsun, detector)

keyword_By_2 = 'KCor__PSI_By.fits'
indexes_By_2, By_list_2 = zip(*[(index, item) for index, item in enumerate(directorylist) if keyword_By_2 in item])

keyword_Bz_2 = 'KCor__PSI_Bz.fits'
indexes_Bz_2, Bz_list_2 = zip(*[(index, item) for index, item in enumerate(directorylist) if keyword_Bz_2 in item])
head = fits.getheader(Bz_list_2[0])
if head['INSTRUME'] == 'COSMO K-Coronagraph':
    rsun = head['RCAM_DCR']
    detector = 'KCor'
    create_six_fig_plot(Bz_list_2, By_list_2, os.path.join(repo_path,'Output/Plots/Test_Vector_Plot_MLSO.png'), rsun, detector)
"""

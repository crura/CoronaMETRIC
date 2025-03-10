#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
Created on Fri Aug 25 19:11:32 2023

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
#matplotlib.use('TkAgg')
#mpl.use('TkAgg')
from functions import KL_div, JS_Div, calculate_KDE, calculate_KDE_statistics, create_results_dictionary
import seaborn as sns


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

datapath = join(repo_path, 'Output/QRaFT_Results','__2017_09_11__COR1__PSI_pB.fits.sav')
#datafiles = [join(datapath,f) for f in listdir(datapath) if isfile(join(datapath,f)) and f !='.DS_Store']

# for i in datapath:
#             idl_save = readsav(i)

            # sub_dict['err_cor1_central'] = idl_save['ERR_SIGNED_ARR_COR1']
            # sub_dict['err_cor1_los'] = idl_save['ERR_SIGNED_ARR_LOS_COR1']
            # sub_dict['err_random'] = idl_save['ERR_SIGNED_ARR_RND']
            # sub_dict['err_psi_central'] = idl_save['ERR_SIGNED_ARR_FORWARD']
            # sub_dict['err_psi_los'] = idl_save['ERR_SIGNED_ARR_LOS_FORWARD']
            # sub_dict['L_cor1'] = idl_save['L_COR1']
            # sub_dict['L_forward'] = idl_save['L_FORWARD']
            # sub_dict['detector'] = detector
            # err_cor1_central_new = np.concatenate([err_cor1_central_new, idl_save['ERR_SIGNED_ARR_COR1']])
            # err_cor1_los_new = np.concatenate([err_cor1_los_new,idl_save['ERR_SIGNED_ARR_LOS_COR1']])
            # err_forward_cor1_central_new = np.concatenate([err_forward_cor1_central_new,idl_save['ERR_SIGNED_ARR_FORWARD']])
            # err_forward_cor1_los_new = np.concatenate([err_forward_cor1_los_new,idl_save['ERR_SIGNED_ARR_LOS_FORWARD']])
            # err_random_new = np.concatenate([err_random_new,idl_save['ERR_SIGNED_ARR_RND']])



idl_save = readsav(datapath)

features = idl_save['features']

features_xx_r = features['XX_R']
plt.imshow(idl_save['img_enh'], norm=matplotlib.colors.LogNorm(),cmap='gray')
for i, feature in enumerate(features):
    color = i * 255.0 / len(features)
    plt.plot(feature.xx_r[:feature.n_nodes], feature.yy_r[:feature.n_nodes], color=(color/255.0, 0, 0), linewidth=3)

#plt.show()
plt.close()


plt.imshow(idl_save['img_p_enh'], norm=matplotlib.colors.LogNorm())
for i, feature in enumerate(features):
    color = i * 255.0 / len(features)
    plt.plot(feature.xx_p[:feature.n_nodes], feature.yy_p[:feature.n_nodes]/idl_save['P'].RS[0], color=(color/255.0, 0, 0), linewidth=2)

#plt.show()
plt.close()











def plot_features(datapath, range=None, old_win=False, title='', save=False, **kwargs):
    
    idl_save = readsav(datapath)
    IMG = idl_save['img_orig']
    FEATURES = idl_save['features']
    P = idl_save['P']
    spath=datapath+'.png'
    
    if not old_win:
        plt.figure(figsize=(7, 7))
        plt.title(title)
        plt.clf()

    position = [0.13, 0.1, 0.86, 0.95]

    if range is not None:
        plt.imshow(IMG, norm=matplotlib.colors.LogNorm(), cmap='gray', vmin=range[0], vmax=range[1], extent=(0, IMG.shape[1], 0, IMG.shape[0]))
    else:
        plt.imshow(IMG, norm=matplotlib.colors.LogNorm(), cmap='gray', extent=(0, IMG.shape[1], 0, IMG.shape[0]))

    if 'polar' not in locals():
        plt.xlabel('X, pixels')
        plt.ylabel('Y, pixels')
        plt.title(title)
        colors = plt.cm.jet(np.linspace(0, 1, len(FEATURES)))
        for i, feature in enumerate(FEATURES):
            plt.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors[i], linewidth=3)
        plt.plot([P['XYCENTER'][0][0], P['XYCENTER'][0][0]], [P['XYCENTER'][0][1] - 30, P['XYCENTER'][0][1] + 30], color='k')
        plt.plot([P['XYCENTER'][0][0] - 30, P['XYCENTER'][0][0] + 30], [P['XYCENTER'][0][1], P['XYCENTER'][0][1]], color='k')
    else:
        X = np.arange(IMG.shape[1]) * P['d_phi']
        Y = np.arange(IMG.shape[0]) * P['d_rho'] / P['Rs']
        plt.xlabel('Position angle, radians')
        plt.ylabel('Radial distance, Rs')
        plt.title(title)
        plt.pcolormesh(X, Y, IMG, cmap='gray', vmin=range[0], vmax=range[1])
        colors = plt.cm.jet(np.linspace(0, 1, len(FEATURES)))
        for i, feature in enumerate(FEATURES):
            plt.plot(feature['xx_p'][:feature['n_nodes']], feature['yy_p'][:feature['n_nodes']] / P['Rs'], color=colors[i], linewidth=2)
            if map:
                map.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors[i], linewidth=3)

    plt.colorbar()
    if save:
        plt.savefig(spath)
    #plt.show()
    plt.close()
    
#plot_features(idl_save['img_orig'], features, idl_save['P'],save=True)
# Example usage:
# plot_features(IMG_enh, FEATURES, P, range=[-0.025, 0.025], title='Detected coronal features')
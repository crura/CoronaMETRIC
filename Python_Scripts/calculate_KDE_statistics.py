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
#matplotlib.use('TkAgg')
#mpl.use('TkAgg')
from functions import KL_div, JS_Div, calculate_KDE, calculate_KDE_statistics
import seaborn as sns


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
    err_cor1_central_new = np.concatenate([err_cor1_central_new, idl_save['ERR_SIGNED_ARR_COR1']])
    err_cor1_los_new = np.concatenate([err_cor1_los_new,idl_save['ERR_SIGNED_ARR_LOS_COR1']])
    err_forward_cor1_central_new = np.concatenate([err_forward_cor1_central_new,idl_save['ERR_SIGNED_ARR_FORWARD']])
    err_forward_cor1_los_new = np.concatenate([err_forward_cor1_los_new,idl_save['ERR_SIGNED_ARR_LOS_FORWARD']])
    err_random_new = np.concatenate([err_random_new,idl_save['ERR_SIGNED_ARR_RND']])


# convert arrays from radians to degrees
err_cor1_central_deg_new = err_cor1_central_new[np.where(err_cor1_central_new != 0)]*180/np.pi
err_forward_cor1_central_deg_new = err_forward_cor1_central_new[np.where(err_forward_cor1_central_new != 0)]*180/np.pi
err_random_deg_new = err_random_new[np.where(err_random_new != 0)]*180/np.pi


x_1_cor1_central_deg_new, KDE_cor1_central_deg_new = calculate_KDE(err_cor1_central_deg_new)
x_1_forward_cor1_central_deg_new, KDE_forward_cor1_central_deg_new = calculate_KDE(err_forward_cor1_central_deg_new)
x_1_random_deg_new, KDE_random_deg_new = calculate_KDE(err_random_deg_new)

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

"""
xmin_random = min(err_random_deg_new)
xmax_random = max(err_random_deg_new)
kde0_random_deg = gaussian_kde(err_random_deg_new)
x_1_random_deg = np.linspace(xmin_random, xmax_random, 200)
kde0_x_random_deg = kde0_random_deg(x_1_random_deg)
"""

what = sns.histplot(err_random_deg_new,kde=True, bins=30)
norm_max_random = max(what.get_lines()[0].get_data()[1])
plt.close()

what2 = sns.histplot(err_cor1_central_deg_new,kde=True, bins=30)
norm_max_cor1 = max(what2.get_lines()[0].get_data()[1])
plt.close()

what3 = sns.histplot(err_forward_cor1_central_deg_new,kde=True, bins=30)
norm_max_forward = max(what3.get_lines()[0].get_data()[1])
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.subplots(1,1)
sns.histplot(err_cor1_central_deg_new,kde=True,label='COR-1',bins=30,ax=ax,color='tab:blue')
sns.histplot(err_forward_cor1_central_deg_new,kde=True,label='PSI/FORWARD pB',bins=30,ax=ax,color='tab:orange')
#sns.histplot(err_random_deg_new,kde=True, bins=30, label='Random',ax=ax, color='tab:green')
#x_axis = np.linspace(-90, 90, len(KDE_cor1_central_deg_new))
plt.plot(x_1_cor1_central_deg_new, (KDE_cor1_central_deg_new/max(KDE_cor1_central_deg_new))*norm_max_cor1, color='tab:blue')
plt.plot(x_1_random_deg_new, (KDE_random_deg_new/max(KDE_random_deg_new))*norm_max_random, color='tab:green', label='random')
plt.plot(x_1_forward_cor1_central_deg_new, (KDE_forward_cor1_central_deg_new/max(KDE_forward_cor1_central_deg_new))*norm_max_forward, color='tab:orange')
norm_kde_random = (KDE_random_deg_new/max(KDE_random_deg_new))*norm_max_random
norm_kde_forward = (KDE_forward_cor1_central_deg_new/max(KDE_forward_cor1_central_deg_new))*norm_max_forward
norm_kde_cor1 = (KDE_cor1_central_deg_new/max(KDE_cor1_central_deg_new))*norm_max_cor1
#sns.kdeplot()
ax.set_xlabel('Angle Discrepancy (Degrees)',fontsize=14)
ax.set_ylabel('Pixel Count',fontsize=14)
ax.set_title('QRaFT Feature Tracing Performance Against Central POS $B$ Field',fontsize=15)
ax.set_xlim(-95,95)
#ax.set_ylim(0,0.07)
ax.legend(fontsize=13)

# plt.text(20,0.045,"COR1 average discrepancy: " + str(np.round(np.average(err_cor1_central_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_cor1_central_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
plt.savefig(os.path.join(repo_path,'Output/Plots/Updated_COR1_vs_FORWARD_Feature_Tracing_Performance.png'))
plt.show()
#plt.close()

JSD_cor1_forward_central_new, KLD_cor1_forward_central_new = calculate_KDE_statistics(norm_kde_cor1, norm_kde_forward)
JSD_cor1_central_random_new, KLD_cor1_central_random_new = calculate_KDE_statistics(norm_kde_cor1, norm_kde_random)
JSD_COR1_Forward_Central_Random_new, KLDcor1_forward_central_random_new = calculate_KDE_statistics(norm_kde_forward, norm_kde_random)

combined_dict = dict(metric=['KL Divergence', 'JS Divergence'],
                    cor1_v_psi=[KLD_cor1_forward_central_new, JSD_cor1_forward_central_new],
                   cor1_v_random=[KLD_cor1_central_random_new, JSD_cor1_central_random_new],
                    psi_v_random=[KLDcor1_forward_central_random_new, JSD_COR1_Forward_Central_Random_new])

pd.set_option('display.float_format', '{:.3E}'.format)
stats_df = pd.DataFrame(combined_dict)
stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
print(stats_df.to_latex(index=False))

print("")

cor1_avg = np.round(np.average(abs(err_cor1_central_deg_new)),5)
forward_avg = np.round(np.average(abs(err_forward_cor1_central_deg_new)),5)
random_avg = np.round(np.average(abs(err_random_deg_new)),5)

cor1_med = np.round(np.median(abs(err_cor1_central_deg_new)),5)
forward_med = np.round(np.median(abs(err_forward_cor1_central_deg_new)),5)
random_med = np.round(np.median(abs(err_random_deg_new)),5)


combined_dict = dict(metric=['average discrepancy', 'median discrepancy'],
                    cor1=[cor1_avg, cor1_med],
                   psi=[forward_avg, forward_med],
                    random=[random_avg, random_med])

pd.set_option('display.float_format', '{:.3f}'.format)
stats_df = pd.DataFrame(combined_dict)
#stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
print(stats_df.to_latex(index=False))
"""
# Generate figure for combined plot of COR-1 and MLSO K-COR datasets
fig, ax = plt.subplots(1,2,figsize=(24,9))

# Generate plots for MLSO K-COR dataset on left axis
sns.distplot(err_mlso_central_deg,hist=True,label='MLSO K-COR',bins=30,ax=ax[0])
sns.distplot(err_forward_central_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax[0])
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax[0])
ax[0].set_xlabel('Angle Discrepancy',fontsize=22)
ax[0].set_ylabel('Probability Density',fontsize=22)
ax[0].set_title('QRaFT Feature Tracing Performance MLSO K-COR vs PSI',fontsize=22)
ax[0].set_xlim(0,90)
ax[0].set_ylim(0,0.07)
ax[0].legend(fontsize=20)

# Generate plots for COR1 dataset on right axis
sns.distplot(err_cor1_central_deg,hist=True,label='COR-1',bins=30,ax=ax[1])
sns.distplot(err_forward_cor1_central_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax[1])
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax[1])
# sns.kdeplot(err_mlso_los_deg,label='KDE')
ax[1].set_xlabel('Angle Discrepancy',fontsize=22)
ax[1].set_ylabel('Probability Density',fontsize=22)
ax[1].set_title('QRaFT Feature Tracing Performance COR-1 vs PSI',fontsize=22)
ax[1].set_xlim(0,90)
ax[1].set_ylim(0,0.07)
ax[1].legend(fontsize=20)
#plt.savefig(os.path.join(repo_path,'Output/Plots/COR1_vs_FORWARD_Feature_Tracing_Performance_Combined.png'))
plt.show()
plt.close()
"""

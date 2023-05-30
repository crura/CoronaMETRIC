#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 16:06:10 2023

@author: crura
"""

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
from functions import KL_div, JS_Div, calculate_KDE, calculate_KDE_statistics, create_results_dictionary
import seaborn as sns
from scipy.optimize import minimize_scalar
import subprocess


#def run_calculations(datafiles, detector):
repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

os.chdir(repo_path)

optimization_array = []
x_list = []

def filter_nan(array):
    nan_indices = np.isnan(array)
    inf_indices = np.isinf(array)
    return array[~nan_indices & ~inf_indices]

def optimize_for_avg(detector, optimization_array):
    
    repo = git.Repo('.', search_parent_directories=True)
    repo_path = repo.working_tree_dir

    datapath = join(repo_path, 'Output/QRaFT_Results')
    datafiles = [join(datapath,f) for f in listdir(datapath) if isfile(join(datapath,f)) and f !='.DS_Store']
    file_path = join(repo_path, 'Output/Plots/results.txt')
    file = open(file_path, 'w')
    
    err_cor1_central_new = np.array([]) # idl_save_new['ERR_ARR_COR1']
    err_cor1_los_new = np.array([])
    err_forward_cor1_central_new = np.array([])
    err_forward_cor1_los_new = np.array([])
    err_random_new = np.array([])
    
    err_cor1_central_masked = np.array([])
    err_forward_central_masked = np.array([])
    err_random_centrak_masked = np.array([])
    
    date_dict = {}
    
    detector = 'K-COR'
    
    for i in datafiles:
        if detector == 'COR-1':
            if i.endswith('COR1__PSI.sav'):
                date_str = i.rstrip('OR1__PSI.sav')[-11:].rstrip('__C')
                sub_dict = {}
                idl_save = readsav(i)
                sub_dict['err_cor1_central'] = idl_save['ERR_SIGNED_ARR_COR1']
                sub_dict['err_cor1_los'] = idl_save['ERR_SIGNED_ARR_LOS_COR1']
                sub_dict['err_random'] = idl_save['ERR_SIGNED_ARR_RND']
                sub_dict['err_psi_central'] = idl_save['ERR_SIGNED_ARR_FORWARD']
                sub_dict['err_psi_los'] = idl_save['ERR_SIGNED_ARR_LOS_FORWARD']
                sub_dict['L_cor1'] = idl_save['L_COR1']
                sub_dict['L_forward'] = idl_save['L_FORWARD']
                sub_dict['detector'] = detector
                err_cor1_central_new = np.concatenate([err_cor1_central_new, idl_save['ERR_SIGNED_ARR_COR1']])
                err_cor1_los_new = np.concatenate([err_cor1_los_new,idl_save['ERR_SIGNED_ARR_LOS_COR1']])
                err_forward_cor1_central_new = np.concatenate([err_forward_cor1_central_new,idl_save['ERR_SIGNED_ARR_FORWARD']])
                err_forward_cor1_los_new = np.concatenate([err_forward_cor1_los_new,idl_save['ERR_SIGNED_ARR_LOS_FORWARD']])
                err_random_new = np.concatenate([err_random_new,idl_save['ERR_SIGNED_ARR_RND']])
                date_dict[date_str] = sub_dict
        elif detector == 'K-COR':
            if i.endswith('KCor__PSI.sav'):
                date_str = i.rstrip('KCor__PSI.sav')[-10:]
                sub_dict = {}
                idl_save = readsav(i)
                sub_dict['err_mlso_central'] = idl_save['ERR_SIGNED_ARR_MLSO']
                sub_dict['err_mlso_los'] = idl_save['ERR_SIGNED_ARR_LOS_MLSO']
                sub_dict['err_random'] = idl_save['ERR_SIGNED_ARR_RND']
                sub_dict['err_psi_central'] = idl_save['ERR_SIGNED_ARR_FORWARD']
                sub_dict['err_psi_los'] = idl_save['ERR_SIGNED_ARR_LOS_FORWARD']
                sub_dict['L_mlso'] = idl_save['L_MLSO']
                sub_dict['L_forward'] = idl_save['L_FORWARD']
                sub_dict['detector'] = detector
                err_cor1_central_new = np.concatenate([err_cor1_central_new, idl_save['ERR_SIGNED_ARR_MLSO']])
                err_cor1_los_new = np.concatenate([err_cor1_los_new,idl_save['ERR_SIGNED_ARR_LOS_MLSO']])
                err_forward_cor1_central_new = np.concatenate([err_forward_cor1_central_new,idl_save['ERR_SIGNED_ARR_FORWARD']])
                err_forward_cor1_los_new = np.concatenate([err_forward_cor1_los_new,idl_save['ERR_SIGNED_ARR_LOS_FORWARD']])
                err_random_new = np.concatenate([err_random_new,idl_save['ERR_SIGNED_ARR_RND']])
                date_dict[date_str] = sub_dict
    
    
    length_dict = {}
    num_features_dict = {}
    
    for i in date_dict.keys():
        results, data = create_results_dictionary(date_dict[i], i, detector, file)
        results_masked, data_masked, mask = create_results_dictionary(date_dict[i], i, detector, file, True)
        print(i)
        if detector == 'COR-1':
            err_cor1_central_masked = np.concatenate([err_cor1_central_masked, data_masked['cor1_central']])
            err_forward_central_masked = np.concatenate([err_forward_central_masked, data_masked['forward_central']])
            err_random_centrak_masked = np.concatenate([err_random_centrak_masked, data_masked['random']])
            # find average feature length based on every unique length, indicating different feature
            length_dict[i] = np.average(np.unique(date_dict[i]['L_cor1']))
            num_features_dict[i] = len(np.unique(date_dict[i]['L_cor1']))
        elif detector == 'K-COR':
            err_cor1_central_masked = np.concatenate([err_cor1_central_masked, data_masked['mlso_central']])
            err_forward_central_masked = np.concatenate([err_forward_central_masked, data_masked['forward_central']])
            err_random_centrak_masked = np.concatenate([err_random_centrak_masked, data_masked['random']])
            # find average feature length based on every unique length, indicating different feature
            length_dict[i] = np.average(np.unique(date_dict[i]['L_mlso']))
            num_features_dict[i] = len(np.unique(date_dict[i]['L_mlso']))
        #print(results)
    
    """
    combined_maksed_dict = {}
    combined_maksed_dict['err_cor1_central_masked'] = err_cor1_central_masked
    combined_maksed_dict['err_forward_central_masked'] = err_forward_central_masked
    results_cor1_masked_combined, data_all = create_results_dictionary([combined_maksed_dict], 'combined')
    """
    
    # convert arrays from radians to degrees
    err_cor1_central_deg_new = filter_nan(err_cor1_central_new[np.where(err_cor1_central_new != 0)]*180/np.pi)
    err_forward_cor1_central_deg_new = filter_nan(err_forward_cor1_central_new[np.where(err_forward_cor1_central_new != 0)]*180/np.pi)
    err_random_deg_new = filter_nan(err_random_new[np.where(err_random_new != 0)]*180/np.pi)
    
    
    err_cor1_central_deg_new_combined = err_cor1_central_deg_new.copy()
    err_forward_cor1_central_deg_new_combined = err_forward_cor1_central_deg_new.copy()
    err_random_deg_new = err_random_deg_new.copy()
    print('lengths of unfiltered dataset: ', err_cor1_central_deg_new.shape, err_forward_cor1_central_deg_new.shape, err_random_deg_new.shape)
    
    
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
    stats_df.columns = ['metric', '{} vs psi pB'.format(detector), '{} vs random'.format(detector), 'psi pB vs random']
    print('\n Combined Results: \n')
    print(stats_df.to_latex(index=False))
    file.write('\n Combined Results: \n')
    file.write(stats_df.to_latex(index=False))
    
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
    sns.histplot(err_cor1_central_deg_new,kde=True,label=detector,bins=30,ax=ax,color='tab:blue')
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
    ax.set_title('QRaFT {} Feature Tracing Performance Against Central POS $B$ Field'.format(detector),fontsize=15)
    ax.set_xlim(-95,95)
    #ax.set_ylim(0,0.07)
    ax.legend(fontsize=13)
    
    # plt.text(20,0.045,"COR1 average discrepancy: " + str(np.round(np.average(err_cor1_central_deg),5)))
    # plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_cor1_central_deg),5)))
    # plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
    plt.savefig(os.path.join(repo_path,'Output/Plots/Updated_{}_vs_FORWARD_Feature_Tracing_Performance.png'.format(detector.replace('-',''))))
    #plt.show()
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
    stats_df.columns = ['metric', '{} vs psi pB'.format(detector), '{} vs random'.format(detector), 'psi pB vs random']
    #print(stats_df.to_latex(index=False))
    #file.write(stats_df.to_latex(index=False))
    
    print("")
    
    cor1_avg = np.round(np.average(abs(err_cor1_central_deg_new)),5)
    forward_avg = np.round(np.average(abs(err_forward_cor1_central_deg_new)),5)
    random_avg = np.round(np.average(abs(err_random_deg_new)),5)
    
    cor1_std = np.round(np.std(abs(err_cor1_central_deg_new)),5)
    forward_avg = np.round(np.average(abs(err_forward_cor1_central_deg_new)),5)
    forward_std = np.round(np.std(abs(err_forward_cor1_central_deg_new)),5)
    
    cor1_confidence_interval = 1.96 * (cor1_std / np.sqrt(len(err_cor1_central_deg_new)))
    
    cor1_med = np.round(np.median(abs(err_cor1_central_deg_new)),5)
    forward_med = np.round(np.median(abs(err_forward_cor1_central_deg_new)),5)
    random_med = np.round(np.median(abs(err_random_deg_new)),5)
    
    
    combined_dict = dict(metric=['average discrepancy', 'median discrepancy'],
                        cor1=[cor1_avg, cor1_med],
                       psi=[forward_avg, forward_med],
                        random=[random_avg, random_med])
    
    pd.set_option('display.float_format', '{:.3f}'.format)
    stats_df = pd.DataFrame(combined_dict)
    stats_df.columns = ['metric', '{}'.format(detector), 'psi', 'random']
    #stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
    print(stats_df.to_latex(index=False))
    file.write(stats_df.to_latex(index=False))
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
    
    
    
    
    
    
    
    
    x_1_cor1_central_deg_new, KDE_cor1_central_deg_new = calculate_KDE(err_cor1_central_masked)
    x_1_forward_cor1_central_deg_new, KDE_forward_cor1_central_deg_new = calculate_KDE(err_forward_central_masked)
    x_1_random_deg_new, KDE_random_deg_new = calculate_KDE(err_random_centrak_masked)
    
    JSD_cor1_forward_central_new, KLD_cor1_forward_central_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_forward_cor1_central_deg_new)
    JSD_cor1_central_random_new, KLD_cor1_central_random_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_random_deg_new)
    JSD_COR1_Forward_Central_Random_new, KLDcor1_forward_central_random_new = calculate_KDE_statistics(KDE_forward_cor1_central_deg_new, KDE_random_deg_new)
    
    combined_dict = dict(metric=['KL Divergence', 'JS Divergence'],
                        cor1_v_psi=[KLD_cor1_forward_central_new, JSD_cor1_forward_central_new],
                       cor1_v_random=[KLD_cor1_central_random_new, JSD_cor1_central_random_new],
                        psi_v_random=[KLDcor1_forward_central_random_new, JSD_COR1_Forward_Central_Random_new])
    
    pd.set_option('display.float_format', '{:.3E}'.format)
    stats_df = pd.DataFrame(combined_dict)
    stats_df.columns = ['metric', '{} vs psi pB (L > {})'.format(detector,mask), '{} vs random (L> {})'.format(detector,mask), 'psi pB vs random (L > {})'.format(mask)]
    print('\n Combined results (L > {}):\n'.format(mask))
    print(stats_df.to_latex(index=False))
    file.write('\n Combined results (L > {}):\n'.format(mask))
    file.write(stats_df.to_latex(index=False))
    
    """
    xmin_random = min(err_random_centrak_masked)
    xmax_random = max(err_random_centrak_masked)
    kde0_random_deg = gaussian_kde(err_random_centrak_masked)
    x_1_random_deg = np.linspace(xmin_random, xmax_random, 200)
    kde0_x_random_deg = kde0_random_deg(x_1_random_deg)
    """
    
    what = sns.histplot(err_random_centrak_masked,kde=True, bins=30)
    norm_max_random = max(what.get_lines()[0].get_data()[1])
    plt.close()
    
    what2 = sns.histplot(err_cor1_central_masked,kde=True, bins=30)
    norm_max_cor1 = max(what2.get_lines()[0].get_data()[1])
    plt.close()
    
    what3 = sns.histplot(err_forward_central_masked,kde=True, bins=30)
    norm_max_forward = max(what3.get_lines()[0].get_data()[1])
    plt.close()
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.subplots(1,1)
    sns.histplot(err_cor1_central_masked,kde=True,label=detector,bins=30,ax=ax,color='tab:blue')
    sns.histplot(err_forward_central_masked,kde=True,label='PSI/FORWARD pB',bins=30,ax=ax,color='tab:orange')
    #sns.histplot(err_random_centrak_masked,kde=True, bins=30, label='Random',ax=ax, color='tab:green')
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
    ax.set_title('QRaFT {} Feature Tracing Performance Against Central POS $B$ Field (L > {})'.format(detector, mask),fontsize=15)
    ax.set_xlim(-95,95)
    #ax.set_ylim(0,0.07)
    ax.legend(fontsize=13)
    
    # plt.text(20,0.045,"COR1 average discrepancy: " + str(np.round(np.average(err_cor1_central_deg),5)))
    # plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_cor1_central_deg),5)))
    # plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
    plt.savefig(os.path.join(repo_path,'Output/Plots/Updated_{}_vs_FORWARD_Feature_Tracing_Performance_masked_L{}.png'.format(detector.replace('-',''), mask)))
    #plt.show()
    plt.close()
    
    
    cor1_avg = np.round(np.average(abs(err_cor1_central_masked)),5)
    forward_avg = np.round(np.average(abs(err_forward_central_masked)),5)
    random_avg = np.round(np.average(abs(err_random_centrak_masked)),5)
    
    cor1_med = np.round(np.median(abs(err_cor1_central_masked)),5)
    forward_med = np.round(np.median(abs(err_forward_central_masked)),5)
    random_med = np.round(np.median(abs(err_random_centrak_masked)),5)
    
    
    combined_dict = dict(metric=['average discrepancy (L>{})'.format(mask), 'median discrepancy (L>{})'.format(mask)],
                        kcor=[cor1_avg, cor1_med],
                       psi=[forward_avg, forward_med],
                        random=[random_avg, random_med])
    
    pd.set_option('display.float_format', '{:.3f}'.format)
    stats_df = pd.DataFrame(combined_dict)
    stats_df.columns = ['metric', '{}'.format(detector), 'psi', 'random']
    #stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
    print(stats_df.to_latex(index=False))
    file.write(stats_df.to_latex(index=False))
    
    optimization_array.append(cor1_avg)
    
    #run_calculations(datafiles, 'COR-1')
    #run_calculations(datafiles, 'K-COR')
    
    file.close()
    return cor1_avg, length_dict, num_features_dict



def objective(x):
    
    cmd = ['./execute_qraft_optimize.sh', str(x)]
    # Run the command and capture the output
    result = subprocess.run(cmd, stdout=subprocess.PIPE)

    # Print the output of the script
    print(result.stdout.decode('utf-8'))
    x_list.append(x)
    
    


    hi, length_dict, num_features_dict = optimize_for_avg('K-COR', optimization_array)
    return hi, length_dict, num_features_dict

# define the bounds
bounds = (0, 5)

# perform constrained optimization
#result = minimize_scalar(lambda x: objective(x), bounds=bounds, method='bounded')

# print the result
#print(f"The minimum value of y is {result.fun:.4f} at x={result.x:.4f}")

test_values = np.arange(bounds[0],bounds[1], 0.1)
test_values_dict = {}
test_num_features_dict = {}
for i in test_values:
    result, length_dict, num_features_dict = objective(i)
    test_values_dict[i] = length_dict
    test_num_features_dict[i] = num_features_dict

test_values_df = pd.DataFrame(test_values_dict)
transposed_length_df = test_values_df.transpose()

test_num_features_df = pd.DataFrame(test_num_features_dict)
transposed_num_length_df = test_num_features_df.transpose()
#value_list = [1,2,3]
#value_dict = {}
#for i in value_list:
#    length_dict = {}
#    for j in date_dict.keys():
#        length_dict[j] = np.average(date_dict[j]['L_mlso'])
#    value_dict[i] = length_dict

# initialize lists to store x and y values
#x_vals = []
#y_vals = []

# perform constrained optimization
#result = minimize_scalar(lambda x: objective(x), bounds=bounds, method='bounded', callback=lambda xk: (x_vals.append(xk), y_vals.append(objective(xk))))

# plot the values of y as x is minimized
fig = plt.figure()
plt.plot(x_list, optimization_array)
plt.xlabel('Iteration')
plt.ylabel('Value of y')
plt.title('Minimization of y')
#plt.show()
detector = 'K-COR'
mask = 25
plt.savefig(os.path.join(repo_path,'Output/Plots/{}_vs_FORWARD_masked_L{}_Optimization.png'.format(detector.replace('-',''), mask)))
plt.close()

fig = plt.figure()
plt.plot(np.arange(0, len(optimization_array),1), optimization_array)
plt.xlabel('Iteration')
plt.ylabel('Value of y')
plt.title('Minimization of y')
#plt.show()
detector = 'K-COR'
mask = 25
plt.savefig(os.path.join(repo_path,'Output/Plots/{}_vs_FORWARD_masked_L{}_Optimization_Indicies.png'.format(detector.replace('-',''), mask)))
plt.close()

# print the result
#print(f"The minimum value of y is {result.fun:.4f} at x={result.x:.4f}")


# Generate a single plot with six line plots
plt.figure(figsize=(10, 6))

# Iterate over each column
for column in transposed_length_df.columns:
    # Plot the values against the row index
    plt.plot(transposed_length_df.index, transposed_length_df[column], label=column)

plt.title('Comparison QRaFT Feature Lengths MLSO K-COR')
plt.xlabel('trend_k parameter')
plt.ylabel('Average Feature Length')
plt.legend()  # Show a legend with column names
plt.savefig(os.path.join(repo_path,'Output/Plots/{}_Average_Feature_Length_Comparison.png'.format(detector)))
plt.show()


# Generate a single plot with six line plots
plt.figure(figsize=(10, 6))

# Iterate over each column
for column in transposed_num_length_df.columns:
    # Plot the values against the row index
    plt.plot(transposed_num_length_df.index, transposed_num_length_df[column], label=column)

plt.title('Comparison QRaFT Number of Features MLSO K-COR')
plt.xlabel('trend_k parameter')
plt.ylabel('Number of Features Detected')
plt.legend()  # Show a legend with column names
plt.savefig(os.path.join(repo_path,'Output/Plots/{}_Number_of_Features_Comparison.png'.format(detector)))
plt.show()




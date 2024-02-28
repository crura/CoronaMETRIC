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
from scipy.stats import gaussian_kde
import seaborn as sns
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import sqlite3
import astropy.units as u
from prettytable import PrettyTable

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir


def filter_nan_singular(array):
    nan_indices = np.isnan(array)
    inf_indices = np.isinf(array)
    return array[~nan_indices & ~inf_indices]

def calculate_KDE(err_array):
    # set minimum and maximum x values for gaussian kde calculation
    xmin = min(err_array)
    xmax = max(err_array)

    # Calculate Gaussian KDE for cor1 pB vs central B field dataset
    kde = gaussian_kde(err_array)
    x_1 = np.linspace(xmin, xmax, 1000)
    kde0 = kde(x_1)
    return x_1, kde0

def KL_div(p_probs, q_probs):
    KL_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(KL_div)

#define JS Divergence
def JS_Div(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (KL_div(p, m) + KL_div(q, m)) / 2

def remove_nans_infs(array, array2):

    # Find the indices of NaNs and Infs in the array
    nan_indices = np.isnan(array)
    inf_indices = np.isinf(array)

    filtered_array = array[~nan_indices & ~inf_indices]
    filtered_array2 = np.delete(array2, np.where(nan_indices | inf_indices))

    if len(filtered_array) < len(array):
        filtered = True
    else:
        filtered = False

    # Filter out NaNs and Infs from the array
    return filtered_array, filtered_array2

def calculate_KDE_statistics(KDE_1, KDE_2):

    #compute JS Divergence
    result_JSD = JS_Div(KDE_1, KDE_2)

    #compute KL Divergence
    result_KLD = KL_div(KDE_1, KDE_2)

    return result_JSD, result_KLD

def create_results_dictionary(input_dict, date, detector, file, masked=False):
    # convert arrays from radians to degrees

    if detector == 'COR-1':
        err_cor1_central_new = input_dict['err_cor1_central']
        err_forward_cor1_central_new = input_dict['err_psi_central']
        err_random_new = input_dict['err_random']
        L_cor1 = input_dict['L_cor1']
        L_forward = input_dict['L_forward']
        detector = input_dict['detector']
    elif detector == 'K-COR':
        err_cor1_central_new = input_dict['err_mlso_central']
        err_forward_cor1_central_new = input_dict['err_psi_central']
        err_random_new = input_dict['err_random']
        L_cor1 = input_dict['L_mlso']
        L_forward = input_dict['L_forward']

    L_cor1_new = L_cor1[np.where(err_cor1_central_new != 0)]
    L_forward_new = L_forward[np.where(err_forward_cor1_central_new != 0)]
    err_cor1_central_deg_new = err_cor1_central_new[np.where(err_cor1_central_new != 0)]*180/np.pi
    err_cor1_central_deg_new, L_cor1_new = remove_nans_infs(err_cor1_central_deg_new, L_cor1_new)
    err_forward_cor1_central_deg_new = err_forward_cor1_central_new[np.where(err_forward_cor1_central_new != 0)]*180/np.pi
    err_forward_cor1_central_deg_new, L_forward_new = remove_nans_infs(err_forward_cor1_central_deg_new, L_forward_new)
    err_random_deg_new = err_random_new[np.where(err_random_new != 0)]*180/np.pi
    err_random_deg_new = filter_nan_singular(err_random_deg_new)

    #L_cor1_new = remove_nans_infs(L_cor1_new)
    #L_forward_new = remove_nans_infs(L_forward_new)

    if masked:
        if detector == 'COR-1':
            mask = np.min(L_cor1_new)
        elif detector == 'K-COR':
            mask = np.min(L_cor1_new)
        print('\n Results for {} (L > {}): \n'.format(date, mask))
        file.write('\n Results for {} (L > {}): \n'.format(date, mask))
        err_cor1_central_deg_new = err_cor1_central_deg_new[np.where(L_cor1_new > mask)]
        err_cor1_central_deg_new, L_cor1_new = remove_nans_infs(err_cor1_central_deg_new, L_cor1_new)
        err_forward_cor1_central_deg_new = err_forward_cor1_central_deg_new[np.where(L_forward_new > mask)]
        err_forward_cor1_central_deg_new, L_forward_new = remove_nans_infs(err_forward_cor1_central_deg_new, L_forward_new)
    else:
        print('\n Results for {}: \n'.format(date))
        file.write('\n Results for {}: \n'.format(date))



    x_1_cor1_central_deg_new, KDE_cor1_central_deg_new = calculate_KDE(err_cor1_central_deg_new)
    x_1_forward_cor1_central_deg_new, KDE_forward_cor1_central_deg_new = calculate_KDE(err_forward_cor1_central_deg_new)
    x_1_random_deg_new, KDE_random_deg_new = calculate_KDE(err_random_deg_new)

    JSD_cor1_forward_central_new, KLD_cor1_forward_central_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_forward_cor1_central_deg_new)
    JSD_cor1_central_random_new, KLD_cor1_central_random_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_random_deg_new)
    JSD_COR1_Forward_Central_Random_new, KLDcor1_forward_central_random_new = calculate_KDE_statistics(KDE_forward_cor1_central_deg_new, KDE_random_deg_new)

    combined_stats_dict = dict(metric=['KL Divergence', 'JS Divergence'],
                        cor1_v_psi=[KLD_cor1_forward_central_new, JSD_cor1_forward_central_new],
                       cor1_v_random=[KLD_cor1_central_random_new, JSD_cor1_central_random_new],
                        psi_v_random=[KLDcor1_forward_central_random_new, JSD_COR1_Forward_Central_Random_new])

    data_dict = {}
    if detector == 'COR-1':
        data_dict['cor1_central'] = err_cor1_central_deg_new
        data_dict['forward_central'] = err_forward_cor1_central_deg_new
        data_dict['random'] = err_random_deg_new
    elif detector == 'K-COR':
        data_dict['mlso_central'] = err_cor1_central_deg_new
        data_dict['forward_central'] = err_forward_cor1_central_deg_new
        data_dict['random'] = err_random_deg_new

    cor1_avg = np.round(np.average(abs(err_cor1_central_deg_new)),5)
    forward_avg = np.round(np.average(abs(err_forward_cor1_central_deg_new)),5)
    random_avg = np.round(np.average(abs(err_random_deg_new)),5)

    cor1_med = np.round(np.median(abs(err_cor1_central_deg_new)),5)
    forward_med = np.round(np.median(abs(err_forward_cor1_central_deg_new)),5)
    random_med = np.round(np.median(abs(err_random_deg_new)),5)

    cor1_std = np.round(np.std(abs(err_cor1_central_deg_new)),5)
    forward_std = np.round(np.std(abs(err_forward_cor1_central_deg_new)),5)
    random_std = np.round(np.std(abs(err_random_deg_new)),5)

    cor1_confidence_interval = 1.96 * (cor1_std / np.sqrt(len(err_cor1_central_deg_new)))
    forward_confidence_interval = 1.96 * (forward_std / np.sqrt(len(err_forward_cor1_central_deg_new)))
    random_confidence_interval = 1.96 * (random_std / np.sqrt(len(err_random_deg_new)))

    cor1_confidence_interval_rounded = np.round(cor1_confidence_interval, 3)
    forward_confidence_interval_rounded = np.round(forward_confidence_interval, 3)
    random_confidence_interval_rounded = np.round(random_confidence_interval, 3)

    cor1_avg_rounded = np.round(cor1_avg, 3)
    forward_avg_rounded = np.round(forward_avg, 3)
    random_avg_rounded = np.round(random_avg, 3)
    #assert type(cor1_avg_rounded) == np.float64



    combined_dict = dict(metric=['average discrepancy', 'median discrepancy'],
                        cor1=['{} +- {}'.format(str(cor1_avg_rounded), str(cor1_confidence_interval_rounded)), cor1_med],
                       psi=['{} +- {}'.format(str(forward_avg_rounded), str(forward_confidence_interval_rounded)), forward_med],
                       random=['{} +- {}'.format(str(random_avg_rounded), str(random_confidence_interval_rounded)), random_med])


    pd.set_option('display.float_format', '{:.3f}'.format)
    accuracy_stats_df = pd.DataFrame(combined_dict)
    accuracy_stats_df.columns = ['metric', '{}'.format(detector), 'psi'.format(detector), 'random']
    #stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
    print(accuracy_stats_df.to_latex(index=False))
    file.write(accuracy_stats_df.to_latex(index=False))


    pd.set_option('display.float_format', '{:.3E}'.format)
    stats_df = pd.DataFrame(combined_stats_dict)
    stats_df.columns = ['metric', '{} vs psi pB'.format(detector), '{} vs random'.format(detector), 'psi pB vs random']
    print(stats_df.to_latex(index=False))
    file.write(stats_df.to_latex(index=False))

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
    #plt.plot(x_1_random_deg_new, (KDE_random_deg_new/max(KDE_random_deg_new))*norm_max_random, color='tab:green', label='random')
    plt.plot(x_1_forward_cor1_central_deg_new, (KDE_forward_cor1_central_deg_new/max(KDE_forward_cor1_central_deg_new))*norm_max_forward, color='tab:orange')
    norm_kde_random = (KDE_random_deg_new/max(KDE_random_deg_new))*norm_max_random
    norm_kde_forward = (KDE_forward_cor1_central_deg_new/max(KDE_forward_cor1_central_deg_new))*norm_max_forward
    norm_kde_cor1 = (KDE_cor1_central_deg_new/max(KDE_cor1_central_deg_new))*norm_max_cor1
    #sns.kdeplot()
    ax.set_xlabel('Angle Discrepancy (Degrees)',fontsize=14)
    ax.set_ylabel('Pixel Count',fontsize=14)
    if masked:
        ax.set_title('QRaFT {} Feature Tracing Performance Against Central POS $B$ Field {} (L > {})'.format(detector, date, mask),fontsize=15)
    else:
        ax.set_title('QRaFT {} Feature Tracing Performance Against Central POS $B$ Field {}'.format(detector, date),fontsize=15)
    ax.set_xlim(-95,95)
    #ax.set_ylim(0,0.07)
    ax.legend(fontsize=13)

    # plt.text(20,0.045,"COR1 average discrepancy: " + str(np.round(np.average(err_cor1_central_deg),5)))
    # plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_cor1_central_deg),5)))
    # plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
    if masked:
        plt.savefig(os.path.join(repo_path,'Output/Plots/Updated_{}_vs_FORWARD_Feature_Tracing_Performance_{}_L_gt_{}.png'.format(detector.replace('-',''), date, mask)))
    else:
        plt.savefig(os.path.join(repo_path,'Output/Plots/Updated_{}_vs_FORWARD_Feature_Tracing_Performance_{}.png'.format(detector.replace('-',''), date)))
    #plt.show()
        
    gaussian_fit_pB = np.random.normal(forward_avg, forward_std, 1000)
    plt.plot(x_1_forward_cor1_central_deg_new, (KDE_forward_cor1_central_deg_new/max(KDE_forward_cor1_central_deg_new))*norm_max_forward, color='tab:orange', label='PSI/FORWARD pB')
    plt.plot(gaussian_fit_pB, label='gaussian fit', color='tab:blue')
    plt.yscale('log')
    plt.show()

    if masked:
        return combined_dict, data_dict, mask
    else:
        return combined_dict, data_dict


def create_six_fig_plot(files_z, files_y, outpath, rsun, detector):
    file1_z, file2_z, file3_z, file4_z, file5_z, file6_z = files_z
    file1_y, file2_y, file3_y, file4_y, file5_y, file6_y = files_y
    # outpath = os.path.join(repo_path,'Output/Plots/Test_Vector_Plot.png')


    fits_dir_bz_los_coaligned = file1_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
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

    ny, nz = data_bz_los_coaligned.shape[0],data_bz_los_coaligned.shape[1]
    dy = np.linspace(0, int(ny), ny)
    dz = np.linspace(0, int(nz), nz)
    R_SUN = rsun
    # rsun = (head['rsun'] / head['cdelt1']) * occlt_list[i]
    widths = np.linspace(0,1024,by_los_coaligned_map.data.size)
    skip_val = int(by_los_coaligned_map.data.shape[0]/73.14285714285714)
    skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    skip1 = slice(None, None, skip_val)
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'projection':wcs},figsize =(10, 10))
    # ax1 = plt.subplot(3,2,1,projection=wcs)
    ax1.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax1.set_aspect('equal')
    ax1.add_patch(Circle((int(data_bz_los_coaligned.shape[0]/2),int(data_bz_los_coaligned.shape[0]/2)), R_SUN, color='black',zorder=100))
    # ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax1.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax1.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file2_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file2_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    # R_SUN = head_bz_los_coaligned['R_SUN']

    # ax2 = plt.subplot(3,2,2,projection=wcs)
    ax2.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax2.set_aspect('equal')
    ax2.add_patch(Circle((int(data_bz_los_coaligned.shape[0]/2),int(data_bz_los_coaligned.shape[0]/2)), R_SUN, color='black',zorder=100))
    # ax2.set_title('7.05600_236.978 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax2.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax2.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file3_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file3_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    # R_SUN = head_bz_los_coaligned['R_SUN']

    # ax3 = plt.subplot(3,2,3,projection=wcs)
    ax3.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax3.set_aspect('equal')
    ax3.add_patch(Circle((int(data_bz_los_coaligned.shape[0]/2),int(data_bz_los_coaligned.shape[0]/2)), R_SUN, color='black',zorder=100))
    # ax3.set_title('7.15300_183.443 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax3.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax3.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file4_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file4_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    # R_SUN = head_bz_los_coaligned['R_SUN']

    # ax4 = plt.subplot(3,2,4,projection=wcs)
    ax4.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax4.set_aspect('equal')
    ax4.add_patch(Circle((int(data_bz_los_coaligned.shape[0]/2),int(data_bz_los_coaligned.shape[0]/2)), R_SUN, color='black',zorder=100))
    # ax4.set_title('7.22000_126.906 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax4.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax4.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file5_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file5_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    # R_SUN = head_bz_los_coaligned['R_SUN']

    # ax5 = plt.subplot(3,2,5,projection=wcs)
    ax5.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax5.set_aspect('equal')
    ax5.add_patch(Circle((int(data_bz_los_coaligned.shape[0]/2),int(data_bz_los_coaligned.shape[0]/2)), R_SUN, color='black',zorder=100))
    # ax5.set_title('7.24700_77.0150 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax5.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax5.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file6_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file6_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = (detector)
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    # R_SUN = head_bz_los_coaligned['R_SUN']

    # ax6 = plt.subplot(3,2,6,projection=wcs)
    ax6.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax6.set_aspect('equal')
    ax6.add_patch(Circle((int(data_bz_los_coaligned.shape[0]/2),int(data_bz_los_coaligned.shape[0]/2)), R_SUN, color='black',zorder=100))
    # ax6.set_title('7.23800_11.5530 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax6.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax6.set_ylabel('Helioprojective Latitude (Solar-Y)')
    plt.subplots_adjust(bottom=0.05, top=0.95)

    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.tight_layout()
    # fig.set_constrained_layout_pads(w_pad=1 / 102, h_pad=1 / 102, hspace=0.0,
    #                                 wspace=0.0)
    plt.savefig(outpath)
    # plt.show()
    plt.close()


    return fig



# import os
# from scipy.io import readsav
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import git
# from matplotlib.patches import Circle
# from astropy.wcs import WCS
# from astropy.io import fits
# import sunpy
# import sunpy.map
# import matplotlib
# import numpy as np
# import scipy as sci
# from tqdm import tqdm_notebook
# import pandas as pd
# import unittest
# from pathlib import Path
# from scipy.interpolate import interp1d
# import matplotlib
# from datetime import datetime, timedelta
# from sunpy.sun import constants
# import astropy.constants
# matplotlib.use('TkAgg')
# mpl.use('TkAgg')
# mpl.rcParams.update(mpl.rcParamsDefault)
# from functions import create_six_fig_plot
# from test_plot_qraft import plot_features


# repo = git.Repo('.', search_parent_directories=True)
# repo_path = repo.working_tree_dir

def display_fits_image_with_3_0_features_and_B_field(fits_file, qraft_file, corresponding_By_file=None, corresponding_Bz_file=None, data_type=None, data_source=None, date=None, PSI=True, enhanced=False):
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
    IMG = idl_save['img_d2_phi_r']
    # fname_save, features, angle_err, angle_err_signed, IMG_d2_phi_r, blob_stat, blob_indices, XYCenter, d_phi, rot_angle, phi_shift, smooth_xy, smooth_phi_rho, detr_phi, rho_range, r_rho, p_range, n_p
    d_phi = idl_save['d_phi']
    d_rho = idl_save['d_rho']
    XYCenter = idl_save['XYCenter']
    rot_angle = idl_save['rot_angle']
    phi_shift = idl_save['phi_shift']
    smooth_xy = idl_save['smooth_xy']
    smooth_phi_rho = idl_save['smooth_phi_rho']
    detr_phi = idl_save['detr_phi']
    rho_range = idl_save['rho_range']
    n_rho = idl_save['n_rho']
    p_range = idl_save['p_range']
    n_p = idl_save['n_p']
    n_nodes_min = idl_save['n_nodes_min']
    intensity_removal_coefficient = idl_save['intensity_removal_coef']

    con = sqlite3.connect("tutorial.db")
    cur = con.cursor()

    qraft_data = [(None, float(d_phi), float(d_rho), int(XYCenter[0]), int(XYCenter[1]), float(rot_angle), float(phi_shift), int(smooth_xy), int(smooth_phi_rho[0]), int(smooth_phi_rho[1]), int(detr_phi), int(rho_range[0]), int(rho_range[1]), int(n_rho), float(p_range[0]), float(p_range[1]), int(n_p), int(n_nodes_min), float(intensity_removal_coefficient))]

    cur.executemany("""INSERT OR IGNORE INTO qraft_input_variables VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", qraft_data)
    con.commit()
    query = """SELECT * from qraft_input_variables where 
            d_phi={} and 
            d_rho={} and 
            XYCenter_x={} and 
            XYCenter_y={} and
            rot_angle={} and
            phi_shift={} and
            smooth_xy={} and
            smooth_phi_rho_lower={} and
            smooth_phi_rho_upper={} and
            detr_phi={} and
            rho_range_lower={} and
            rho_range_upper={} and
            n_rho={} and
            p_range_lower={} and
            p_range_upper={} and
            n_p={} and
            n_nodes_min={} and 
            intensity_removal_coefficient={};""".format(float(d_phi), float(d_rho), int(XYCenter[0]), int(XYCenter[1]), float(rot_angle), float(phi_shift), int(smooth_xy), int(smooth_phi_rho[0]), int(smooth_phi_rho[1]), int(detr_phi), int(rho_range[0]), int(rho_range[1]), int(n_rho), float(p_range[0]), float(p_range[1]), int(n_p), int(n_nodes_min), float(intensity_removal_coefficient))

    cur.execute(query)
    row = cur.fetchone()
    (qraft_parameters_id, d_phi_db, d_rho_db, XYCenter_x_db, XYCenter_y_db, rot_angle_db, phi_shift_db, smooth_xy_db, smooth_phi_rho_lower_db, smooth_phi_rho_upper_db, detr_phi_db, rho_range_lower_db, rho_range_upper_db, n_rho_db, p_range_lower_db, p_range_upper_db, n_p_db, n_nodes_min_db, intensity_removal_coefficient_db) = row
    foreign_key = qraft_parameters_id
    # img_enh = idl_save['img_enh']
    FEATURES = idl_save['features']
    # P = idl_save['P']
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
            file1_y = corresponding_By_file
            file1_z = corresponding_Bz_file

        elif detector == 'COR1':
            file1_y = corresponding_By_file
            file1_z = corresponding_Bz_file






    fits_dir_bz_los_coaligned = file1_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    if PSI:
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


    angles_arr = np.array(angles)
    angles_arr_finite = angles_arr[~np.isnan(angles_arr)]*180/np.pi
    angles_arr_mean = np.round(np.mean(angles_arr[~np.isnan(angles_arr)])*180/np.pi,5)
    angles_arr_median = np.round(np.median(angles_arr[~np.isnan(angles_arr)])*180/np.pi,5)
    std = np.round(np.std(abs(angles_arr_finite)),5)
    confidence_interval = np.round(1.96 * (std / np.sqrt(len(angles_arr_finite))),5)
    n = len(angles_arr_finite)
    angles_signed_arr = np.array(angles_signed)
    angles_signed_arr_finite = angles_signed_arr[~np.isnan(angles_signed_arr)]*180/np.pi

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

    colors = plt.cm.jet(np.linspace(0, 1, len(FEATURES)))
    # for i, feature in enumerate(FEATURES):
        # axes.plot(feature['xx_r'][:feature['n_nodes']], feature['yy_r'][:feature['n_nodes']], color=colors[i], linewidth=3)
    # Scatter plot for angle errors
    norm = mpl.colors.Normalize(vmin=-90, vmax=90)
    sc = axes.scatter(angles_xx_positions, angles_yy_positions, c=np.degrees(angles_signed), cmap='coolwarm', label=False, norm=norm)
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
        if data_type:
            axes.set_title('PSI/FORWARD {} Eclipse Model Corresponding to {} {} Observation'.format(data_type, date, data_source.strip('_PSI')))
            plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Angle_Error_{}_{}_{}_PSI.png'.format(string_print, detector, data_type)))
        else:
            axes.set_title('Corresponding PSI/FORWARD pB Eclipse Model')
            plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Angle_Error_{}_{}_PSI.png'.format(string_print, detector)))
    else:
        axes.set_title('{} Observation {}'.format(detector, str_strip))
        if data_type:
            plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Angle_Error_{}_{}_{}.png'.format(string_print, detector, data_type)))
        else:
            plt.savefig(os.path.join(repo_path,'Output/Plots/Features_Angle_Error_{}_{}.png'.format(string_print, detector)))
    # plt.show()
    plt.close()


    return angles_signed_arr_finite, angles_arr_finite, angles_arr_mean, angles_arr_median, std, confidence_interval, n, foreign_key



def get_files_from_pattern(directory, pattern):
    # Use glob to get all files with the '.fits' extension in the specified directory
    fits_files = glob.glob(f"{directory}/*{pattern}")
    return sorted(fits_files)


def determine_paths(fits_file, PSI=True):

    filename = fits_file.split('/')[-1]

    data = fits.getdata(fits_file)
    head = fits.getheader(fits_file)

    telescope = head['telescop']
    instrument = head['instrume']

    if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
        head['detector'] = ('KCor')

    detector = head['detector']
    if PSI:
        if detector == 'KCor':
            if 'KCor' in filename:
                keyword = 'KCor__PSI'
                keyword_By = 'KCor__PSI_By.fits'
                keyword_Bz = 'KCor__PSI_Bz.fits'
                file1_y = os.path.join(repo_path, 'Output/fits_images/' + filename.split('KCor')[0] + keyword_By)
                file1_z = os.path.join(repo_path, 'Output/fits_images/' + filename.split('KCor')[0] + keyword_Bz)

        elif detector == 'COR1':
            if 'COR1' in filename:
                keyword = 'COR1__PSI'
                keyword_By = 'COR1__PSI_By.fits'
                keyword_Bz = 'COR1__PSI_Bz.fits'
                file1_y = os.path.join(repo_path, 'Output/fits_images/' + filename.split('COR1')[0] + keyword_By)
                file1_z = os.path.join(repo_path, 'Output/fits_images/' + filename.split('COR1')[0] + keyword_Bz)
    
    date_obs = head['date-obs']
    date = date_obs.split('T',1)[0]
    string_print = date_obs.split('T')[0].replace('-','_')
    data_type = filename.split('_')[-1].strip('.fits')
    if data_type == 'LOS':
        data_type = 'ne_LOS'
    if data_type == 'med':
        data_type = 'COR1 median filtered'
    if data_type == 'avg':
        data_type = 'KCor l2 avg'
    if PSI:
        data_source = keyword
    else:
        data_source = detector

    return data_source, date, data_type



def print_sql_query(dbName, query):
    con = sqlite3.connect(dbName)
    cur = con.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    column_names = [description[0] for description in cur.description]

    table = PrettyTable()
    table.field_names = column_names
    for row in rows:
        table.add_row(row)
    print(table)
    con.close()


def plot_sql_query(dbName, query, parameter_x, parameter_y, title=None, xlabel=None, ylabel=None):
    con = sqlite3.connect(dbName)
    df = pd.read_sql_query(query, con)
    fig, ax = plt.subplots()
    ax.plot(df[parameter_x], df[parameter_y])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # plt.savefig(outpath)
    plt.show()

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
            mask = 50
        elif detector == 'K-COR':
            mask = 25
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

    combined_dict = dict(metric=['KL Divergence', 'JS Divergence'],
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


    combined_dict = dict(metric=['average discrepancy', 'median discrepancy'],
                        cor1=[cor1_avg, cor1_med],
                       psi=[forward_avg, forward_med],
                        random=[random_avg, random_med])


    pd.set_option('display.float_format', '{:.3f}'.format)
    accuracy_stats_df = pd.DataFrame(combined_dict)
    accuracy_stats_df.columns = ['metric', '{}'.format(detector), 'psi'.format(detector), 'random']
    #stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
    print(accuracy_stats_df.to_latex(index=False))
    file.write(accuracy_stats_df.to_latex(index=False))


    pd.set_option('display.float_format', '{:.3E}'.format)
    stats_df = pd.DataFrame(combined_dict)
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

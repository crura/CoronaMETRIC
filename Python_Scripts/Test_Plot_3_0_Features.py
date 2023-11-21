
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
from scipy.stats import gaussian_kde
from test_plot_qraft import plot_features
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functions import display_fits_image_with_3_0_features_and_B_field
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from functions import calculate_KDE_statistics
import glob
import sqlite3
con = sqlite3.connect("tutorial.db")

cur = con.cursor()

# cur.execute("CREATE TABLE stats(comparison, data_source, date, JSD, KLD)")

# cur.execute("CREATE TABLE stats2_new(data_type, data_source, date, mean, median, confidence interval, n)")

cur.execute("DROP TABLE IF EXISTS stats3_new")

cur.execute("CREATE TABLE IF NOT EXISTS stats3_new(data_type, data_source, date, mean, median, confidence interval, n)")


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

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
    data_source = keyword

    return data_source, date, data_type

fits_path = os.path.join(repo_path, 'QRaFT/3.0_PSI_Tests')
fits_files_pB = get_files_from_pattern(fits_path, 'COR1__PSI_pB.fits')
fits_files_ne = get_files_from_pattern(fits_path, 'COR1__PSI_ne.fits')
fits_files_ne_LOS = get_files_from_pattern(fits_path, 'COR1__PSI_ne_LOS.fits')

combined_pB = []
combined_ne = []
combined_ne_LOS = []

for i in range(len(fits_files_pB)-1):
    data_stats_2 = []

    file_pB = fits_files_pB[i]
    data_source, date, data_type = determine_paths(file_pB)
    angles_arr_finite_pB, angles_arr_mean_pB, angles_arr_median_pB, confidence_interval_pB, n_pB = display_fits_image_with_3_0_features_and_B_field(file_pB, file_pB+'.sav')
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_pB, angles_arr_median_pB, confidence_interval_pB, n_pB))

    file_ne = fits_files_ne[i]
    data_source, date, data_type = determine_paths(file_ne)
    angles_arr_finite_ne, angles_arr_mean_ne, angles_arr_median_ne, confidence_interval_ne, n_ne = display_fits_image_with_3_0_features_and_B_field(file_ne, file_ne+'.sav')
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_ne, angles_arr_median_ne, confidence_interval_ne, n_ne))

    file_ne_LOS = fits_files_ne_LOS[i]
    data_source, date, data_type = determine_paths(file_ne_LOS)
    angles_arr_finite_ne_LOS, angles_arr_mean_ne_LOS, angles_arr_median_ne_LOS, confidence_interval_ne_LOS, n_ne_LOS = display_fits_image_with_3_0_features_and_B_field(file_ne_LOS, file_ne_LOS+'.sav')
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_ne_LOS, angles_arr_median_ne_LOS, confidence_interval_ne_LOS, n_ne_LOS))

    cur.executemany("INSERT INTO stats3_new VALUES(?, ?, ?, ?, ?, ?, ?)", data_stats_2)
    con.commit()  # Remember to commit the transaction after executing INSERT.

    # retrieve probability density data from seaborne distplots
    x_dist_values_pB = sns.distplot(angles_arr_finite_pB).get_lines()[0].get_data()[0]
    xmin_pB = x_dist_values_pB.min()
    xmax_pB = x_dist_values_pB.max()
    # plt.show()
    plt.close()

    kde0_pB = gaussian_kde(angles_arr_finite_pB)
    x_1_pB = np.linspace(xmin_pB, xmax_pB, 200)
    kde0_x_pB = kde0_pB(x_1_pB)

    # retrieve probability density data from seaborne distplots
    x_dist_values_ne = sns.distplot(angles_arr_finite_ne).get_lines()[0].get_data()[0]
    xmin_ne = x_dist_values_ne.min()
    xmax_ne = x_dist_values_ne.max()
    # plt.show()
    plt.close()

    kde0_ne = gaussian_kde(angles_arr_finite_ne)
    x_1_ne = np.linspace(xmin_ne, xmax_ne, 200)
    kde0_x_ne = kde0_ne(x_1_ne)



    # retrieve probability density data from seaborne distplots
    x_dist_values_ne_LOS = sns.distplot(angles_arr_finite_ne_LOS).get_lines()[0].get_data()[0]
    xmin_ne_LOS = x_dist_values_ne_LOS.min()
    xmax_ne_LOS = x_dist_values_ne_LOS.max()
    # plt.show()
    plt.close()

    kde0_ne_LOS = gaussian_kde(angles_arr_finite_ne_LOS)
    x_1_ne_LOS = np.linspace(xmin_ne_LOS, xmax_ne_LOS, 200)
    kde0_x_ne_LOS = kde0_ne_LOS(x_1_ne_LOS)

    plt.plot(x_1_ne_LOS, kde0_x_ne_LOS, color='g', label='ne LOS KDE')
    plt.plot(x_1_ne, kde0_x_ne, color='b', label='ne KDE')
    plt.plot(x_1_pB, kde0_x_pB, color='r', label='pB KDE')
    plt.legend()
    plt.show()


    #compute JS Divergence

    data_source, date, data_type = determine_paths(file_pB)

    JSD_cor1_psi_pB_ne, KLD_cor1_psi_pB_ne = calculate_KDE_statistics(kde0_x_pB, kde0_x_ne_LOS)
    JSD_cor1_psi_pB_ne_LOS, KLD_cor1_psi_pB_ne_LOS = calculate_KDE_statistics(kde0_x_pB, kde0_x_ne)
    JSD_cor1_psi_ne_ne_LOS, KLD_cor1_psi_ne_ne_LOS = calculate_KDE_statistics(kde0_x_ne, kde0_x_ne_LOS)

    data = [
        ("pB vs ne", data_source, date, JSD_cor1_psi_pB_ne, KLD_cor1_psi_pB_ne),
        ("pB vs ne_LOS", data_source, date, JSD_cor1_psi_pB_ne_LOS, KLD_cor1_psi_pB_ne_LOS),
        ("ne vs ne_LOS", data_source, date, JSD_cor1_psi_ne_ne_LOS, KLD_cor1_psi_ne_ne_LOS),
    ]

    # cur.executemany("INSERT INTO stats VALUES(?, ?, ?, ?, ?)", data)
    # con.commit()  # Remember to commit the transaction after executing INSERT.

    # JSD_cor1_central_random_new, KLD_cor1_central_random_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_random_deg_new)
    # JSD_COR1_Forward_Central_Random_new, KLDcor1_forward_central_random_new = calculate_KDE_statistics(KDE_forward_cor1_central_deg_new, KDE_random_deg_new)

    # combined_dict = dict(metric=['KL Divergence', 'JS Divergence'],
    #                     cor1_v_psi=[KLD_cor1_forward_central_new, JSD_cor1_forward_central_new],
    #                     cor1_v_random=[KLD_cor1_central_random_new, JSD_cor1_central_random_new],
    #                     psi_v_random=[KLDcor1_forward_central_random_new, JSD_COR1_Forward_Central_Random_new])

    # pd.set_option('display.float_format', '{:.3E}'.format)
    # stats_df = pd.DataFrame(combined_dict)
    # stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
    # print(stats_df.to_latex(index=False))

    combined_pB.append(angles_arr_finite_pB)
    combined_ne.append(angles_arr_finite_ne)
    combined_ne_LOS.append(angles_arr_finite_ne_LOS)

data_stats_2_combined = []

combined_pB_ravel = [item for sublist in combined_pB for item in sublist]
combined_ne_ravel = [item for sublist in combined_ne for item in sublist]
combined_ne_LOS_ravel = [item for sublist in combined_ne_LOS for item in sublist]

combined_pB_ravel_arr = np.array(combined_pB_ravel)
angles_arr_mean_pB_combined = np.round(np.mean(combined_pB_ravel_arr), 5)
angles_arr_median_pB_combined = np.round(np.median(combined_pB_ravel_arr), 5)
n_pB_combined = len(combined_pB_ravel_arr)
std_pB_combined = np.round(np.std(abs(combined_pB_ravel_arr)),5)
confidence_interval_pB_combined = np.round(1.96 * (std_pB_combined / np.sqrt(len(combined_pB_ravel_arr))),5)
data_type_pB_combined = 'pB'
date_combined = 'combined'
data_stats_2_combined.append((data_type_pB_combined, data_source, date_combined, angles_arr_mean_pB_combined, angles_arr_median_pB_combined, confidence_interval_pB_combined, n_pB_combined))

combined_ne_ravel_arr = np.array(combined_ne_ravel)
angles_arr_mean_ne_combined = np.round(np.mean(combined_ne_ravel_arr), 5)
angles_arr_median_ne_combined = np.round(np.median(combined_ne_ravel_arr), 5)
n_ne_combined = len(combined_ne_ravel_arr)
std_ne_combined = np.round(np.std(abs(combined_ne_ravel_arr)),5)
confidence_interval_ne_combined = np.round(1.96 * (std_ne_combined / np.sqrt(len(combined_ne_ravel_arr))),5)
data_type_ne_combined = 'ne'
date_combined = 'combined'
data_stats_2_combined.append((data_type_ne_combined, data_source, date_combined, angles_arr_mean_ne_combined, angles_arr_median_ne_combined, confidence_interval_ne_combined, n_ne_combined))

combined_ne_LOS_ravel_arr = np.array(combined_ne_LOS_ravel)
angles_arr_mean_ne_LOS_combined = np.round(np.mean(combined_ne_LOS_ravel_arr), 5)
angles_arr_median_ne_LOS_combined = np.round(np.median(combined_ne_LOS_ravel_arr), 5)
n_ne_LOS_combined = len(combined_ne_LOS_ravel_arr)
std_ne_LOS_combined = np.round(np.std(abs(combined_ne_LOS_ravel_arr)),5)
confidence_interval_ne_LOS_combined = np.round(1.96 * (std_ne_LOS_combined / np.sqrt(len(combined_ne_LOS_ravel_arr))),5)
data_type_ne_LOS_combined = 'ne_LOS'
date_combined = 'combined'
data_stats_2_combined.append((data_type_ne_LOS_combined, data_source, date_combined, angles_arr_mean_ne_LOS_combined, angles_arr_median_ne_LOS_combined, confidence_interval_ne_LOS_combined, n_ne_LOS_combined))



cur.executemany("INSERT INTO stats3_new VALUES(?, ?, ?, ?, ?, ?, ?)", data_stats_2_combined)
con.commit()  # Remember to commit the transaction after executing INSERT.


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
# from test_plot_qraft import plot_features
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functions import display_fits_image_with_3_0_features_and_B_field
from scipy.stats import norm
from matplotlib import pyplot as plt
import seaborn as sns
from functions import calculate_KDE_statistics, determine_paths, get_files_from_pattern
import sqlite3
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import tukey_hsd


con = sqlite3.connect("tutorial.db")

cur = con.cursor()

# cur.execute("CREATE TABLE stats(comparison, data_source, date, JSD, KLD)")

# cur.execute("CREATE TABLE stats2_new(data_type, data_source, date, mean, median, confidence interval, n)")

cur.execute("DROP TABLE IF EXISTS central_tendency_stats_cor1_new")

cur.execute("CREATE TABLE IF NOT EXISTS central_tendency_stats_cor1_new(data_type, data_source, date, mean, median, confidence interval, n)")

cur.execute("DROP TABLE IF EXISTS central_tendency_stats_kcor_new")

cur.execute("CREATE TABLE IF NOT EXISTS central_tendency_stats_kcor_new(data_type, data_source, date, mean, median, confidence interval, n)")


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir




fits_path = os.path.join(repo_path, 'Output/QRaFT_Results')
fits_files_pB = get_files_from_pattern(fits_path, 'COR1__PSI_pB.fits')
fits_files_ne = get_files_from_pattern(fits_path, 'COR1__PSI_ne.fits')
fits_files_ne_LOS = get_files_from_pattern(fits_path, 'COR1__PSI_ne_LOS.fits')

combined_pB = []
combined_ne = []
combined_ne_LOS = []

combined_pB_signed = []
combined_ne_signed = []
combined_ne_signed_LOS = []

for i in range(len(fits_files_pB)):
    data_stats_2 = []

    file_pB = fits_files_pB[i]
    data_source, date, data_type = determine_paths(file_pB)
    angles_signed_arr_finite_pB, angles_arr_finite_pB, angles_arr_mean_pB, angles_arr_median_pB, confidence_interval_pB, n_pB = display_fits_image_with_3_0_features_and_B_field(file_pB, file_pB+'.sav', data_type=data_type, data_source=data_source, date=date)
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_pB, angles_arr_median_pB, confidence_interval_pB, n_pB))

    file_ne = fits_files_ne[i]
    data_source, date, data_type = determine_paths(file_ne)
    angles_signed_arr_finite_ne, angles_arr_finite_ne, angles_arr_mean_ne, angles_arr_median_ne, confidence_interval_ne, n_ne = display_fits_image_with_3_0_features_and_B_field(file_ne, file_ne+'.sav', data_type=data_type, data_source=data_source, date=date)
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_ne, angles_arr_median_ne, confidence_interval_ne, n_ne))

    file_ne_LOS = fits_files_ne_LOS[i]
    data_source, date, data_type = determine_paths(file_ne_LOS)
    angles_signed_arr_finite_ne_LOS, angles_arr_finite_ne_LOS, angles_arr_mean_ne_LOS, angles_arr_median_ne_LOS, confidence_interval_ne_LOS, n_ne_LOS = display_fits_image_with_3_0_features_and_B_field(file_ne_LOS, file_ne_LOS+'.sav', data_type=data_type, data_source=data_source, date=date)
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_ne_LOS, angles_arr_median_ne_LOS, confidence_interval_ne_LOS, n_ne_LOS))

    cur.executemany("INSERT INTO central_tendency_stats_cor1_new VALUES(?, ?, ?, ?, ?, ?, ?)", data_stats_2)
    con.commit()  # Remember to commit the transaction after executing INSERT.

    # retrieve probability density data from seaborne distplots
    x_dist_values_pB = sns.distplot(angles_signed_arr_finite_pB).get_lines()[0].get_data()[0]
    xmin_pB = x_dist_values_pB.min()
    xmax_pB = x_dist_values_pB.max()
    # plt.show()
    plt.close()

    kde0_pB = gaussian_kde(angles_signed_arr_finite_pB)
    x_1_pB = np.linspace(xmin_pB, xmax_pB, 200)
    kde0_x_pB = kde0_pB(x_1_pB)

    # retrieve probability density data from seaborne distplots
    x_dist_values_ne = sns.distplot(angles_signed_arr_finite_ne).get_lines()[0].get_data()[0]
    xmin_ne = x_dist_values_ne.min()
    xmax_ne = x_dist_values_ne.max()
    # plt.show()
    plt.close()

    kde0_ne = gaussian_kde(angles_signed_arr_finite_ne)
    x_1_ne = np.linspace(xmin_ne, xmax_ne, 200)
    kde0_x_ne = kde0_ne(x_1_ne)



    # retrieve probability density data from seaborne distplots
    x_dist_values_ne_LOS = sns.distplot(angles_signed_arr_finite_ne_LOS).get_lines()[0].get_data()[0]
    xmin_ne_LOS = x_dist_values_ne_LOS.min()
    xmax_ne_LOS = x_dist_values_ne_LOS.max()
    # plt.show()
    plt.close()

    kde0_ne_LOS = gaussian_kde(angles_signed_arr_finite_ne_LOS)
    x_1_ne_LOS = np.linspace(xmin_ne_LOS, xmax_ne_LOS, 200)
    kde0_x_ne_LOS = kde0_ne_LOS(x_1_ne_LOS)

    plt.plot(x_1_ne_LOS, kde0_x_ne_LOS, color='g', label='ne LOS KDE')
    plt.plot(x_1_ne, kde0_x_ne, color='b', label='ne KDE')
    plt.plot(x_1_pB, kde0_x_pB, color='r', label='pB KDE')
    plt.legend()
    # plt.show()
    plt.close()


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


    combined_pB_signed.append(angles_signed_arr_finite_pB)
    combined_ne_signed.append(angles_signed_arr_finite_ne)
    combined_ne_signed_LOS.append(angles_signed_arr_finite_ne_LOS)

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
data_source = 'COR1_PSI'
data_stats_2_combined.append((data_type_pB_combined, data_source, date_combined, angles_arr_mean_pB_combined, angles_arr_median_pB_combined, confidence_interval_pB_combined, n_pB_combined))

combined_ne_ravel_arr = np.array(combined_ne_ravel)
angles_arr_mean_ne_combined = np.round(np.mean(combined_ne_ravel_arr), 5)
angles_arr_median_ne_combined = np.round(np.median(combined_ne_ravel_arr), 5)
n_ne_combined = len(combined_ne_ravel_arr)
std_ne_combined = np.round(np.std(abs(combined_ne_ravel_arr)),5)
confidence_interval_ne_combined = np.round(1.96 * (std_ne_combined / np.sqrt(len(combined_ne_ravel_arr))),5)
data_type_ne_combined = 'ne'
date_combined = 'combined'
data_source = 'COR1_PSI'
data_stats_2_combined.append((data_type_ne_combined, data_source, date_combined, angles_arr_mean_ne_combined, angles_arr_median_ne_combined, confidence_interval_ne_combined, n_ne_combined))

combined_ne_LOS_ravel_arr = np.array(combined_ne_LOS_ravel)
angles_arr_mean_ne_LOS_combined = np.round(np.mean(combined_ne_LOS_ravel_arr), 5)
angles_arr_median_ne_LOS_combined = np.round(np.median(combined_ne_LOS_ravel_arr), 5)
n_ne_LOS_combined = len(combined_ne_LOS_ravel_arr)
std_ne_LOS_combined = np.round(np.std(abs(combined_ne_LOS_ravel_arr)),5)
confidence_interval_ne_LOS_combined = np.round(1.96 * (std_ne_LOS_combined / np.sqrt(len(combined_ne_LOS_ravel_arr))),5)
data_type_ne_LOS_combined = 'ne_LOS'
date_combined = 'combined'
data_source = 'COR1_PSI'
data_stats_2_combined.append((data_type_ne_LOS_combined, data_source, date_combined, angles_arr_mean_ne_LOS_combined, angles_arr_median_ne_LOS_combined, confidence_interval_ne_LOS_combined, n_ne_LOS_combined))



cur.executemany("INSERT INTO central_tendency_stats_cor1_new VALUES(?, ?, ?, ?, ?, ?, ?)", data_stats_2_combined)
con.commit()  # Remember to commit the transaction after executing INSERT.


combined_pB_signed_ravel = [item for sublist in combined_pB_signed for item in sublist]
combined_ne_signed_ravel = [item for sublist in combined_ne_signed for item in sublist]
combined_ne_signed_LOS_ravel = [item for sublist in combined_ne_signed_LOS for item in sublist]

combined_pB_signed_ravel_arr = np.array(combined_pB_signed_ravel)
combined_ne_signed_ravel_arr = np.array(combined_ne_signed_ravel)
combined_ne_signed_LOS_ravel_arr = np.array(combined_ne_signed_LOS_ravel)

print(combined_ne_signed_ravel_arr)

query = "SELECT mean, median, date, data_type, data_source, n, confidence FROM central_tendency_stats_cor1_new WHERE date!='combined' ORDER BY mean ASC;"
cur.execute(query)
rows = cur.fetchall()
print(rows)

what = sns.histplot(combined_ne_signed_ravel_arr,kde=True, bins=30)
print(what.get_lines()[0].get_data()[0])
print(what.get_lines()[0].get_data()[1])
norm_max_ne = max(what.get_lines()[0].get_data()[1])
plt.close()

what2 = sns.histplot(combined_pB_signed_ravel,kde=True, bins=30)
norm_max_pB = max(what2.get_lines()[0].get_data()[1])
plt.close()

what3 = sns.histplot(combined_ne_signed_LOS_ravel,kde=True, bins=30)
norm_max_ne_los = max(what3.get_lines()[0].get_data()[1])
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.subplots(1,1)
sns.histplot(combined_ne_signed_ravel_arr,kde=True,label='ne',bins=30,ax=ax,color='tab:blue')
sns.histplot(combined_pB_signed_ravel,kde=True,label='pB',bins=30,ax=ax,color='tab:orange')
sns.histplot(combined_ne_signed_LOS_ravel,kde=True, bins=30, label='ne_LOS',ax=ax, color='tab:green')
#x_axis = np.linspace(-90, 90, len(KDE_cor1_central_deg_new))


# plt.plot(x_1_cor1_central_deg_new, (KDE_cor1_central_deg_new/max(KDE_cor1_central_deg_new))*norm_max_cor1, color='tab:blue')
# plt.plot(x_1_random_deg_new, (KDE_random_deg_new/max(KDE_random_deg_new))*norm_max_random, color='tab:green', label='random')
# plt.plot(x_1_forward_cor1_central_deg_new, (KDE_forward_cor1_central_deg_new/max(KDE_forward_cor1_central_deg_new))*norm_max_forward, color='tab:orange')
# norm_kde_random = (KDE_random_deg_new/max(KDE_random_deg_new))*norm_max_random
# norm_kde_forward = (KDE_forward_cor1_central_deg_new/max(KDE_forward_cor1_central_deg_new))*norm_max_forward
# norm_kde_cor1 = (KDE_cor1_central_deg_new/max(KDE_cor1_central_deg_new))*norm_max_cor1


#sns.kdeplot()
ax.set_xlabel('Angle Discrepancy (Degrees)',fontsize=14)
ax.set_ylabel('Pixel Count',fontsize=14)
detector = 'COR1_PSI'
ax.set_title('QRaFT {} Feature Tracing Performance Against Central POS $B$ Field'.format(detector),fontsize=15)
ax.set_xlim(-95,95)
#ax.set_ylim(0,0.07)
ax.legend(fontsize=13)

# plt.text(20,0.045,"COR1 average discrepancy: " + str(np.round(np.average(err_cor1_central_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_cor1_central_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
plt.savefig(os.path.join(repo_path,'Output/Plots/Updated_{}_vs_FORWARD_Feature_Tracing_Performance.png'.format(detector.replace('-',''))))
# plt.show()
#plt.close()

query = "SELECT mean, median, date, data_type, data_source, n, confidence FROM central_tendency_stats_cor1_new WHERE date!='combined' ORDER BY mean ASC;"
cur.execute(query)
rows = cur.fetchall()

# Close the cursor and the connection
# cur.close()
# con.close()

# Process the data for plotting
data_by_date = {}  # Dictionary to store data by date

for row in rows:
    mean, median, date, data_type, data_source, n, confidence = row
    if date not in data_by_date:
        data_by_date[date] = {'mean': [], 'confidence': [], 'data_type': []}
    data_by_date[date]['mean'].append(mean)
    data_by_date[date]['confidence'].append(confidence)
    data_by_date[date]['data_type'].append(data_type)

# Plot the scatter plot with error bars by date
dates = sorted(list(data_by_date.keys()))
data_types = sorted(list(set(data_by_date[dates[0]]['data_type'])))  # Assuming data types are consistent across dates

fig = plt.figure(figsize=(8, 8))
# Create a scatter plot for each date
for i, date in enumerate(dates):
    data_to_plot = [data_by_date[date]['mean'][j] for j in range(len(data_by_date[date]['data_type']))]
    confidence_to_plot = [data_by_date[date]['confidence'][j] for j in range(len(data_by_date[date]['data_type']))]
    data_type_to_plot = [data_by_date[date]['data_type'][j] for j in range(len(data_by_date[date]['data_type']))]
    for j in range(len(data_to_plot)):
        if data_type_to_plot[j] == data_types[0]:
            plt.errorbar(x=[i], y=data_to_plot[j], yerr=confidence_to_plot[j], fmt='o', color='C0' ,label=data_type_to_plot[j] if i == 0 else "")
        elif data_type_to_plot[j] == data_types[1]:
            plt.errorbar(x=[i], y=data_to_plot[j], yerr=confidence_to_plot[j], fmt='o', color='C2' ,label=data_type_to_plot[j] if i == 0 else "")
        elif data_type_to_plot[j] == data_types[2]:
            plt.errorbar(x=[i], y=data_to_plot[j], yerr=confidence_to_plot[j], fmt='o', color='C1' ,label=data_type_to_plot[j] if i == 0 else "")

# Customize the plot
plt.xlabel('Date of Corresponding Observation')
plt.ylabel('Mean Value (Degrees)')
plt.title('PSI COR-1 Projection Angle Discrepancy by Date')
plt.legend()
plt.ylim(0,20)

# Set x-axis ticks and labels
plt.xticks(range(len(dates)), dates)
plt.savefig(os.path.join(repo_path, 'Output/Plots', '{}_Angle_Discrepancy_By_Date.png'.format(data_type)))
plt.show()

# Combine data into a single array
all_data = np.concatenate([combined_ne_signed_ravel_arr, combined_ne_signed_LOS_ravel_arr, combined_pB_signed_ravel_arr])

# Create labels for the data types
labels = ['ne'] * len(combined_ne_signed_ravel_arr) + ['ne_LOS'] * len(combined_ne_signed_LOS_ravel_arr) + ['pB'] * len(combined_pB_signed_ravel_arr)

# Perform Tukey's HSD post-hoc test
tukey_result = pairwise_tukeyhsd(all_data, labels)
print(tukey_result)


f_statistic, p_value = f_oneway(combined_ne_signed_ravel_arr, combined_ne_signed_LOS_ravel_arr, combined_pB_signed_ravel_arr)
# Check for statistical significance
if p_value < 0.05:
    print("There are significant differences between at least two data types.")
else:
    print("No significant differences detected between data types.")


res = tukey_hsd(combined_ne_signed_ravel_arr, combined_ne_signed_LOS_ravel_arr, combined_pB_signed_ravel_arr)
print(res)


# Combine data into a single array
all_data = np.concatenate([combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr])

# Create labels for the data types
labels = ['ne'] * len(combined_ne_ravel_arr) + ['ne_LOS'] * len(combined_ne_LOS_ravel_arr) + ['pB'] * len(combined_pB_ravel_arr)

# Perform Tukey's HSD post-hoc test
tukey_result = pairwise_tukeyhsd(all_data, labels)
print(tukey_result)
fig, ax = plt.subplots(1, 1)
# ax.boxplot([combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr], showfliers=False)
# ax.set_xticklabels(["ne", "ne_LOS", "pB"]) 
ax.set_xlabel("Mean (Degrees)") 
ax.set_ylabel("Data Type") 
ax.set_title('HSD Comparison of Data Types for PSI_COR1 Combined Results')
tukey_result.plot_simultaneous(xlabel='Mean (Degrees)', ax=ax)
plt.savefig(os.path.join(repo_path,'Output/Plots/testfig1.png'))
plt.show()

f_statistic, p_value = f_oneway(combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr)
# Check for statistical significance
if p_value < 0.05:
    print("There are significant differences between at least two data types.")
else:
    print("No significant differences detected between data types.")


fig, ax = plt.subplots(1, 1)
ax.boxplot([combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr], showfliers=False)
ax.set_xticklabels(["ne", "ne_LOS", "pB"]) 
ax.set_ylim(0, 40)
ax.set_ylabel("Mean (Degrees)") 
ax.set_xlabel("Data Type") 
ax.set_title('Box Plot Comparison of Data Types for PSI_COR1 Combined Results')
plt.savefig(os.path.join(repo_path, 'Output/Plots/testfig2.png'))
plt.show()

res = tukey_hsd(combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr)
print(res)


fits_path = os.path.join(repo_path, 'Output/QRaFT_Results')
fits_files_pB = get_files_from_pattern(fits_path, 'KCor__PSI_pB.fits')
fits_files_ne = get_files_from_pattern(fits_path, 'KCor__PSI_ne.fits')
fits_files_ne_LOS = get_files_from_pattern(fits_path, 'KCor__PSI_ne_LOS.fits')

combined_pB = []
combined_ne = []
combined_ne_LOS = []

combined_pB_signed = []
combined_ne_signed = []
combined_ne_signed_LOS = []

for i in range(len(fits_files_pB)):
    data_stats_2 = []

    file_pB = fits_files_pB[i]
    data_source, date, data_type = determine_paths(file_pB)
    angles_signed_arr_finite_pB, angles_arr_finite_pB, angles_arr_mean_pB, angles_arr_median_pB, confidence_interval_pB, n_pB = display_fits_image_with_3_0_features_and_B_field(file_pB, file_pB+'.sav', data_type=data_type, data_source=data_source, date=date)
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_pB, angles_arr_median_pB, confidence_interval_pB, n_pB))

    file_ne = fits_files_ne[i]
    data_source, date, data_type = determine_paths(file_ne)
    angles_signed_arr_finite_ne, angles_arr_finite_ne, angles_arr_mean_ne, angles_arr_median_ne, confidence_interval_ne, n_ne = display_fits_image_with_3_0_features_and_B_field(file_ne, file_ne+'.sav', data_type=data_type, data_source=data_source, date=date)
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_ne, angles_arr_median_ne, confidence_interval_ne, n_ne))

    file_ne_LOS = fits_files_ne_LOS[i]
    data_source, date, data_type = determine_paths(file_ne_LOS)
    angles_signed_arr_finite_ne_LOS, angles_arr_finite_ne_LOS, angles_arr_mean_ne_LOS, angles_arr_median_ne_LOS, confidence_interval_ne_LOS, n_ne_LOS = display_fits_image_with_3_0_features_and_B_field(file_ne_LOS, file_ne_LOS+'.sav', data_type=data_type, data_source=data_source, date=date)
    data_stats_2.append((data_type, data_source, date, angles_arr_mean_ne_LOS, angles_arr_median_ne_LOS, confidence_interval_ne_LOS, n_ne_LOS))

    cur.executemany("INSERT INTO central_tendency_stats_kcor_new VALUES(?, ?, ?, ?, ?, ?, ?)", data_stats_2)
    con.commit()  # Remember to commit the transaction after executing INSERT.

    # retrieve probability density data from seaborne distplots
    x_dist_values_pB = sns.distplot(angles_signed_arr_finite_pB).get_lines()[0].get_data()[0]
    xmin_pB = x_dist_values_pB.min()
    xmax_pB = x_dist_values_pB.max()
    # plt.show()
    plt.close()

    kde0_pB = gaussian_kde(angles_signed_arr_finite_pB)
    x_1_pB = np.linspace(xmin_pB, xmax_pB, 200)
    kde0_x_pB = kde0_pB(x_1_pB)

    # retrieve probability density data from seaborne distplots
    x_dist_values_ne = sns.distplot(angles_signed_arr_finite_ne).get_lines()[0].get_data()[0]
    xmin_ne = x_dist_values_ne.min()
    xmax_ne = x_dist_values_ne.max()
    # plt.show()
    plt.close()

    kde0_ne = gaussian_kde(angles_signed_arr_finite_ne)
    x_1_ne = np.linspace(xmin_ne, xmax_ne, 200)
    kde0_x_ne = kde0_ne(x_1_ne)



    # retrieve probability density data from seaborne distplots
    x_dist_values_ne_LOS = sns.distplot(angles_signed_arr_finite_ne_LOS).get_lines()[0].get_data()[0]
    xmin_ne_LOS = x_dist_values_ne_LOS.min()
    xmax_ne_LOS = x_dist_values_ne_LOS.max()
    # plt.show()
    plt.close()

    kde0_ne_LOS = gaussian_kde(angles_signed_arr_finite_ne_LOS)
    x_1_ne_LOS = np.linspace(xmin_ne_LOS, xmax_ne_LOS, 200)
    kde0_x_ne_LOS = kde0_ne_LOS(x_1_ne_LOS)

    plt.plot(x_1_ne_LOS, kde0_x_ne_LOS, color='g', label='ne LOS KDE')
    plt.plot(x_1_ne, kde0_x_ne, color='b', label='ne KDE')
    plt.plot(x_1_pB, kde0_x_pB, color='r', label='pB KDE')
    plt.legend()
    # plt.show()
    plt.close()


    #compute JS Divergence

    data_source, date, data_type = determine_paths(file_pB)

    JSD_kcor_psi_pB_ne, KLD_kcor_psi_pB_ne = calculate_KDE_statistics(kde0_x_pB, kde0_x_ne_LOS)
    JSD_kcor_psi_pB_ne_LOS, KLD_kcor_psi_pB_ne_LOS = calculate_KDE_statistics(kde0_x_pB, kde0_x_ne)
    JSD_kcor_psi_ne_ne_LOS, KLD_kcor_psi_ne_ne_LOS = calculate_KDE_statistics(kde0_x_ne, kde0_x_ne_LOS)

    data = [
        ("pB vs ne", data_source, date, JSD_kcor_psi_pB_ne, KLD_kcor_psi_pB_ne),
        ("pB vs ne_LOS", data_source, date, JSD_kcor_psi_pB_ne_LOS, KLD_kcor_psi_pB_ne_LOS),
        ("ne vs ne_LOS", data_source, date, JSD_kcor_psi_ne_ne_LOS, KLD_kcor_psi_ne_ne_LOS),
    ]

    # cur.executemany("INSERT INTO stats VALUES(?, ?, ?, ?, ?)", data)
    # con.commit()  # Remember to commit the transaction after executing INSERT.

    # JSD_kcor_central_random_new, KLD_kcor_central_random_new = calculate_KDE_statistics(KDE_kcor_central_deg_new, KDE_random_deg_new)
    # JSD_kcor_Forward_Central_Random_new, KLDkcor_forward_central_random_new = calculate_KDE_statistics(KDE_forward_kcor_central_deg_new, KDE_random_deg_new)

    # combined_dict = dict(metric=['KL Divergence', 'JS Divergence'],
    #                     kcor_v_psi=[KLD_kcor_forward_central_new, JSD_kcor_forward_central_new],
    #                     kcor_v_random=[KLD_kcor_central_random_new, JSD_kcor_central_random_new],
    #                     psi_v_random=[KLDkcor_forward_central_random_new, JSD_kcor_Forward_Central_Random_new])

    # pd.set_option('display.float_format', '{:.3E}'.format)
    # stats_df = pd.DataFrame(combined_dict)
    # stats_df.columns = ['metric', 'kcor vs psi pB', 'kcor vs random', 'psi pB vs random']
    # print(stats_df.to_latex(index=False))

    combined_pB.append(angles_arr_finite_pB)
    combined_ne.append(angles_arr_finite_ne)
    combined_ne_LOS.append(angles_arr_finite_ne_LOS)


    combined_pB_signed.append(angles_signed_arr_finite_pB)
    combined_ne_signed.append(angles_signed_arr_finite_ne)
    combined_ne_signed_LOS.append(angles_signed_arr_finite_ne_LOS)

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
data_source = 'KCor_PSI'
data_stats_2_combined.append((data_type_pB_combined, data_source, date_combined, angles_arr_mean_pB_combined, angles_arr_median_pB_combined, confidence_interval_pB_combined, n_pB_combined))

combined_ne_ravel_arr = np.array(combined_ne_ravel)
angles_arr_mean_ne_combined = np.round(np.mean(combined_ne_ravel_arr), 5)
angles_arr_median_ne_combined = np.round(np.median(combined_ne_ravel_arr), 5)
n_ne_combined = len(combined_ne_ravel_arr)
std_ne_combined = np.round(np.std(abs(combined_ne_ravel_arr)),5)
confidence_interval_ne_combined = np.round(1.96 * (std_ne_combined / np.sqrt(len(combined_ne_ravel_arr))),5)
data_type_ne_combined = 'ne'
date_combined = 'combined'
data_source = 'KCor_PSI'
data_stats_2_combined.append((data_type_ne_combined, data_source, date_combined, angles_arr_mean_ne_combined, angles_arr_median_ne_combined, confidence_interval_ne_combined, n_ne_combined))

combined_ne_LOS_ravel_arr = np.array(combined_ne_LOS_ravel)
angles_arr_mean_ne_LOS_combined = np.round(np.mean(combined_ne_LOS_ravel_arr), 5)
angles_arr_median_ne_LOS_combined = np.round(np.median(combined_ne_LOS_ravel_arr), 5)
n_ne_LOS_combined = len(combined_ne_LOS_ravel_arr)
std_ne_LOS_combined = np.round(np.std(abs(combined_ne_LOS_ravel_arr)),5)
confidence_interval_ne_LOS_combined = np.round(1.96 * (std_ne_LOS_combined / np.sqrt(len(combined_ne_LOS_ravel_arr))),5)
data_type_ne_LOS_combined = 'ne_LOS'
date_combined = 'combined'
data_source = 'KCor_PSI'
data_stats_2_combined.append((data_type_ne_LOS_combined, data_source, date_combined, angles_arr_mean_ne_LOS_combined, angles_arr_median_ne_LOS_combined, confidence_interval_ne_LOS_combined, n_ne_LOS_combined))



cur.executemany("INSERT INTO central_tendency_stats_kcor_new VALUES(?, ?, ?, ?, ?, ?, ?)", data_stats_2_combined)
con.commit()  # Remember to commit the transaction after executing INSERT.


combined_pB_signed_ravel = [item for sublist in combined_pB_signed for item in sublist]
combined_ne_signed_ravel = [item for sublist in combined_ne_signed for item in sublist]
combined_ne_signed_LOS_ravel = [item for sublist in combined_ne_signed_LOS for item in sublist]

combined_pB_signed_ravel_arr = np.array(combined_pB_signed_ravel)
combined_ne_signed_ravel_arr = np.array(combined_ne_signed_ravel)
combined_ne_signed_LOS_ravel_arr = np.array(combined_ne_signed_LOS_ravel)

what = sns.histplot(combined_ne_signed_ravel_arr,kde=True, bins=30)
norm_max_ne = max(what.get_lines()[0].get_data()[1])
plt.close()

what2 = sns.histplot(combined_pB_signed_ravel,kde=True, bins=30)
norm_max_pB = max(what2.get_lines()[0].get_data()[1])
plt.close()

what3 = sns.histplot(combined_ne_signed_LOS_ravel,kde=True, bins=30)
norm_max_ne_los = max(what3.get_lines()[0].get_data()[1])
plt.close()

fig = plt.figure(figsize=(10,10))
ax = fig.subplots(1,1)
sns.histplot(combined_ne_signed_ravel_arr,kde=True,label='ne',bins=30,ax=ax,color='tab:blue')
sns.histplot(combined_pB_signed_ravel,kde=True,label='pB',bins=30,ax=ax,color='tab:orange')
sns.histplot(combined_ne_signed_LOS_ravel,kde=True, bins=30, label='ne_LOS',ax=ax, color='tab:green')
#x_axis = np.linspace(-90, 90, len(KDE_kcor_central_deg_new))


# plt.plot(x_1_kcor_central_deg_new, (KDE_kcor_central_deg_new/max(KDE_kcor_central_deg_new))*norm_max_kcor, color='tab:blue')
# plt.plot(x_1_random_deg_new, (KDE_random_deg_new/max(KDE_random_deg_new))*norm_max_random, color='tab:green', label='random')
# plt.plot(x_1_forward_kcor_central_deg_new, (KDE_forward_kcor_central_deg_new/max(KDE_forward_kcor_central_deg_new))*norm_max_forward, color='tab:orange')
# norm_kde_random = (KDE_random_deg_new/max(KDE_random_deg_new))*norm_max_random
# norm_kde_forward = (KDE_forward_kcor_central_deg_new/max(KDE_forward_kcor_central_deg_new))*norm_max_forward
# norm_kde_kcor = (KDE_kcor_central_deg_new/max(KDE_kcor_central_deg_new))*norm_max_kcor


#sns.kdeplot()
ax.set_xlabel('Angle Discrepancy (Degrees)',fontsize=14)
ax.set_ylabel('Pixel Count',fontsize=14)
detector = 'KCor_PSI'
ax.set_title('QRaFT {} Feature Tracing Performance Against Central POS $B$ Field'.format(detector),fontsize=15)
ax.set_xlim(-95,95)
#ax.set_ylim(0,0.07)
ax.legend(fontsize=13)

# plt.text(20,0.045,"kcor average discrepancy: " + str(np.round(np.average(err_kcor_central_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_kcor_central_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
plt.savefig(os.path.join(repo_path,'Output/Plots/Updated_{}_vs_FORWARD_Feature_Tracing_Performance.png'.format(detector.replace('-',''))))
# plt.show()
#plt.close()

query = "SELECT mean, median, date, data_type, data_source, n, confidence FROM central_tendency_stats_kcor_new WHERE date!='combined' ORDER BY mean ASC;"
cur.execute(query)
rows = cur.fetchall()

# Close the cursor and the connection
# cur.close()
# con.close()

# Process the data for plotting
data_by_date = {}  # Dictionary to store data by date

for row in rows:
    mean, median, date, data_type, data_source, n, confidence = row
    if date not in data_by_date:
        data_by_date[date] = {'mean': [], 'confidence': [], 'data_type': []}
    data_by_date[date]['mean'].append(mean)
    data_by_date[date]['confidence'].append(confidence)
    data_by_date[date]['data_type'].append(data_type)

# Plot the scatter plot with error bars by date
dates = sorted(list(data_by_date.keys()))
data_types = sorted(list(set(data_by_date[dates[0]]['data_type'])))  # Assuming data types are consistent across dates

fig = plt.figure(figsize=(8, 8))
# Create a scatter plot for each date
for i, date in enumerate(dates):
    data_to_plot = [data_by_date[date]['mean'][j] for j in range(len(data_by_date[date]['data_type']))]
    confidence_to_plot = [data_by_date[date]['confidence'][j] for j in range(len(data_by_date[date]['data_type']))]
    data_type_to_plot = [data_by_date[date]['data_type'][j] for j in range(len(data_by_date[date]['data_type']))]
    for j in range(len(data_to_plot)):
        if data_type_to_plot[j] == data_types[0]:
            plt.errorbar(x=[i], y=data_to_plot[j], yerr=confidence_to_plot[j], fmt='o', color='C0' ,label=data_type_to_plot[j] if i == 0 else "")
        elif data_type_to_plot[j] == data_types[1]:
            plt.errorbar(x=[i], y=data_to_plot[j], yerr=confidence_to_plot[j], fmt='o', color='C2' ,label=data_type_to_plot[j] if i == 0 else "")
        elif data_type_to_plot[j] == data_types[2]:
            plt.errorbar(x=[i], y=data_to_plot[j], yerr=confidence_to_plot[j], fmt='o', color='C1' ,label=data_type_to_plot[j] if i == 0 else "")

# Customize the plot
plt.xlabel('Date of Corresponding Observation')
plt.ylabel('Mean Value (Degrees)')
plt.title('PSI K-COR Projection Angle Discrepancy by Date')
plt.legend()
plt.ylim(0,20)

# Set x-axis ticks and labels
plt.xticks(range(len(dates)), dates)
plt.savefig(os.path.join(repo_path, 'Output/Plots', '{}_Angle_Discrepancy_By_Date.png'.format(data_type)))
plt.show()


# Combine data into a single array
all_data = np.concatenate([combined_ne_signed_ravel_arr, combined_ne_signed_LOS_ravel_arr, combined_pB_signed_ravel_arr])

# Create labels for the data types
labels = ['ne'] * len(combined_ne_signed_ravel_arr) + ['ne_LOS'] * len(combined_ne_signed_LOS_ravel_arr) + ['pB'] * len(combined_pB_signed_ravel_arr)

# Perform Tukey's HSD post-hoc test
tukey_result = pairwise_tukeyhsd(all_data, labels)
print(tukey_result)


f_statistic, p_value = f_oneway(combined_ne_signed_ravel_arr, combined_ne_signed_LOS_ravel_arr, combined_pB_signed_ravel_arr)
# Check for statistical significance
if p_value < 0.05:
    print("There are significant differences between at least two data types.")
else:
    print("No significant differences detected between data types.")


res = tukey_hsd(combined_ne_signed_ravel_arr, combined_ne_signed_LOS_ravel_arr, combined_pB_signed_ravel_arr)
print(res)


# Combine data into a single array
all_data = np.concatenate([combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr])

# Create labels for the data types
labels = ['ne'] * len(combined_ne_ravel_arr) + ['ne_LOS'] * len(combined_ne_LOS_ravel_arr) + ['pB'] * len(combined_pB_ravel_arr)

# Perform Tukey's HSD post-hoc test
tukey_result = pairwise_tukeyhsd(all_data, labels)
print(tukey_result)
fig, ax = plt.subplots(1, 1)
# ax.boxplot([combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr], showfliers=False)
# ax.set_xticklabels(["ne", "ne_LOS", "pB"]) 
ax.set_xlabel("Mean (Degrees)") 
ax.set_ylabel("Data Type") 
ax.set_title('HSD Comparison of Data Types for PSI_KCor Combined Results')
tukey_result.plot_simultaneous(xlabel='Mean (Degrees)', ax=ax)
plt.savefig(os.path.join(repo_path, 'Output/Plots/testfig1_kcor.png'))
plt.show()

f_statistic, p_value = f_oneway(combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr)
# Check for statistical significance
if p_value < 0.05:
    print("There are significant differences between at least two data types.")
else:
    print("No significant differences detected between data types.")


fig, ax = plt.subplots(1, 1)
ax.boxplot([combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr], showfliers=False)
ax.set_xticklabels(["ne", "ne_LOS", "pB"]) 
ax.set_ylim(0, 40)
ax.set_ylabel("Mean (Degrees)") 
ax.set_xlabel("Data Type") 
ax.set_title('Box Plot Comparison of Data Types for PSI_KCor Combined Results')
plt.savefig(os.path.join(repo_path, 'Output/Plots/testfig2_kcor.png'))
plt.show()

res = tukey_hsd(combined_ne_ravel_arr, combined_ne_LOS_ravel_arr, combined_pB_ravel_arr)
print(res)

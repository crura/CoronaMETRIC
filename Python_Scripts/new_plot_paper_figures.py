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


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
data_dir = os.path.join(repo_path,'Data/QRaFT/errors.sav')

idl_save_qraft = readsav(data_dir)
err_mlso_central = idl_save_qraft['ERR_ARR_MLSO']
err_mlso_los = idl_save_qraft['ERR_ARR_LOS_MLSO']
err_forward_central = idl_save_qraft['ERR_ARR_FORWARD']
err_forward_los = idl_save_qraft['ERR_ARR_LOS_FORWARD']
err_random = idl_save_qraft['ERR_ARR_RND']

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
directory_list_1 = idl_save_outstring['directory_list']
directory_list_2 = idl_save_outstring['directory_list_2']
occlt_list = idl_save_outstring['occlt_list']

for i in range(len(directory_list_1)):
    directory_list_1[i] = os.path.join(repo_path, str(directory_list_1[i], 'utf-8'))

for i in range(len(directory_list_2)):
    directory_list_2[i] = os.path.join(repo_path, str(directory_list_2[i], 'utf-8'))


# remove blank first element of list
for i in range(len(outstring_list)):
    if outstring_list[i] == '':
        outstring_list_new = np.delete(outstring_list, i)
    else:
        pass
# translate all utf-8 strings into normal strings
for i in range(len(outstring_list_new)):
    outstring_list_new[i] = str(outstring_list_new[i],'utf-8')

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

occlt_list_1 = [occlt_list_new[index] for index in indexes1]
occlt_list_2 = [occlt_list_new[index] for index in indexes2]

print(directory_list_1, outstring_list_1, directory_list_2, outstring_list_2)
print(occlt_list_1, occlt_list_2)


os.path.join(repo_path,'Data/QRaFT/errors.sav')


def display_fits_images(fits_files, occlt_list, outpath):
    # fig, axes = plt.subplots(nrows=int(n/2), ncols=2, figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))


    for i in range(len(fits_files)):

        data = fits.getdata(fits_files[i])
        head = fits.getheader(fits_files[i])

        map = sunpy.map.Map(data, head)

        telescope = head['telescop']
        instrument = head['instrume']
        print(telescope)
        # print(head)
        if telescope == 'COSMO K-Coronagraph' or instrument == 'COSMO K-Coronagraph':
          head['detector'] = ('KCor')

        if head['detector'] == 'COR1':
            map.plot_settings['cmap'] = matplotlib.colormaps['Greys_r']
            rsun = (head['rsun'] / head['cdelt1']) * occlt_list[i] # number of pixels in radius of sun
        else:
            rsun = (head['rsun'] / head['cdelt1']) * occlt_list[i] # number of pixels in radius of sun
        axes = fig.add_subplot(int(len(fits_files)/2), 2, i+1, projection=map)
        if head['detector'] == 'PSI-MAS Forward Model' or head['telescop'] == 'PSI-MAS Forward Model':
            map.plot(axes=axes,title=False,norm=matplotlib.colors.LogNorm())
        elif head['detector'] == 'COR1':
            map.plot(axes=axes,title=False,clip_interval=(1, 99.99)*u.percent)
        else:
            map.plot(axes=axes,title=False)
        axes.add_patch(Circle((int(data.shape[0]/2),int(data.shape[1]/2)), rsun, color='black',zorder=100))
        # axes[i].imshow(data, cmap='gray')
        # axes[i].set_title(fits_file)

    plt.subplots_adjust(bottom=0.05, top=0.95)
    plt.savefig(outpath)
    # plt.show()
    # plt.close()


display_fits_images(outstring_list_1, occlt_list_1,os.path.join(repo_path,'Output/Plots/COR1_PSI_Plots.png'))
display_fits_images(directory_list_1, occlt_list_1 ,os.path.join(repo_path,'Output/Plots/COR1_Plots.png'))
display_fits_images(outstring_list_2, occlt_list_2 ,os.path.join(repo_path,'Output/Plots/MLSO_PSI_Plots.png'))
display_fits_images(directory_list_2, occlt_list_2 ,os.path.join(repo_path,'Output/Plots/MLSO_Plots.png'))





# carrington lat/lon in degrees
files = directory_list_2
longitudes = []
latitudes = []
small_angle_const = (3600 * 360)/(2 * np.pi)
x_radius = []
y_radius = []
z_radius = []
for i in files:
    path = os.path.join(repo_path, i)
    head = fits.getheader(path)
    time = datetime.strptime(head['DATE-OBS'],'%Y-%m-%dT%H:%M:%S')
    latitudes.append(head['CRLT_OBS'])
    longitudes.append(head['CRLN_OBS'])
    d_sun_obs = (constants.radius.to_value() * small_angle_const) / head['RSUN']
    x_radius.append(d_sun_obs)
    y_radius.append(d_sun_obs)
    z_radius.append(d_sun_obs)

files2 = directory_list_1
longitudes2 = []
latitudes2 = []
x2_radius = []
y2_radius = []
z2_radius = []
for i in files2:
    path2 = os.path.join(repo_path, i)
    head2 = fits.getheader(path2)
    time2 = datetime.strptime(head['DATE-OBS'],'%Y-%m-%dT%H:%M:%S')
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

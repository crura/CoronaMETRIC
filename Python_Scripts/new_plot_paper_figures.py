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
matplotlib.use('TkAgg')
mpl.use('TkAgg')


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

# filter filenames into separate lists based on detector
keyword = outstring_list_new[0].split('__')[2]
outstring_list_1 = [item for item in outstring_list_new if keyword in item]
outstring_list_2 = [item for item in outstring_list_new if keyword not in item]

print(directory_list_1, outstring_list_1, directory_list_2, outstring_list_2)


os.path.join(repo_path,'Data/QRaFT/errors.sav')

def create_six_figure_plot(file_string_list, occlt, outpath):

    fits_dir_1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')

    data1 = fits.getdata(fits_dir_psi)
    head1 = fits.getheader(fits_dir_psi)
    head1['detector'] = ('KCor')
    psimap = sunpy.map.Map(data1, head1)
    # psimap.plot_settings['norm'] = plt.Normalize(psimap.min(), psimap.max())

    fig2 = plt.figure(figsize=(10, 10))
    ax1 = fig2.add_subplot(3, 2, 1, projection=psimap)
    psimap.plot(axes=ax1,title=False,norm=matplotlib.colors.LogNorm())
    R_SUN = head1['R_SUN']
    ax1.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))

    fits_dir_psi = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_pB.fits')

    data2 = fits.getdata(fits_dir_psi)
    head2 = fits.getheader(fits_dir_psi)
    head2['detector'] = ('KCor')
    psimap = sunpy.map.Map(data2, head2)

    ax2 = fig2.add_subplot(3, 2, 2, projection=psimap)
    psimap.plot(axes=ax2,title=False,norm=matplotlib.colors.LogNorm())
    R_SUN = head2['R_SUN']
    ax2.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


    fits_dir_psi = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_pB.fits')

    data3 = fits.getdata(fits_dir_psi)
    head3 = fits.getheader(fits_dir_psi)
    head3['detector'] = ('KCor')
    psimap = sunpy.map.Map(data3, head3)

    ax3 = fig2.add_subplot(3, 2, 3, projection=psimap)
    psimap.plot(axes=ax3,title=False,norm=matplotlib.colors.LogNorm())
    R_SUN = head3['R_SUN']
    ax3.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


    fits_dir_psi = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_pB.fits')

    data4 = fits.getdata(fits_dir_psi)
    head4 = fits.getheader(fits_dir_psi)
    head4['detector'] = ('KCor')
    psimap = sunpy.map.Map(data4, head4)

    ax4 = fig2.add_subplot(3, 2, 4, projection=psimap)
    psimap.plot(axes=ax4,title=False,norm=matplotlib.colors.LogNorm())
    R_SUN = head4['R_SUN']
    ax4.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


    fits_dir_psi = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_pB.fits')

    data5 = fits.getdata(fits_dir_psi)
    head5 = fits.getheader(fits_dir_psi)
    head5['detector'] = ('KCor')
    psimap = sunpy.map.Map(data5, head5)


    ax5 = fig2.add_subplot(3, 2, 5, projection=psimap)
    psimap.plot(axes=ax5,title=False,norm=matplotlib.colors.LogNorm())
    R_SUN = head5['R_SUN']
    ax5.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


    fits_dir_psi = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_pB.fits')

    data6 = fits.getdata(fits_dir_psi)
    head6 = fits.getheader(fits_dir_psi)
    head6['detector'] = ('KCor')
    psimap = sunpy.map.Map(data6, head6)

    ax6 = fig2.add_subplot(3, 2, 6, projection=psimap)
    ax6.set_xlabel(' ')
    psimap.plot(axes=ax6,title=False,norm=matplotlib.colors.LogNorm())
    R_SUN = head6['R_SUN']
    ax6.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    plt.subplots_adjust(bottom=0.05, top=0.95)
    plt.savefig(os.path.join(repo_path,'Output/Plots/PSI_Plots.png'))
    # plt.show()


def display_fits_images(fits_files, outpath):
    # fig, axes = plt.subplots(nrows=int(n/2), ncols=2, figsize=(10, 10))
    fig = plt.figure(figsize=(10, 10))


    for i in range(len(fits_files)):

        data = fits.getdata(fits_files[i])
        head = fits.getheader(fits_files[i])
        map = sunpy.map.Map(data, head)
        axes = fig.add_subplot(int(len(fits_files)), 2, i+1, projection=map)
        map.plot(axes=axes,title=False,norm=matplotlib.colors.LogNorm())
        rsun = head['rsun'] / head['cdelt1'] # number of pixels in radius of sun
        axes.add_patch(Circle((int(data.shape[0]/2),int(data.shape[1]/2)), rsun, color='black',zorder=100))
        # axes[i].imshow(data, cmap='gray')
        # axes[i].set_title(fits_file)

    plt.savefig(outpath)
    # plt.show()
    # plt.close()




display_fits_images(outstring_list_1 ,os.path.join(repo_path,'Output/Plots/Test_Plot.png'))
display_fits_images(directory_list_1 ,os.path.join(repo_path,'Output/Plots/Test_Plot2.png'))
display_fits_images(outstring_list_2 ,os.path.join(repo_path,'Output/Plots/Test_Plot3.png'))
display_fits_images(directory_list_2 ,os.path.join(repo_path,'Output/Plots/Test_Plot4.png'))


# params = date_print + str(detector,'utf-8') + '_PSI'

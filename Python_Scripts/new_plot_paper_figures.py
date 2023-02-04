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
mpl.rcParams.update(mpl.rcParamsDefault)


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
    if occlt_list[i] == '':
        occlt_list_new = np.delete(occlt_list, i)
    else:
        pass
# translate all utf-8 strings into normal strings
for i in range(len(occlt_list_new)):
    occlt_list_new[i] = float(occlt_list_new[i])

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
        else:
            map.plot(axes=axes,title=False)
        axes.add_patch(Circle((int(data.shape[0]/2),int(data.shape[1]/2)), rsun, color='black',zorder=100))
        # axes[i].imshow(data, cmap='gray')
        # axes[i].set_title(fits_file)

    plt.subplots_adjust(bottom=0.05, top=0.95)
    plt.savefig(outpath)
    # plt.show()
    # plt.close()


display_fits_images(outstring_list_1 ,os.path.join(repo_path,'Output/Plots/Test_Plot.png'))
display_fits_images(directory_list_1 ,os.path.join(repo_path,'Output/Plots/Test_Plot2.png'))
display_fits_images(outstring_list_2 ,os.path.join(repo_path,'Output/Plots/Test_Plot3.png'))
display_fits_images(directory_list_2 ,os.path.join(repo_path,'Output/Plots/Test_Plot4.png'))



# params = date_print + str(detector,'utf-8') + '_PSI'

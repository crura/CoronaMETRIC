#!'/Users/crura/.conda/envs/test_env/lib/python3.9'
import sunpy
import matplotlib.pyplot as plt
import sunpy.version
import sunpy.map
import matplotlib.colors
import astropy.units as u
from astropy.io import fits
from pathlib import Path
import pandas as pd
import git
from scipy.io import readsav
import os
from matplotlib.patches import Circle


import subprocess
subprocess.run(["mkdir","Output/Plots","-p"])

path = __file__
pathnew = Path(path)

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
date_obs =idl_save['DATE_OBS']
crln_obs_print = idl_save['crln_obs_print']
crlt_obs_print = idl_save['crlt_obs_print']
date_print = str(idl_save['date_print'],'utf-8')
fits_directory = str(idl_save['fits_directory'][0],'utf-8')
occlt = idl_save['occlt']
shape = idl_save['shape']
params = date_print + '_PSI'



fits_dir_cor1 = os.path.join(fits_directory)

data2 = fits.getdata(fits_dir_cor1)
head2 = fits.getheader(fits_dir_cor1)
# head2['detector'] = ('Cor-1')
cor1map = sunpy.map.Map(data2, head2)


fits_dir_psi = os.path.join(repo_path,'Output/fits_images/{}_pB.fits'.format(params))
data1 = fits.getdata(fits_dir_psi)
head1 = fits.getheader(fits_dir_psi)
head1['detector'] = ('Cor-1')
psimap = sunpy.map.Map(data1, head1)




fig1 = plt.figure(figsize=(15, 8))
ax1 = fig1.add_subplot(1, 2, 1, projection=cor1map)

cor1map.plot(axes=ax1,title=False)

R_SUN = occlt * (head2['rsun'] / head2['cdelt1'])
ax1.add_patch(Circle((int(shape/2),int(shape/2)), R_SUN, color='black',zorder=100))



ax2 = fig1.add_subplot(1, 2, 2, projection=cor1map)
psimap.plot_settings['norm'] = plt.Normalize(cor1map.min(), cor1map.max())

psimap.plot(axes=ax2,title=False,norm=matplotlib.colors.LogNorm())
R_SUN = occlt * (head1['rsun'] / head1['cdelt1'])
ax2.add_patch(Circle((int(shape/2),int(shape/2)), R_SUN, color='black',zorder=100))
ax1.set_title('COR-1 Observation {}'.format(str(date_obs,'utf-8') + 'Z'), fontsize=18)
ax2.set_title('Corresponding PSI/FORWARD pB Eclipse Model', fontsize=18)

string_print = str(date_obs,'utf-8').split('T')[0].replace('-','_') + 'cor1'

plt.savefig(os.path.join(repo_path,'Output/Plots/Model_Comparison_{}.png'.format(string_print)))
plt.show()
plt.close()

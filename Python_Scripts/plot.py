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

path = __file__
pathnew = Path(path)

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
date_obs =idl_save['DATE_OBS']
crln_obs_print = idl_save['crln_obs_print']
crlt_obs_print = idl_save['crlt_obs_print']
shape = idl_save['shape']
params = str(crlt_obs_print,'utf-8') + '_' +  str(crln_obs_print,'utf-8')

fits_dir = pathnew.parent.parent.joinpath('Data/PSI/Forward_MLSO_Projection.fits')
path_fits_dir = Path(fits_dir)


data = fits.getdata(fits_dir)
head = fits.getheader(fits_dir)

psimap = sunpy.map.Map(data, head)

fits_dir_mlso = pathnew.parent.parent.joinpath(str(idl_save['fits_directory'],'utf-8'))

data = fits.getdata(fits_dir_mlso)
head = fits.getheader(fits_dir_mlso)
head['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data, head)

psimap.plot_settings['norm'] = plt.Normalize(mlsomap.min(), mlsomap.max())

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection=psimap)
psimap.plot(axes=ax1,norm=matplotlib.colors.LogNorm())
ax2 = fig.add_subplot(1, 2, 2, projection=mlsomap)
mlsomap.plot(axes=ax2)
plt.show()



# Import Coaligned Integrated Electron Density
integrated_dens_coaligned =pd.read_csv(pathnew.parent.parent.joinpath('Output/FORWARD_MLSO_Rotated_Data/PSI_MLSO_Integrated_Electron_Density_Coalignment.csv'), sep=',',header=None)
integrated_dens_coaligned_values = integrated_dens_coaligned.values
print(integrated_dens_coaligned_values.shape)

# add KCor as detector to fits file header
from astropy.io import fits
data = integrated_dens_coaligned_values
head = fits.getheader(fits_dir_mlso)
#mlsomap.fits_header.insert('Detector', 'kcor')
head['Observatory'] = ('PSI-MAS')
head['detector'] = ('Cor-1')

psi_coalign_map_pb = sunpy.map.Map(data, head)
#psi_coalign_map_pb


fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection=psi_coalign_map_pb)
ax1.set_title('PSI-Forward Model')
psi_coalign_map_pb.plot(axes=ax1,norm=matplotlib.colors.LogNorm())
ax2 = fig.add_subplot(1, 2, 2, projection=mlsomap)
mlsomap.plot(axes=ax2)
ax2.title.set_text('MLSO K-COR Observation {}'.format(str(date_obs,'utf-8') + 'Z'))
ax1.title.set_text('PSI/FORWARD Polarized Brightness Eclipse Model ' + params)
plt.savefig(pathnew.parent.parent.joinpath('Data/Output/FORWARD_MLSO_Rotated_Data/Plots/Model_Comparison_{}.png'.format(params)))
plt.show()




fits_dir_cor1 = os.path.join(repo_path,'Data/COR1/2017_08_20_rep_avg.fts')

data2 = fits.getdata(fits_dir_cor1)
head2 = fits.getheader(fits_dir_cor1)
head2['detector'] = ('KCor')
cor1map = sunpy.map.Map(data1, head1)


fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(1, 2, 1, projection=cor1map)
cor1map.plot(axes=ax1,title=False)
R_SUN = head1['R_SUN']
ax1.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))




fits_dir_psi = os.path.join(repo_path,'Output/fits_images/{}_pB.fits'.format(params))

data1 = fits.getdata(fits_dir_psi)
head1 = fits.getheader(fits_dir_psi)
head1['detector'] = ('Cor-1')
psimap = sunpy.map.Map(data1, head1)
# psimap.plot_settings['norm'] = plt.Normalize(psimap.min(), psimap.max())

fig2 = plt.figure(figsize=(10, 10))
ax1 = fig2.add_subplot(1, 2, 2, projection=cor1map)
psimap.plot(axes=ax1,title=False,norm=matplotlib.colors.LogNorm())
R_SUN = head1['R_SUN']
ax1.add_patch(Circle((shape,shape), R_SUN, color='black',zorder=100))
plt.savefig(os.path.join(repo_path,'Output/Plots/Model_Comparison.png'))
# plt.show()
plt.close()

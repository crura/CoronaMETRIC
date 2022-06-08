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

fits_dir = pathnew.parent.parent.joinpath('Data/PSI/Forward_MLSO_Projection.fits')
path_fits_dir = Path(fits_dir)


data = fits.getdata(fits_dir)
head = fits.getheader(fits_dir)

psimap = sunpy.map.Map(data, head)

fits_dir_mlso = pathnew.parent.parent.joinpath('Data/MLSO/20170829_200801_kcor_l2_avg_2.fts')

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



# Import Coaligned Images
psi_mlso_proj_pb =pd.read_csv(pathnew.parent.parent.joinpath('Output/FORWARD_MLSO_Rotated_Data/PSI_MLSO_Coalignment.csv'), sep=',',header=None)
psi_mlso_proj_pb_values = psi_mlso_proj_pb.values
print(psi_mlso_proj_pb_values.shape)

# add KCor as detector to fits file header
from astropy.io import fits
data = psi_mlso_proj_pb_values
head = fits.getheader(fits_dir_mlso)
#mlsomap.fits_header.insert('Detector', 'kcor')
head['Observatory'] = ('PSI-MAS')
head['detector'] = ('KCor')

psi_coalign_map_pb = sunpy.map.Map(data, head)
#psi_coalign_map_pb


fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection=psi_coalign_map_pb)
ax1.set_title('PSI-Forward Model')
psi_coalign_map_pb.plot(axes=ax1,norm=matplotlib.colors.LogNorm())
ax2 = fig.add_subplot(1, 2, 2, projection=mlsomap)
mlsomap.plot(axes=ax2)
ax2.title.set_text('MLSO K-COR Observation 2017-08-29')
ax1.title.set_text('PSI/FORWARD Synthetic Polarized Brightness Eclipse Model')
plt.savefig(pathnew.parent.parent.joinpath('Data/Output/FORWARD_MLSO_Rotated_Data/Plots/Model_Comparison.png'))
plt.show()

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
import matplotlib as mpl

b_central_idl_saved = readsav(os.path.join(repo_path,'Output/Central_B_Field_MLSO_Coaligned.sav'))
bx_central_coaligned = b_central_idl_saved['bx_central_coaligned']
by_central_coaligned = b_central_idl_saved['by_central_coaligned']
bz_central_coaligned = b_central_idl_saved['bz_central_coaligned']

idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
crln_obs_print = idl_save['crln_obs_print']
crlt_obs_print = idl_save['crlt_obs_print']
params = str(crlt_obs_print,'utf-8') + '_' +  str(crln_obs_print,'utf-8')
original_data = readsav(os.path.join(repo_path,'Output',params +'.sav'))

original_dens_2d_center = original_data['dens_2d_center']
original_dens_integrated_2d = original_data['dens_integrated_2d']
original_forward_pb_image = original_data['forward_pb_image']
original_bx_2d_center = original_data['bx_2d_center']
original_bx_2d_integrated = original_data['bx_2d_integrated']
original_by_2d_center = original_data['by_2d_center']
original_by_2d_integrated = original_data['by_2d_integrated']
original_bz_2d_center = original_data['bz_2d_center']
original_bz_2d_integrated = original_data['bz_2d_integrated']

from mpl_toolkits.axes_grid1 import make_axes_locatable
f = plt.figure()
f.set_figheight(10)
f.set_figwidth(10)
mpl.rcParams.update(mpl.rcParamsDefault)
ax1 = plt.subplot(1, 2, 1)

im1 = ax1.imshow(original_bx_2d_center,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('left', size='5%', pad=0.35)
f.colorbar(im1,cax=cax)
cax.yaxis.set_ticks_position('left')
ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(bx_central_coaligned,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.35)
f.colorbar(im2, cax=cax, orientation='vertical');
ax1.set_title('B_x Central Original')
ax2.set_title('B_x Central Reshaped')
plt.show()
plt.close()


f = plt.figure()
f.set_figheight(10)
f.set_figwidth(10)
mpl.rcParams.update(mpl.rcParamsDefault)
ax1 = plt.subplot(1, 2, 1)

im1 = ax1.imshow(original_by_2d_center,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('left', size='5%', pad=0.35)
f.colorbar(im1,cax=cax)
cax.yaxis.set_ticks_position('left')
ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(by_central_coaligned,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.35)
f.colorbar(im2, cax=cax, orientation='vertical');
ax1.set_title('B_y Central Original')
ax2.set_title('B_y Central Reshaped')
plt.show()
plt.close()


f = plt.figure()
f.set_figheight(10)
f.set_figwidth(10)
mpl.rcParams.update(mpl.rcParamsDefault)
ax1 = plt.subplot(1, 2, 1)

im1 = ax1.imshow(original_bz_2d_center,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('left', size='5%', pad=0.35)
f.colorbar(im1,cax=cax)
cax.yaxis.set_ticks_position('left')
ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(bz_central_coaligned,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.35)
f.colorbar(im2, cax=cax, orientation='vertical');
ax1.set_title('B_z Central Original')
ax2.set_title('B_z Central Reshaped')
plt.show()
plt.close()

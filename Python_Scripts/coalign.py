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

path = __file__
pathnew = Path(path)

fits_dir = pathnew.parent.parent.joinpath('Data/PSI/Forward_MLSO_Projection.fits')
path_fits_dir = Path(fits_dir)


data = fits.getdata(fits_dir)
head = fits.getheader(fits_dir)

psimap = sunpy.map.Map(data, head)

fits_dir_mlso = pathnew.parent.parent.joinpath('Data/MLSO/20170829_200801_kcor_l2_avg.fts')

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
psi_mlso_proj_pb =pd.read_csv('/Users/crura/Desktop/Research/Magnetic_Field/FORWARD_MLSO_Rotated_Data/PSI_PB_MLSO_Coalignment.csv', sep=',',header=None)


psi_mlso_proj_pb_values = psi_mlso_proj_pb.values


fits_dir_mlso = '/Users/crura/Desktop/Research/Magnetic_Field/Coalign/20170829_200801_kcor_l2_avg.fts'

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
plt.savefig('/Users/crura/Desktop/Research/Presentations/URD_22_Pictures/Model_Comparison.png')
plt.show()

#!'/Users/crura/.conda/envs/test_env/lib/python3.9'
import sunpy
import matplotlib.pyplot as plt
import sunpy.version
import sunpy.map
import matplotlib.colors
import astropy.units as u
from astropy.io import fits
from pathlib import Path

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

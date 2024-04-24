# interpolate to rebin kcor fits images from 1024x1024 to 512x512

from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from reproject import reproject_adaptive
import matplotlib
hdu1 = fits.open('/Users/crura/Desktop/Research/Test_Space/Naty_Images_Experiments/Image-Coalignment/Data/MLSO/20170820_180657_kcor_l2_avg.fts')[0]
hdu2 = fits.open('/Users/crura/Desktop/Research/Test_Space/Naty_Images_Experiments/Image-Coalignment/Data/COR1/cor1a_bff_20170820001000.fits')[0]

ax1 = plt.subplot(1,2,1, projection=WCS(hdu1.header))
ax1.imshow(hdu1.data, origin='lower', norm=matplotlib.colors.LogNorm())
# ax1.coords['ra'].set_axislabel('Right Ascension')
# ax1.coords['dec'].set_axislabel('Declination')
ax1.set_title('KCOR')

ax2 = plt.subplot(1,2,2, projection=WCS(hdu2.header))
ax2.imshow(hdu2.data, origin='lower', norm=matplotlib.colors.LogNorm())
# ax2.coords['glon'].set_axislabel('Galactic Longitude')
# ax2.coords['glat'].set_axislabel('Galactic Latitude')
# ax2.coords['glat'].set_axislabel_position('r')
# ax2.coords['glat'].set_ticklabel_position('r')
ax2.set_title('COR1')
plt.show()


# array, footprint = reproject_adaptive(hdu1, WCS(hdu2.header), shape_out=(512,512))

# # Update the header
# hdu2.header['NAXIS1'] = 512
# hdu2.header['NAXIS2'] = 512
# hdu2.header['CRPIX1'] = hdu2.header['CRPIX1'] / 2  # Assuming the reference pixel should also be scaled
# hdu2.header['CRPIX2'] = hdu2.header['CRPIX2'] / 2

# ax1 = plt.subplot(1,2,1, projection=WCS(hdu2.header))
# ax1.imshow(array, origin='lower')
# # ax1.coords['ra'].set_axislabel('Right Ascension')
# # ax1.coords['dec'].set_axislabel('Declination')
# ax1.set_title('KCOR reprojected')

# ax2 = plt.subplot(1,2,2, projection=WCS(hdu2.header))
# ax2.imshow(footprint, origin='lower')
# # ax2.coords['ra'].set_axislabel('Right Ascension')
# # ax2.coords['dec'].set_axislabel('Declination')
# # ax2.coords['dec'].set_axislabel_position('r')
# # ax2.coords['dec'].set_ticklabel_position('r')
# ax2.set_title('COR1')














from scipy.ndimage import zoom

# Open the FITS file and get the data
hdu1 = fits.open('/Users/crura/Desktop/Research/Test_Space/Naty_Images_Experiments/Image-Coalignment/Data/MLSO/20170820_180657_kcor_l2_avg.fts')[0]
data = hdu1.data

# Calculate the zoom factors
zoom_factors = (512 / data.shape[0], 512 / data.shape[1])

# Interpolate the data to the new shape
interpolated_data = zoom(data, zoom_factors)

# Update the header
hdu1.header['NAXIS1'] = 512
hdu1.header['NAXIS2'] = 512
hdu1.header['CRPIX1'] = hdu1.header['CRPIX1'] / 2  # Assuming the reference pixel should also be scaled
hdu1.header['CRPIX2'] = hdu1.header['CRPIX2'] / 2
hdu1.header['CDELT1'] = hdu1.header['CDELT1'] * 2  # Double the pixel size in the X direction
hdu1.header['CDELT2'] = hdu1.header['CDELT2'] * 2  # Double the pixel size in the Y direction

# Create a new FITS HDU with the interpolated data and the original header
new_hdu = fits.PrimaryHDU(interpolated_data, header=hdu1.header)

ax1 = plt.subplot(1,2,1, projection=WCS(new_hdu.header))
ax1.imshow(new_hdu.data, origin='lower', norm=matplotlib.colors.LogNorm())
# ax1.coords['ra'].set_axislabel('Right Ascension')
# ax1.coords['dec'].set_axislabel('Declination')
ax1.set_title('KCOR')

ax2 = plt.subplot(1,2,2, projection=WCS(hdu2.header))
ax2.imshow(hdu2.data, origin='lower', norm=matplotlib.colors.LogNorm())
# ax2.coords['glon'].set_axislabel('Galactic Longitude')
# ax2.coords['glat'].set_axislabel('Galactic Latitude')
# ax2.coords['glat'].set_axislabel_position('r')
# ax2.coords['glat'].set_ticklabel_position('r')
ax2.set_title('COR1')

plt.show()
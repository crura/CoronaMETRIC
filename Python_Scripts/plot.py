data_dir = '/Users/crura/Desktop/Research/Vadim/errors.sav'
from scipy.io import readsav
import matplotlib as mpl
import matplotlib.pyplot as plt
import git
from matplotlib.patches import Circle

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

idl_save = readsav(data_dir)
err_mlso_central = idl_save['ERR_ARR_MLSO']
err_mlso_los = idl_save['ERR_ARR_LOS_MLSO']
err_forward_central = idl_save['ERR_ARR_FORWARD']
err_forward_los = idl_save['ERR_ARR_LOS_FORWARD']
err_random = idl_save['ERR_ARR_RND']

# Generate plots for Central arrays
mpl.rcParams.update(mpl.rcParamsDefault)
import numpy as np
err_mlso_central_deg = err_mlso_central[np.where(err_mlso_central > 0)]*180/np.pi
err_forward_central_deg = err_forward_central[np.where(err_forward_central > 0)]*180/np.pi
err_random_deg = err_random[np.where(err_random > 0)]*180/np.pi

import subprocess
subprocess.run(["mkdir","Output/Plots","-p"])

import seaborn as sns
import os
sns.distplot(err_mlso_central_deg,hist=True,label='MLSO',bins=30)
sns.distplot(err_forward_central_deg,hist=True,label='FORWARD',bins=30)
sns.distplot(err_random_deg,hist=False,label='Random')
plt.xlabel('Angle Discrepancy')
plt.ylabel('Probability Density')
plt.title('Feature Tracing Performance against Central POS $B$ Field')
plt.xlim(0,90)
plt.ylim(0,0.07)
plt.legend()
# plt.text(20,0.045,"MLSO average discrepancy: " + str(np.round(np.average(err_mlso_central_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_central_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
plt.savefig(os.path.join(repo_path,'Output/Plots/MLSO_vs_FORWARD_Feature_Tracing_Performance.png'))
plt.show()
plt.close()
print(np.min(err_forward_central_deg))
# Generate plots for LOS arrays
err_mlso_los_deg = err_mlso_los[np.where(err_mlso_los > 0)]*180/np.pi
err_forward_los_deg = err_forward_los[np.where(err_forward_los > 0)]*180/np.pi

sns.distplot(err_mlso_los_deg,hist=True,label='MLSO LOS',bins=30)
sns.distplot(err_forward_los_deg,hist=True,label='FORWARD LOS',bins=30)
sns.distplot(err_random_deg,hist=False,label='Random')
plt.xlabel('Angle Discrepancy')
plt.ylabel('Probability Density')
plt.title('Feature Tracing Performance Against LOS Integrated $B$ Field')
plt.xlim(0,90)
plt.ylim(0,0.07)
# plt.text(20,0.045,"MLSO average discrepancy: " + str(np.round(np.average(err_mlso_los_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_los_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
plt.legend()
plt.savefig(os.path.join(repo_path,'Output/Plots/MLSO_vs_FORWARD_Feature_Tracing_Performance_LOS.png'))
plt.show()
plt.close()


fig, ax = plt.subplots(1,2,figsize=(16,8))
sns.distplot(err_mlso_central_deg,hist=True,label='MLSO K-COR',bins=30,ax=ax[0])
sns.distplot(err_forward_central_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax[0])
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax[0])
ax[0].set_xlabel('Angle Discrepancy')
ax[0].set_ylabel('Probability Density')
ax[0].set_title('QRaFT Feature Tracing Performance Against Central POS $B$ Field')
ax[0].set_xlim(0,90)
ax[0].set_ylim(0,0.07)
ax[0].legend()
# plt.text(20,0.045,"MLSO average discrepancy: " + str(np.round(np.average(err_mlso_central_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_central_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
# plt.savefig(os.path.join(repo_path,'Output/Plots/MLSO_vs_FORWARD_Feature_Tracing_Performance.png'))
# plt.show()
# plt.close()
print(np.min(err_forward_central_deg))
# Generate plots for LOS arrays
err_mlso_los_deg = err_mlso_los[np.where(err_mlso_los > 0)]*180/np.pi
err_forward_los_deg = err_forward_los[np.where(err_forward_los > 0)]*180/np.pi

sns.distplot(err_mlso_los_deg,hist=True,label='MLSO K-COR',bins=30,ax=ax[1])
sns.distplot(err_forward_los_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax[1])
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax[1])
ax[1].set_xlabel('Angle Discrepancy')
ax[1].set_ylabel('Probability Density')
ax[1].set_title('QRaFT Feature Tracing Performance Against LOS Integrated $B$ Field')
ax[1].set_xlim(0,90)
ax[1].set_ylim(0,0.07)
# plt.text(20,0.045,"MLSO average discrepancy: " + str(np.round(np.average(err_mlso_los_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_los_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
ax[1].legend()
plt.savefig(os.path.join(repo_path,'Output/Plots/MLSO_vs_FORWARD_Feature_Tracing_Performance_Combined.png'))
plt.show()
plt.close()


from astropy.io import fits
import sunpy
import sunpy.map
fits_dir_bz_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/6.89000_303.470_Bz_LOS.fits'
data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
head_bz_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

fits_dir_by_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/6.89000_303.470_By_LOS.fits'
data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
head_by_los_coaligned['Observatory'] = ('PSI-MAS')
head_by_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

ny, nz = 1024,1024
dy = np.linspace(-int(ny/2), int(ny/2), ny)
dz = np.linspace(-int(nz/2), int(nz/2), nz)
R_SUN = head_bz_los_coaligned['R_SUN']
widths = np.linspace(0,500,by_los_coaligned_map.data.size)
skip_val = 14
skip = (slice(None, None, skip_val), slice(None, None, skip_val))
skip1 = slice(None, None, skip_val)
fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize =(10, 10))
ax1.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
ax1.set_aspect('equal')
ax1.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
ax1.set_xlabel('Y Position')
ax1.set_ylabel('Z Position')

fits_dir_bz_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.05600_236.978_Bz_LOS.fits'
data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
head_bz_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

fits_dir_by_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.05600_236.978_By_LOS.fits'
data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
head_by_los_coaligned['Observatory'] = ('PSI-MAS')
head_by_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

R_SUN = head_bz_los_coaligned['R_SUN']

ax2.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
ax2.set_aspect('equal')
ax2.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax2.set_title('7.05600_236.978 LOS $B_z$ vs $B_y$ Field Vector Plot')
ax2.set_xlabel('Y Position')
ax2.set_ylabel('Z Position')

fits_dir_bz_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.15300_183.443_Bz_LOS.fits'
data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
head_bz_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

fits_dir_by_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.15300_183.443_By_LOS.fits'
data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
head_by_los_coaligned['Observatory'] = ('PSI-MAS')
head_by_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

R_SUN = head_bz_los_coaligned['R_SUN']

ax3.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
ax3.set_aspect('equal')
ax3.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax3.set_title('7.15300_183.443 LOS $B_z$ vs $B_y$ Field Vector Plot')
ax3.set_xlabel('Y Position')
ax3.set_ylabel('Z Position')

fits_dir_bz_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.22000_126.906_Bz_LOS.fits'
data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
head_bz_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

fits_dir_by_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.22000_126.906_By_LOS.fits'
data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
head_by_los_coaligned['Observatory'] = ('PSI-MAS')
head_by_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

R_SUN = head_bz_los_coaligned['R_SUN']

ax4.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
ax4.set_aspect('equal')
ax4.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax4.set_title('7.22000_126.906 LOS $B_z$ vs $B_y$ Field Vector Plot')
ax4.set_xlabel('Y Position')
ax4.set_ylabel('Z Position')

fits_dir_bz_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.23800_11.5530_Bz_LOS.fits'
data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
head_bz_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

fits_dir_by_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.23800_11.5530_By_LOS.fits'
data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
head_by_los_coaligned['Observatory'] = ('PSI-MAS')
head_by_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

R_SUN = head_bz_los_coaligned['R_SUN']

ax6.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
ax6.set_aspect('equal')
ax6.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax6.set_title('7.23800_11.5530 LOS $B_z$ vs $B_y$ Field Vector Plot')
ax6.set_xlabel('Y Position')
ax6.set_ylabel('Z Position')

fits_dir_bz_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.24700_77.0150_Bz_LOS.fits'
data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
head_bz_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

fits_dir_by_los_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.24700_77.0150_By_LOS.fits'
data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
head_by_los_coaligned['Observatory'] = ('PSI-MAS')
head_by_los_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

R_SUN = head_bz_los_coaligned['R_SUN']

ax5.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
ax5.set_aspect('equal')
ax5.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax5.set_title('7.24700_77.0150 LOS $B_z$ vs $B_y$ Field Vector Plot')
ax5.set_xlabel('Y Position')
ax5.set_ylabel('Z Position')
mpl.rcParams.update(mpl.rcParamsDefault)
plt.tight_layout()
plt.savefig(os.path.join(repo_path,'Output/Plots/LOS_B_Field_Vector_Plots.png'))
plt.show()
plt.close()


fits_dir_bz_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/6.89000_303.470_Bz.fits'
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/6.89000_303.470_By.fits'
data_by_central_coaligned = fits.getdata(fits_dir_by_central_coaligned)
head_by_central_coaligned = fits.getheader(fits_dir_by_central_coaligned)
head_by_central_coaligned['Observatory'] = ('PSI-MAS')
head_by_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_central_coaligned_map = sunpy.map.Map(data_by_central_coaligned, head_by_central_coaligned)

ny, nz = 1024,1024
dy = np.linspace(-int(ny/2), int(ny/2), ny)
dz = np.linspace(-int(nz/2), int(nz/2), nz)
R_SUN = head_bz_central_coaligned['R_SUN']
widths = np.linspace(0,500,by_central_coaligned_map.data.size)
skip_val = 14
skip = (slice(None, None, skip_val), slice(None, None, skip_val))
skip1 = slice(None, None, skip_val)
fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, figsize =(10, 10))
ax1.quiver(dy[skip1],dz[skip1],by_central_coaligned_map.data[skip],bz_central_coaligned_map.data[skip],linewidths=widths)
ax1.set_aspect('equal')
ax1.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax1.set_title('6.89000_303.470 Central $B_z$ vs $B_y$ Field Vector Plot')
ax1.set_xlabel('Y Position')
ax1.set_ylabel('Z Position')

fits_dir_bz_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.05600_236.978_Bz.fits'
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.05600_236.978_By.fits'
data_by_central_coaligned = fits.getdata(fits_dir_by_central_coaligned)
head_by_central_coaligned = fits.getheader(fits_dir_by_central_coaligned)
head_by_central_coaligned['Observatory'] = ('PSI-MAS')
head_by_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_central_coaligned_map = sunpy.map.Map(data_by_central_coaligned, head_by_central_coaligned)

R_SUN = head_bz_central_coaligned['R_SUN']

ax2.quiver(dy[skip1],dz[skip1],by_central_coaligned_map.data[skip],bz_central_coaligned_map.data[skip],linewidths=widths)
ax2.set_aspect('equal')
ax2.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax2.set_title('7.05600_236.978 Central $B_z$ vs $B_y$ Field Vector Plot')
ax2.set_xlabel('Y Position')
ax2.set_ylabel('Z Position')

fits_dir_bz_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.15300_183.443_Bz.fits'
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.15300_183.443_By.fits'
data_by_central_coaligned = fits.getdata(fits_dir_by_central_coaligned)
head_by_central_coaligned = fits.getheader(fits_dir_by_central_coaligned)
head_by_central_coaligned['Observatory'] = ('PSI-MAS')
head_by_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_central_coaligned_map = sunpy.map.Map(data_by_central_coaligned, head_by_central_coaligned)

R_SUN = head_bz_central_coaligned['R_SUN']

ax3.quiver(dy[skip1],dz[skip1],by_central_coaligned_map.data[skip],bz_central_coaligned_map.data[skip],linewidths=widths)
ax3.set_aspect('equal')
ax3.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax3.set_title('7.15300_183.443 Central $B_z$ vs $B_y$ Field Vector Plot')
ax3.set_xlabel('Y Position')
ax3.set_ylabel('Z Position')

fits_dir_bz_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.22000_126.906_Bz.fits'
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.22000_126.906_By.fits'
data_by_central_coaligned = fits.getdata(fits_dir_by_central_coaligned)
head_by_central_coaligned = fits.getheader(fits_dir_by_central_coaligned)
head_by_central_coaligned['Observatory'] = ('PSI-MAS')
head_by_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_central_coaligned_map = sunpy.map.Map(data_by_central_coaligned, head_by_central_coaligned)

R_SUN = head_bz_central_coaligned['R_SUN']

ax4.quiver(dy[skip1],dz[skip1],by_central_coaligned_map.data[skip],bz_central_coaligned_map.data[skip],linewidths=widths)
ax4.set_aspect('equal')
ax4.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax4.set_title('7.22000_126.906 Central $B_z$ vs $B_y$ Field Vector Plot')
ax4.set_xlabel('Y Position')
ax4.set_ylabel('Z Position')

fits_dir_bz_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.23800_11.5530_Bz.fits'
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.23800_11.5530_By.fits'
data_by_central_coaligned = fits.getdata(fits_dir_by_central_coaligned)
head_by_central_coaligned = fits.getheader(fits_dir_by_central_coaligned)
head_by_central_coaligned['Observatory'] = ('PSI-MAS')
head_by_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_central_coaligned_map = sunpy.map.Map(data_by_central_coaligned, head_by_central_coaligned)

R_SUN = head_bz_central_coaligned['R_SUN']

ax6.quiver(dy[skip1],dz[skip1],by_central_coaligned_map.data[skip],bz_central_coaligned_map.data[skip],linewidths=widths)
ax6.set_aspect('equal')
ax6.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax6.set_title('7.23800_11.5530 Central $B_z$ vs $B_y$ Field Vector Plot')
ax6.set_xlabel('Y Position')
ax6.set_ylabel('Z Position')

fits_dir_bz_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.24700_77.0150_Bz.fits'
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = '/Users/crura/Desktop/Research/github/Image-Coalignment/Output/7.24700_77.0150_By.fits'
data_by_central_coaligned = fits.getdata(fits_dir_by_central_coaligned)
head_by_central_coaligned = fits.getheader(fits_dir_by_central_coaligned)
head_by_central_coaligned['Observatory'] = ('PSI-MAS')
head_by_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
by_central_coaligned_map = sunpy.map.Map(data_by_central_coaligned, head_by_central_coaligned)

R_SUN = head_bz_central_coaligned['R_SUN']

ax5.quiver(dy[skip1],dz[skip1],by_central_coaligned_map.data[skip],bz_central_coaligned_map.data[skip],linewidths=widths)
ax5.set_aspect('equal')
ax5.add_patch(Circle((0,0), R_SUN, color='black',zorder=100))
ax5.set_title('7.24700_77.0150 Central $B_z$ vs $B_y$ Field Vector Plot')
ax5.set_xlabel('Y Position')
ax5.set_ylabel('Z Position')
mpl.rcParams.update(mpl.rcParamsDefault)
plt.tight_layout()
plt.savefig(os.path.join(repo_path,'Output/Plots/Central_B_Field_Vector_Plots.png'))
plt.show()
plt.close()

# retrieve probability density data from seaborne distplots
dist_values_mlso_central = sns.distplot(err_mlso_central_deg).get_lines()[0].get_data()[1]
plt.close()
dist_values_forward_central = sns.distplot(err_forward_central_deg).get_lines()[0].get_data()[1]
plt.close()

dist_values_mlso_los = sns.distplot(err_mlso_los_deg).get_lines()[0].get_data()[1]
plt.close()
dist_values_forward_los = sns.distplot(err_forward_los_deg).get_lines()[0].get_data()[1]
plt.close()

dist_values_random = sns.distplot(err_random_deg).get_lines()[0].get_data()[1]
plt.close()

import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
# creating the data distribution
# x = np.arange(-5, 5, 1)
# p = norm.pdf(x, 0, 2)
# q = norm.pdf(x, 2, 2)
#define KL Divergence
"""KL Divergence(P|Q)"""
def KL_div(p_probs, q_probs):
    KL_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(KL_div)

#define JS Divergence
def JS_Div(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    # normalize
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (KL_div(p, m) + KL_div(q, m)) / 2


print("")
#compute JS Divergence
result_JSD_MLSO_FORWARD_LOS= JS_Div(dist_values_mlso_los, dist_values_forward_los)
print("JS Divergence between MLSO LOS and PSI/FORWARD LOS",result_JSD_MLSO_FORWARD_LOS)
result_JSD_MLSO_FORWARD_Central= JS_Div(dist_values_mlso_central, dist_values_forward_central)
print("JS Divergence between MLSO Central and PSI/FORWARD Central",result_JSD_MLSO_FORWARD_Central)

result_JSD12= JS_Div(dist_values_mlso_los, dist_values_random)
print("JS Divergence between MLSO LOS and Random",result_JSD12)
result_JSD21= JS_Div(dist_values_mlso_central, dist_values_random)
print("JS Divergence between MLSO Central and Random",result_JSD21)

result_JSD12= JS_Div(dist_values_forward_los, dist_values_random)
print("JS Divergence between PSI/FORWARD LOS and Random",result_JSD12)
result_JSD21= JS_Div(dist_values_forward_central, dist_values_random)
print("JS Divergence between PSI/FORWARD Central and Random",result_JSD21)

print("")

#compute KL Divergence
KL_Div_mlso_forward_central = KL_div(dist_values_mlso_central,dist_values_forward_central)
print("KL Divergence between MLSO Central and PSI/FORWARD Central",KL_Div_mlso_forward_central)
KL_Div_mlso_forward_los = KL_div(dist_values_mlso_los, dist_values_forward_los)
print("KL Divergence between MLSO LOS and PSI/FORWARD LOS",KL_Div_mlso_forward_los)

KL_Div_mlso_central_random = KL_div(dist_values_mlso_central,dist_values_random)
print("KL Divergence between MLSO Central and Random",KL_Div_mlso_central_random)
KL_Div_mlso_los_random = KL_div(dist_values_mlso_los, dist_values_random)
print("KL Divergence between MLSO LOS and Random",KL_Div_mlso_los_random)

KL_Div_forward_central_random = KL_div(dist_values_forward_central,dist_values_random)
print("KL Divergence between PSI/FORWARD Central and Random",KL_Div_forward_central_random)
KL_Div_forward_los_random = KL_div(dist_values_forward_los, dist_values_random)
print("KL Divergence between PSI/FORWARD LOS and Random",KL_Div_forward_los_random)

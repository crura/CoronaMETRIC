# Copyright 2025 Christopher Rura

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

idl_save = readsav(data_dir)
err_mlso_central = idl_save['ERR_ARR_MLSO']
err_mlso_los = idl_save['ERR_ARR_LOS_MLSO']
err_forward_central = idl_save['ERR_ARR_FORWARD']
err_forward_los = idl_save['ERR_ARR_LOS_FORWARD']
err_random = idl_save['ERR_ARR_RND']

# Generate plots for Central arrays
mpl.rcParams.update(mpl.rcParamsDefault)

err_mlso_central_deg = err_mlso_central[np.where(err_mlso_central > 0)]*180/np.pi
err_forward_central_deg = err_forward_central[np.where(err_forward_central > 0)]*180/np.pi
err_random_deg = err_random[np.where(err_random > 0)]*180/np.pi

import subprocess
subprocess.run(["mkdir","Output/Plots","-p"])

import seaborn as sns
import os

from scipy.stats import gaussian_kde
xmin_mlso_central = -14.5
xmax_mlso_central = 104.0


kde0_mlso_central_deg = gaussian_kde(err_mlso_central_deg)
x_1_mlso_central_deg = np.linspace(xmin_mlso_central, xmax_mlso_central, 200)
kde0_x_mlso_central_deg = kde0_mlso_central_deg(x_1_mlso_central_deg)
# plt.plot(x_1_mlso_central_deg, kde0_x_mlso_central_deg, color='b', label='mlso central KDE scipy')

xmin_forward_central = -17.24
xmax_forward_central = 106.13

kde0_forward_central_deg = gaussian_kde(err_forward_central_deg)
x_1_forward_central_deg = np.linspace(xmin_forward_central, xmax_forward_central, 200)
kde0_x_forward_central_deg = kde0_forward_central_deg(x_1_forward_central_deg)
# plt.plot(x_1_forward_central_deg, kde0_x_forward_central_deg, color='b', label='forward central KDE scipy')

xmin_random = -18.395
xmax_random = 108.39

kde0_random_deg = gaussian_kde(err_random_deg)
x_1_random_deg = np.linspace(xmin_random, xmax_random, 200)
kde0_x_random_deg = kde0_random_deg(x_1_random_deg)
# plt.plot(x_1_random_deg, kde0_x_random_deg, color='b', label='random KDE scipy')



#
# sns.distplot(err_mlso_central_deg,hist=True,label='MLSO',bins=30)
# sns.distplot(err_forward_central_deg,hist=True,label='FORWARD',bins=30)
# sns.distplot(err_random_deg,hist=False,label='Random')
# plt.xlabel('Angle Discrepancy')
# plt.ylabel('Probability Density')
# plt.title('Feature Tracing Performance against Central POS $B$ Field')
# plt.xlim(0,90)
# plt.ylim(0,0.07)
# plt.legend()

fig = plt.figure(figsize=(8,8))
ax = fig.subplots(1,1)
sns.distplot(err_mlso_central_deg,hist=True,label='MLSO K-COR',bins=30,ax=ax)
sns.distplot(err_forward_central_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax)
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax)
ax.set_xlabel('Angle Discrepancy')
ax.set_ylabel('Probability Density')
ax.set_title('QRaFT Feature Tracing Performance Against Central POS $B$ Field')
ax.set_xlim(0,90)
ax.set_ylim(0,0.07)
ax.legend()

# plt.text(20,0.045,"MLSO average discrepancy: " + str(np.round(np.average(err_mlso_central_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_central_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
plt.savefig(os.path.join(repo_path,'Output/Plots/MLSO_vs_FORWARD_Feature_Tracing_Performance.png'))
# #plt.show()
plt.close()
print(np.min(err_forward_central_deg))
# Generate plots for LOS arrays
err_mlso_los_deg = err_mlso_los[np.where(err_mlso_los > 0)]*180/np.pi
err_forward_los_deg = err_forward_los[np.where(err_forward_los > 0)]*180/np.pi


xmin_mlso_los = -13.9
xmax_mlso_los = 103.9

kde0_mlso_los_deg = gaussian_kde(err_mlso_los_deg)
x_1_mlso_los_deg = np.linspace(xmin_mlso_los, xmax_mlso_los, 200)
kde0_x_mlso_los_deg = kde0_mlso_los_deg(x_1_mlso_los_deg)
plt.plot(x_1_mlso_los_deg, kde0_x_mlso_los_deg, color='b', label='mlso los KDE scipy')

xmin_forward_los = -19.893
xmax_forward_los = 109.395

kde0_forward_los_deg = gaussian_kde(err_forward_los_deg)
x_1_forward_los_deg = np.linspace(xmin_forward_los, xmax_forward_los, 200)
kde0_x_forward_los_deg = kde0_forward_los_deg(x_1_forward_los_deg)
plt.plot(x_1_forward_los_deg, kde0_x_forward_los_deg, color='b', label='forward los KDE scipy')


kde0_random_deg = gaussian_kde(err_random_deg)
x_1_random_deg = np.linspace(xmin_random, xmax_random, 200)
kde0_x_random_deg = kde0_random_deg(x_1_random_deg)
plt.plot(x_1_random_deg, kde0_x_random_deg, color='b', label='random KDE scipy')

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
# #plt.show()
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
# #plt.show()
# plt.close()
print(np.min(err_forward_central_deg))
# Generate plots for LOS arrays
err_mlso_los_deg = err_mlso_los[np.where(err_mlso_los > 0)]*180/np.pi
err_forward_los_deg = err_forward_los[np.where(err_forward_los > 0)]*180/np.pi

sns.distplot(err_mlso_los_deg,hist=True,label='MLSO K-COR',bins=30,ax=ax[1])
sns.distplot(err_forward_los_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax[1])
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax[1])
# sns.kdeplot(err_mlso_los_deg,label='KDE')
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
# #plt.show()
plt.close()

from sklearn.neighbors import KernelDensity


"""
n =int(len(err_mlso_los_deg)/3)
Y = err_mlso_los_deg
hi, bins, patches = plt.hist(Y,bins=n-1,range=(0,90))
x = np.concatenate((bins[:-1],hi))
x_train = x[:,np.newaxis]
kde = KernelDensity(kernel="gaussian", bandwidth=0.75).fit(x_train)
log_dens = kde.score_samples(x_train)
"""


fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170820_180657_kcor_l2_avg.fts')

data1 = fits.getdata(fits_dir_mlso)
head1 = fits.getheader(fits_dir_mlso)
head1['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data1, head1)
# psimap.plot_settings['norm'] = plt.Normalize(mlsomap.min(), mlsomap.max())

fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(3, 2, 1, projection=mlsomap)
mlsomap.plot(axes=ax1,title=False)
R_SUN = head1['R_SUN']
ax1.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))

fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170825_185258_kcor_l2_avg.fts')

data2 = fits.getdata(fits_dir_mlso)
head2 = fits.getheader(fits_dir_mlso)
head2['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data2, head2)

ax2 = fig1.add_subplot(3, 2, 2, projection=mlsomap)
mlsomap.plot(axes=ax2,title=False)
R_SUN = head2['R_SUN']
ax2.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170829_200801_kcor_l2_avg.fts')

data3 = fits.getdata(fits_dir_mlso)
head3 = fits.getheader(fits_dir_mlso)
head3['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data3, head3)

ax3 = fig1.add_subplot(3, 2, 3, projection=mlsomap)
mlsomap.plot(axes=ax3,title=False)
R_SUN = head3['R_SUN']
ax3.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170903_025117_kcor_l2_avg.fts')

data4 = fits.getdata(fits_dir_mlso)
head4 = fits.getheader(fits_dir_mlso)
head4['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data4, head4)

ax4 = fig1.add_subplot(3, 2, 4, projection=mlsomap)
mlsomap.plot(axes=ax4,title=False)
R_SUN = head4['R_SUN']
ax4.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170906_213054_kcor_l2_avg.fts')

data5 = fits.getdata(fits_dir_mlso)
head5 = fits.getheader(fits_dir_mlso)
head5['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data5, head5)


ax5 = fig1.add_subplot(3, 2, 5, projection=mlsomap)
mlsomap.plot(axes=ax5,title=False)
R_SUN = head5['R_SUN']
ax5.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170911_202927_kcor_l2_avg.fts')

data6 = fits.getdata(fits_dir_mlso)
head6 = fits.getheader(fits_dir_mlso)
head6['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data6, head6)

ax6 = fig1.add_subplot(3, 2, 6, projection=mlsomap)
ax6.set_xlabel(' ')
mlsomap.plot(axes=ax6,title=False)
R_SUN = head6['R_SUN']
ax6.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
plt.subplots_adjust(bottom=0.05, top=0.95)
plt.savefig(os.path.join(repo_path,'Output/Plots/MLSO_Plots.png'))
# #plt.show()


Bz1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')
# By1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')
Bz2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_pB.fits')
# By2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_By.fits')
Bz3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_pB.fits')
# By3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_By.fits')
Bz4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_pB.fits')
# By4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_By.fits')
Bz5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_pB.fits')
# By5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_By.fits')
Bz6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_pB.fits')


fits_dir_psi = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')

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
# #plt.show()




"""
def create_six_fig_plot_dens(files_z, outpath):
    file1_z, file2_z, file3_z, file4_z, file5_z, file6_z = files_z

    fits_dir_bz_los_coaligned = file1_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    wcs = WCS(head_bz_los_coaligned)

    # ny, nz = 1024,1024
    # dy = np.linspace(0, int(ny), ny)
    # dz = np.linspace(0, int(nz), nz)
    R_SUN = head_bz_los_coaligned['R_SUN']
    # widths = np.linspace(0,500,by_los_coaligned_map.data.size)
    # skip_val = 14
    # skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    # skip1 = slice(None, None, skip_val)
    fig = plt.figure(figsize =(10, 10))
    # ax1 = plt.subplot(3,2,1,projection=wcs)
    # fig.add_subplot(rows, columns, i+1,projection=wcs)
    ax1 = fig.add_subplot(3, 2, 1, projection=bz_los_coaligned_map)
    bz_los_coaligned_map.plot(axes=ax1,norm=matplotlib.colors.LogNorm())
    # ax1.imshow(data_bz_los_coaligned,origin='lower',cmap='Greys_r',aspect='auto')
    ax1.set_aspect('equal')
    ax1.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax1.set_xlabel(' ')
    ax1.set_ylabel('Helioprojective Latitude (Solar-Y)')


    fits_dir_bz_los_coaligned = file2_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    wcs = WCS(head_bz_los_coaligned)

    # ny, nz = 1024,1024
    # dy = np.linspace(0, int(ny), ny)
    # dz = np.linspace(0, int(nz), nz)
    R_SUN = head_bz_los_coaligned['R_SUN']
    # widths = np.linspace(0,500,by_los_coaligned_map.data.size)
    # skip_val = 14
    # skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    # skip1 = slice(None, None, skip_val)
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'projection':wcs},figsize =(15, 15))
    # ax1 = plt.subplot(3,2,1,projection=wcs)
    ax2 = fig.add_subplot(3, 2, 2, projection=bz_los_coaligned_map)
    bz_los_coaligned_map.plot(axes=ax2,norm=matplotlib.colors.LogNorm())
    # ax2.imshow(data_bz_los_coaligned,norm=matplotlib.colors.LogNorm(),origin='lower',cmap='Greys_r',aspect='auto')
    ax2.set_aspect('equal')
    ax2.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax2.set_xlabel(' ')
    ax2.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file3_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    wcs = WCS(head_bz_los_coaligned)

    # ny, nz = 1024,1024
    # dy = np.linspace(0, int(ny), ny)
    # dz = np.linspace(0, int(nz), nz)
    R_SUN = head_bz_los_coaligned['R_SUN']
    # widths = np.linspace(0,500,by_los_coaligned_map.data.size)
    # skip_val = 14
    # skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    # skip1 = slice(None, None, skip_val)
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'projection':wcs},figsize =(10, 10))
    # ax1 = plt.subplot(3,2,1,projection=wcs)
    ax3 = fig.add_subplot(3, 2, 3, projection=bz_los_coaligned_map)
    bz_los_coaligned_map.plot(axes=ax3,norm=matplotlib.colors.LogNorm())
    # ax3.imshow(data_bz_los_coaligned,norm=matplotlib.colors.LogNorm(),origin='lower',cmap='Greys_r',aspect='auto')
    ax3.set_aspect('equal')
    ax3.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax3.set_xlabel(' ')
    ax3.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file4_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    wcs = WCS(head_bz_los_coaligned)

    # ny, nz = 1024,1024
    # dy = np.linspace(0, int(ny), ny)
    # dz = np.linspace(0, int(nz), nz)
    R_SUN = head_bz_los_coaligned['R_SUN']
    # widths = np.linspace(0,500,by_los_coaligned_map.data.size)
    # skip_val = 14
    # skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    # skip1 = slice(None, None, skip_val)
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'projection':wcs},figsize =(10, 10))
    # ax1 = plt.subplot(3,2,1,projection=wcs)
    ax4 = fig.add_subplot(3, 2, 4, projection=bz_los_coaligned_map)
    bz_los_coaligned_map.plot(axes=ax4,norm=matplotlib.colors.LogNorm())
    # ax4.imshow(data_bz_los_coaligned,norm=matplotlib.colors.LogNorm(),origin='lower',cmap='Greys_r',aspect='auto')
    ax4.set_aspect('equal')
    ax4.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax4.set_xlabel(' ')
    ax4.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file5_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    wcs = WCS(head_bz_los_coaligned)

    # ny, nz = 1024,1024
    # dy = np.linspace(0, int(ny), ny)
    # dz = np.linspace(0, int(nz), nz)
    R_SUN = head_bz_los_coaligned['R_SUN']
    # widths = np.linspace(0,500,by_los_coaligned_map.data.size)
    # skip_val = 14
    # skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    # skip1 = slice(None, None, skip_val)
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'projection':wcs},figsize =(10, 10))
    # ax1 = plt.subplot(3,2,1,projection=wcs)
    ax5 = fig.add_subplot(3, 2, 5, projection=bz_los_coaligned_map)
    bz_los_coaligned_map.plot(axes=ax5,norm=matplotlib.colors.LogNorm())
    # ax5.imshow(data_bz_los_coaligned,norm=matplotlib.colors.LogNorm(),origin='lower',cmap='Greys_r',aspect='auto')
    ax5.set_aspect('equal')
    ax5.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax5.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax5.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file6_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    wcs = WCS(head_bz_los_coaligned)

    # ny, nz = 1024,1024
    # dy = np.linspace(0, int(ny), ny)
    # dz = np.linspace(0, int(nz), nz)
    R_SUN = head_bz_los_coaligned['R_SUN']
    # widths = np.linspace(0,500,by_los_coaligned_map.data.size)
    # skip_val = 14
    # skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    # skip1 = slice(None, None, skip_val)
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'projection':wcs},figsize =(10, 10))
    # ax1 = plt.subplot(3,2,1,projection=wcs)
    ax6 = fig.add_subplot(3, 2, 6, projection=bz_los_coaligned_map)
    bz_los_coaligned_map.plot(axes=ax6,norm=matplotlib.colors.LogNorm())
    # ax6.imshow(data_bz_los_coaligned,norm=matplotlib.colors.LogNorm(),origin='lower',cmap='Greys_r',aspect='auto')
    ax6.set_aspect('equal')
    ax6.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax6.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax6.set_ylabel('Helioprojective Latitude (Solar-Y)')
    plt.subplots_adjust(bottom=0.05, top=0.95)

    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.tight_layout()
    # fig.set_constrained_layout_pads(w_pad=1 / 102, h_pad=1 / 102, hspace=0.0,
    #                                 wspace=0.0)
    plt.savefig(outpath)

    #plt.show()
    plt.close()

    return fig
"""
#
# n =int(len(err_mlso_los_deg)/3)
# x_vals = np.linspace(0,90,n)
# Y = err_mlso_los_deg
# hi, bins = np.histogram(x_vals, Y,bins=n)
# X = np.concatenate((Y,x_vals))[:, np.newaxis]
# kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
# plt.plot(kde.score_samples(X))
# #plt.show()

def create_six_fig_plot(files_z, files_y, outpath):
    file1_z, file2_z, file3_z, file4_z, file5_z, file6_z = files_z
    file1_y, file2_y, file3_y, file4_y, file5_y, file6_y = files_y

    fits_dir_bz_los_coaligned = file1_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    wcs = WCS(head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file1_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    ny, nz = 1024,1024
    dy = np.linspace(0, int(ny), ny)
    dz = np.linspace(0, int(nz), nz)
    R_SUN = head_bz_los_coaligned['R_SUN']
    widths = np.linspace(0,500,by_los_coaligned_map.data.size)
    skip_val = 14
    skip = (slice(None, None, skip_val), slice(None, None, skip_val))
    skip1 = slice(None, None, skip_val)
    fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3, 2, subplot_kw={'projection':wcs},figsize =(10, 10))
    # ax1 = plt.subplot(3,2,1,projection=wcs)
    ax1.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax1.set_aspect('equal')
    ax1.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax1.set_title('6.89000_303.470 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax1.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax1.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file2_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file2_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    R_SUN = head_bz_los_coaligned['R_SUN']

    # ax2 = plt.subplot(3,2,2,projection=wcs)
    ax2.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax2.set_aspect('equal')
    ax2.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax2.set_title('7.05600_236.978 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax2.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax2.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file3_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file3_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    R_SUN = head_bz_los_coaligned['R_SUN']

    # ax3 = plt.subplot(3,2,3,projection=wcs)
    ax3.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax3.set_aspect('equal')
    ax3.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax3.set_title('7.15300_183.443 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax3.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax3.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file4_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file4_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    R_SUN = head_bz_los_coaligned['R_SUN']

    # ax4 = plt.subplot(3,2,4,projection=wcs)
    ax4.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax4.set_aspect('equal')
    ax4.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax4.set_title('7.22000_126.906 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax4.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax4.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file5_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file5_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    R_SUN = head_bz_los_coaligned['R_SUN']

    # ax5 = plt.subplot(3,2,5,projection=wcs)
    ax5.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax5.set_aspect('equal')
    ax5.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax5.set_title('7.24700_77.0150 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax5.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax5.set_ylabel('Helioprojective Latitude (Solar-Y)')

    fits_dir_bz_los_coaligned = file6_z
    data_bz_los_coaligned = fits.getdata(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned = fits.getheader(fits_dir_bz_los_coaligned)
    head_bz_los_coaligned['Observatory'] = ('PSI-MAS')
    head_bz_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    bz_los_coaligned_map = sunpy.map.Map(data_bz_los_coaligned, head_bz_los_coaligned)

    fits_dir_by_los_coaligned = file6_y
    data_by_los_coaligned = fits.getdata(fits_dir_by_los_coaligned)
    head_by_los_coaligned = fits.getheader(fits_dir_by_los_coaligned)
    head_by_los_coaligned['Observatory'] = ('PSI-MAS')
    head_by_los_coaligned['detector'] = ('KCor')
    # print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
    by_los_coaligned_map = sunpy.map.Map(data_by_los_coaligned, head_by_los_coaligned)

    R_SUN = head_bz_los_coaligned['R_SUN']

    # ax6 = plt.subplot(3,2,6,projection=wcs)
    ax6.quiver(dy[skip1],dz[skip1],by_los_coaligned_map.data[skip],bz_los_coaligned_map.data[skip],linewidths=widths)
    ax6.set_aspect('equal')
    ax6.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
    # ax6.set_title('7.23800_11.5530 LOS $B_z$ vs $B_y$ Field Vector Plot')
    ax6.set_xlabel('Helioprojective Longitude (Solar-X)')
    ax6.set_ylabel('Helioprojective Latitude (Solar-Y)')
    plt.subplots_adjust(bottom=0.05, top=0.95)

    mpl.rcParams.update(mpl.rcParamsDefault)
    # fig.tight_layout()
    # fig.set_constrained_layout_pads(w_pad=1 / 102, h_pad=1 / 102, hspace=0.0,
    #                                 wspace=0.0)
    plt.savefig(outpath)
    # #plt.show()
    plt.close()

    return fig

Bz1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_Bz_LOS.fits')
By1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_By_LOS.fits')
Bz2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_Bz_LOS.fits')
By2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_By_LOS.fits')
Bz3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_Bz_LOS.fits')
By3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_By_LOS.fits')
Bz4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_Bz_LOS.fits')
By4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_By_LOS.fits')
Bz5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_Bz_LOS.fits')
By5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_By_LOS.fits')
Bz6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_Bz_LOS.fits')
By6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_By_LOS.fits')
file_list_Bz_LOS = [Bz1, Bz2, Bz3, Bz4, Bz5, Bz6]
file_list_By_LOS = [By1, By2, By3, By4, By5, By6]

create_six_fig_plot(file_list_Bz_LOS,file_list_By_LOS,os.path.join(repo_path,'Output/Plots/LOS_B_Field_Vector_Plots.png'))

Bz1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_Bz.fits')
By1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_By.fits')
Bz2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_Bz.fits')
By2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_By.fits')
Bz3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_Bz.fits')
By3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_By.fits')
Bz4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_Bz.fits')
By4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_By.fits')
Bz5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_Bz.fits')
By5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_By.fits')
Bz6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_Bz.fits')
By6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_By.fits')
file_list_Bz_LOS = [Bz1, Bz2, Bz3, Bz4, Bz5, Bz6]
file_list_By_LOS = [By1, By2, By3, By4, By5, By6]

create_six_fig_plot(file_list_Bz_LOS,file_list_By_LOS,os.path.join(repo_path,'Output/Plots/Central_B_Field_Vector_Plots.png'))


Bz1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')
# By1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')
Bz2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_pB.fits')
# By2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_By.fits')
Bz3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_pB.fits')
# By3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_By.fits')
Bz4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_pB.fits')
# By4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_By.fits')
Bz5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_pB.fits')
# By5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_By.fits')
Bz6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_pB.fits')
# By6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_By.fits')
file_list_pB = [Bz1, Bz2, Bz3, Bz4, Bz5, Bz6]
# file_list_By_LOS = [By1, By2, By3, By4, By5, By6]




By1 = os.path.join(repo_path,'Data/MLSO/20170820_180657_kcor_l2_avg.fts')
# By1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')
By2 = os.path.join(repo_path,'Data/MLSO/20170825_185258_kcor_l2_avg.fts')
# By2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_By.fits')
By3 = os.path.join(repo_path,'Data/MLSO/20170829_200801_kcor_l2_avg.fts')
# By3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_By.fits')
By4 = os.path.join(repo_path,'Data/MLSO/20170903_025117_kcor_l2_avg.fts')
# By4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_By.fits')
By5 = os.path.join(repo_path,'Data/MLSO/20170906_213054_kcor_l2_avg.fts')
# By5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_By.fits')
By6 = os.path.join(repo_path,'Data/MLSO/20170911_202927_kcor_l2_avg.fts')
# By6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_By.fits')
file_list_pB = [Bz1, Bz2, Bz3, Bz4, Bz5, Bz6]
file_list_MLSO = [By1, By2, By3, By4, By5, By6]
# file_list_By_LOS = [By1, By2, By3, By4, By5, By6]

# fig1 = create_six_fig_plot_dens(file_list_pB,os.path.join(repo_path,'Output/Plots/Forward_pB_Plots.png'))
# fig2 = create_six_fig_plot_dens(file_list_MLSO,os.path.join(repo_path,'Output/Plots/MLSO_Plots.png'))




# fig1 = create_six_fig_plot(file_list_Bz_LOS,file_list_By_LOS,os.path.join(repo_path,'Output/Plots/LOS_B_Field_Vector_Plots.png'))


# fig2 = create_six_fig_plot(file_list_Bz_LOS,file_list_By_LOS,os.path.join(repo_path,'Output/Plots/LOS_B_Field_Vector_Plots.png'))

backend = mpl.get_backend()
mpl.use('agg')


c1 = fig1.canvas
c2 = fig2.canvas

c1.draw()
c2.draw()

a1 = np.array(c1.buffer_rgba())
a2 = np.array(c2.buffer_rgba())
a = np.hstack((a1,a2))


mpl.use(backend)
fig,ax = plt.subplots(figsize=(15, 15),dpi=100)
fig.subplots_adjust(0, 0, 1, 1)
ax.set_axis_off()
ax.matshow(a)
plt.subplots_adjust(bottom=0.05, top=1.25)
# #plt.show()
plt.close()

"""
fits_dir_bz_central_coaligned = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_Bz.fits')
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_By.fits')
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
ax1.set_xlabel('Helioprojective Longitude (Solar-X)')
ax1.set_ylabel('Helioprojective Latitude (Solar-Y)')

fits_dir_bz_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_Bz.fits')
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_By.fits')
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
ax2.set_xlabel('Helioprojective Longitude (Solar-X)')
ax2.set_ylabel('Helioprojective Latitude (Solar-Y)')

fits_dir_bz_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_Bz.fits')
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_By.fits')
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
ax3.set_xlabel('Helioprojective Longitude (Solar-X)')
ax3.set_ylabel('Helioprojective Latitude (Solar-Y)')

fits_dir_bz_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_Bz.fits')
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_By.fits')
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
ax4.set_xlabel('Helioprojective Longitude (Solar-X)')
ax4.set_ylabel('Helioprojective Latitude (Solar-Y)')

fits_dir_bz_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_Bz.fits')
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_By.fits')
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
ax6.set_xlabel('Helioprojective Longitude (Solar-X)')
ax6.set_ylabel('Helioprojective Latitude (Solar-Y)')

fits_dir_bz_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_Bz.fits')
data_bz_central_coaligned = fits.getdata(fits_dir_bz_central_coaligned)
head_bz_central_coaligned = fits.getheader(fits_dir_bz_central_coaligned)
head_bz_central_coaligned['Observatory'] = ('PSI-MAS')
head_bz_central_coaligned['detector'] = ('KCor')
# print('CRLT_OBS: ' + str(head['CRLT_OBS']),'CRLN_OBS: ' + str(head['CRLN_OBS']))
bz_central_coaligned_map = sunpy.map.Map(data_bz_central_coaligned, head_bz_central_coaligned)

fits_dir_by_central_coaligned = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_By.fits')
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
ax5.set_xlabel('Helioprojective Longitude (Solar-X)')
ax5.set_ylabel('Helioprojective Latitude (Solar-Y)')
mpl.rcParams.update(mpl.rcParamsDefault)
plt.tight_layout()
plt.savefig(os.path.join(repo_path,'Output/Plots/Central_B_Field_Vector_Plots.png'))
#plt.show()
plt.close()
"""

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

print("")

print("MLSO average discrepancy: " + str(np.round(np.average(err_mlso_central_deg),5)))
print("FORWARD average discrepancy: " + str(np.round(np.average(err_forward_central_deg),5)))
print("Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))

print("MLSO LOS average discrepancy: " + str(np.round(np.average(err_mlso_los_deg),5)))
print("FORWARD LOS average discrepancy: " + str(np.round(np.average(err_forward_los_deg),5)))
print("Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
print("")

print("KDE Results: ")
print("")
#compute JS Divergence
result_JSD_MLSO_FORWARD_LOS= JS_Div(kde0_x_mlso_los_deg, kde0_x_forward_los_deg)
print("JS Divergence between MLSO LOS and PSI/FORWARD LOS",result_JSD_MLSO_FORWARD_LOS)
result_JSD_MLSO_FORWARD_Central= JS_Div(kde0_x_mlso_central_deg, kde0_x_forward_central_deg)
print("JS Divergence between MLSO Central and PSI/FORWARD Central",result_JSD_MLSO_FORWARD_Central)

result_JSD12= JS_Div(kde0_x_mlso_los_deg, kde0_x_random_deg)
print("JS Divergence between MLSO LOS and Random",result_JSD12)
result_JSD21= JS_Div(kde0_x_mlso_central_deg, kde0_x_random_deg)
print("JS Divergence between MLSO Central and Random",result_JSD21)

result_JSD12= JS_Div(kde0_x_forward_los_deg, kde0_x_random_deg)
print("JS Divergence between PSI/FORWARD LOS and Random",result_JSD12)
result_JSD21= JS_Div(kde0_x_forward_central_deg, dist_values_random)
print("JS Divergence between PSI/FORWARD Central and Random",result_JSD21)

print("")

#compute KL Divergence
KL_Div_mlso_forward_central = KL_div(kde0_x_mlso_central_deg, kde0_x_forward_central_deg)
print("KL Divergence between MLSO Central and PSI/FORWARD Central",KL_Div_mlso_forward_central)
KL_Div_mlso_forward_los = KL_div(kde0_x_mlso_los_deg, kde0_x_forward_los_deg)
print("KL Divergence between MLSO LOS and PSI/FORWARD LOS",KL_Div_mlso_forward_los)

KL_Div_mlso_central_random = KL_div(kde0_x_mlso_central_deg, kde0_x_random_deg)
print("KL Divergence between MLSO Central and Random",KL_Div_mlso_central_random)
KL_Div_mlso_los_random = KL_div(kde0_x_mlso_los_deg, kde0_x_random_deg)
print("KL Divergence between MLSO LOS and Random",KL_Div_mlso_los_random)

KL_Div_forward_central_random = KL_div(kde0_x_forward_central_deg, dist_values_random)
print("KL Divergence between PSI/FORWARD Central and Random",KL_Div_forward_central_random)
KL_Div_forward_los_random = KL_div(kde0_x_forward_los_deg, kde0_x_random_deg)
print("KL Divergence between PSI/FORWARD LOS and Random",KL_Div_forward_los_random)

print("")
print("Funzies")

result_JSD_MLSO_FORWARD_LOS= JS_Div(kde0_x_mlso_los_deg, dist_values_mlso_los)
print("JS Divergence between MLSO LOS KDE to SNS.DISTPLOT",result_JSD_MLSO_FORWARD_LOS)
result_JSD_MLSO_FORWARD_Central= JS_Div(kde0_x_mlso_central_deg, dist_values_mlso_central)
print("JS Divergence between MLSO Central KDE to SNS.DISTPLOT",result_JSD_MLSO_FORWARD_Central)

from scipy.integrate import quad
import scipy.stats


mean = np.average(err_mlso_los_deg)
std = np.std(err_mlso_los_deg)

"""
def normal_distribution_function(x):
    value = scipy.stats.norm.pdf(x,mean,std)
    return value
x1 = mean + std
x2 = mean + 2.0 * std


x_min = 0
x_max = 90

x = np.linspace(x_min, x_max, 100)

y = scipy.stats.norm.pdf(x,mean,std)

plt.plot(x,y, color='black')

res, err = quad(normal_distribution_function, x1, x2)


print('Normal Distribution (mean,std):',mean,std)
print('Integration bewteen {} and {} --> '.format(x1,x2),res)

#----------------------------------------------------------------------------------------#
# plot integration surface

ptx = np.linspace(x1, x2, 10)
pty = scipy.stats.norm.pdf(ptx,mean,std)

distribution_names = dict({'name': 'Dionysia' ,'age': 28,'location': 'Athens'})

plt.fill_between(ptx, pty, color='#0b559f', alpha=1.0)

#----------------------------------------------------------------------------------------#

plt.grid()

plt.xlim(x_min,x_max)
plt.ylim(0,max(y))

plt.title('How to integrate a normal distribution in python ?',fontsize=10)

plt.xlabel('x')
plt.ylabel('Normal Distribution')

plt.savefig("integrate_normal_distribution.png")
#plt.show()
plt.close()
"""

def integrate_distribution(dist, x1, x2, x_min, x_max):


    mean = np.average(dist)
    std = np.std(dist)


    kde0 = gaussian_kde(dist)
    # x_1_mlso_central_deg = np.linspace(xmin_mlso_central, xmax_mlso_central, 200)


    x = np.linspace(x_min, x_max, 200)
    y = kde0(x)

    plt.plot(x,y, color='black')

    res, err = quad(kde0_forward_central_deg, x1, x2)
    print(' Distribution (mean,std):',mean,std)
    print('Integration bewteen {} and {} --> '.format(x1,x2),res)

    #----------------------------------------------------------------------------------------#
    # plot integration surface

    ptx = np.linspace(x1, x2, 200)
    pty = kde0(ptx)
    plt.fill_between(ptx, pty, color='#0b559f', alpha=1.0)

    plt.grid()
    #
    plt.xlim(x_min,x_max)
    # plt.ylim(0,max(y))

    dist_name = 'No Name'

    if dist.all() == err_mlso_central_deg.all():
        dist_name = 'MLSO Central'
    if dist.all() == err_forward_central_deg.all():
        dist_name = 'PSI/FORWARD Central'
    if dist.all() == err_mlso_los_deg.all():
        dist_name = 'MLSO LOS'
    if dist.all() == err_forward_los_deg.all():
        dist_name = 'PSI/FORWARD LOS'


    plt.title('Probability Density Integral for {} between points {}, {}: {}'.format(dist_name,x1,x2,round(res,5)),fontsize=10)

    plt.xlabel('x')
    plt.ylabel('Normal Distribution')

    plt.savefig(os.path.join(repo_path,'Output/Plots/integrate_normal_distribution.png'))
    # return res

    # #plt.show()
    plt.close()

    return res

integrate_distribution(err_mlso_central_deg,-14.5,0,xmin_mlso_central, xmax_mlso_central)




# Make Combined Plot



fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170820_180657_kcor_l2_avg.fts')

data1 = fits.getdata(fits_dir_mlso)
head1 = fits.getheader(fits_dir_mlso)
head1['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data1, head1)
# psimap.plot_settings['norm'] = plt.Normalize(mlsomap.min(), mlsomap.max())

fig1 = plt.figure(figsize=(10, 10))
ax1 = fig1.add_subplot(3, 2, 1, projection=mlsomap)
mlsomap.plot(axes=ax1,title=False)
R_SUN = head1['R_SUN']
ax1.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))

fits_dir_mlso = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')

data2 = fits.getdata(fits_dir_mlso)
head2 = fits.getheader(fits_dir_mlso)
head2['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data2, head2)

ax2 = fig1.add_subplot(3, 2, 2, projection=mlsomap)
mlsomap.plot(axes=ax2,title=False,norm=matplotlib.colors.LogNorm())
R_SUN = head2['R_SUN']
ax2.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170825_185258_kcor_l2_avg.fts')

data3 = fits.getdata(fits_dir_mlso)
head3 = fits.getheader(fits_dir_mlso)
head3['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data3, head3)

ax3 = fig1.add_subplot(3, 2, 3, projection=mlsomap)
mlsomap.plot(axes=ax3,title=False)
R_SUN = head3['R_SUN']
ax3.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_mlso = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_pB.fits')

data4 = fits.getdata(fits_dir_mlso)
head4 = fits.getheader(fits_dir_mlso)
head4['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data4, head4)

ax4 = fig1.add_subplot(3, 2, 4, projection=mlsomap)
mlsomap.plot(axes=ax4,title=False,norm=matplotlib.colors.LogNorm())
R_SUN = head4['R_SUN']
ax4.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_mlso = os.path.join(repo_path,'Data/MLSO/20170829_200801_kcor_l2_avg.fts')

data5 = fits.getdata(fits_dir_mlso)
head5 = fits.getheader(fits_dir_mlso)
head5['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data5, head5)


ax5 = fig1.add_subplot(3, 2, 5, projection=mlsomap)
mlsomap.plot(axes=ax5,title=False)
R_SUN = head5['R_SUN']
ax5.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_mlso = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_pB.fits')

data6 = fits.getdata(fits_dir_mlso)
head6 = fits.getheader(fits_dir_mlso)
head6['detector'] = ('KCor')
mlsomap = sunpy.map.Map(data6, head6)

ax6 = fig1.add_subplot(3, 2, 6, projection=mlsomap)
ax6.set_xlabel(' ')
mlsomap.plot(axes=ax6,title=False,norm=matplotlib.colors.LogNorm())
R_SUN = head6['R_SUN']
ax6.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))
plt.subplots_adjust(bottom=0.05, top=0.95)
# #plt.show()
plt.savefig(os.path.join(repo_path,'Output/Plots/Combined_Plots_1.png'))
plt.close()
# #plt.show()

By1 = os.path.join(repo_path,'Data/MLSO/20170820_180657_kcor_l2_avg.fts')
# By1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')
By2 = os.path.join(repo_path,'Data/MLSO/20170825_185258_kcor_l2_avg.fts')
# By2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_By.fits')
By3 = os.path.join(repo_path,'Data/MLSO/20170829_200801_kcor_l2_avg.fts')
# By3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_By.fits')
By4 = os.path.join(repo_path,'Data/MLSO/20170903_025117_kcor_l2_avg.fts')
# By4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_By.fits')
By5 = os.path.join(repo_path,'Data/MLSO/20170906_213054_kcor_l2_avg.fts')
# By5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_By.fits')
By6 = os.path.join(repo_path,'Data/MLSO/20170911_202927_kcor_l2_avg.fts')

Bz1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')
# By1 = os.path.join(repo_path,'Output/fits_images/6.89000_303.470_pB.fits')
Bz2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_pB.fits')
# By2 = os.path.join(repo_path,'Output/fits_images/7.05600_236.978_By.fits')
Bz3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_pB.fits')
# By3 = os.path.join(repo_path,'Output/fits_images/7.15300_183.443_By.fits')
Bz4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_pB.fits')
# By4 = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_By.fits')
Bz5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_pB.fits')
# By5 = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_By.fits')
Bz6 = os.path.join(repo_path,'Output/fits_images/7.23800_11.5530_pB.fits')


fits_dir_psi = os.path.join(repo_path,'Data/MLSO/20170903_025117_kcor_l2_avg.fts')

data1 = fits.getdata(fits_dir_psi)
head1 = fits.getheader(fits_dir_psi)
head1['detector'] = ('KCor')
psimap = sunpy.map.Map(data1, head1)
# psimap.plot_settings['norm'] = plt.Normalize(psimap.min(), psimap.max())

fig2 = plt.figure(figsize=(10, 10))
ax1 = fig2.add_subplot(3, 2, 1, projection=psimap)
psimap.plot(axes=ax1,title=False)
R_SUN = head1['R_SUN']
ax1.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))

fits_dir_psi = os.path.join(repo_path,'Output/fits_images/7.22000_126.906_pB.fits')

data2 = fits.getdata(fits_dir_psi)
head2 = fits.getheader(fits_dir_psi)
head2['detector'] = ('KCor')
psimap = sunpy.map.Map(data2, head2)

ax2 = fig2.add_subplot(3, 2, 2, projection=psimap)
psimap.plot(axes=ax2,title=False,norm=matplotlib.colors.LogNorm())
R_SUN = head2['R_SUN']
ax2.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_psi = os.path.join(repo_path,'Data/MLSO/20170906_213054_kcor_l2_avg.fts')

data3 = fits.getdata(fits_dir_psi)
head3 = fits.getheader(fits_dir_psi)
head3['detector'] = ('KCor')
psimap = sunpy.map.Map(data3, head3)

ax3 = fig2.add_subplot(3, 2, 3, projection=psimap)
psimap.plot(axes=ax3,title=False)
R_SUN = head3['R_SUN']
ax3.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_psi = os.path.join(repo_path,'Output/fits_images/7.24700_77.0150_pB.fits')

data4 = fits.getdata(fits_dir_psi)
head4 = fits.getheader(fits_dir_psi)
head4['detector'] = ('KCor')
psimap = sunpy.map.Map(data4, head4)

ax4 = fig2.add_subplot(3, 2, 4, projection=psimap)
psimap.plot(axes=ax4,title=False,norm=matplotlib.colors.LogNorm())
R_SUN = head4['R_SUN']
ax4.add_patch(Circle((512,512), R_SUN, color='black',zorder=100))


fits_dir_psi = os.path.join(repo_path,'Data/MLSO/20170911_202927_kcor_l2_avg.fts')

data5 = fits.getdata(fits_dir_psi)
head5 = fits.getheader(fits_dir_psi)
head5['detector'] = ('KCor')
psimap = sunpy.map.Map(data5, head5)


ax5 = fig2.add_subplot(3, 2, 5, projection=psimap)
psimap.plot(axes=ax5,title=False)
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
# #plt.show()
plt.savefig(os.path.join(repo_path,'Output/Plots/Combined_Plots_2.png'))
plt.close()
# #plt.show()


# import matplotlib as mpl
# backend = mpl.get_backend()
# mpl.use('TkAgg')
# matplotlib.use('TkAgg')


c1 = fig1.canvas
c2 = fig2.canvas

c1.draw()
c2.draw()

a1 = np.array(c1.buffer_rgba())
a2 = np.array(c2.buffer_rgba())
a = np.hstack((a1,a2))


# mpl.use(backend)
fig,ax = plt.subplots(figsize=(15, 15),dpi=100)
fig.subplots_adjust(0, 0, 1, 1)
ax.set_axis_off()
ax.matshow(a)
plt.subplots_adjust(bottom=0.05, top=1.25)
plt.savefig(os.path.join(repo_path,'Output/Plots/Combined_Plot.png'))
# #plt.show()
plt.close()

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
colnames=['year', 'day', 'rad_au', 'hg_lat','hg_lon']
hi = pd.read_csv(os.path.join(repo_path,'Data/Ephemeris/helios_PKexb4xQt9.lst.txt'),delim_whitespace=True,names=colnames,skiprows=1)
epoch = datetime(2017,1,1,0)
xnew = np.linspace(229,255,1000000)
f1 = interp1d(hi['day'].values,hi['hg_lon'].values,kind='linear')
ynew = f1(xnew)

plt.gcf().clear()
fig = plt.figure(1,figsize=(10,10))
ax = fig.add_subplot(111)
ax.scatter(hi['day'].values,hi['hg_lon'].values,label='Earth position from JPL ephemeris')
ax.plot(xnew,ynew,label='Interpolated earth position')
ax.set_ylabel('Heliographic Longitude of Earth')
ax.set_xlabel('Day in Year')
ax.set_title('Position of Earth in August-September 2017')
delta_time = epoch + timedelta(days=xnew[np.where(np.round(ynew,3)==303.470)[0][1]]-1)
# [line1] = plt.axvline(xnew[np.where(np.round(ynew,3)==303.470)[0][1]],linestyle='--',label='Locations of PSI/FORWARD Model Slices')
ax.axvline(xnew[np.where(np.round(ynew,3)==303.470)[0][1]],linestyle='--',label='Locations of PSI/FORWARD Model Slices')
ax.axvline(xnew[np.where(np.round(ynew,3)==236.978)[0][1]],linestyle='--')
ax.axvline(xnew[np.where(np.round(ynew,3)==183.443)[0][1]],linestyle='--')
ax.axvline(xnew[np.where(np.round(ynew,3)==126.906)[0][1]],linestyle='--')
ax.axvline(xnew[np.where(np.round(ynew,3)==77.015)[0][1]],linestyle='--')
ax.axvline(xnew[np.where(np.round(ynew,3)==11.553)[0][1]],linestyle='--')
handles, labels = ax.get_legend_handles_labels()
lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.04, 1))
plt.legend(bbox_to_anchor=(1.04, 1))
plt.savefig(os.path.join(repo_path,'Output/Plots/Ephemeris_Plot.png'),bbox_extra_artists=(lgd))
# #plt.show()

#
# # new bandwith STUFF
# bandwidth = 0.5
# kde0 = gaussian_kde(err_mlso_los_deg, bw_method=bandwidth)
# dist_values_mlso_los = sns.kdeplot(err_mlso_los_deg,bw_adjust=bandwidth).get_lines()[0].get_data()[1]
# dist_values_mlso_los_norm = dist_values_mlso_los / dist_values_mlso_los.max()
# kde0_x_norm = kde0_x / kde0_x.max()
# plt.plot(x_1,kde0_x_norm,label='scipy')
# plt.plot(x_1, dist_values_mlso_los_norm, label='seaborne')
# plt.legend()
# #plt.show()



data_dir_2 = os.path.join(repo_path,'Data/QRaFT/errors_cor1.sav')
idl_save_2 = readsav(data_dir_2)
err_cor1_central = idl_save_2['ERR_ARR_COR1']
err_cor1_los = idl_save_2['ERR_ARR_LOS_COR1']
err_forward_cor1_central = idl_save_2['ERR_ARR_FORWARD']
err_forward_cor1_los = idl_save_2['ERR_ARR_LOS_FORWARD']
# err_random = idl_save_2['ERR_ARR_RND']


# Generate plots for Central arrays
mpl.rcParams.update(mpl.rcParamsDefault)

# convert arrays from radians to degrees
err_cor1_central_deg = err_cor1_central[np.where(err_cor1_central > 0)]*180/np.pi
err_forward_cor1_central_deg = err_forward_cor1_central[np.where(err_forward_cor1_central > 0)]*180/np.pi
err_random_deg = err_random[np.where(err_random > 0)]*180/np.pi

# set minimum and maximum x values for gaussian kde calculation
xmin_cor1_central = -14.5
xmax_cor1_central = 104.0

# Calculate Gaussian KDE for cor1 pB vs central B field dataset
kde0_cor1_central_deg = gaussian_kde(err_cor1_central_deg)
x_1_cor1_central_deg = np.linspace(xmin_cor1_central, xmax_cor1_central, 1000000)
kde0_x_cor1_central_deg = kde0_cor1_central_deg(x_1_cor1_central_deg)
# plt.plot(x_1_mlso_central_deg, kde0_x_mlso_central_deg, color='b', label='mlso central KDE scipy')

xmin_forward_cor1_central = -17.24
xmax_forward_cor1_central = 106.13

# Calculate Gaussian KDE for forward pB vs central B field dataset (cor-1 version)
kde0_forward_cor1_central = gaussian_kde(err_forward_cor1_central_deg)
x_1_forward_cor1_central_deg = np.linspace(xmin_forward_cor1_central, xmax_forward_cor1_central, 1000000)
kde0_x_forward_cor1_central_deg = kde0_forward_cor1_central(x_1_forward_cor1_central_deg)
# plt.plot(x_1_forward_central_deg, kde0_x_forward_central_deg, color='b', label='forward central KDE scipy')

xmin_random = -18.395
xmax_random = 108.39

# Calculate Gaussian KDE for forward pB vs random B field dataset
kde0_random_deg = gaussian_kde(err_random_deg)
x_1_random_deg = np.linspace(xmin_random, xmax_random, 1000000)
kde0_x_random_deg = kde0_random_deg(x_1_random_deg)


fig = plt.figure(figsize=(8,8))
ax = fig.subplots(1,1)
sns.distplot(err_cor1_central_deg,hist=True,label='COR-1',bins=30,ax=ax)
sns.distplot(err_forward_cor1_central_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax)
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax)
ax.set_xlabel('Angle Discrepancy')
ax.set_ylabel('Probability Density')
ax.set_title('QRaFT Feature Tracing Performance Against Central POS $B$ Field')
ax.set_xlim(0,90)
ax.set_ylim(0,0.07)
ax.legend()

# plt.text(20,0.045,"COR1 average discrepancy: " + str(np.round(np.average(err_cor1_central_deg),5)))
# plt.text(20,0.04,"FORWARD average discrepancy: " + str(np.round(np.average(err_forward_cor1_central_deg),5)))
# plt.text(20,0.035,"Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))
plt.savefig(os.path.join(repo_path,'Output/Plots/COR1_vs_FORWARD_Feature_Tracing_Performance.png'))
# #plt.show()
plt.close()




# Generate figure for combined plot of COR-1 and MLSO K-COR datasets
fig, ax = plt.subplots(1,2,figsize=(24,9))

# Generate plots for MLSO K-COR dataset on left axis
sns.distplot(err_mlso_central_deg,hist=True,label='MLSO K-COR',bins=30,ax=ax[0])
sns.distplot(err_forward_central_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax[0])
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax[0])
ax[0].set_xlabel('Angle Discrepancy',fontsize=22)
ax[0].set_ylabel('Probability Density',fontsize=22)
ax[0].set_title('QRaFT Feature Tracing Performance MLSO K-COR vs PSI',fontsize=22)
ax[0].set_xlim(0,90)
ax[0].set_ylim(0,0.07)
ax[0].legend(fontsize=20)

# Generate plots for COR1 dataset on right axis
sns.distplot(err_cor1_central_deg,hist=True,label='COR-1',bins=30,ax=ax[1])
sns.distplot(err_forward_cor1_central_deg,hist=True,label='PSI/FORWARD pB',bins=30,ax=ax[1])
sns.distplot(err_random_deg,hist=False,label='Random',ax=ax[1])
# sns.kdeplot(err_mlso_los_deg,label='KDE')
ax[1].set_xlabel('Angle Discrepancy',fontsize=22)
ax[1].set_ylabel('Probability Density',fontsize=22)
ax[1].set_title('QRaFT Feature Tracing Performance COR-1 vs PSI',fontsize=22)
ax[1].set_xlim(0,90)
ax[1].set_ylim(0,0.07)
ax[1].legend(fontsize=20)
plt.savefig(os.path.join(repo_path,'Output/Plots/COR1_vs_FORWARD_Feature_Tracing_Performance_Combined.png'))
# #plt.show()
plt.close()



# retrieve probability density data from seaborne distplots
dist_values_cor1_central = sns.distplot(err_cor1_central_deg).get_lines()[0].get_data()[1]
plt.close()
dist_values_forward_cor1_central = sns.distplot(err_forward_cor1_central_deg).get_lines()[0].get_data()[1]
plt.close()

# dist_values_mlso_los = sns.distplot(err_mlso_los_deg).get_lines()[0].get_data()[1]
# plt.close()
# dist_values_forward_los = sns.distplot(err_forward_los_deg).get_lines()[0].get_data()[1]
# plt.close()

dist_values_random = sns.distplot(err_random_deg).get_lines()[0].get_data()[1]
plt.close()




"""
print("")
#compute JS Divergence
# result_JSD_MLSO_FORWARD_LOS= JS_Div(dist_values_mlso_los, dist_values_forward_los)
# print("JS Divergence between MLSO LOS and PSI/FORWARD LOS",result_JSD_MLSO_FORWARD_LOS)
result_JSD_COR1_FORWARD_Central= JS_Div(dist_values_cor1_central, dist_values_forward_cor1_central)
print("JS Divergence between COR-1 Central and PSI/FORWARD Central",result_JSD_COR1_FORWARD_Central)

# result_JSD12= JS_Div(dist_values_mlso_los, dist_values_random)
# print("JS Divergence between MLSO LOS and Random",result_JSD12)
result_JSD_COR1_Random= JS_Div(dist_values_cor1_central, dist_values_random)
print("JS Divergence between COR1 Central and Random",result_JSD_COR1_Random)

# result_JSD12= JS_Div(dist_values_forward_los, dist_values_random)
# print("JS Divergence between PSI/FORWARD LOS and Random",result_JSD12)
result_JSD_COR1_FORWARD_Random= JS_Div(dist_values_forward_cor1_central, dist_values_random)
print("JS Divergence between PSI/FORWARD Central (COR-1) and Random",result_JSD_COR1_FORWARD_Random)

print("")

#compute KL Divergence
KL_Div_cor1_forward_central = KL_div(dist_values_cor1_central,dist_values_forward_cor1_central)
print("KL Divergence between COR-1 Central and PSI/FORWARD Central",KL_Div_cor1_forward_central)
# KL_Div_mlso_forward_los = KL_div(dist_values_mlso_los, dist_values_forward_los)
# print("KL Divergence between MLSO LOS and PSI/FORWARD LOS",KL_Div_mlso_forward_los)

KL_Div_cor1_central_random = KL_div(dist_values_cor1_central,dist_values_random)
print("KL Divergence between COR-1 Central and Random",KL_Div_cor1_central_random)
# KL_Div_mlso_los_random = KL_div(dist_values_mlso_los, dist_values_random)
# print("KL Divergence between MLSO LOS and Random",KL_Div_mlso_los_random)

KL_Div_cor1_forward_central_random = KL_div(dist_values_forward_cor1_central,dist_values_random)
print("KL Divergence between PSI/FORWARD Central (COR-1) and Random",KL_Div_cor1_forward_central_random)
# KL_Div_forward_los_random = KL_div(dist_values_forward_los, dist_values_random)
# print("KL Divergence between PSI/FORWARD LOS and Random",KL_Div_forward_los_random)
"""
print("")

print("COR-1 Central average discrepancy: " + str(np.round(np.average(err_cor1_central_deg),5)))
print("FORWARD Central average discrepancy: " + str(np.round(np.average(err_forward_cor1_central_deg),5)))
print("Random average discrepancy: " + str(np.round(np.average(err_random_deg),5)))

print("COR-1 Central median discrepancy: " + str(np.round(np.median(err_cor1_central_deg),5)))
print("FORWARD Central median discrepancy: " + str(np.round(np.median(err_forward_cor1_central_deg),5)))
print("Random average discrepancy: " + str(np.round(np.median(err_random_deg),5)))
print("")

# print("KDE Results: ")
# print("")

# err_cor1_central_deg, err_forward_cor1_central_deg, err_random_deg
# kde0_x_cor1_central_deg, kde0_x_forward_cor1_central_deg, kde0_x_random_deg
#compute JS Divergence
# result_JSD_MLSO_FORWARD_LOS= JS_Div(kde0_x_mlso_los_deg, kde0_x_forward_los_deg)
# print("JS Divergence between MLSO LOS and PSI/FORWARD LOS",result_JSD_MLSO_FORWARD_LOS)
result_JSD_COR1_FORWARD_Central= JS_Div(kde0_x_cor1_central_deg, kde0_x_forward_cor1_central_deg)
print("JS Divergence between COR-1 Central and PSI/FORWARD Central: ",result_JSD_COR1_FORWARD_Central)

# result_JSD12= JS_Div(kde0_x_mlso_los_deg, kde0_x_random_deg)
# print("JS Divergence between MLSO LOS and Random",result_JSD12)
result_JSD_COR1_Central_Random= JS_Div(kde0_x_cor1_central_deg, kde0_x_random_deg)
print("JS Divergence between COR1 Central and Random: ",result_JSD_COR1_Central_Random)

# result_JSD12= JS_Div(kde0_x_forward_los_deg, kde0_x_random_deg)
# print("JS Divergence between PSI/FORWARD LOS and Random",result_JSD12)
result_JSD_COR1_Forward_Central_Random= JS_Div(kde0_x_forward_cor1_central_deg, kde0_x_random_deg)
print("JS Divergence between PSI/FORWARD Central and Random",result_JSD_COR1_Forward_Central_Random)

print("")

#compute KL Divergence
KL_Div_cor1_forward_central = KL_div(kde0_x_cor1_central_deg, kde0_x_forward_cor1_central_deg)
print("KL Divergence between COR1 Central and PSI/FORWARD Central: ",KL_Div_cor1_forward_central)
# KL_Div_mlso_forward_los = KL_div(kde0_x_mlso_los_deg, kde0_x_forward_los_deg)
# print("KL Divergence between MLSO LOS and PSI/FORWARD LOS",KL_Div_mlso_forward_los)

KL_Div_cor1_central_random = KL_div(kde0_x_cor1_central_deg, kde0_x_random_deg)
print("KL Divergence between COR1 Central and Random: ",KL_Div_cor1_central_random)
# KL_Div_mlso_los_random = KL_div(kde0_x_mlso_los_deg, kde0_x_random_deg)
# print("KL Divergence between MLSO LOS and Random",KL_Div_mlso_los_random)

KL_Div_cor1_forward_central_random = KL_div(kde0_x_forward_cor1_central_deg, kde0_x_random_deg)
print("KL Divergence between PSI/FORWARD Central (COR1) and Random: ",KL_Div_cor1_forward_central_random)
# KL_Div_forward_los_random = KL_div(kde0_x_forward_los_deg, kde0_x_random_deg)
# print("KL Divergence between PSI/FORWARD LOS and Random",KL_Div_forward_los_random)

print("")
print("yay")

# result_JSD_MLSO_FORWARD_LOS= JS_Div(kde0_x_mlso_los_deg, dist_values_mlso_los)
# print("JS Divergence between MLSO LOS KDE to SNS.DISTPLOT",result_JSD_MLSO_FORWARD_LOS)
# result_JSD_MLSO_FORWARD_Central= JS_Div(kde0_x_mlso_central_deg, dist_values_mlso_central)
# print("JS Divergence between MLSO Central KDE to SNS.DISTPLOT",result_JSD_MLSO_FORWARD_Central)

import sunpy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import os
import git
import numpy as np
from scipy.interpolate import interp1d
from sunpy.coordinates.ephemeris import *
from sunpy.coordinates.frames import *
from sunpy.coordinates.sun import L0
from astropy.io import fits
from sunpy.sun import constants
import astropy.constants
import astropy.units as u


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir


# carrington lat/lon in degrees
files = ['Data/MLSO/20170820_180657_kcor_l2_avg.fts', 'Data/MLSO/20170825_185258_kcor_l2_avg.fts', 'Data/MLSO/20170829_200801_kcor_l2_avg.fts', 'Data/MLSO/20170903_025117_kcor_l2_avg.fts', 'Data/MLSO/20170906_213054_kcor_l2_avg.fts', 'Data/MLSO/20170911_202927_kcor_l2_avg.fts']
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

files2 = ['Data/COR1/2017_08_20_rep_med.fts', 'Data/COR1/2017_08_25_rep_med.fts', 'Data/COR1/2017_08_29_rep_med.fts', 'Data/COR1/2017_09_03_rep_med.fts', 'Data/COR1/2017_09_06_rep_med.fts', 'Data/COR1/2017_09_11_rep_med.fts']
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
# ax.scatter(np.deg2rad(longitudes), x_radius/astropy.constants.au.to_value(u.m), color='red',label='MLSO K-COR Observations')
ax.scatter(np.deg2rad(longitudes2), x2_radius/astropy.constants.au.to_value(u.m), color='blue',label='COR-1 Observations')
ax.set_theta_zero_location("S")
ax.legend(bbox_to_anchor=(1, 1.05), loc="upper right")
ax.set_title('Locations of Each Observation in Dataset')
ax.set_rlim(0, 1.3)
plt.savefig(os.path.join(repo_path,'Output/Plots/Polar_Observations_Plot.png'))
# #plt.show()

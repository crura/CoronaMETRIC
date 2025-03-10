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
from os.path import join, isfile
from os import listdir
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

datapath = join(repo_path, 'Data/QRaFT/COR-1_Errors_New')
datafiles = [join(datapath,f) for f in listdir(datapath) if isfile(join(datapath,f)) and f !='.DS_Store']

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
plt.close()









data_dir_2 = os.path.join(repo_path,'Data/QRaFT/errors_cor1.sav')
data_dir_new = os.path.join(repo_path,'Data/QRaFT/COR-1_Errors/errors_all.sav')


idl_save_2 = readsav(data_dir_2)
idl_save_new = readsav(data_dir_new)

err_cor1_central = idl_save_2['ERR_ARR_COR1']
err_cor1_los = idl_save_2['ERR_ARR_LOS_COR1']
err_forward_cor1_central = idl_save_2['ERR_ARR_FORWARD']
err_forward_cor1_los = idl_save_2['ERR_ARR_LOS_FORWARD']
# err_random = idl_save_2['ERR_ARR_RND']

err_cor1_central_new = idl_save_new['ERR_ARR_COR1']
err_cor1_los_new = idl_save_new['ERR_ARR_LOS_COR1']
err_forward_cor1_central_new = idl_save_new['ERR_ARR_FORWARD']
err_forward_cor1_los_new = idl_save_new['ERR_ARR_LOS_FORWARD']
err_random_new = idl_save_new['ERR_ARR_RND']

# convert arrays from radians to degrees
err_cor1_central_deg_new = err_cor1_central_new[np.where(err_cor1_central_new > 0)]*180/np.pi
err_forward_cor1_central_deg_new = err_forward_cor1_central_new[np.where(err_forward_cor1_central_new > 0)]*180/np.pi
err_random_deg_new = err_random_new[np.where(err_random_new > 0)]*180/np.pi

def calculate_KDE(err_array):
    # set minimum and maximum x values for gaussian kde calculation
    xmin = min(err_array)
    xmax = max(err_array)

    # Calculate Gaussian KDE for cor1 pB vs central B field dataset
    kde = gaussian_kde(err_array)
    x_1 = np.linspace(xmin, xmax, 1000000)
    kde0 = kde(x_1)
    return kde0

def calculate_KDE_statistics(KDE_1, KDE_2):

    #compute JS Divergence
    result_JSD = JS_Div(KDE_1, KDE_2)

    #compute KL Divergence
    result_KLD = KL_div(KDE_1, KDE_2)

    return result_JSD, result_KLD

KDE_cor1_central_deg_new = calculate_KDE(err_cor1_central_deg_new)
KDE_forward_cor1_central_deg_new = calculate_KDE(err_forward_cor1_central_deg_new)
KDE_random_deg_new = calculate_KDE(err_random_deg_new)

JSD_cor1_forward_central_new, KLD_cor1_forward_central_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_forward_cor1_central_deg_new)
JSD_cor1_central_random_new, KLD_cor1_central_random_new = calculate_KDE_statistics(KDE_cor1_central_deg_new, KDE_random_deg_new)
JSD_COR1_Forward_Central_Random_new, KLDcor1_forward_central_random_new = calculate_KDE_statistics(KDE_forward_cor1_central_deg_new, KDE_random_deg_new)

combined_dict = dict(metric=['KL Divergence', 'JS Divergence'],
                    cor1_v_psi=[KLD_cor1_forward_central_new, JSD_cor1_forward_central_new],
                    cor1_v_random=[KLD_cor1_central_random_new, JSD_cor1_central_random_new],
                    psi_v_random=[KLDcor1_forward_central_random_new, JSD_COR1_Forward_Central_Random_new])

pd.set_option('display.float_format', '{:.3E}'.format)
stats_df = pd.DataFrame(combined_dict)
stats_df.columns = ['metric', 'cor1 vs psi pB', 'cor1 vs random', 'psi pB vs random']
print(stats_df.to_latex(index=False))

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

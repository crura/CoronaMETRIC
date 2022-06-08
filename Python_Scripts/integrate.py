import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import matplotlib
from tqdm import tqdm_notebook
import pandas as pd
import matplotlib as mpl
import git
from scipy.io import readsav
import unittest

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
crln_obs = idl_save['crln_obs']
crlt_obs = idl_save['crlt_obs']
crln_obs_print = idl_save['crln_obs_print']
crlt_obs_print = idl_save['crlt_obs_print']
occlt = idl_save['occlt']
r_sun_range = idl_save['range']

params = str(crlt_obs_print,'utf-8') + '_' +  str(crln_obs_print,'utf-8')
repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

print('generating integrated electron density')
# Generate integrated electron density
input_path = os.path.join(repo_path,'Data/Rotated_Density_LOS')
path = input_path + "/"
parent_list = os.listdir(input_path)
imagelist = []
headlist = []
for child in parent_list:


    file = str(path + child)
    df_dens=pd.read_csv(file, sep=',',header=None)
    image_data = df_dens.values
    imagelist.append(image_data)

image_sum = np.sum(imagelist, axis=0)

outpath = 'Data/Integrated_Parameters'

np.savetxt(os.path.join(repo_path,outpath,'Integrated_Electron_Density.csv'),image_sum.ravel(),delimiter=',') # save the integrated electron density as a 1D array in a csv file


# Generate integrated Bx
print('generating LOS-integrated Bx')
input_path_bx = os.path.join(repo_path,'Data/Bx_Rotated')
path_bx = input_path_bx + "/"
parent_list_bx = os.listdir(input_path_bx)
imagelist_bx = []
headlist = []
for child in parent_list_bx:


    file = str(path_bx + child)
    df_bx=pd.read_csv(file, sep=',',header=None)
    image_data_bx = df_bx.values
    imagelist_bx.append(image_data_bx)

image_sum_bx = np.sum(imagelist_bx, axis=0)

outpath = 'Data/Integrated_Parameters'

np.savetxt(os.path.join(repo_path,outpath,'Integrated_LOS_Bx.csv'),image_sum_bx.ravel(),delimiter=',') # save the LOS integrated Bx as a 1D array in a csv file


# Generate integrated By
print('generating LOS-integrated By')
input_path_by = os.path.join(repo_path,'Data/By_Rotated')
path_by = input_path_by + "/"
parent_list_by = os.listdir(input_path_by)
imagelist_by = []
headlist = []
for child in parent_list_by:


    file = str(path_by + child)
    df_by=pd.read_csv(file, sep=',',header=None)
    image_data_by = df_by.values
    imagelist_by.append(image_data_by)

image_sum_by = np.sum(imagelist_by, axis=0)

outpath = 'Data/Integrated_Parameters'

np.savetxt(os.path.join(repo_path,outpath,'Integrated_LOS_By.csv'),image_sum_by.ravel(),delimiter=',') # save the LOS integrated By as a 1D array in a csv file


# Generate integrated Bz
print('generating LOS-integrated Bz')
input_path_bz = os.path.join(repo_path,'Data/Bz_Rotated')
path_bz = input_path_bz + "/"
parent_list_bz = os.listdir(input_path_bz)
imagelist_bz = []
headlist = []
for child in parent_list_bz:


    file = str(path_bz + child)
    df_bz=pd.read_csv(file, sep=',',header=None)
    image_data_bz = df_bz.values
    imagelist_bz.append(image_data_bz)

image_sum_bz = np.sum(imagelist_bz, axis=0)

outpath = 'Data/Integrated_Parameters'

np.savetxt(os.path.join(repo_path,outpath,'Integrated_LOS_Bz.csv'),image_sum_bz.ravel(),delimiter=',') # save the LOS integrated Bz as a 1D array in a csv file

# generate Bz vs By vector plot showing LOS-Integrated B field line tracing
print('generating LOS Bz vs By vector plot')
# define n x n grid with spacing dy, dz
dy = np.linspace(0,image_data_by.shape[0]-1, image_data_by.shape[0])
dz = np.linspace(0,image_data_bz.shape[0]-1,image_data_bz.shape[0])
# make plot large
mpl.rcParams.update(mpl.rcParamsDefault)
plt.quiver(dy,dz,image_sum_by,image_sum_bz) #Make B_x, B_y the x and y directions of each field vector
# zoom in plot
plt.axis('equal')
plt.xlim(80,175)
plt.ylim(80,175)
# make title
plt.title('Vector Plot of MLSO Rotated LOS-Integrated By , Bz ' + params)
plt.xlabel('Y Position')
plt.ylabel('Z Position')
plt.savefig(os.path.join(repo_path,'Data/Integrated_Parameters/Plots/LOS_Integrated_By_Bz_Vector_Plot' + params + '.png'))
plt.close()



# generate Bz vs By vector plot showing Central B field line tracing
center_input_path = os.path.join(repo_path,'Data/Central_Parameters')
path_bx_central = os.path.join(center_input_path,'rotated_Bx_2d.csv')
path_by_central = os.path.join(center_input_path,'rotated_By_2d.csv')
path_bz_central = os.path.join(center_input_path,'rotated_Bz_2d.csv')

bx_central_image_data=pd.read_csv(path_bx_central, sep=',',header=None).values
by_central_image_data=pd.read_csv(path_by_central, sep=',',header=None).values
bz_central_image_data=pd.read_csv(path_bz_central, sep=',',header=None).values

print('generating Central Bz vs By vector plot')
# define n x n grid with spacing dy, dz
dy = np.linspace(0,by_central_image_data.shape[0]-1, by_central_image_data.shape[0])
dz = np.linspace(0,bz_central_image_data.shape[0]-1,bz_central_image_data.shape[0])
# make plot large
mpl.rcParams.update(mpl.rcParamsDefault)
plt.quiver(dy,dz,by_central_image_data,bz_central_image_data) #Make B_x, B_y the x and y directions of each field vector
# zoom in plot
plt.axis('equal')
plt.xlim(80,175)
plt.ylim(80,175)
# make title
plt.title('Vector Plot of MLSO Rotated Central By , Bz ' + params)
plt.xlabel('Y Position')
plt.ylabel('Z Position')
plt.savefig(os.path.join(repo_path,'Data/Central_Parameters/Plots/Central_By_Bz_Vector_Plot' + params + '.png'))
plt.close()


plt.imshow(image_sum,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
plt.colorbar(label='log scale electron density')
plt.title('LOS Integrated Electron Density ' + params)
plt.savefig(os.path.join(repo_path,'Data/Integrated_Parameters/Plots/LOS_Integrated_Electron_Density' + params + '.png'))
plt.close()

path_dens_central = os.path.join(center_input_path,'rotated_Dens_2d.csv')
dfDens_central=pd.read_csv(path_dens_central, sep=',',header=None)
arrDens_central = dfDens_central.values
plt.imshow(arrDens_central,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
plt.colorbar(label='log scale electron density')
plt.title('Central Electron Density ' + params)
plt.savefig(os.path.join(repo_path,'Data/Central_Parameters/Plots/Central_Electron_Density' + params + '.png'))
plt.close()

path_dens_forward = os.path.join(repo_path,'Data/Forward_PB_data.csv')
dfDens_forward=pd.read_csv(path_dens_forward, sep=',',header=None)
arrDens_forward = dfDens_forward.values
plt.imshow(arrDens_forward,norm=matplotlib.colors.LogNorm(),cmap='magma',origin='lower')
plt.colorbar(label='log scale electron density')
plt.title('Forward Electron Density ' + params)
plt.savefig(os.path.join(repo_path,'Data/Forward_Parameters/Plots/Forward_Electron_Density_' + params + '.png'))
plt.close()

# Change the duplicate MLSO image carrington latitude and longitude to be equivalent to the model parameter's to allow for interpolation using WCS
mlso_dir = os.path.join(repo_path,'Data/MLSO/20170829_200801_kcor_l2_avg_2.fts')
repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir
idl_save = readsav(os.path.join(repo_path,'Data/model_parameters.sav'))
crln_obs = idl_save['crln_obs']
crlt_obs = idl_save['crlt_obs']
crln_obs_print = float(str(idl_save['crln_obs_print'],'utf-8'))
crlt_obs_print = float(str(idl_save['crlt_obs_print'], 'utf-8'))
occlt = idl_save['occlt']
r_sun_range = idl_save['range']
data = fits.getdata(mlso_dir)
head = fits.getheader(mlso_dir)
head['CRLT_OBS'] = crlt_obs_print
head['CRLN_OBS'] = crln_obs_print
# head['hgln_obs'] = head['CRLN_OBS']
# head['hglt_obs'] = head['CRLT_OBS']
hdunew = fits.PrimaryHDU(data=data,header=head)
hdunew.writeto(mlso_dir,overwrite=True)

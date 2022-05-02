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

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_dir

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
plt.title('Vector Plot of MLSO Rotated LOS-Integrated By , Bz')
plt.xlabel('Y Position')
plt.ylabel('Z Position')
plt.savefig(os.path.join(repo_path,'Data/Integrated_Parameters/Plots/LOS_Integrated_By_Bz_Vector_Plot.png'))

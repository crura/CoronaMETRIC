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

np.savetxt(os.path.join(repo_path,outpath,'Integrated_LOS_Bx.csv'),image_sum_bx.ravel(),delimiter=',') # save the integrated electron density as a 1D array in a csv file

plt.imshow(image_sum_bx,cmap='gist_gray',origin='lower')
plt.colorbar()
plt.title('integrated LOS B_x')
plt.show()

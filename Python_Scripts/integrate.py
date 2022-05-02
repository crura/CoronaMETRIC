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

np.savetxt(os.path.join(repo_path,outpath,'Integrated_Electron_Density.csv'),image_sum.ravel(),delimiter=',')
plt.imshow(image_sum,norm=matplotlib.colors.LogNorm(),cmap='gist_gray',origin='lower')
plt.colorbar()
plt.title('integrated electron density')
plt.show()

import os
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import matplotlib
from tqdm import tqdm_notebook
import pandas as pd
import matplotlib as mpl

# Generate integrated electron density
path = "/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Rotated_Density_LOS/"
parent_list = os.listdir("/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Rotated_Density_LOS")
imagelist = []
headlist = []
for child in parent_list:


    file = str(path + child)
    df_dens=pd.read_csv(file, sep=',',header=None)
    image_data = df_dens.values
    imagelist.append(image_data)

image_sum = np.sum(imagelist, axis=0)
plt.imshow(image_sum,norm=matplotlib.colors.LogNorm(),cmap='gist_gray',origin='lower')
plt.colorbar()
plt.title('integrated electron density')
plt.show()

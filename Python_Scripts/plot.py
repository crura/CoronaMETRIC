data_dir = '/Users/crura/Desktop/Research/Vadim/errors.sav'
from scipy.io import readsav
import matplotlib as mpl
import matplotlib.pyplot as plt
import git

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

import seaborn as sns
sns.distplot(err_mlso_central_deg,hist=True,label='MLSO')
sns.distplot(err_forward_central_deg,hist=True,label='FORWARD')
sns.distplot(err_random_deg,hist=False,label='Random')
plt.xlabel('Angle Discrepancy')
plt.ylabel('Probability Density')
plt.title('MLSO vs FORWARD Feature Tracing Performance')
plt.xlim(0,90)
plt.ylim(0,0.06)
plt.legend()
plt.show()

# Generate plots for LOS arrays
err_mlso_los_deg = err_mlso_los[np.where(err_mlso_los > 0)]*180/np.pi
err_forward_los_deg = err_forward_los[np.where(err_forward_los > 0)]*180/np.pi

sns.distplot(err_mlso_los_deg,hist=True,label='MLSO LOS')
sns.distplot(err_forward_los_deg,hist=True,label='FORWARD LOS')
sns.distplot(err_random_deg,hist=False,label='Random')
plt.xlabel('Angle Discrepancy')
plt.ylabel('Probability Density')
plt.title('MLSO vs FORWARD LOS Feature Tracing Performance')
plt.xlim(0,90)
plt.ylim(0,0.06)
plt.legend()
plt.show()

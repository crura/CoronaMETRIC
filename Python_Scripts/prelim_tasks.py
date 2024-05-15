# interpolate to rebin kcor fits images from 1024x1024 to 512x512

from functions import rescale_kcor_file_to_512x512, get_files_from_pattern
import git
import os


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

fits_path = os.path.join(repo_path, 'Data/MLSO')

fits_files_kcor = get_files_from_pattern(fits_path, 'kcor_l2_avg', '.fts')

for fits_file in fits_files_kcor:
    rescale_kcor_file_to_512x512(fits_file)
    print(f'Finished processing {fits_file}')
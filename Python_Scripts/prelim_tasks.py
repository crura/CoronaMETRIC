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

# interpolate to rebin kcor fits images from 1024x1024 to 512x512

from functions import rescale_kcor_file_to_512x512, get_files_from_pattern
import git
import os
import json


repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir


config_file = os.path.join(repo_path, 'config.json')
with open(config_file) as f:
    config = json.load(f)

fits_path = os.path.join(repo_path, config['kcor_data_path'])
fits_files_kcor = get_files_from_pattern(fits_path, 'kcor_l2_avg', '.fts')

for fits_file in fits_files_kcor:
    rescale_kcor_file_to_512x512(fits_file)
    print(f'Finished processing {fits_file}')
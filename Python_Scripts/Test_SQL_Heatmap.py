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
from functions import calculate_KDE_statistics, determine_paths, get_files_from_pattern, calculate_KDE, plot_histogram_with_JSD_Gaussian_Analysis, correct_fits_header, heatmap_sql_query
import sqlite3
import os
import git

repo = git.Repo('.', search_parent_directories=True)
repo_path = repo.working_tree_dir

con = sqlite3.connect("tutorial.db")

cur = con.cursor()

# Read SQL Query File
with open(os.path.join(repo_path, 'Python_Scripts', 'Test_SQL_Queries.sql'), 'r') as file:
    script = file.read()

cur.executescript(script)
con.commit()

query = "SELECT group1, group2, mean_diff from tukey_hsd_mean_diff_combined_cor1;"
dbName = "tutorial.db"
heatmap_sql_query(dbName, query, print_to_file=True, output_file=os.path.join(repo_path, 'Output/Plots/Test_COR1_Combined_HSD_mean_diff_heatmap.png'), colorbar_label='Absolute Mean Difference (Degrees)', title='Heatmap of Mean Differences by Population', x_label='group 1', y_label='group 2')

query = "SELECT group1, group2, reject from tukey_hsd_reject_combined_cor1;"
dbName = "tutorial.db"
heatmap_sql_query(dbName, query, print_to_file=True, output_file=os.path.join(repo_path, 'Output/Plots/Test_COR1_Combined_HSD_reject_heatmap.png'), colorbar_label='Reject Null Hypothesis?', title='Heatmap of Reject Value by Population', x_label='group 1', y_label='group 2')
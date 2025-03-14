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
source /Users/crura/miniconda3/etc/profile.d/conda.sh &&
conda init bash &&
conda activate test_env &&
python3 -m venv env &&
source env/bin/activate &&
pip install -r requirements.txt &&
python Python_Scripts/new_plot_paper_figures.py
python Python_Scripts/calculate_KDE_statistics.py
deactivate

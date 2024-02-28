source /Users/crura/miniconda3/etc/profile.d/conda.sh &&
conda init bash &&
conda activate test_env &&
python3 -m venv env &&
source env/bin/activate &&
pip install -r requirements.txt &&
python Python_Scripts/new_plot_paper_figures.py
python Python_Scripts/calculate_KDE_statistics.py
deactivate

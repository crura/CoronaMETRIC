source /Users/crura/miniconda3/etc/profile.d/conda.sh &&
conda init bash &&
conda activate test_env &&
python3 -m venv env &&
source env/bin/activate &&
pip install -r requirements.txt &&
# python Python_Scripts/new_plot_paper_figures.py
# python Python_Scripts/test_new_plot_paper_figs.py
# python Python_Scripts/calculate_KDE_statistics.py
python Python_Scripts/Test_Plot_3_0_Features.py
deactivate
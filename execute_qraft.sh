#!/bin/sh -x
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
git_repo=$(git rev-parse --show-toplevel) &&
mkdir -p $git_repo/Output/QRaFT_Results &&
cd $git_repo/QRaFT &&
(echo "ssw_path" &&
echo "the variable is : $1" &&
echo ".compile -v '/Users/crura/SSW/gen/idl/string/strjustify.pro'" &&
echo ".compile -v '/Users/crura/SSW/gen/idl/system/strrep_logenv.pro'" &&
echo ".compile -v '/Users/crura/SSW/gen/idl/string/prstr.pro'" &&
echo ".compile -v '/Users/crura/SSW/gen/idl/genutil/uniqo.pro'" &&
echo "ssw_path, '/Users/crura/SSW/gen'" &&
echo "ssw_path, '/Users/crura/SSW/hinode'" &&
echo "ssw_path, '/Users/crura/SSW/offline'" &&
echo "ssw_path, '/Users/crura/SSW/proba2'" &&
echo "ssw_path, '/Users/crura/SSW/sdo'" &&
echo "ssw_path, '/Users/crura/SSW/site'" &&
echo "ssw_path, '/Users/crura/SSW/so'" &&
echo "ssw_path, '/Users/crura/SSW/soho'" &&
echo "ssw_path, '/Users/crura/SSW/stereo'" &&
echo "ssw_path, '/Users/crura/SSW/trace'" &&
echo "ssw_path, '//Users/crura/SSW/vobs'" &&
echo "ssw_path, '/Users/crura/SSW/packages'" &&
echo ".compile -v '/Users/crura/Desktop/Research/idlroutines/download.pro'" &&
echo ".compile -v '/Users/crura/SSW/packages/forward/idl/DEFAULTS/for_settingdefaults.pro'" &&
echo ".compile -v '/Users/crura/SSW/gen/idl/util/default.pro'" &&
echo ".compile -v '/Users/crura/IDLWorkspace/Default/linspace.pro'" &&
echo ".compile -v '$git_repo/IDL_Scripts/write_psi_image_as_fits.pro'" &&
echo ".compile -v '$git_repo/IDL_Scripts/write_psi_fits.pro'" &&
echo ".compile -v '$git_repo/IDL_Scripts/write_psi_mlso_fits.pro'" &&
echo ".compile -v '$git_repo/IDL_Scripts/rtp2xyz.pro'" &&
echo ".compile -v '$git_repo/IDL_Scripts/repstr.pro'" &&
echo ".compile -v '$git_repo/get_fordump.pro'" &&
echo ".compile -v '$git_repo/image_coalignment.pro'" &&
echo ".compile -v '$git_repo/save_parameters.pro'" &&
echo ".compile -v '$git_repo/generate_forward_model.pro'" &&
echo ".compile -v '$git_repo/run_code.pro'" &&
echo ".compile -v '$git_repo/QRaFT/open_PSI.pro'" &&
echo ".compile -v '$git_repo/QRaFT/QRaFT_PSI.pro'" &&
echo "script4, '$git_repo/Output/fits_images/', '$git_repo/Output/QRaFT_Results/',  err_arr_COR1,  err_arr_LOS_COR1, err_arr_FORWARD,  err_arr_LOS_FORWARD,  err_arr_rnd, L_COR1, L_FORWARD, /manual" &&
echo "script3, '$git_repo/Output/fits_images/', '$git_repo/Output/QRaFT_Results/',  err_arr_MLSO,  err_arr_LOS_MLSO, err_arr_FORWARD,  err_arr_LOS_FORWARD,  err_arr_rnd, L_MLSO, L_FORWARD, /manual" || true &&
# Simulate CTRL-D to exit the cat command
echo -e "\n" | cat) |
/Users/crura/Documents/bin/ssw
mv $git_repo/Output/fits_images/*.sav $git_repo/Output/QRaFT_Results
mv $git_repo/Output/fits_images/*.png $git_repo/Output/QRaFT_Results
deactivate

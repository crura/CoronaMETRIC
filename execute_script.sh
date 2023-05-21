
source /Users/crura/miniconda3/etc/profile.d/conda.sh &&
conda init bash &&
conda activate test_env &&
git_repo=$(git rev-parse --show-toplevel) &&
(echo "ssw_path" &&
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
echo ".compile -v '$git_repo/generate_forward_model.pro'" &&
echo ".compile -v '$git_repo/get_fordump.pro'" &&
echo ".compile -v '$git_repo/image_coalignment.pro'" &&
echo ".compile -v '$git_repo/save_parameters.pro'" &&
echo ".compile -v '$git_repo/run_code.pro'" &&
echo "run_code" &&
cat) | /Users/crura/Documents/bin/ssw

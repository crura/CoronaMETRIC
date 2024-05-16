pro run_code

  ; set path variable for rest of idl code defined by git
  spawn, 'git rev-parse --show-toplevel', git_repo
  ; directory_search_cor1_1 = git_repo + '/Data/COR1_Original/*.fts'
  ; directory_search_cor1_2 = git_repo + '/Data/COR1_Original/*.fits'
  ; directory_list_cor1_1 = file_search(directory_search_cor1_1)
  ; directory_list_cor1_2 = file_search(directory_search_cor1_2)
  ; directory_list = [directory_list_cor1_1, directory_list_cor1_2]

  ; directory_search_kcor_1 = git_repo + '/Data/MLSO/*.fts'
  ; directory_search_kcor_2 = git_repo + '/Data/MLSO/*.fits'
  ; directory_list_kcor_1 = file_search(directory_search_kcor_1)
  ; directory_list_kcor_2 = file_search(directory_search_kcor_2)
  ; directory_list_2 = [directory_list_kcor_1, directory_list_kcor_2]
  
  config = JSON_PARSE('/Users/crura/Desktop/Research/Test_Space/Naty_Images_Experiments/Vadim_QRaFT_Experiments/Start_From_Scrap_2/Image-Coalignment/config.json')
  cor1_data_path = string(config['cor1_data_path'])
  cor1_data_extension = string(config['cor1_data_extension'])
  cor1_pattern_search = string(config['cor1_pattern_search'])
  cor1_pattern_middle = config['cor1_pattern_middle']

  kcor_data_path = string(config['kcor_data_path'])
  kcor_data_extension = string(config['kcor_data_extension'])
  kcor_pattern_search = string(config['kcor_pattern_search'])
  kcor_pattern_middle = config['kcor_pattern_middle']
  
  if (cor1_pattern_middle) then begin
    cor1_search_string = cor1_data_path + '/*' + cor1_pattern_search + '*' + cor1_data_extension
  endif else begin
    cor1_search_string = cor1_data_path + '/*' + cor1_pattern_search + cor1_data_extension
  endelse

  if (kcor_pattern_middle) then begin
    kcor_search_string = kcor_data_path + '/*' + kcor_pattern_search + '*' + kcor_data_extension
  endif else begin
    kcor_search_string = kcor_data_path + '/*' + kcor_pattern_search + kcor_data_extension
  endelse
  
  directory_list = FILE_SEARCH(cor1_search_string)
  directory_list_2 = FILE_SEARCH(kcor_search_string)
  outstring_list = ['']
  occlt_list = ['']
  

  ; directory_list = ['Data/COR1/cor1a_bff_20170820001000.fits', 'Data/COR1/cor1a_bff_20170824233500.fits', 'Data/COR1/cor1a_bff_20170829000500.fits', 'Data/COR1/cor1a_bff_20170903000500.fits', 'Data/COR1/cor1a_bff_20170906000500.fits', 'Data/COR1/cor1a_bff_20170911000500.fits']
  ; directory_list_2 = ['Data/MLSO/20170820_180657_kcor_l2_avg.fts', 'Data/MLSO/20170825_185258_kcor_l2_avg.fts', 'Data/MLSO/20170829_200801_kcor_l2_avg.fts', 'Data/MLSO/20170903_025117_kcor_l2_avg.fts', 'Data/MLSO/20170906_213054_kcor_l2_avg.fts', 'Data/MLSO/20170911_202927_kcor_l2_avg.fts']
  outstring_list = ['']
  occlt_list = ['']
  spawn, 'mkdir -p ' + git_repo + '/Output/fits_images'
  save,outstring_list, directory_list, directory_list_2, occlt_list, filename=git_repo + '/Data/outstrings.sav'
  spawn, 'python Python_Scripts/prelim_tasks.py'

  FOREACH element, directory_list DO BEGIN
    spawn, 'cp ' + element + ' ' + git_repo + '/Output/fits_images'
    hi = generate_forward_model(element)
  ENDFOREACH

  FOREACH element2, directory_list_2 DO BEGIN
    spawn, 'cp ' + element2 + ' ' + git_repo + '/Output/fits_images'
    hi2 = generate_forward_model(element2)
  ENDFOREACH

  spawn, 'python Python_Scripts/new_plot_paper_figures.py'

END

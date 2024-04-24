pro run_code

  directory_list = ['Data/COR1/cor1a_bff_20170820001000.fits', 'Data/COR1/cor1a_bff_20170824233500.fits', 'Data/COR1/cor1a_bff_20170829000500.fits', 'Data/COR1/cor1a_bff_20170903000500.fits', 'Data/COR1/cor1a_bff_20170906000500.fits', 'Data/COR1/cor1a_bff_20170911000500.fits']
  directory_list_2 = ['Data/MLSO/20170820_180657_kcor_l2_avg.fts', 'Data/MLSO/20170825_185258_kcor_l2_avg.fts', 'Data/MLSO/20170829_200801_kcor_l2_avg.fts', 'Data/MLSO/20170903_025117_kcor_l2_avg.fts', 'Data/MLSO/20170906_213054_kcor_l2_avg.fts', 'Data/MLSO/20170911_202927_kcor_l2_avg.fts']
  outstring_list = ['']
  occlt_list = ['']
  ; set path variable for rest of idl code defined by git
  spawn, 'git rev-parse --show-toplevel', git_repo
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

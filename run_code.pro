pro run_code

  directory_list = ['Data/COR1/2017_08_20_rep_med.fts', 'Data/COR1/2017_08_25_rep_med.fts', 'Data/COR1/2017_08_29_rep_med.fts', 'Data/COR1/2017_09_03_rep_med.fts', 'Data/COR1/2017_09_06_rep_med.fts', 'Data/COR1/2017_09_11_rep_med.fts'];'Data/MLSO/20170820_180657_kcor_l2_avg.fts', 'Data/MLSO/20170825_185258_kcor_l2_avg.fts', 'Data/MLSO/20170829_200801_kcor_l2_avg.fts', 'Data/MLSO/20170903_025117_kcor_l2_avg.fts', 'Data/MLSO/20170906_213054_kcor_l2_avg.fts', 'Data/MLSO/20170911_202927_kcor_l2_avg.fts']
  FOREACH element, directory_list DO BEGIN
    hi = generate_forward_model(element)
  ENDFOREACH

  directory_list_2 = ['Data/MLSO/20170820_180657_kcor_l2_avg.fts', 'Data/MLSO/20170825_185258_kcor_l2_avg.fts', 'Data/MLSO/20170829_200801_kcor_l2_avg.fts', 'Data/MLSO/20170903_025117_kcor_l2_avg.fts', 'Data/MLSO/20170906_213054_kcor_l2_avg.fts', 'Data/MLSO/20170911_202927_kcor_l2_avg.fts'];'Data/MLSO/20170820_180657_kcor_l2_avg.fts', 'Data/MLSO/20170825_185258_kcor_l2_avg.fts', 'Data/MLSO/20170829_200801_kcor_l2_avg.fts', 'Data/MLSO/20170903_025117_kcor_l2_avg.fts', 'Data/MLSO/20170906_213054_kcor_l2_avg.fts', 'Data/MLSO/20170911_202927_kcor_l2_avg.fts']
  FOREACH element2, directory_list_2 DO BEGIN
    hi2 = generate_forward_model(element2)
  ENDFOREACH

END

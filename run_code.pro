pro run_code

  directory_list = ['Data/MLSO/20170820_180657_kcor_l2_avg.fts', 'Data/MLSO/20170825_185258_kcor_l2_avg.fts', 'Data/MLSO/20170829_200801_kcor_l2_avg.fts', 'Data/MLSO/20170903_025117_kcor_l2_avg.fts', 'Data/MLSO/20170906_213054_kcor_l2_avg.fts', 'Data/MLSO/20170911_202927_kcor_l2_avg.fts']
  FOREACH element, directory_list DO BEGIN
    hi = generate_forward_model(element)
  ENDFOREACH

END

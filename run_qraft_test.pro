PRO run_qraft_test, input_file_path

  dir = input_file_path + '/Output/fits_images'
  json_config = input_file_path + '/config.json'
  config = JSON_PARSE(json_config)
  ; cor1_data_path = string(config['cor1_data_path'])
  cor1_data_extension = string(config['cor1_data_extension'])
  cor1_pattern_search = string(config['cor1_pattern_search'])
  cor1_pattern_middle = config['cor1_pattern_middle']

  ; kcor_data_path = string(config['kcor_data_path'])
  kcor_data_extension = string(config['kcor_data_extension'])
  kcor_pattern_search = string(config['kcor_pattern_search'])
  kcor_pattern_middle = config['kcor_pattern_middle']
  
  if (cor1_pattern_middle) then begin
    cor1_search_string = '/*' + cor1_pattern_search + '*' + cor1_data_extension
  endif else begin
    cor1_search_string = '/*' + cor1_pattern_search + cor1_data_extension
  endelse

  if (kcor_pattern_middle) then begin
    kcor_search_string = '/*' + kcor_pattern_search + '*' + kcor_data_extension
  endif else begin
    kcor_search_string = '/*' + kcor_pattern_search + kcor_data_extension
  endelse

  ; print, dir
  f_COR1 = file_search(dir+cor1_search_string)
  f_pB = file_search(dir+'/*COR1__PSI_pB.fits')
  f_ne = file_search(dir+'/*COR1__PSI_ne.fits')
  f_ne_LOS = file_search(dir+'/*COR1__PSI_ne_LOS.fits')

  f_By =  file_search(dir+'/*COR1__PSI_By.fits')
  f_Bz =  file_search(dir+'/*COR1__PSI_Bz.fits')
  f_By_LOS =  file_search(dir+'*COR1__PSI_By_LOS.fits')
  f_Bz_LOS =  file_search(dir+'*COR1__PSI_Bz_LOS.fits')
  
  output_image_path = input_file_path + "/Output/Plots/QRaFT_Figures"
  spawn, 'mkdir -p ' + output_image_path
  
  for i=0, n_elements(f_pB)-1 do begin
    head = headfits(f_pB[i])
    head_struct = fitshead2struct(head)
    date_obs = head_struct.date_obs
    date_split = strsplit(date_obs,'T',/extract)
    date_str = date_split[0]
    QRaFT_TEST, 1, f_pB[i], f_By[i], f_Bz[i], 110.0, output_image_path+"/"+date_str+"_pB_COR1_fig_1.png", output_image_path+"/"+date_str+"_pB_COR1_fig_2.png", output_image_path+"/"+date_str+"_pB_COR1_fig_3.png", output_image_path+"/"+date_str+"_pB_COR1_fig_4.png", output_image_path+"/"+date_str+"_pB_COR1_fig_5.png", 0
    QRaFT_TEST, 1, f_ne[i], f_By[i], f_Bz[i], 110.0, output_image_path+"/"+date_str+"_ne_COR1_fig_1.png", output_image_path+"/"+date_str+"_ne_COR1_fig_2.png", output_image_path+"/"+date_str+"_ne_COR1_fig_3.png", output_image_path+"/"+date_str+"_ne_COR1_fig_4.png", output_image_path+"/"+date_str+"_ne_COR1_fig_5.png", 0
    QRaFT_TEST, 1, f_ne_LOS[i], f_By[i], f_Bz[i], 110.0, output_image_path+"/"+date_str+"_ne_LOS_COR1_fig_1.png", output_image_path+"/"+date_str+"_ne_LOS_COR1_fig_2.png", output_image_path+"/"+date_str+"_ne_LOS_COR1_fig_3.png", output_image_path+"/"+date_str+"_ne_LOS_COR1_fig_4.png", output_image_path+"/"+date_str+"_ne_LOS_COR1_fig_5.png", 0
    QRaFT_TEST, 1, f_COR1[i], f_By[i], f_Bz[i], 110.0, output_image_path+"/"+date_str+"_COR1_fig_1.png", output_image_path+"/"+date_str+"_COR1_fig_2.png", output_image_path+"/"+date_str+"_COR1_fig_3.png", output_image_path+"/"+date_str+"_COR1_fig_4.png", output_image_path+"/"+date_str+"_COR1_fig_5.png", 0
    
  endfor
  
  f_KCor = file_search(dir+kcor_search_string)
  f_pB = file_search(dir+'/*KCor__PSI_pB.fits')
  f_ne = file_search(dir+'/*KCor__PSI_ne.fits')
  f_ne_LOS = file_search(dir+'/*KCor__PSI_ne_LOS.fits')

  f_By =  file_search(dir+'/*KCor__PSI_By.fits')
  f_Bz =  file_search(dir+'/*KCor__PSI_Bz.fits')
  f_By_LOS =  file_search(dir+'*KCor__PSI_By_LOS.fits')
  f_Bz_LOS =  file_search(dir+'*KCor__PSI_Bz_LOS.fits')

  for i=0, n_elements(f_pB)-1 do begin
    head = headfits(f_pB[i])
    head_struct = fitshead2struct(head)
    date_obs = head_struct.date_obs
    date_split = strsplit(date_obs,'T',/extract)
    date_str = date_split[0]
    QRaFT_TEST, 1, f_pB[i], f_By[i], f_Bz[i], 110.0, output_image_path+"/"+date_str+"_pB_KCOR_fig_1.png", output_image_path+"/"+date_str+"_pB_KCOR_fig_2.png", output_image_path+"/"+date_str+"_pB_KCOR_fig_3.png", output_image_path+"/"+date_str+"_pB_KCOR_fig_4.png", output_image_path+"/"+date_str+"_pB_KCOR_fig_5.png", 0
    QRaFT_TEST, 1, f_ne[i], f_By[i], f_Bz[i], 110.0, output_image_path+"/"+date_str+"_ne_KCOR_fig_1.png", output_image_path+"/"+date_str+"_ne_KCOR_fig_2.png", output_image_path+"/"+date_str+"_ne_KCOR_fig_3.png", output_image_path+"/"+date_str+"_ne_KCOR_fig_4.png", output_image_path+"/"+date_str+"_ne_KCOR_fig_5.png", 0
    QRaFT_TEST, 1, f_ne_LOS[i], f_By[i], f_Bz[i], 110.0, output_image_path+"/"+date_str+"_ne_LOS_KCOR_fig_1.png", output_image_path+"/"+date_str+"_ne_LOS_KCOR_fig_2.png", output_image_path+"/"+date_str+"_ne_LOS_KCOR_fig_3.png", output_image_path+"/"+date_str+"_ne_LOS_KCOR_fig_4.png", output_image_path+"/"+date_str+"_ne_LOS_KCOR_fig_5.png", 0
    QRaFT_TEST, 1, f_KCor[i], f_By[i], f_Bz[i], 110.0, output_image_path+"/"+date_str+"_KCOR_fig_1.png", output_image_path+"/"+date_str+"_KCOR_fig_2.png", output_image_path+"/"+date_str+"_KCOR_fig_3.png", output_image_path+"/"+date_str+"_KCOR_fig_4.png", output_image_path+"/"+date_str+"_KCOR_fig_5.png", 0

  endfor
  
  end
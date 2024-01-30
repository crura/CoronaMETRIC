PRO run_qraft_test, input_file_path

  dir = input_file_path + '/Output/fits_images'
  ; print, dir
  f_COR1 = file_search(dir+'/*rep_med*')
  f_pB = file_search(dir+'/*COR1__PSI_pB.fits')
  f_ne = file_search(dir+'/*COR1__PSI_ne.fits')
  f_ne_LOS = file_search(dir+'/*COR1__PSI_ne_LOS.fits')

  f_By =  file_search(dir+'/*COR1__PSI_By.fits')
  f_Bz =  file_search(dir+'/*COR1__PSI_Bz.fits')
  f_By_LOS =  file_search(dir+'*COR1__PSI_By_LOS.fits')
  f_Bz_LOS =  file_search(dir+'*COR1__PSI_Bz_LOS.fits')
  
  for i=0, n_elements(f_pB)-1 do begin
    QRaFT_TEST, 1, f_pB[i], f_By[i], f_Bz[i], 110.0
    QRaFT_TEST, 1, f_ne[i], f_By[i], f_Bz[i], 110.0
    QRaFT_TEST, 1, f_ne_LOS[i], f_By[i], f_Bz[i], 110.0
    
  endfor
  
  f_KCor = file_search(dir+'/*rep_med*')
  f_pB = file_search(dir+'/*KCor__PSI_pB.fits')
  f_ne = file_search(dir+'/*KCor__PSI_ne.fits')
  f_ne_LOS = file_search(dir+'/*KCor__PSI_ne_LOS.fits')

  f_By =  file_search(dir+'/*KCor__PSI_By.fits')
  f_Bz =  file_search(dir+'/*KCor__PSI_Bz.fits')
  f_By_LOS =  file_search(dir+'*KCor__PSI_By_LOS.fits')
  f_Bz_LOS =  file_search(dir+'*KCor__PSI_Bz_LOS.fits')

  for i=0, n_elements(f_pB)-1 do begin
    QRaFT_TEST, 1, f_pB[i], f_By[i], f_Bz[i], 220.0
    QRaFT_TEST, 1, f_ne[i], f_By[i], f_Bz[i], 220.0
    QRaFT_TEST, 1, f_ne_LOS[i], f_By[i], f_Bz[i], 220.0

  endfor
  
  end
function save_parameters

  restore,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Electron_Density_Center.sav',/v
  restore,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Bx_2d_Center.sav',/v
  restore,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/By_2d_Center.sav',/v
  restore,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Bz_2d_Center.sav',/v
  restore, '/Users/crura/SSW/packages/forward/datadump',/v
  restore, 'model_parameters.sav',/v

  out_string = strcompress(string(CRLT_OBS),/remove_all) + '_' + strcompress(string(CRLN_OBS),/remove_all) + '.sav'
  out_path = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data'
  str = [out_path,out_string]
  save_path = str.join('/')

  BX_2d_center = BX_2d
  BY_2d_center = BY_2d
  BZ_2d_center = BZ_2d


  save,Dens_2d_center,BX_2d_center,BY_2d_center,BZ_2d_center,filename=save_path
End

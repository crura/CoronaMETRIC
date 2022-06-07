function save_parameters

  spawn, 'git rev-parse --show-toplevel', git_repo

  restore, git_repo + '/Data/Electron_Density_Center.sav',/v
  restore, git_repo + '/Data/Bx_2d_Center.sav',/v
  restore, git_repo + '/Data/By_2d_Center.sav',/v
  restore, git_repo + '/Data/Bz_2d_Center.sav',/v
  restore, '/Users/crura/SSW/packages/forward/datadump',/v
  restore, git_repo + '/Data/model_parameters.sav',/v


  out_string = strcompress(string(CRLT_OBS),/remove_all) + '_' + strcompress(string(CRLN_OBS),/remove_all) + '.sav'
  out_path = git_repo + '/Output'
; need to generate if statement to execute this if Output directory does not exist
  ;spawn, 'rm -r ' + out_path + '; mkdir ' + out_path
  str = [out_path,out_string]
  save_path = str.join('/')

  BX_2d_center = BX_2d
  BY_2d_center = BY_2d
  BZ_2d_center = BZ_2d





  dens_integrated = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_Electron_Density.csv')
  dens_integrated_2d = reform(dens_integrated.field1,256,256)

  bx_integrated = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_LOS_Bx.csv')
  bx_2d_integrated = reform(bx_integrated.field1,256,256)

  by_integrated = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_LOS_By.csv')
  by_2d_integrated = reform(by_integrated.field1,256,256)

  bz_integrated = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_LOS_Bz.csv')
  bz_2d_integrated = reform(bz_integrated.field1,256,256)

  restore, '/Users/crura/SSW/packages/forward/output.sav',/v
  forward_pb_image = quantmap.data

  save,Dens_2d_center,dens_integrated_2d,forward_pb_image,BX_2d_center,bx_2d_integrated,BY_2d_center,by_2d_integrated,BZ_2d_center,bz_2d_integrated,filename=save_path

End

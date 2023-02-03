function image_coalignment, directory
  ;restore, '/Users/crura/Desktop/Research/Magnetic_Field/Forward_PB_data.sav',/v
  spawn, 'git rev-parse --show-toplevel', git_repo
  restore, git_repo + '/Data/model_parameters.sav',/v
  restore, git_repo + '/Data/outstrings.sav', /v
  spawn, 'mkdir -p ' + git_repo + '/Output'
  spawn, 'mkdir -p ' + git_repo + '/Output/fits_images'
  ;.compile -v '/Users/crura/SSW/gen/idl/util/default.pro'
  ;.compile -v '/Users/crura/IDLWorkspace/Default/linspace.pro'
  ;.compile -v '/Users/crura/Desktop/Research/Magnetic_Field/write_psi_image_as_fits.pro'
  ;.compile -v '/Users/crura/Desktop/Research/Magnetic_Field/rtp2xyz.pro'

  ;Inputs:
  ;
  ;   OutFile: Path of the output file name.
  ;
  ;   Im: A 2D array containing the image values.
  ;
  ;    x: 1D array of pixel coordinates in x [Rs] or [Degrees].
  ;
  ;    y: 1D array of pixel coordinates in y [Rs] or [Degrees].
  ;
  ;    ObsDate: UT time string in the this format: '2014-04-13T18:00:00'
  ;
  ;    Lon: Carrington longitude of the viewpoint [0-360]
  ;
  ;    B0: Carrington latitude of the viewpoint [-90,90].







  ; Construct and write fits image for PSI integrated electron density

  dens = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_Electron_Density.csv')
  len = fix(sqrt(n_elements(forward_pb_image)))
  dens_2d = reform(dens.field1,len,len)
  out_string = date_print + '__' + detector + '__PSI';strcompress(string(CRLT_OBS),/remove_all) + '_' + strcompress(string(CRLN_OBS),/remove_all)
  rsun_range = range + range
  dx = rsun_range/len
  ; convert x,y arrays to rsun
  x_array = linspace(0,len-1,len) * dx
  y_array = linspace(0,len-1,len) * dx



  DATE='2017-08-30T02:23:16.968Z' ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/Integrated_Electron_Density_MLSO_Projection.fits'
  data = dens_2d

  WRITE_PSI_IMAGE_AS_FITS,output,data,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1



  ; Construct and write fits image for PSI Bx central 2d Plane of Sky cut
  restore, git_repo + '/Data/Bx_2d_Center.sav',/v
  bx_central = bx_2d
  output_bx_central = git_repo + '/Output/Forward_Bx_Central_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bx_central,bx_central,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI By central 2d Plane of Sky cut
  restore,git_repo + '/Data/By_2d_Center.sav',/v
  by_central = by_2d
  output_by_central = git_repo + '/Output/Forward_By_Central_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_by_central,by_central,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI Bz central 2d Plane of Sky cut
  restore,git_repo + '/Data/Bz_2d_Center.sav',/v
  bz_central = bz_2d
  output_bz_central = git_repo + '/Output/Forward_Bz_Central_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bz_central,bz_central,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI Bx LOS field
  bx_los_1d = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_LOS_Bx.csv')
  bx_los_2d = reform(bx_los_1d.field1,len,len)
  output_bx_los = git_repo + '/Output/Forward_Bx_LOS_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bx_los,bx_los_2d,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI Bx LOS field
  by_los_1d = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_LOS_By.csv')
  by_los_2d = reform(by_los_1d.field1,len,len)
  output_by_los = git_repo + '/Output/Forward_By_LOS_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_by_los,by_los_2d,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI Bx LOS field
  bz_los_1d = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_LOS_Bz.csv')
  bz_los_2d = reform(bz_los_1d.field1,len,len)
  output_bz_los = git_repo + '/Output/Forward_Bz_LOS_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bz_los,bz_los_2d,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  ; Construct and write fits image for PSI Central Electron Density
  rsun_range = range + range
  dx = rsun_range/len
  ; convert x,y arrays to rsun
  x_array = linspace(0,len-1,len) * dx
  y_array= linspace(0,len-1,len) * dx

  ; dens = read_csv(git_repo + '/Data/Central_Parameters/rotated_Dens_2d.csv')
  ; dens_2d = reform(dens.field1,256,256)
  restore, git_repo + '/Data/Electron_Density_Center.sav',/v
  dens_2d = Dens_2d_center

  DATE='2017-08-30T02:23:16.968Z' ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/Central_Electron_Density_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output,dens_2d,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI integrated electron density

  rsun_range = range + range
  dx = rsun_range/len
  ; convert x,y arrays to rsun
  x_array = linspace(0,len-1,len) * dx
  y_array= linspace(0,len-1,len) * dx
  ; x_1d = read_csv('/Users/crura/Desktop/Research/Rotated_x_Test.csv')
  ; x_1 = x_1d.FIELD1
  ; y_1d = read_csv('/Users/crura/Desktop/Research/Rotated_y_Test.csv')
  ; y_1 = y_1d.FIELD1


  ;restore, git_repo + '/Data/model_parameters.sav',/v
  dens_2d_forward = forward_pb_image



  DATE='2017-08-30T02:23:16.968Z' ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/Forward_PB_MLSO_Projection.fits'
  data = dens_2d

  WRITE_PSI_IMAGE_AS_FITS,output,dens_2d_forward,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  ; Construct and write fits image for PSI Bx LOS field
  bz_los_1d = read_csv(git_repo + '/Data/Integrated_Parameters/Integrated_LOS_Bz.csv')
  bz_los_2d = reform(bz_los_1d.field1,len,len)
  output_bz_los = git_repo + '/Output/Forward_Bz_LOS_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bz_los,bz_los_2d,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1






  ; Coalign PSI Integrated Electron density with MLSO image
  mlso_im = readfits(fits_directory)
  mlso_head = headfits(fits_directory)
  wcs_mlso = fitshead2wcs( mlso_head )
  coord = wcs_get_coord( wcs_mlso )


  psi_integrated_dens_im = readfits(git_repo + '/Output/Integrated_Electron_Density_MLSO_Projection.fits')
  psi_integrated_dens_head = headfits(git_repo + '/Output/Integrated_Electron_Density_MLSO_Projection.fits')

  wcs_psi = fitshead2wcs( psi_integrated_dens_head )

  pixel = wcs_get_pixel( wcs_psi, coord)
  new_psi_integrated_dens = reform( interpolate( psi_integrated_dens_im, pixel[0,*,*], pixel[1,*,*] ))
  spawn, 'mkdir -p ' + git_repo + '/Output/FORWARD_MLSO_Rotated_Data'
  spath = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_MLSO_Integrated_Electron_Density_Coalignment' + '.csv'
  write_csv,spath,new_psi_integrated_dens


  ; Coalign PSI Central Electron density with MLSO image
  psi_central_im = readfits(git_repo + '/Output/Central_Electron_Density_MLSO_Projection.fits')
  psi_central_head = headfits(git_repo + '/Output/Central_Electron_Density_MLSO_Projection.fits')

  wcs_psi = fitshead2wcs( psi_central_head )

  pixel = wcs_get_pixel( wcs_psi, coord)
  new_psi_central_dens = reform( interpolate( psi_central_im, pixel[0,*,*], pixel[1,*,*] ))
  spawn, 'mkdir -p ' + git_repo + '/Output/FORWARD_MLSO_Rotated_Data'
  spath = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_MLSO_Central_Electron_Density_Coalignment' + '.csv'
  write_csv,spath,new_psi_central_dens


  ; Coalign PSI Forward PB Image with MLSO image
  psi_pb_im = readfits(git_repo + '/Output/Forward_PB_MLSO_Projection.fits')
  psi_pb_head = headfits(git_repo + '/Output/Forward_PB_MLSO_Projection.fits')

  wcs_psi_pb = fitshead2wcs( psi_pb_head )

  pixel_pb = wcs_get_pixel( wcs_psi_pb, coord)
  new_psi_pb = reform( interpolate( psi_pb_im, pixel_pb[0,*,*], pixel_pb[1,*,*] ))
  spath_pb = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_PB_MLSO_Coalignment.csv'
  write_csv,spath_pb,new_psi_pb


  ; Coalign PSI Forward B_x Central 2d Plane of Sky cut with MLSO image
  psi_bx_central_im = readfits(output_bx_central)
  psi_bx_central_head = headfits(output_bx_central)

  wcs_psi_bx_central = fitshead2wcs( psi_bx_central_head )

  pixel_bx_central = wcs_get_pixel( wcs_psi_bx_central, coord)
  new_psi_bx_central = reform( interpolate( psi_bx_central_im, pixel_bx_central[0,*,*], pixel_bx_central[1,*,*] ))
  spath_bx_central = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_Bx_Central_MLSO_Coalignment' + '.csv'
  write_csv,spath_bx_central,new_psi_bx_central


  ; Coalign PSI Forward B_y Central 2d Plane of Sky cut with MLSO image
  psi_by_central_im = readfits(output_by_central)
  psi_by_central_head = headfits(output_by_central)

  wcs_psi_by_central = fitshead2wcs( psi_by_central_head )

  pixel_by_central = wcs_get_pixel( wcs_psi_by_central, coord)
  new_psi_by_central = reform( interpolate( psi_by_central_im, pixel_by_central[0,*,*], pixel_by_central[1,*,*] ))
  spath_by_central = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_By_Central_MLSO_Coalignment' + '.csv'
  write_csv,spath_by_central,new_psi_by_central


  ; Coalign PSI Forward B_z Central 2d Plane of Sky cut with MLSO image
  psi_bz_central_im = readfits(output_bz_central)
  psi_bz_central_head = headfits(output_bz_central)

  wcs_psi_bz_central = fitshead2wcs( psi_bz_central_head )

  pixel_bz_central = wcs_get_pixel( wcs_psi_bz_central, coord)
  new_psi_bz_central = reform( interpolate( psi_bz_central_im, pixel_bz_central[0,*,*], pixel_bz_central[1,*,*] ))
  spath_bz_central = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_Bz_Central_MLSO_Coalignment' + '.csv'
  write_csv,spath_bz_central,new_psi_bz_central


  ; Coalign PSI Forward B_x LOS Integrated field with MLSO image
  psi_bx_los_im = readfits(output_bx_los)
  psi_bx_los_head = headfits(output_bx_los)

  wcs_psi_bx_los = fitshead2wcs( psi_bx_los_head )

  pixel_bx_los = wcs_get_pixel( wcs_psi_bx_los, coord)
  new_psi_bx_los = reform( interpolate( psi_bx_los_im, pixel_bx_los[0,*,*], pixel_bx_los[1,*,*] ))
  spath_bx_los = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_Bx_LOS_MLSO_Coalignment' + '.csv'
  write_csv,spath_bx_los,new_psi_bx_los


  ; Coalign PSI Forward B_y LOS Integrated field with MLSO image
  psi_by_los_im = readfits(output_by_los)
  psi_by_los_head = headfits(output_by_los)

  wcs_psi_by_los = fitshead2wcs( psi_by_los_head )

  pixel_by_los = wcs_get_pixel( wcs_psi_by_los, coord)
  new_psi_by_los = reform( interpolate( psi_by_los_im, pixel_by_los[0,*,*], pixel_by_los[1,*,*] ))
  spath_by_los = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_By_LOS_MLSO_Coalignment' + '.csv'
  write_csv,spath_by_los,new_psi_by_los


  ; Coalign PSI Forward B_z LOS Integrated field with MLSO image
  psi_bz_los_im = readfits(output_bz_los)
  psi_bz_los_head = headfits(output_bz_los)

  wcs_psi_bz_los = fitshead2wcs( psi_bz_los_head )

  pixel_bz_los = wcs_get_pixel( wcs_psi_bz_los, coord)
  new_psi_bz_los = reform( interpolate( psi_bz_los_im, pixel_bz_los[0,*,*], pixel_bz_los[1,*,*] ))
  spath_bz_los = git_repo + '/Output/FORWARD_MLSO_Rotated_Data/PSI_Bz_LOS_MLSO_Coalignment' + '.csv'
  write_csv,spath_bz_los,new_psi_bz_los



  bx_central_coaligned = new_psi_bx_central
  by_central_coaligned = new_psi_by_central
  bz_central_coaligned = new_psi_bz_central

  bx_integrated_coaligned = new_psi_bx_los
  by_integrated_coaligned = new_psi_by_los
  bz_integrated_coaligned = new_psi_bz_los

  psi_central_dens_coaligned = new_psi_central_dens
  psi_integrated_dens_coaligned = new_psi_integrated_dens
  psi_forward_pb_coaligned = new_psi_pb



  save,bx_central_coaligned,by_central_coaligned,bz_central_coaligned,filename=git_repo + '/Output/Central_B_Field_MLSO_Coaligned.sav'
  save,bx_integrated_coaligned,by_integrated_coaligned,bz_integrated_coaligned,filename=git_repo + '/Output/LOS_B_Field_MLSO_Coaligned.sav'

  head_fits_mlso = headfits(fits_directory)

  ; find length of reference image from fits header to use for coalignment
  len_new = sxpar(head_fits_mlso,'NAXIS1')

  dx_new = rsun_range/len_new
  x_array_new = linspace(0,len_new-1,len_new) * dx_new
  y_array_new= linspace(0,len_new-1,len_new) * dx_new
  sxaddpar,head_fits_mlso,'TELESCOP','PSI-MAS Forward Model' ;Modify value of TELESCOP

  DATE=date_obs ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/fits_images/' + out_string + '_By.fits'
  WRITE_PSI_IMAGE_AS_FITS,output,by_central_coaligned,x_array_new,y_array_new,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  writefits, output, by_central_coaligned, head_fits_mlso

  x_array_new = linspace(0,len_new-1,len_new) * dx_new
  y_array_new= linspace(0,len_new-1,len_new) * dx_new

  DATE=date_obs ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/fits_images/' + out_string + '_By_LOS.fits'
  WRITE_PSI_IMAGE_AS_FITS,output,by_integrated_coaligned,x_array_new,y_array_new,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  writefits, output, by_integrated_coaligned, head_fits_mlso

  x_array_new = linspace(0,len_new-1,len_new) * dx_new
  y_array_new= linspace(0,len_new-1,len_new) * dx_new

  DATE=date_obs ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/fits_images/' + out_string + '_Bz.fits'
  WRITE_PSI_IMAGE_AS_FITS,output,bz_central_coaligned,x_array_new,y_array_new,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  writefits, output, bz_central_coaligned, head_fits_mlso

  x_array_new = linspace(0,len_new-1,len_new) * dx_new
  y_array_new= linspace(0,len_new-1,len_new) * dx_new

  DATE=date_obs ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/fits_images/' + out_string + '_Bz_LOS.fits'
  WRITE_PSI_IMAGE_AS_FITS,output,bz_integrated_coaligned,x_array_new,y_array_new,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  writefits, output, bz_integrated_coaligned, head_fits_mlso

  x_array_new = linspace(0,len_new-1,len_new) * dx_new
  y_array_new= linspace(0,len_new-1,len_new) * dx_new

  DATE=date_obs ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/fits_images/' + out_string + '_ne.fits'
  WRITE_PSI_IMAGE_AS_FITS,output,psi_central_dens_coaligned,x_array_new,y_array_new,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  writefits, output, psi_central_dens_coaligned, head_fits_mlso

  x_array_new = linspace(0,len_new-1,len_new) * dx_new
  y_array_new= linspace(0,len_new-1,len_new) * dx_new

  DATE=date_obs ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/fits_images/' + out_string + '_ne_LOS.fits'
  WRITE_PSI_IMAGE_AS_FITS,output,psi_integrated_dens_coaligned,x_array_new,y_array_new,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  writefits, output, psi_integrated_dens_coaligned, head_fits_mlso

  x_array_new = linspace(0,len_new-1,len_new) * dx_new
  y_array_new= linspace(0,len_new-1,len_new) * dx_new

  DATE=date_obs ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = git_repo + '/Output/fits_images/' + out_string + '_pB.fits'
  WRITE_PSI_IMAGE_AS_FITS,output,psi_forward_pb_coaligned,x_array_new,y_array_new,DATE,CMER,BANG,/ForceXyRs,/GetCoords,ObsDistanceAU=1
  outstring_list = [outstring_list, output]

  writefits, output, psi_forward_pb_coaligned, head_fits_mlso

  ;spawn, 'mkdir -p ' git_repo + '/Output/Coaligned_Parameters/'
  out_string = strcompress(string(CRLT_OBS),/remove_all) + '_' + strcompress(string(CRLN_OBS),/remove_all) + '.sav'
  out_path = git_repo + '/Output/Coaligned_Parameters'
  spawn, 'mkdir -p ' + out_path
  str = [out_path,out_string]
  save_path = str.join('/')


  save,bx_central_coaligned,by_central_coaligned,bz_central_coaligned,bx_integrated_coaligned,by_integrated_coaligned,bz_integrated_coaligned,psi_central_dens_coaligned,psi_integrated_dens_coaligned,psi_forward_pb_coaligned,filename=save_path
  save,crln_obs,crlt_obs,occlt,range,crlt_obs_print,crln_obs_print,forward_pb_image,date_obs,fits_directory, shape, rsun, date_print, detector, filename=git_repo + '/Data/model_parameters.sav'
  save, outstring_list, filename = git_repo + '/Data/outstrings.sav'
  return, 0
END

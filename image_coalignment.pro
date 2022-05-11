function image_coalignment, directory
  ;restore, '/Users/crura/Desktop/Research/Magnetic_Field/Forward_PB_data.sav',/v
  restore, 'model_parameters.sav',/v
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

  rsun_range = range + range
  dx = rsun_range/256
  ; convert x,y arrays to rsun
  x_array = linspace(0,255,256) * dx
  y_array= linspace(0,255,256) * dx
  ; x_1d = read_csv('/Users/crura/Desktop/Research/Rotated_x_Test.csv')
  ; x_1 = x_1d.FIELD1
  ; y_1d = read_csv('/Users/crura/Desktop/Research/Rotated_y_Test.csv')
  ; y_1 = y_1d.FIELD1



  dens = read_csv('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Integrated_Parameters/Integrated_Electron_Density.csv')
  dens_2d = reform(dens.field1,256,256)



  DATE='2017-08-30T02:23:16.968Z' ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=crln_obs
  BANG=crlt_obs
  output = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/Forward_MLSO_Projection.fits'
  data = dens_2d

;  WRITE_PSI_IMAGE_AS_FITS,output,data,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1



  ; Construct and write fits image for PSI Bx central 2d Plane of Sky cut
  restore,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Bx_2d_Center.sav',/v
  bx_central = bx_2d
  output_bx_central = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/Forward_Bx_Central_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bx_central,bx_central,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI By central 2d Plane of Sky cut
  restore,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/By_2d_Center.sav',/v
  by_central = by_2d
  output_by_central = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/Forward_By_Central_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_by_central,by_central,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI Bz central 2d Plane of Sky cut
  restore,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Bz_2d_Center.sav',/v
  bz_central = bz_2d
  output_bz_central = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/Forward_Bz_Central_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bz_central,bz_central,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI Bx LOS field
  bx_los_1d = read_csv('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Integrated_Parameters/Integrated_LOS_Bx.csv')
  bx_los_2d = reform(bx_los_1d.field1,256,256)
  output_bx_los = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/Forward_Bx_LOS_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bx_los,bx_los_2d,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI Bx LOS field
  by_los_1d = read_csv('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Integrated_Parameters/Integrated_LOS_By.csv')
  by_los_2d = reform(by_los_1d.field1,256,256)
  output_by_los = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/Forward_By_LOS_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_by_los,by_los_2d,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1


  ; Construct and write fits image for PSI Bx LOS field
  bz_los_1d = read_csv('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Integrated_Parameters/Integrated_LOS_Bz.csv')
  bz_los_2d = reform(bz_los_1d.field1,256,256)
  output_bz_los = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/Forward_Bz_LOS_MLSO_Projection.fits'
  WRITE_PSI_IMAGE_AS_FITS,output_bz_los,bz_los_2d,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1






  ; Coalign PSI Integrated Electron density with MLSO image
  mlso_im = readfits('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/MLSO/20170829_200801_kcor_l2_avg.fts')
  mlso_head = headfits('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/MLSO/20170829_200801_kcor_l2_avg.fts')
  wcs_mlso = fitshead2wcs( mlso_head )
  coord = wcs_get_coord( wcs_mlso )


  psi_im = readfits('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/PSI/Forward_MLSO_Projection.fits')
  psi_head = headfits('/Users/crura/Desktop/Research/github/Image-Coalignment/Data/PSI/Forward_MLSO_Projection.fits')

  wcs_psi = fitshead2wcs( psi_head )

  pixel = wcs_get_pixel( wcs_psi, coord)
  new_psi = reform( interpolate( psi_im, pixel[0,*,*], pixel[1,*,*] ))
  spath = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/FORWARD_MLSO_Rotated_Data/PSI_MLSO_Coalignment' + '.csv'
  write_csv,spath,new_psi


  ; Coalign PSI Forward PB Image with MLSO image
  ;psi_pb_im = readfits('/Users/crura/Desktop/Research/Magnetic_Field/Forward_PB_MLSO_Projection.fits')
  ;psi_pb_head = headfits('/Users/crura/Desktop/Research/Magnetic_Field/Forward_PB_MLSO_Projection.fits')

  ;wcs_psi_pb = fitshead2wcs( psi_pb_head )

  ;pixel_pb = wcs_get_pixel( wcs_psi_pb, coord)
  ;new_psi_pb = reform( interpolate( psi_pb_im, pixel_pb[0,*,*], pixel_pb[1,*,*] ))
  ;spath_pb = '/Users/crura/Desktop/Research/Magnetic_Field/FORWARD_MLSO_Rotated_Data/PSI_PB_MLSO_Coalignment' + '.csv'
  ;write_csv,spath_pb,new_psi_pb


  ; Coalign PSI Forward B_x Central 2d Plane of Sky cut with MLSO image
  psi_bx_central_im = readfits(output_bx_central)
  psi_bx_central_head = headfits(output_bx_central)

  wcs_psi_bx_central = fitshead2wcs( psi_bx_central_head )

  pixel_bx_central = wcs_get_pixel( wcs_psi_bx_central, coord)
  new_psi_bx_central = reform( interpolate( psi_bx_central_im, pixel_bx_central[0,*,*], pixel_bx_central[1,*,*] ))
  spath_bx_central = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/FORWARD_MLSO_Rotated_Data/PSI_Bx_Central_MLSO_Coalignment' + '.csv'
  write_csv,spath_bx_central,new_psi_bx_central


  ; Coalign PSI Forward B_y Central 2d Plane of Sky cut with MLSO image
  psi_by_central_im = readfits(output_by_central)
  psi_by_central_head = headfits(output_by_central)

  wcs_psi_by_central = fitshead2wcs( psi_by_central_head )

  pixel_by_central = wcs_get_pixel( wcs_psi_by_central, coord)
  new_psi_by_central = reform( interpolate( psi_by_central_im, pixel_by_central[0,*,*], pixel_by_central[1,*,*] ))
  spath_by_central = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/FORWARD_MLSO_Rotated_Data/PSI_By_Central_MLSO_Coalignment' + '.csv'
  write_csv,spath_by_central,new_psi_by_central


  ; Coalign PSI Forward B_z Central 2d Plane of Sky cut with MLSO image
  psi_bz_central_im = readfits(output_bz_central)
  psi_bz_central_head = headfits(output_bz_central)

  wcs_psi_bz_central = fitshead2wcs( psi_bz_central_head )

  pixel_bz_central = wcs_get_pixel( wcs_psi_bz_central, coord)
  new_psi_bz_central = reform( interpolate( psi_bz_central_im, pixel_bz_central[0,*,*], pixel_bz_central[1,*,*] ))
  spath_bz_central = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/FORWARD_MLSO_Rotated_Data/PSI_Bz_Central_MLSO_Coalignment' + '.csv'
  write_csv,spath_bz_central,new_psi_bz_central


  ; Coalign PSI Forward B_x LOS Integrated field with MLSO image
  psi_bx_los_im = readfits(output_bx_los)
  psi_bx_los_head = headfits(output_bx_los)

  wcs_psi_bx_los = fitshead2wcs( psi_bx_los_head )

  pixel_bx_los = wcs_get_pixel( wcs_psi_bx_los, coord)
  new_psi_bx_los = reform( interpolate( psi_bx_los_im, pixel_bx_los[0,*,*], pixel_bx_los[1,*,*] ))
  spath_bx_los = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/FORWARD_MLSO_Rotated_Data/PSI_Bx_LOS_MLSO_Coalignment' + '.csv'
  write_csv,spath_bx_los,new_psi_bx_los


  ; Coalign PSI Forward B_y LOS Integrated field with MLSO image
  psi_by_los_im = readfits(output_by_los)
  psi_by_los_head = headfits(output_by_los)

  wcs_psi_by_los = fitshead2wcs( psi_by_los_head )

  pixel_by_los = wcs_get_pixel( wcs_psi_by_los, coord)
  new_psi_by_los = reform( interpolate( psi_by_los_im, pixel_by_los[0,*,*], pixel_by_los[1,*,*] ))
  spath_by_los = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/FORWARD_MLSO_Rotated_Data/PSI_By_LOS_MLSO_Coalignment' + '.csv'
  write_csv,spath_by_los,new_psi_by_los


  ; Coalign PSI Forward B_z LOS Integrated field with MLSO image
  psi_bz_los_im = readfits(output_bz_los)
  psi_bz_los_head = headfits(output_bz_los)

  wcs_psi_bz_los = fitshead2wcs( psi_bz_los_head )

  pixel_bz_los = wcs_get_pixel( wcs_psi_bz_los, coord)
  new_psi_bz_los = reform( interpolate( psi_bz_los_im, pixel_bz_los[0,*,*], pixel_bz_los[1,*,*] ))
  spath_bz_los = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/FORWARD_MLSO_Rotated_Data/PSI_Bz_LOS_MLSO_Coalignment' + '.csv'
  write_csv,spath_bz_los,new_psi_bz_los



  bx_central_coaligned = new_psi_bx_central
  by_central_coaligned = new_psi_by_central
  bz_central_coaligned = new_psi_bz_central

  bx_los_coaligned = new_psi_bx_los
  by_los_coaligned = new_psi_by_los
  bz_los_coaligned = new_psi_bz_los



  save,bx_central_coaligned,by_central_coaligned,bz_central_coaligned,filename='/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/Central_B_Field_MLSO_Coaligned.sav'
  save,bx_los_coaligned,by_los_coaligned,bz_los_coaligned,filename='/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Output/LOS_B_Field_MLSO_Coaligned.sav'





  return, 0
END

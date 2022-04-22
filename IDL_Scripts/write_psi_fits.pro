function write_psi_fits, directory
  restore, '/Users/crura/Desktop/Research/Magnetic_Field/Forward_PB_data.sav',/v
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
  
  x_array = linspace(0,255,256)
  y_array= linspace(0,255,256)
  x_1d = read_csv('/Users/crura/Desktop/Research/Rotated_x_Test.csv')
  x_1 = x_1d.FIELD1
  y_1d = read_csv('/Users/crura/Desktop/Research/Rotated_y_Test.csv')
  y_1 = y_1d.FIELD1
  
  
  
  
  
  
  
  
  DATE='2017-08-30T02:23:16.968Z' ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=290.70154
  BANG=6.9252958
  output = '/Users/crura/Desktop/Research/Magnetic_Field/Forward_PB_data.fits'
  data = FORWARD_PB_IMAGE
  
  WRITE_PSI_IMAGE_AS_FITS,output,data,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  return, 0
END

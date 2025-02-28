; Copyright 2025 Christopher Rura

; Licensed under the Apache License, Version 2.0 (the "License");
; you may not use this file except in compliance with the License.
; You may obtain a copy of the License at

;     http://www.apache.org/licenses/LICENSE-2.0

; Unless required by applicable law or agreed to in writing, software
; distributed under the License is distributed on an "AS IS" BASIS,
; WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
; See the License for the specific language governing permissions and
; limitations under the License.
function write_psi_mlso_fits, directory
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
  
  
  restore,'/Users/crura/SSW/packages/forward/MLSO_projection.sav',/v
  forward_pb_image = quantmap.data
  

  rsun_range = 6.0799999237060547 + 6.0799999237060547
  dx = rsun_range/256
; convert x,y arrays to rsun
  x_array = linspace(0,255,256) * dx
  y_array= linspace(0,255,256) * dx
  x_1d = read_csv('/Users/crura/Desktop/Research/Rotated_x_Test.csv')
  x_1 = x_1d.FIELD1
  y_1d = read_csv('/Users/crura/Desktop/Research/Rotated_y_Test.csv')
  y_1 = y_1d.FIELD1



  dens = read_csv('/Users/crura/Desktop/Research/Magnetic_Field/MLSO_LOS_Integrated_Elctron_Density.csv')
  dens_2d = reform(dens.field1,256,256)



  DATE='2017-08-30T02:23:16.968Z' ;'2014-04-13T18:00:00' '1988-01-18T17:20:43.123Z'
  CMER=183.44300
  BANG=7.1530000
  output = '/Users/crura/Desktop/Research/Magnetic_Field/Forward_PB_MLSO_Projection.fits'
  data = dens_2d

  WRITE_PSI_IMAGE_AS_FITS,output,data,x_array,y_array,DATE,CMER,BANG,/GetCoords;,ObsDistanceAU=1

  return, 0
END

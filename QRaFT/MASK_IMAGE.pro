
function MASK_IMAGE,  IMG, dyn_range

  ;+
  ; NAME:
  ;     MASK_IMAGE
  ;
  ; PURPOSE:
  ;     Returns a boolean mask of solar image pixels with values
  ;     within a specified dynamic range     
  ;
  ; CALLING SEQUENCE:
  ;     mask = MASK_IMAGE(IMG, dyn_range)
  ;
  ; INPUTS:
  ;     IMG       - 2D array representing solar image
  ;     dyn_range - 2-element array containing minimum (dyn_range[0]) and maximum (dyn_range[1])
  ;                 pixel values satisfying the required dynamic range
  ; OUTPUTS:
  ;     mask      - 2D byte array of the same size as IMG. It contains the value 1 for the image 
  ;                 positions within the dynamic range; the value 0 for all other positions.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;    mask =  MASK_IMAGE(IMG, [10, 1000])
  ;    
  ;    IMG is a processed image array.
  ;    The output array "mask" contains 1s where 10 < IMG < 1000; all other positions have 0s.
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------


  mask = (img gt dyn_range[0]) and (img lt dyn_range[1])
   
  return, mask
  
End

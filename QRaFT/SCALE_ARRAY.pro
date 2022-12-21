function SCALE_ARRAY, arr1, arr2

  ;+
  ; NAME:
  ;     SCALE_ARRAY
  ;
  ; PURPOSE:
  ;     Linearly rescales the values of first provided data array (arr1) to match 
  ;     the dynamic range of the second (arr2).
  ;
  ; CALLING SEQUENCE:
  ;     arr1_rescaled = SCALE_ARRAY( arr1, arr2)
  ;
  ; INPUTS:
  ;     arr1  -  data array to be rescaled
  ;     arr2  -  data array whose dynamic range should be matched
  ;              Note: arr1 and arr2 may have different sizes and mumbers of dimensions. 
  ;
  ; OUTPUTS:
  ;     arr1_rescaled   - rescaled version of arr1
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     arr1 = [1.0,2.0,3.0]
  ;l    arr2 = [3.0,4.0,5.0,7.0,8.0]
  ;     print, minmax(arr2)
  ;         3.00000      8.00000
  ;     print, minmax(SCALE_ARRAY(arr1,arr2))    
  ;         3.00000      8.00000
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  ;
  
  a =  (max(arr2)-min(arr2))/(1.0*(max(arr1)-min(arr1)))
  b = min(arr2) - a*min(arr1)
  
  return, a*arr1+b
End

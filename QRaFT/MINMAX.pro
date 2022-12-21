function MINMAX, x

  ;+
  ; NAME:
  ;     MINMAX
  ;
  ; PURPOSE:
  ;     Finds the minimun and the maximum values of the submitted data array; returns them as a 2-element array [min, max].
  ;
  ; CALLING SEQUENCE:
  ;     result = MINMAX(x)
  ;
  ; INPUTS:
  ;     x   - analyized data array, arbitrary dimensionality.
  ;
  ; OUTPUTS:
  ;     result  - 2-element array [min, max] containing the minimun and the maximum values of the array "x".
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     print, minmax([1,2,3,4,5])
  ;         1       5
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------

  return, [min(x) , max(x)]

End

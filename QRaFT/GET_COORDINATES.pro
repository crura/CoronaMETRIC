function GET_COORDINATES, IMG

  ;+
  ; NAME:
  ;     GET_COORDINATES
  ;     
  ; PURPOSE:
  ;     Prepares an empty structure to be filled with the values of 
  ;     polar coordinates of data pixels of the submitted solar image.
  ;     
  ; CALLING SEQUENCE:
  ;     polar_coord = GET_COORDINATES(IMG)
  ;
  ; INPUTS:
  ;     IMG - 2D data array representing solar image
  ;
  ; OUTPUTS:
  ;     IDL structure of two empty 2D arrays, X and Y, to be filled
  ;     with polar coordinates by an external code.  
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     help, GET_COORDINATES(fltarr(100,100))
  ;     X   FLOAT     Array[100, 100]
  ;     Y   FLOAT     Array[100, 100]
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  ;
; 
;  
  X_ = findgen(n_elements(img[*,0])) & Y_ = findgen(n_elements(img[0,*]))
  X = IMG & X[*,*]=0.0 & Y=X
  for j=0,n_elements(X[0,*])-1 do X[*,j] = X_
  for i=0,n_elements(Y[*,0])-1 do Y[i,*] = Y_
  
  return, {X:X, Y:Y}
  
End
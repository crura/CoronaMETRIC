
function ATAN2, yy, xx ; returns angles in the  0... 2Pi range

  ;+
  ; NAME:
  ;     ATAN2
  ;
  ; PURPOSE:
  ;     Computes an array of polar angles within the  0... 2Pi range
  ;     for the provided arrays of vertical and horizontal coordinates.
  ;     
  ;     Notice the coordinate arrays (yy, xx) when calling the funciton.
  ;
  ; CALLING SEQUENCE:
  ;     angles = ATAN2(yy,xx)
  ; 
  ; INPUTS:
  ;     yy      - 1D array of Y coordinates of data points in the rectangular coordinate system
  ;     xx      - 1D array of X coordinates of data points in the rectangular coordinate system
  ;
  ; OUTPUTS:
  ;     angles  - 1D array of polar angles raging from 0 to 2 Pi, in radians.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     print, ATAN2([2.0, 5.7, -250.0, -20.0], [1.5, -4.0, -300, 33.0])/!Pi
  ;     0.295167     0.694775      1.22114      1.82656
  ;     
  ;     (polar angles of 4 data points located respectively in the 1st, 2nd, 3rd, and 4th quadrants, 
  ;      printed in Pi units).  
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  
  N = n_elements(xx) 
    angles = fltarr(N) 
    
    for i=0, N-1 do $
      angles[i] =  atan(yy[i],xx[i])+(yy[i] lt 0. ? 2*!PI : 0.)
    
    if N eq 1 then angles=angles[0]
    
    return, angles

  end

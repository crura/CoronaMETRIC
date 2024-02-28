Function LIMITRANGE1, Data, Range ;

  ;+
  ; NAME:
  ;     LIMITRANGE1
  ;
  ; PURPOSE:
  ;     Enforces the dynamic range of a 2D data array 
  ;     using the prescribed minimum and maximum values:
  ;       1) Array values elements below the required minimum are set to that minimum value;       
  ;       2) Array elements above the required maximum are set to that maximum value;
  ;       3) If all input array ellements are above the minimum, the element [0,0] is assigned the required minimum. 
  ;       4) If all input array ellements are below the maximum, the element [0,1] is assigned the required maximum.
  ;        
  ;     Returns the result as a new array of the same type.
  ;     
  ;     Used for plotting purposes.   
  ;
  ; CALLING SEQUENCE:
  ;     Data_adjusted = LIMITRANGE1(Data, Range)
  ;
  ; INPUTS:
  ;     Data  - input data array of any dimension 
  ;     Range - 2-element array containing minimum (Range[0]) and maximum (Range[1]) values
  ;
  ; OUTPUTS:
  ;     Data_adjusted - output data array with the dynamic range matching Range
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     IMG_NEW = LIMITRANGE1(IMG, [100, 1000]) 
  ;     
  ;     Array IMAGE is adjusted to the dynamic range [100,1000];
  ;     the resulting array is assigned to variable IMG_NEW.
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------

 
; truncates range and adds min-max pixels if necessary
; call :  Rescaled_Image = limitrange1(Image, [-100,100] )

  Data1 = Data; to preserve the input array

  w_min = where(data lt Range[0])
  w_max = where(data gt Range[1])
  
  if w_min[0] ne -1 then Data1[w_min] = Range[0] else Data1[0,0]=Range[0]
  if w_max[0] ne -1 then Data1[w_max] = Range[1] else Data1[0,1]=Range[1]
  
  return, Data1
  
End

FUNCTION REMOVE_OCCULT, IMG, P

  ;+
  ; NAME:
  ;     REMOVE_OCCULT
  ;
  ; PURPOSE:  
  ;     Removes the footprint of the occulting disk or the Moon from a solar coronal image
  ;     by settings the pixels covered by the disk to 0.
  ;     
  ;     This funciton should be used before detecting coronal features, to ensure that 
  ;     [some segments of] the solar or occulting disc edges are not erroneously interpreted 
  ;     as coronal features.  
  ;
  ; CALLING SEQUENCE:
  ;     IMG_cleaned = REMOVE_OCCULT(IMG, P)
  ;
  ; INPUTS:
  ;     IMG - 2D data array represented the processed image
  ;     P   - structure of processing parameters, retrieved from file using P=READ_PARAMETERS().
  ;           The information about the circular region to be removed is contained in
  ;           P.XYCenter (coordinates of the disak center) and P.Rs (radius of the region).
  ;
  ; OUTPUTS:
  ;     IMG_cleaned - modified version of array IMG in which the pixels of hte removed circular regions
  ;                   are set to 0. 
  ;     
  ; RESTRICTIONS:
  ;     none
  ;     
  ; EXAMPLE:
  ;     IMG_cleaned = REMOVE_OCCULT(IMG, P)
  ;     
  ;       Here, IMG is the solar image to be cleaned; 
  ;       P is the processing parameters obtianed by P=READ_PARAMETERS()
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------  
  
  img_=img
  
  x2 = (indgen(n_elements(img_[*,0])) - P.XYCenter[0])^2
  y2 = (indgen(n_elements(img_[0,*])) - P.XYCenter[1])^2
  R2 = (P.Rs)^2
  
  for i=0, n_elements(img_[*,0])-1 do $
    for j=0, n_elements(img_[0,*])-1 do $
      if x2[i] + y2[j] le R2 then img_[i,j]=0 
      
  ;for x=0, n_elements(img_[*,0])-1 do $
  ;  for y=0, n_elements(img_[0,*])-1 do $      
  ;    if (x-P.XYCenter[0])^2+(y-P.XYCenter[1])^2 le (P.Rs)^2 then img_[x,y]=0      
      
  return, img_  
END
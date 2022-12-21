
function FIND_CLOSEST_FEATURE, FEATURES, X, Y

  ;+
  ; NAME:
  ;     FIND_CLOSEST_FEATURE
  ;
  ; PURPOSE:
  ;     Finds the index of the coronal feature in the array FEATURES of all the detected features
  ;     which is the closest to a given search point [X,Y].
  ;     X and Y are discrete rectangular coordinates defined by the image indices.
  ;
  ;     The funciton is used to locate the feature closest to a mouse click point,
  ;     for manual interactive manipulaitons with the detected features.
  ;
  ; CALLING SEQUENCE:
  ;     result = FIND_CLOSEST_FEATURE(FEATURES, X, Y)
  ;
  ; INPUTS:
  ;     FEATURES  -   array of IDL structures containing locations
  ;                   of the detected features
  ;     X, Y      -   respoectively the horizontal (first image index) and the vertical (second image index) positions
  ;                   of the image piixel near which the feature should be found
  ;
  ; OUTPUTS:
  ;     result    -   IDL structure with information about the located closest feature:
  ;       result.i    - index of the closest feature in the input array FEATURES
  ;       result.k    - index of the closest node inside the closest feature
  ;       result.dist - Eucledian distance (in image pixel units) between the searh point (X,Y)
  ;                     and the closest node.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     help, FIND_CLOSEST_FEATURE(FEATURES, 30, 50)
  ;     I               INT             32
  ;     K               INT              5
  ;     DIST            FLOAT           1054.71
  ;
   ;     where FEATURES is an array of features detected via a chain of
   ;     processing steps implemented by this code package
   ;     (see PROCESS_FILE.pro for more details).
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  ;
  ;
  ; plotting example:
  ; q = find_closest_feature(A2.ELP_PARAM, 206,610)
  ; oplot, A2.ELP_PARAM[q.i].xx_r, A2.ELP_PARAM[q.i].yy_r, color=0
  ;
  i_=0 & k_=0
  sqr_dist_ = (FEATURES[i_].xx_r[k_] - X)^2 + (FEATURES[i_].yy_r[k_] - Y)^2

  for i=0, n_elements(FEATURES)-1 do begin
    for k=0, n_elements(FEATURES[i].xx_r)-1 do begin
      sqr_dist = (FEATURES[i].xx_r[k] - X)^2 + (FEATURES[i].yy_r[k] - Y)^2
      if sqr_dist lt sqr_dist_ then begin
        i_=i & k_=k & sqr_dist_ = sqr_dist
      endif
    endfor
  endfor
  return, {i:i_, k:k_,dist:sqrt(sqr_dist)}
End

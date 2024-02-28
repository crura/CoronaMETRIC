function FILTER_FEATURES_MANUAL, FEATURES, IMG_enh, P,  w=w

  ;+
  ; NAME:
  ;     FILTER_FEATURES_MANUAL
  ;
  ; PURPOSE:
  ;     Interactive manual filtration (removal) of unreliable coronal features
  ;     Filtration is performed based on a decision-making by a human operator
  ;     through a series of mouse clicks.
  ;
  ;     The function provides on-screen instructions on how to process the features.
  ;
  ;     It is recommended to use this function _after_ the automatic filtration
  ;     function FILTER_FEATURES(), to ensure that all the features returned by
  ;     this code package are valid.
  ;
  ; CALLING SEQUENCE:
  ;     FEATURES_filtered = FILTER_FEATURES_MANUAL(FEATURES, w=w)
  ;
  ; INPUTS:
  ;     FEATURES    - array of IDL structures containing locations of the detected features
  ;
  ; OUTPUTS:
  ;     FEATURES_filtered - array of IDL structures of manually filtered features
  ;     w - array positions of the filtered features in the original (non-filtered) array FEATURES
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     FEATURES_filtered = FILTER_FEATURES_MANUAL(FEATURES)
  ;
  ;     where FEATURES is an array of features detected in image IMG via a chain of
  ;     processing steps implemented by this code package
  ;     (see PROCESS_FILE.pro for more details).
  ;
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------


  ;i_arr = [0]

  ii = intarr(n_elements(FEATURES)) & ii[*]=1
  w = indgen(n_elements(FEATURES))
  !mouse.button=1
  WHILE(!MOUSE.BUTTON NE 4) DO BEGIN
    CURSOR,X,Y,/Data,/DOWN
    if !MOUSE.BUTTON EQ 1 THEN begin
      c = find_closest_feature(FEATURES[w], X,Y)
	  ;=============================
	  ; for old IDL:
      if n_elements(i_arr) eq 0 then i_arr=[w[c.i]] else $
      i_arr=[i_arr,w[c.i]]
	  ;=============================

      ;PRINT,'X = ', string(X, format = '(D30.20)'), '  Y = ', string(Y, format = '(D30.20)'), c.i, 'Chi-SQ = ', features[c.i].chisq
      if n_elements(i_arr) ne 0 then ii[i_arr] = 0
      w = where(ii ne 0)
      PLOT_FEATURES, IMG_enh, FEATURES[w], P, range=minmax(P.blob_detect_thresholds), /old

    endif

  ENDWHILE
  ;ii = intarr(n_elements(FEATURES)) & ii[*]=1
  ;if n_elements(i_arr) ne 0 then ii[i_arr] = 0
  w = where(ii ne 0)

  return, FEATURES[w]

End
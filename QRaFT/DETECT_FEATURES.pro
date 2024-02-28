 FUNCTION DETECT_FEATURES, IMG_P, P, X_p, Y_p, Phi_p, Rho_p, dyn_range, FIT_MODE, silent=silent

  ;+
  ; NAME:
  ;     DETECT_FEATURES
  ;
  ; PURPOSE:
  ;     Identifies quasi-radial structure in a coronal image transform to polar coordinates.
  ;     Uses blob detection and polynomial or ellipse-based interpolation of the detected
  ;     features.
   ;
  ; CALLING SEQUENCE:
  ;     FEATURES_ = DETECT_FEATURES(IMG_p, P, X_p, Y_p, Phi_p, Rho_p, dyn_range, FIT_MODE, /silent)
  ;
  ; INPUTS:
  ;     IMG_p   - 2D array representing a gradient-enhanced solar coronal image transormed to plane polar coordinates
  ;     P       - IDL structure containg processing parameters, retrieved from file using P=READ_PARAMETERS()
  ;     X_p       - 2D array of cartesian X-positions of image IMG_p pixels
  ;     Y_p       - 2D array of cartesian Y-positions of image IMG_p pixels
  ;     Phi_p     - 2D array of angular coordinates of image IMG_p pixels
  ;     Rho_p     - 2D array of radial coordinates of image IMG_p pixels
  ;     dyn_range - 2-element array [min, max] containing the lower (min) and upper (max) image intensity
  ;                 thresholds used to detect the features. Image pixels falling in this intensity interval
  ;                 are assigned to one of the detected feature; other pixels are disregarded.
  ;     FIT_MODE  - string variable (case-insensitive) defining the feature interpolation method
  ;                 used by this funciton.
  ;		silent   - integer keyword turning on/off silent mode of Polyfit_decomp(). If silent=1, no messages are displayed. Default value 0.
  ;
  ;                 Allowed values (case-insensitive):
  ;
  ;                 FIT_MODE = 'ELP'  - features are interpolated by ellipses
  ;                 FIT_MODE = 'POLY1'  - features are interpolated by 1st degree polynomials (straight lines) in polar coordinates
  ;                 FIT_MODE = 'POLY2'  - features are interpolated by 2nd degree polynomials in polar coordinates
  ;                 FIT_MODE = 'POLY3'  - features are interpolated by 3rd degree polynomials in polar coordinates
  ;
  ; OUTPUTS:
  ;     FEATURES  - array of IDL structures containing location and orientation parameters of the detected features.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     FEATURES = DETECT_FEATURES(IMG_p, P, X_p, Y_p, Phi_p, Rho_p, [0.0, 0.025], 'POLY1', /silent)

  ;   Most important data fields of the FEATURE[i] structure:

  ;   XX_R, YY_R: array indices of defining feature segments used to evaluate       feature angles
  ;   ANGLES_XX_R, ANGLES_YY_R: array indices of the intermediate points to         which feature angles are assigned
  ;   ANGLES_P: polar angles of the feature segments
  ;   ANGLES_P_RADIAL: polar angles of feature segments relative to the local       radial direction
  ;   ANGLE_AVR: average angle of the feature
  ;   ANGLE_RADIAL_AVR: average angle of the feature relative to            the local radial direction
  ;   L: length of the feature in the array coordinates
  ;   NPOINTS: number of feature pixels before approximation
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------


  dx = 3

  case strupcase(FIT_MODE) of

    'ELP':    FEATURES = Ellipse_decomp(img_p, dyn_range, P.blob_detect_min, P.blob_detect_n_segments)

    'POLY1':  FEATURES = Polyfit_decomp(img_p, dyn_range, P.blob_detect_min, dx, 1, /tr, silent=silent );, /aver)
    'POLY2':  FEATURES = Polyfit_decomp(img_p, dyn_range, P.blob_detect_min, dx, 2, /tr, silent=silent );, /aver)
    'POLY3':  FEATURES = Polyfit_decomp(img_p, dyn_range, P.blob_detect_min, dx, 3, /tr , silent=silent);, /aver)

    ;'POLY1':  FEATURES = Polyfit_decomp(img_p, dyn_range, P.blob_detect_min, P.blob_detect_n_segments, 1, /tr );, /aver)
    ;'POLY2':  FEATURES = Polyfit_decomp(img_p, dyn_range, P.blob_detect_min, P.blob_detect_n_segments, 2, /tr );, /aver)
    ;'POLY3':  FEATURES = Polyfit_decomp(img_p, dyn_range, P.blob_detect_min, P.blob_detect_n_segments, 3, /tr );, /aver)
  endcase


  if n_elements(FEATURES) gt 0 then begin
    ;FEATURES=Compute_Feature_Coord(FEATURES, P.XYcenter, X_p, Y_p, Phi_p, Rho_p)
    FEATURES=Compute_Feature_Coord(FEATURES, P.XYcenter, X_p, Y_p, Phi_p, Rho_p)


    ; Make_Angle_Arrays, Ellipse_fits.elp_param, Angles, X, Y
    ; LATER

    ;RETURN, {Angles:Angles, X:X, Y:Y, Elp_param:Ellipse_fits.elp_param}
    ;RETURN, Ellipse_fits.elp_param
    RETURN, FEATURES
  endif else $
    RETURN, [-1]

 END

 FUNCTION FILTER_FEATURES, FEATURES, IMG, P, w=w

   ;+
   ; NAME:
   ;     FILTER_FEATURES
   ;
   ; PURPOSE:
   ;     Automatic filtration of quasi-linear coronal features detected in image IMG
   ;     Filtration is performed based on a set of parameters (passed by the structure P)
   ;     empirically optimized for a particular image type.
   ;
   ;     The filtering algorithm applies lower and upper bounds to the following
   ;     characteristics of the features:
   ;        1) feature length (linear size), to exclude small noisy features;
   ;        2) polar angle range, to remove artifacts associated with a specific image sector
   ;           (e.g., to exclude the supporting structure the occulting disk);
   ;        3) average relative angle of the feature with respect to the local radial direciton,
   ;           to remove spurious features characterized by unrealistic orientaiton;
   ;        4) maximum phi-averaged image intensity along the feature,  to remove faint and unreliable features
   ;   			 detected in the rho-locations of small overall signal.
   ;
   ; CALLING SEQUENCE:
   ;     FEATURES_filtered = FILTER_FEATURES(FEATURES, IMG, P, w=w)
   ;
   ; INPUTS:
   ;     FEATURES    - array of IDL structures containing locations of the detected features
   ;     IMG         - original solar image (rectangular coordinates, non-enhanced) in which the features were detected
   ;      P   - structure of processing parameters, retrieved from file using P=READ_PARAMETERS()
   ;
   ; OUTPUTS:
   ;     FEATURES_filtered - array of IDL structures of filtered features
   ;     w - array positions of the filtered features in the original (non-filtered) array FEATURES
   ;
   ; RESTRICTIONS:
   ;     none
   ;
   ; EXAMPLE:
   ;     FEATURES_filtered = FILTER_FEATURES(FEATURES, IMG, P)
   ;
   ;     where FEATURES is an array of features detected in image IMG via a chain of
   ;     processing steps implemented by this code package
   ;     (see PROCESS_FILE.pro for more details),
   ;     P is a structure of processing parameters.
   ;
   ; MODIFICATION HISTORY:
   ;       V. Uritsky, 2019-2022
   ;-
   ;----------------------------------------------------------------


   ; THE FOLLOWING SET OF FILTIRING PARAMETERS IS OPTIMIZED FOR MLSO and STEREO COR1
   ; AND MAY NEED TO  BE RE-ADJUSTED FOR OTHER IMAGE TYPES (e.g. ECLIPSE IMAGES)

   ;L_range = [20,1000] ; allowed feature sizes, in discrete rectangular coordinates
   ;Phi_range = [2.77, 2.85] ; excluded polar angle range, used to remove the "handle" (supporting rod)
                            ; of the MLSO occulting disk and the associated artifacts across the image
   ;Alpha_range = 1*[-1,1]*!Pi/4.0 ; allowed range of average relative angles measured wrt local radial direciton
   ;Flux_range = [100,1E10] ; allowed range of average fluxes along the feature
   ;Flux_rho_trend_range = [0.170,1E10] ; allowed range of average fluxes along the feature computed using averaged radial trend
   ;Flux_rel_range = [0.0,1E10] ; allowed range of relative fluxes (actual flux along feature / averaged flux along feature using averaged radial trend )

   Fluxes = fltarr(n_elements(features))
   for i=0,n_elements(fluxes)-1 do fluxes[i] = mean( img[features[i].xx_r, features[i].yy_r] )

   ; prepares empty structure for polar coordinates
   Rect_to_Polar, img, P.XYcenter, P.d_phi, P.d_rho, IMG_P, Y_p=Y_p, X_p=X_p, Phi_p=Phi_p, Rho_p=Rho_p

   ;========================
   ;Rho_trend = mean(img_p,dim=1)  ; use abs(img_p) if img is a signed array
   ; for old IDL:
   Rho_trend = fltarr(n_elements(img_p[0,*]))
   for rho_ind=0, n_elements(img_p[0,*])-1 do Rho_trend[rho_ind] = mean(img_p[*,rho_ind])
   ;========================


   Fluxes_rho_trend = fltarr(n_elements(features))
   for i=0,n_elements(Fluxes_rho_trend)-1 do Fluxes_rho_trend[i] = max( Rho_trend[features[i].yy_p] ) ;mean( Rho_trend[features[i].yy_p )

   Fluxes_rel = Fluxes/Fluxes_rho_trend

   w = where( $
     ((features.L gt P.L_range[0]) and (features.L lt P.L_range[1])) $
     and ((features.x_p_avr lt P.Phi_range[0]) or (features.x_p_avr gt P.Phi_range[1])) $
     and ( (features.angle_radial_avr gt P.Alpha_range[0]) and (features.angle_radial_avr lt P.Alpha_range[1]) ) $
     and ((fluxes_rho_trend gt P.flux_rho_trend_range[0]) and (fluxes_rho_trend lt P.flux_rho_trend_range[1])) $
   )

   if w[0] ne -1 then  RETURN, FEATURES[w] $
   else RETURN, -1

 END
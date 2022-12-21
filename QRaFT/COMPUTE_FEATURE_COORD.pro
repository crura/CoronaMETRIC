Function COMPUTE_FEATURE_COORD, FEATURES, XYcenter, X_p, Y_p, Phi_p, Rho_p 

  ;+
  ; NAME:
  ;     COMPUTE_FEATURE_COORD
  ;
  ; PURPOSE:
  ;     Calculates coordinates and local angles of nodes and linear segments used to interpolate
  ;     the detected solar features,both in rectangular and plane coolar coordinate systems.
  ;     The computed parameters are appended as additional tags in the input data structure
  ;     containing the features.
  ;
  ; CALLING SEQUENCE:
  ;     FEATURES_with_coord = Compute_Feature_Coord(FEATURES, XYcenter, X_p, Y_p, Phi_p, Rho_p)
  ;
  ; INPUTS:
  ;     FEATURES    - array of IDL structures containing parameters of the detected features
  ;                   as defined by the feature-tracking funcitons "Ellipse_decomp" or "Polyfit_decomp"
  ;     XYcenter    - 2-element array [X,Y] containing horizontal and vertical pixel positions
  ;                   of the Sun disk center on studied images
  ;     X_p       - 2D array of cartesian X-positions of image pixels
  ;     Y_p       - 2D array of cartesian Y-positions of image pixels
  ;     Phi_p     - 2D array of angular coordinates of image pixels
  ;     Rho_p     - 2D array of radial coordinates of image pixels
  ;
  ; OUTPUTS:
  ;     FEATURES_with_coord   - appended array of structures containing all the fields of the original array FEATURES,
  ;                             as well as a set of new variables describing local positions and angles
  ;                             of each of the straight-line segments interpolating the features
  ;                             (the segments are linear in polar coordinats but not in rectangular coordinates).
  ;
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     FEATURES_with_coord = Compute_Feature_Coord(FEATURES, P.XYcenter, X_p, Y_p, Phi_p, Rho_p)
  ;
  ;     P is a structure containing processing parameters retrieved from a parameter file using function READ_PARAMETERS(),
  ;     X_p, Y_p, Phi_p, Rho_p are the coordinate arrays returned by function ENHANCE_IMNAGE().
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;
  ;
  n_points = n_elements(FEATURES[0].xx)

  for i=0, n_elements(FEATURES)-1 do begin

    ; polar coordinates of corner points
    x1_p = reform(Phi_p[FEATURES[i].x1,0])
    y1_p = reform(Rho_p[0,FEATURES[i].y1])
    x2_p = reform(Phi_p[FEATURES[i].x2,0])
    y2_p = reform(Rho_p[0,FEATURES[i].y2])
    ;y2_p = 0

    ;;; discrete rectangular coordinates of corner points
    ;x1_r = reform(Rho_p[0,A.Elp_param[i].y1]*cos(phi_p[A.Elp_param[i].x1,0]) + XYCenter[0])
    ;y1_r = reform(Rho_p[0,A.Elp_param[i].y1]*sin(phi_p[A.Elp_param[i].x1,0]) + XYCenter[1])
    ;x2_r = reform(Rho_p[0,A.Elp_param[i].y2]*cos(phi_p[A.Elp_param[i].x2,0]) + XYCenter[0])
    ;y2_r = reform(Rho_p[0,A.Elp_param[i].y2]*sin(phi_p[A.Elp_param[i].x2,0]) + XYCenter[1])

    x1_r = X_p[FEATURES[i].x1, FEATURES[i].y1]
    y1_r = Y_p[FEATURES[i].x1, FEATURES[i].y1]
    x2_r = X_p[FEATURES[i].x2, FEATURES[i].y2]
    y2_r = Y_p[FEATURES[i].x2, FEATURES[i].y2]

    ; length of the feature ini descrete rectan. coord
    L = ( (x2_r-x1_r)^2 + (y2_r-y1_r)^2 )^0.5

    ; polar coordinates of points along major axes

    ;xx_p = reform(Phi_p[FEATURES[i].xx,0])
    ;yy_p = reform(Rho_p[0,FEATURES[i].yy])

    xx_p = scale_array(FEATURES[i].xx, reform(Phi_p[FEATURES[i].xx,0]))
    yy_p = scale_array(FEATURES[i].yy, reform(Rho_p[0,FEATURES[i].yy]))

    ;;; rectangular coordinates of points along major axes
    ;xx_r = reform(X_p[FEATURES[i].xx, FEATURES[i].yy])
    ;yy_r = reform(Y_p[FEATURES[i].xx, FEATURES[i].yy])
    ;reform(Rho_p[0,A.Elp_param[i].yy]*sin(phi_p[A.Elp_param[i].xx,0]) + XYCenter[1])

    xx_r = yy_p*cos(xx_p) + XYCenter[0]
    yy_r = yy_p*sin(xx_p) + XYCenter[1]

    ;N = n_elements(xx_r)-1
    N = features[i].n_nodes-1
    angles = fltarr(n_points) & angles_p = fltarr(n_points) & angles_p_radial = fltarr(n_points)
    angles_xx_r = fltarr(n_points) & angles_yy_r = fltarr(n_points)
    for k=0,N-1 do begin
      angles[k] = atan2(yy_r[k+1]-yy_r[k], xx_r[k+1]-xx_r[k])

      ; computing orientation angles of the k-th segment
      ; in polar coordinates
      delta_phi = xx_p[k+1]-xx_p[k]
      alpha = delta_phi + atan(  sin(delta_phi)/( (yy_p[k+1]/yy_p[k]) - cos(delta_phi) )  )
      angles_p_radial[k] = alpha    ; angle relative to the local radial direcition [phi=const]
      angles_p[k] = xx_p[k] + alpha ; counter-clockwise polar angle relative to X>0 direction
      if angles_p[k] lt 0 then angles_p[k]=angles_p[k]+2*!Pi
      if angles_p[k] gt 2*!Pi then angles_p[k]=angles_p[k]-2*!Pi
      ;---

      angles_xx_r[k] = (xx_r[k] + xx_r[k+1])/2.0
      angles_yy_r[k] = (yy_r[k] + yy_r[k+1])/2.0
    endfor

    ;angle_avr = mean(angles)
    angle_avr = atan2( y2_r - y1_r, x2_r - x1_r )
    angle_radial_avr = mean(angles_p_radial[0:N-1])

    x_p_avr = mean(xx_p[0:N]);(x1_p+x2_p)/2.0
    y_p_avr = mean(yy_p[0:N])
    x_r_avr = mean(xx_r[0:N])
    y_r_avr = mean(yy_r[0:N])

    ; 'xx_elp', reform(A.Ellipses[2*i,*]), 'yy_elp', reform(A.Ellipses[2*i+1,*]),
    struc = create_struct(FEATURES[i], 'x1_p',x1_p, 'y1_p', y1_p, 'x2_p',x2_p, 'y2_p',y2_p, $
                'x1_r',x1_r, 'y1_r',y1_r, 'x2_r',x2_r, 'y2_r',y2_r, 'L', L, $
                'xx_p',xx_p, 'yy_p',yy_p,'xx_r',xx_r, 'yy_r',yy_r, 'angles',angles, $
                'angles_p',angles_p, 'angles_p_radial', angles_p_radial, $
                'angle_avr', angle_avr, 'angle_radial_avr', angle_radial_avr, $
                'x_p_avr', x_p_avr, 'y_p_avr', y_p_avr, 'x_r_avr', x_r_avr, 'y_r_avr', y_r_avr, $
                'angles_xx_r', angles_xx_r, 'angles_yy_r', angles_yy_r)

    if n_elements(FEATURES_) eq 0 then FEATURES_ =  struc else FEATURES_ = [FEATURES_, struc ]

  endfor
  ;A = {ELP_BLOBS:A.ELP_BLOBS,ELP_PARAM:ELP_PARAM}

  return, FEATURES_


End

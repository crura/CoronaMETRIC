
FUNCTION ELLIPSE_DECOMP, IMG, dyn_range, min_blob, n_segments

  ;+
  ; NAME:
  ;     ELLIPSE_DECOMP
  ;
  ; PURPOSE:
  ;     Detects quasi-radial features in a solar image and interpolates them with ellipses:
  ;       1) Finds image features based on lower and upper detection thresholds;
  ;       2) Removes smallest features caused by noise (initial filtering);
  ;       3) Replaces each feature with a best-fit ellipse;
  ;       4) Computes coordinates of equally-spaced nodes along the major axes of the ellipses.
  ;
  ;     Should be applied to a gradient-enhanced solar image transformed into plane polar coordinates.
  ;
  ; CALLING SEQUENCE:
  ;     FEATURES = ELLIPSE_DECOMP(IMG, dyn_range, min_blob, n_segments)
  ;
  ; INPUTS:
  ;     IMG         - 2D array containing the enhanced image in plane polar coordinates (Phi_p, Rho_p), obtained using adaptive 2nd-order azimuthal differencing
  ;     dyn_range   - 2-element array [min, max] containing the lower (min) and upper (max) image intensity
  ;                 thresholds used to detect the features. Image pixels falling in this intensity interval
  ;                 are assigned to one of the detected feature; other pixels are disregarded.
  ;     min_blob    - size of the smallest feature (in image pixels) to be included
  ;     n_segments  - number of segments in each interpolated feature. Each feature is devided into n_segments pieces
  ;                   separated by n_segments + 1 equally spaced nodal points. The positions of the nodes are calculated and returned.
  ;
  ; OUTPUTS:
  ;     FEATURES  - array of IDL structures containing location and orientation parameters of the detected features.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:

  ;     FEATURES = ELLIPSE_DECOMP(IMG_P, [0.0, 0.025], 20, 5)
  ;     (using manually entered processing parameteres)
  ;
  ;     FEATURES = ELLIPSE_DECOMP(IMG_P, [P.blob_detect_thresholds[0,0], P.blob_detect_thresholds[1,0]], P.blob_detect_min, P.blob_detect_n_segments)
  ;     (same, using processing parameteres stored in the data structure P=READ_PARAMETERS() obtained from a parameter file)
  ;
  ;     In either example, IMG_P is an enhanced image in polar coordinates; see PROCESS_FILE for more details on the processing steps.
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  ;

  mask = mask_image(img, dyn_range)

  ELP_BLOBS = Obj_New('blob_analyzer', img, mask = mask, scale=[1,1])
  count = ELP_BLOBS -> NumberOfBlobs()
  ;w = []
  ; filtering-out smallest blobs
  for i=0, count-1 do begin
    stats = ELP_BLOBS -> GetStats(i, /NoScale)
    if stats.count ge min_blob then begin
      if n_elements(w) eq 0 then w = [i] else w=[w,i]
    endif
  endfor
  ;Ellipses=[]
  if n_elements(w) gt 0 then begin ; valid features found
    Nx = n_elements(img[*,0]) & Ny = n_elements(img[0,*])

    for i=0, n_elements(w)-1 do begin

    q = ELP_BLOBS -> FitEllipse(w[i], elp_struc = elp_struc, /noscale)

    if n_elements(Ellipses) eq 0 then Ellipses = [q] else Ellipses=[Ellipses, q]

    ; constraining x and y ranges of ellipses to avoid array indices outside of IMG
    ; primitive but better than nothing.
    ; A better method would use x,y interpolation with image boundaries...

    if elp_struc.x1 lt 0 then elp_struc.x1=0
    if elp_struc.x2 lt 0 then elp_struc.x2=0
    if elp_struc.y1 lt 0 then elp_struc.y1=0
    if elp_struc.y2 lt 0 then elp_struc.y2=0

    if elp_struc.x1 gt Nx-1 then elp_struc.x1=Nx-1
    if elp_struc.x2 gt Nx-1 then elp_struc.x2=Nx-1
    if elp_struc.y1 gt Ny-1 then elp_struc.y1=Ny-1
    if elp_struc.y2 gt Ny-1 then elp_struc.y2=Ny-1


    dx = (elp_struc.x2-elp_struc.x1)/(1.0*n_segments)
    dy = (elp_struc.y2-elp_struc.y1)/(1.0*n_segments)
    xx = elp_struc.x1 + findgen(n_segments+1)*dx
    yy = elp_struc.y1 + findgen(n_segments+1)*dy

    ;elp_struc = {elp_struc, xx:xx, yy:yy}
    elp_struc = create_struct(elp_struc, 'xx',xx, 'yy',yy)
    if n_elements(ELP_PARAM) eq 0 then ELP_PARAM =  elp_struc else ELP_PARAM = [ELP_PARAM, elp_struc ]

    ;stats = ELP_BLOBS -> GetStats(w[i], /NoScale)
    ;;if n_elements(ELP_STAT) eq 0 then ELP_STAT = stats else ELP_STAT = [ELP_STAT, stats]
    ; need to exclude perimeter_pts array which has a variable size

    endfor
     ;return, {ELP_BLOBS:ELP_BLOBS, ELP_PARAM:ELP_PARAM, Ellipses:Ellipses}
     FEATURES = ELP_PARAM
     return, FEATURES
   endif else $
     return, [] ; no features found

End

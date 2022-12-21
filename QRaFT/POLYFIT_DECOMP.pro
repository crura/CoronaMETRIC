FUNCTION POLYFIT_DECOMP, IMG, dyn_range, min_blob, dx, order, transpose_img=transpose_img, silent=silent
  ;+
  ; NAME:
  ;     POLYFIT_DECOMP
  ;
  ; PURPOSE:
  ;     Detects quasi-radial features in a solar image and interpolates them with polynomials:
  ;       1) Finds image features based on lower and upper detection thresholds;
  ;       2) Removes smallest features caused by noise (initial filtering);
  ;       3) Replaces each feature with a best-fit polynomial;
  ;       4) Computes coordinates of equally-spaced nodes along the polynomial lines.
  ;
  ;     Should be applied to a gradient-enhanced solar image transformed into plane polar coordinates.
  ;
  ; CALLING SEQUENCE:
  ;     FEATURES = POLYFIT_DECOMP(IMG, dyn_range, min_blob, dx, order, /transpose, /average)
  ;
  ; INPUTS:
  ;     IMG         - 2D array containing the enhanced image in plane polar coordinates (Phi_p, Rho_p),
  ;                   obtained using adaptive 2nd-order azimuthal differencing
  ;     dyn_range   - 2-element array [min, max] containing the lower (min) and upper (max) image intensity
  ;                   thresholds used to detect the features. Image pixels falling in this intensity interval
  ;                   are assigned to one of the detected feature; other pixels are disregarded.
  ;     min_blob    - size of the smallest feature (in image pixels) to be included
  ;	  dx		  - x-distance (in pixel units) between neighboring nodes used to interpolate the feature.
  ; 			    The number of nodal points and the segments between them varies
  ;			    depending on feature length; dx is same for all features.
  ;                   The sampling is equidistant along the X direction (along the Y direction if /transpose is used).
  ;                   The positions of the nodes are calculated and returned.
  ;     order       - order of the fitting polynomials
  ;     transpose_img   - if this keyword is set (transpose_img=1) the image is transposed before any other processing steps.
  ;                   This could lead to more robust fits if many features are quasi-vertical.
  ;		silent   - integer keyword turning on/off silent mode. If silent=1, no messages are displayed. Default value 0.
  ;
  ; OUTPUTS:
  ;     FEATURES    - array of IDL structures containing location and orientation parameters of the detected features.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLES:

  ;     FEATURES = POLYFIT_DECOMP(IMG_P, [0.0, 0.025], 20, 2, /transpose)
  ;     (using manually entered processing parameteres, 2nd order polynomial fitting)
  ;
  ;     FEATURES = POLYFIT_DECOMP(IMG_P, [P.blob_detect_thresholds[0,0], P.blob_detect_thresholds[1,0]], P.blob_detect_min, P.blob_detect_n_segments, 2, /transpose )
  ;     (same, using processing parameteres stored in the data structure P=READ_PARAMETERS() obtained from a parameter file)
  ;
  ;     In either example, IMG_P is an enhanced image in polar coordinates; see PROCESS_FILE for more details on the processing steps.
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  ;

;
; pfits = Polyfit_decomp(img_p_enh_, [2,50], 20, 10, 3, /tr, /aver)
; w_good = where(pfits[*].status eq 0)
; tvscl_, img_p_enh_
; for k=1, n_elements(w_good)-1 do plots, pfits[w_good[k]].xx, pfits[w_good[k]].yy, thick=2
;

  coord = Get_coordinates(img)

  if keyword_set(transpose_img) then begin
    mask = mask_image(transpose(img), dyn_range)
    X = transpose(coord.Y) & Y = transpose(coord.X)
  endif else begin
    mask = mask_image(img, dyn_range)
    X = coord.X & Y = coord.Y
  endelse

  lbl = label_region(mask, /ulong, /all_neighbors)
  hst = histogram(lbl, reverse_indices = hst_r)

  for i=1, n_elements(hst)-1 do begin  ; exclude i=0, it's a huge cluster of all other pixels

    w = hst_r[hst_r[i]:hst_r[i+1]-1]

    if n_elements(w) ge min_blob then begin

 		; to keep spacing between nodes fixed:
 		; 1) define dx outside of this funciton
 		; 2) recompute n_segments dynamically for each feature
 		; future improvement: save the resulting coordinate arrays of variable length
 		; to the FEATURES structure

		n_nodes = max([2, 1+round( (max(X[w])-min(X[w]))/(1.0*dx) ) ])
        ;dx = (max(X[w])-min(X[w]))/(1.0*n_segments)
        if n_nodes eq 2 then $
          xx = [min(X[w]), max(X[w])] else $
          xx = min(X[w]) + findgen(n_nodes)*dx

        yy = fltarr(n_nodes)

        X_ = X[w] & Y_ = Y[w]
  		if min(X_) eq max(X_) then X_[0] = 1.01*X_[0] ; small perturbation, to avoid fitting error when feature is parallel to Y axis

        polyfit = poly_fit(X_,Y_, order, sigma=sigma, yband=yband, yfit=yfit, yerror=yerror, chisq=chisq, status=status)

        if status eq 0 then begin
          for k=0, order do yy = yy + polyfit[k]*(xx^k)
        endif else begin
          polyfit = fltarr(order+1) & sigma=polyfit
        endelse

        if keyword_set(transpose_img) then begin
          xx_=yy & yy=xx & xx=xx_
        endif

		if (min(xx) lt 0) and (not keyword_set(silent)) then print, 'INDEX: ', i, '   # OF BAD NODES IN THE FIT: ', n_elements(where(xx lt 0))

		w_ = where(xx lt 0) & if w_[0] ne -1 then xx[w_]=0
		w_ = where(yy lt 0) & if w_[0] ne -1 then yy[w_]=0

		w_ = where(xx gt ( n_elements(IMG[*,0])-1) ) & if w_[0] ne -1 then xx[w_]=n_elements(IMG[*,0])-1
		w_ = where(yy gt ( n_elements(IMG[0,*])-1) ) & if w_[0] ne -1 then yy[w_]=n_elements(IMG[0,*])-1

        x1 = xx[0] & x2 = xx[n_nodes-1]
        y1 = yy[0] & y2 = yy[n_nodes-1]


		; making an empty fixed-length array of polyfit coefficients up to order 3, filling with computed values:
	    polyfit_ = dblarr(4) & polyfit_[0:order] = polyfit
	    sigma_ = dblarr(4) & sigma_[0:order] = sigma

		xx_ = dblarr(max(size(IMG, /dim))) & yy_=xx_ ;& xx_[*]=-1 & yy_ = xx_ ; -1 : missing coordinate flag (beyond n_segments)
		xx_[0:n_nodes-1] = xx
		yy_[0:n_nodes-1] = yy

      struc = {x1:x1, y1:y1, x2:x2, y2:y2, xx:xx_, yy:yy_, n_nodes:n_nodes, polyfit_order: order, polyfit:polyfit_, sigma:sigma_,chisq:chisq, status:status}
      if n_elements(FEATURES) eq 0 then FEATURES =  struc else FEATURES = [FEATURES, struc ]

    endif
  endfor

  ;stop
  ;if min(features.x1) lt 0 then stop

  return, FEATURES

End
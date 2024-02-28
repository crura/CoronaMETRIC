PRO RECT_TO_POLAR, IMG, XYcenter, d_phi, d_rho, IMG_p, Rho_range=Rho_range, $
                    X_r=X_r, Y_r=Y_r, Phi_r=Phi_r, Rho_r=Rho_r, X_p=X_p, Y_p=Y_p, Phi_p=Phi_p, Rho_p=Rho_p

  ;+
  ; NAME:
  ;     RECT_TO_POLAR
  ;
  ; PURPOSE:
  ;     Transformes solar image given in rectangular (Cartesian) coordinates
  ;     into the plane polar coordinate system centered at X = XYCenter[0], Y = XYCenter[1].
  ;     Computes polar and Cartesian positions of the pixels of the original and the transformed images.
  ;
  ; CALLING SEQUENCE:
  ;     RECT_TO_POLAR, IMG, XYcenter, d_phi, d_rho, IMG_p, Rho_range=Rho_range, X_r=X_r, Y_r=Y_r, Phi_r=Phi_r, Rho_r=Rho_r, X_p=X_p, Y_p=Y_p, Phi_p=Phi_p, Rho_p=Rho_p
  ;
  ; INPUTS:
  ;     IMG       - original image in rectangular coordinates
  ;     XYCenter  - 2-element array containing X and Y positions (respectively XYCenter[0] and XYCenter[1]) of the central pixel in image "IMG".
  ;     d_phi     - bin size of the angular coordinate
  ;     d_rho     - bin size of the radial coordinate
  ;     Rho_range - range of valid radial coordinate values (dimensional, in X,Y units)
  ;
  ; OUTPUTS:
  ;     Img_p     - 2D array containing the transformed image (in polar coordinates)
  ;
  ;     Positions of pixels in the original (rectangular) image:
  ;     X_r       - 2D array of cartesian X-positions of IMG pixels
  ;     Y_r       - 2D array of cartesian Y-positions of IMG pixels
  ;     Phi_r     - 2D array of angular coordinates of IMG pixels
  ;     Rho_r     - 2D array of radial coordinates of IMG pixels
  ;
  ;     Positions of pixels in the transformd (polar) image:
  ;     X_p       - 2D array of cartesian X-positions of IMG_p pixels
  ;     Y_p       - 2D array of cartesian Y-positions of IMG_p pixels
  ;     Phi_p     - 2D array of angular coordinates of IMG_p pixels
  ;     Rho_p     - 2D array of radial coordinates of IMG_p pixels
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     IMG = readfits("20140404_172405_kcor_l1.fts")
  ;     RECT_TO_POLAR, IMG, [513.0,513.0], 0.00873, 1.0, IMG_p
  ;     tvscl, IMG_p
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-

  if n_elements(X_r)*n_elements(Y_r) eq 0 then begin ; one or both coord arrays are not provided

    coord = Get_coordinates(img) & X_r = coord.X & Y_r = coord.Y

  endif

  Rho_r =(( X_r - XYcenter[0])^2.0+( Y_r - XYcenter[1])^2.0)^0.5
  Phi_r =atan( Y_r - XYcenter[1], X_r - XYcenter[0] )
    w = where(Phi_r lt 0) & Phi_r[w] = Phi_r[w] + 2*!Pi
  ;+++++++++++++++++

  x_count= n_elements(X_r[*,0]) & y_count=n_elements(X_r[0,*])

  phi_min = min(Phi_r) & phi_max = max(Phi_r)
  rho_min = min(Rho_r) & rho_max = max(Rho_r)

  phi_count = ceil((phi_max - phi_min)/d_phi ); + 1

  rho_count = ceil((rho_max - rho_min)/d_rho); + 1

  Phi_p = fltarr(phi_count,rho_count)
  Rho_p = fltarr(phi_count,rho_count)
  X_p   = fltarr(phi_count,rho_count)
  Y_p   = fltarr(phi_count,rho_count)
  Img_p = fltarr(phi_count,rho_count)
  counts_p = fltarr(phi_count,rho_count)

  for i=0, phi_count-1 do Phi_p[i,*] = phi_min + (i)*d_phi; + d_phi/2.0
  for j=0, rho_count-1 do Rho_p[*,j] = rho_min + (j)*d_rho; + d_rho/2.0

  X_p = Rho_p*cos(Phi_p)+XYCenter[0]
  Y_p = Rho_p*sin(Phi_p)+XYCenter[1]

  for i=0, x_count-1 do $
    for j=0, y_count-1 do begin


      i_p = (Phi_r[i,j] - phi_min)/d_phi
      j_p = (Rho_r[i,j] - rho_min)/d_rho

	  if i_p ge phi_count then i_p = phi_count-1
	  if j_p ge rho_count then j_p = rho_count-1

      Img_p[i_p,j_p] = Img_p[i_p,j_p] + Img[i,j]

      Counts_p[i_p,j_p] = Counts_p[i_p,j_p]+1

    endfor

   ; averaging inside non-empty pixels
   w = where(Counts_p gt 0)
   Img_p[w] = Img_p[w]/Counts_p[w]

  if keyword_set(Rho_range) then begin

    w = where((Rho_p gt Rho_range[0]) and (Rho_p lt Rho_range[1]))
    rho_1 = min(Rho_p[w]) & rho_2 = max(Rho_p[w])

    Phi_p = Phi_p[*,rho_1:rho_2]
    Rho_p = Rho_p[*,rho_1:rho_2]
    X_p     = X_p[*,rho_1:rho_2]
    Y_p     = Y_p[*,rho_1:rho_2]
    Img_p = Img_p[*,rho_1:rho_2]
    Counts_p = Counts_p[*,rho_1:rho_2]

  endif

End
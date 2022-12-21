function RADIAL_DETRENDING, IMG, XYCenter, r_trend=r_trend, img_trend=img_trend

  ;+
  ; NAME:
  ;     RADIAL_DETRENDING
  ;
  ; PURPOSE:
  ;     Removes average radial intensity trend from a solar coronal image; returns the
  ;     detrended image and the arrays.
  ;
  ;     Used to equalize the dynamic range of the intensities of the solar coronal features
  ;     located at different radial distances from the Sun.
  ;
  ; CALLING SEQUENCE:
  ;     IMG_detr = RADIAL_DETRENDING(IMG, XYCenter, r_trend=, img_trend=)
  ;
  ;
  ; INPUTS:
  ;     IMG       - 2D array representing the studied solar image
  ;     XYCenter  - 2-element array containing X = XYCenter[0] and Y = XYCenter[1] positions
  ;                 of the Sun center in image IMG, in pixel units.
  ;
  ; OUTPUTS:
  ;     IMG_detr  - the resulting detrended solar image, of the same size as IMG.
  ;     r_trend   - 1D array containing the values of the average radial trend of the image IMG.
  ;                 The index of the "r_trend" array equals the radial distance r from the Sun center
  ;                 in pixel units, i.e. r = sqrt((x- XYCenter[0])^2.0+(y- XYCenter[1])^2.0).
  ;     img_trend - 2D trend array of the same size as IMG containing the average radial trend mapped onto
  ;                 rectangular image coordinats.
  ;                 The original data array, the trend arrays and the detrended array
  ;                 are related through IMG_detr = IMG - img_trend.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     tvscl, RADIAL_DETRENDING(IMG, [513, 513])
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  ;

  ; P.S. Replaces the obsolete Radial_detrend()
  ; which uses a 1D scanning in Cartesian coordinates to remove the radial trend.

  Nx = n_elements(img[*,0]) & Ny = n_elements(img[0,*])

  ; coordinates of the solar disc center in MLSO images (513, 513)
  if n_elements(XYCenter) eq 0 then XYCenter=[Nx/2 + 1, Ny/2 + 1]
  x0 = XYCenter[0] & y0 = XYCenter[1]

  ; radial trend array
  r_trend=fltarr(Nx) & r_trend_n=r_trend
  ;r_max = sqrt(((Nx-1)-x0)^2.0+((Ny-1)-y0)^2.0)
  ;r_trend=fltarr(r_max+1) & r_trend_n=r_trend

  ; finding radial trend:
  for x=0,Nx-1 do $
    for y=0,Ny-1 do begin
      r = sqrt((x-x0)^2.0+(y-y0)^2.0)
      r_trend[r]=r_trend[r]+img[x,y]
      r_trend_n[r]=r_trend_n[r]+1
    endfor
  w = where(r_trend_n gt 0)
  r_trend[w] = r_trend[w]/r_trend_n[w]

 ;r_trend = smooth(r_trend, 8)  ;#########

  ; constructing averaged image following this trend
  img_trend = img & img_trend[*,*]=0.0
  for x=0,Nx-1 do for y=0,Ny-1 do begin & r = sqrt((x-x0)^2.0+(y-y0)^2.0) & img_trend[x,y]=r_trend[r] & endfor
  ; in some cases, this might leave "holes" in the "trend image" because some elements of r_trend[*] could be empty

  ;dedtrended image
  img_detr = img - smooth( img_trend, [8,8] )

  return, img_detr
end

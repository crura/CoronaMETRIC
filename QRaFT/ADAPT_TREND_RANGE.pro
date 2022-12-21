
function ADAPT_TREND_RANGE, IMG_orig, XYCenter, d_phi, d_rho, k=k

  ;+
  ; NAME:
  ;     ADAPT_TREND_RANGE
  ;
  ; PURPOSE:
  ;     Calculates adaptive lower threshold of phi-averaged image intensity; used by the FILTER_FEATURES() funciton
  ;
  ;
  ; CALLING SEQUENCE:
  ;    thrend_range = ADAPT_TREND_RANGE( IMG_orig, XYCenter, d_phi, d_rho, k=)
  ;
  ; INPUTS:
  ;     IMG_orig 	 - 2D array representing original solar image before detrending and enhancement.
  ;     XYCenter   - 2-element array containing X and Y positions (respectively XYCenter[0] and XYCenter[1]) of the central solar pixel in the image "IMG_orig".
  ;     d_phi     		- bin size of the angular coordinate
  ;     d_rho     	    - bin size of the radial coordinate ;
  ;		k					-  coefficient applied to the minimum value of intensity along the averaged radial trend. Default value is 1.0.
  ;
  ; OUTPUTS:
  ;     2-element array [lower_trend_limit, upper_trend_limit]
  ;
  ;		At present, upper_trend_limit is set to "infinity" (can be adjusted if needed)
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     thresholds = ADAPT_TREND_RANGE(IMG_orig, P.XYCenter, P.d_phi, P.d_rho, k=1.5)
  ;	    			where P is the structure of processing parameters retrieved  using P=READ_PARAMETERS()
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------

   if n_elements(k) eq 0 then k=0.5

   Rect_to_Polar, IMG_orig, XYcenter, d_phi, d_rho, IMG_p

   Rho_trend = fltarr(n_elements(IMG_p[0,*]))
   for rho_ind=0, n_elements(IMG_p[0,*])-1 do Rho_trend[rho_ind] = mean(IMG_p[*,rho_ind])
   w = where(Rho_trend gt 0)

   return, [ k*min(Rho_trend[w]), 1E10]

End
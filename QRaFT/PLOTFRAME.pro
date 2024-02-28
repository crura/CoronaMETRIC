PRO PLOTFRAME, ylog=ylog

  ;+
  ; NAME:
  ;     PLOTFRAME
  ;
  ; PURPOSE:
  ;     Draws a frame around the current plotting area using the current color.
  ;     Used by IMAGE_PLOT_1.
  ;
  ; CALLING SEQUENCE:
  ;     PLOTFRAME, /ylog
  ;
  ; INPUTS:
  ;     ylog  - boolean keyword specifying whether or not the Y axis is logarithimic. 
  ;
  ; OUTPUTS:
  ;     Graphic output (plotted frame), no numerical outputs.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     PLOTFRAME, ylog=ylog
  ;     where ylog is set prior to calling PLOTFRAME.
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  ;
  
  
  xx = [ !x.crange[0], !x.crange[0], !x.crange[1], !x.crange[1], !x.crange[0] ]
  yy = [ !y.crange[0], !y.crange[1], !y.crange[1], !y.crange[0], !y.crange[0] ]
  if keyword_set(ylog) then yy = 10.0^yy
  plots, xx,yy

End


PRO IMAGE_PLOT_1, IMAGE, X, Y, CTABLE=ctable, RANGE=range, XRANGE=xrange,  XTITLE = xtitle, YTITLE = ytitle, TITLE=title, PlotTitle=PlotTitle, POSITION=position, $
                  COLORBAR_PLACE = colorbar_PLACE, CLRBAR_FORMAT = clrbar_format, YLOG=ylog, no_xlabels=no_xlabels, no_ylabels=no_ylabels

  ;+
  ; NAME:
  ;     IMAGE_PLOT_1
  ;
  ; PURPOSE:
  ;     Creates a color-coded map of a 2D data array (solar image) according
  ;     to a specified pattern of plotting parameters.
  ;
  ; CALLING SEQUENCE:
  ;     IMAGE_PLOT_1, IMAGE, X, Y, CTABLE=, RANGE=, XRANGE=,  XTITLE =, YTITLE =, TITLE=,
  ;                   PlotTitle=, POSITION=, COLORBAR_PLACE =, CLRBAR_FORMAT =,
  ;                   /YLOG, /no_xlabels, /no_ylabels
  ;
  ; INPUTS:
  ;     IMAGE     - 2D array containing plotted image. Must be defined on a regular grid.
  ;     X         - 1D array of X coordinates of IMAGE pixels Optional, contains pixel indices by default.
  ;     Y         - 1D array of Y coordinates of IMAGE pixels. Optional, contains pixel indices by default.
  ;     CTABLE    - index of the IDL color table used to plot the image (called by loadct).
  ;                 Optional. If not provided, no color table is set inside the procedure and the current external
  ;                 color table is used instead.
  ;     RANGE     - 2-element array [min, max] containing forced min and max values of the image to be plotted.
  ;                 Optional; the image range is unrestricted if RANGE is not provided.
  ;     XRANGE    - X axis dynamic range. Optional.
  ;     XTITLE    - X axis title. Optional.
  ;     YTITLE    - Y axis title. Optional.
  ;     PLOTTITLE - standard IDL plot title, centered above the plot. Optional.
  ;     TITLE     - additional plot title, left-aligned close to the upper edge of the plotting area. Optional.
  ;     POSITION  - 4-element 1D array with IDL plot position settings [left, bottom, right, top] in device unites. Optional.
  ;     COLORBAR_PLACE  - integer parameter specifying the position of the olor bar.
  ;                       COLORBAR_PLACE = 1  - vertical color bar on the right side of the image plot
  ;                       COLORBAR_PLACE = 2  - horizontal color bar on the top of the image plot
  ;                       Optional, default value 1.
  ;     CLRBAR_FORMAT - FORTRAN-style format code applied to the values on the color bar scale. Optional, default value '(F7.2)'.
  ;     YLOG      - boolean keyword controling logarithmic appearance of the Y axis. Optional, YLOG=0 be default.
  ;     no_xlabels - boolean keyword; if set, the X labels are not shown on the plot. Optional, default value 0.
  ;     no_ylabels - boolean keyword; if set, the Y labels are not shown on the plot. Optional, default value 0.
  ;
  ; OUTPUTS:
  ;     No numerical outputs.
  ;     The image plot will appear in the currently active plotting window or in a postcrip file, depending on external graphic settings.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     IMAGE_PLOT_1, IMAGE, ctable=13, xtitle='X', ytitle='Y', PlotTitle = 'Example'
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  ;
;  PLOTTING a 2-D IMAGE

  ; initializing missing auxiliary parameters:
  ;setkeyword, CTABLE, 0
;stop
  setkeyword, RANGE, [min(IMAGE), max(IMAGE)]
  setkeyword, X, indgen(n_elements(Image[*,0]))
  setkeyword, Y, indgen(n_elements(Image[0,*]))
  setkeyword, POSITION, [0.1, 0.1, 0.9, 0.9]
  setkeyword, COLORBAR_PLACE, 1
  setkeyword, CLRBAR_FORMAT, '(E11.1)';'(I4)' ;'(F7.2)' ; '(E8.1)'
  setkeyword, ylog, 0
  noerase = !P.noerase ; remember initial value set outside
  ; loading color table unless ctable=-1 (controled from outside):
  if n_elements(ctable) gt 0 then loadct, CTABLE, /silent
  ;if keyword_set(ctable) then if ctable ne -1 then loadct, CTABLE, /silent
  ; preparing graphic device  ( ??? )
  ;;device, decomposed=0
  ; correcting color indexing for a device-independent output of BLACK axes/labels on WHITE background :

  ;setbw

  ; rescaling image to make it fit the required range :
  image_rescaled = limitrange1(IMAGE, RANGE)

  ; array of shades used by shade_surf
  shades= bytscl(image_rescaled)

  ; avoiding color indexes used by BLACK and WHITE (redefined by SETBW) :
  w = where(shades eq 255) & shades[w] = 254
  w = where(shades eq 0) & shades[w] = 1

  ;stop
    ; drawing color bar :

  ;erase

  if COLORBAR_PLACE eq 1 then $ ; vertical, on the right
    color_bar, min=RANGE[0], max=RANGE[1], /vertical, /right, divisions=4, format = CLRBAR_FORMAT,$
              position = [position[2]+0.015, position[1], position[2]+0.03, position[3] ], color=!P.color   ;0 for EPS output ?
              ;print, [position[2]+0.01, position[1], position[2]+0.03, position[3] ]
  if COLORBAR_PLACE eq 2 then $ ; horizontal, on the top
    color_bar, min=RANGE[0], max=RANGE[1],  format = CLRBAR_FORMAT, $
              position = [position[0],position[3]+0.03, position[2], position[3]+0.05 ], color=!P.color   ;0 for EPS output ?

  if keyword_set(no_xlabels) then begin
    xtickname=emptylabels()
    xtitle = ''
  endif
  if keyword_set(no_ylabels) then begin
    ytickname=emptylabels()
    ytitle = ''
  endif

  !P.noerase=1

  ; plotting the image :

  PLOT, [min(X),max(X)], [min(Y),max(Y)], /xstyle, /ystyle, position=position, Title=PlotTitle, xtickname=emptylabels(), ytickname=emptylabels(), $
       /noerase, /nodata;, XTICKLEN = -0.05, YTICKLEN = -0.01  ; adding outward ticks
  ;PLOT, minmax_(X), minmax_(Y), xrange = xrange, yrange = yrange, ylog=ylog, /xstyle, /ystyle, position=position, Title=PlotTitle, /noerase, /nodata, TICKLEN = -0.015 ; adding outward ticks
  ;help, image_rescaled, X, Y, ylog
  shade_surf, image_rescaled, X, Y, shades=shades, az=0, ax=90, zstyle=5, xtitle=xtitle, ytitle=ytitle, ylog=ylog, $
              position=position, xticklen=1, xtickname=xtickname,  ytickname=ytickname, yticklen=1, xgridstyle=1, ygridstyle=1, xthick=1.0, ythick=1.0, $
              /xstyle, /ystyle
  ;PLOT, X, Y, xrange=!x.crange, yrange=!y.crange, /xstyle, /ystyle, position=position, xtickname=emptylabels(), ytickname=emptylabels(), /noerase, /nodata, TICKLEN = -0.015 ; adding outward ticks


  ; drawing frame around the plotting area (otherwise top and right axes are invisible) :

  ;setbw
  ;
  plotframe, ylog=ylog

  if n_elements(title) ne 0 then xyouts, position[0], position[3]+0.005, title, /normal ; 0.01

  !P.noerase=noerase
  ;stop
END

;=======================================================================================================================================

;=======================================================================================================================================

;=======================================================================================================================================
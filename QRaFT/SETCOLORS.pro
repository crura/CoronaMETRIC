PRO SETCOLORS

  ;+
  ; NAME:
  ;     SETCOLORS
  ;
  ; PURPOSE:
  ;     Plotting utility. Redifines values of several color table values to simplify using colors in 
  ;	  graphic procedures by assigning a short color index to the "color" keyword.
  ;
  ;     Since the color table is distorted it needs to be loaded again for image plotting.
  ;
  ; CALLING SEQUENCE:
  ;     SETCOLORS
  ;
  ; INPUTS:
  ;     none
  ;
  ; OUTPUTS:
  ;     redefined color table; no numerical outputs
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     
  ;     SETCOLORS
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------

    TVLCT, [0,255,255, 0,0,150, 255, 160], [0,255,0,255,0,150, 230, 100], [0,255,0,0,255,150, 0, 0]
    ;- for Postscript output Color=
    ; 0 - black, 1 - white, 2 - red, 3 - green, 4 - blue, 5 - grey, 6 - yellow, 7 - brown
    ;
End
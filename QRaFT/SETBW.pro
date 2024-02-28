PRO SETBW

  ;+
  ; NAME:
  ;     SETBW
  ;
  ; PURPOSE:
  ;     Plotting utility. Swaps BLACK and WHITE color indices depending on the device
  ;     Used to unsure device-independent plotting of black lines and symbols
  ;     against the white background.
  ;
  ;     Should be called after
  ;     device, decom=0
  ;
  ; CALLING SEQUENCE:
  ;     SETBW
  ;
  ; INPUTS:
  ;     none
  ;
  ; OUTPUTS:
  ;     adjusted color table; no numerical outputs.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     device, decom=0
  ;     SETBW
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------


  if !D.NAME eq 'PS' then begin
    if !P.background ne 0 then begin
       !P.color = !P.background & !P.background = 0
    endif
  endif else begin
    if !P.color ne 0 then begin
       !P.background = !P.color & !P.color = 0
    endif
  endelse


End

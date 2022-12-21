Function EMPTYLABELS, N

  ;+
  ; NAME:
  ;     EMPTYLABELS
  ;
  ; PURPOSE:
  ;     Creates a string array of empty axis labels. Used to suppress tick labels in plots.
  ;
  ; CALLING SEQUENCE:
  ;     empty_labels =  EMPTYLABELS()
  ;
  ; INPUTS:
  ;     N - number or labels (optional, N=60 by default)
  ;
  ; OUTPUTS:
  ;     1D string array of empty labels (' ')
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     plot, [1,2], [1,2], xtickname=emptylabels() ; x tick labels are suppressed
  ;  
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  
  if n_elements(N) eq 0 then N=60
  empty_labels=replicate(' ',N)
  return, empty_labels

End

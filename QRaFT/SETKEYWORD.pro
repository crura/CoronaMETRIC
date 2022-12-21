PRO SETKEYWORD, keyword, value

  ;+
  ; NAME:
  ;     SETKEYWORD
  ;
  ; PURPOSE:
  ;     Sets a keyword to a required value, usually for setting the default value.
  ;
  ; CALLING SEQUENCE:
  ;     SETKEYWORD, keyword, value
  ;
  ; INPUTS:
  ;     value   - the value that the keyword will takes. 
  ;
  ; OUTPUTS:
  ;     keyword - the keyword variable set to the value "value". 
  ;               The type of "value" defines the type of "keyword".
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     SETKEYWORD, my_keyword, {a:string, b:float}
  ;     (the keyword "my_keyword" takes the value of the specified 2-tag structure)
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------


  ;if not keyword_set(keyword) then keyword=value 
  if n_elements(keyword) eq 0 then keyword=value

End 

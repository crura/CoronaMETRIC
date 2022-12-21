
function EXTRACT_PARAMETER, fname_par, varname, vartype

  ;+
  ; NAME:
  ;     EXTRACT_PARAMETER
  ;
  ; PURPOSE:
  ;     Returns the value of a specified processing parameter
  ;     from the text file "fname_par" all parameters
  ;
  ; CALLING SEQUENCE:
  ;     parameter = EXTRACT_PARAMETER(fname_par, varname, vartype)
  ;
  ; INPUTS:
  ;     fname_par   - name of ASCII file containing values of the processing parameters
  ;                   optimized for a particular image type
  ;     varname     - string variable containing the name of the parameter top be returned
  ;                   by this function (case - insensitive)
  ;     vartype     - string variable containing the type of the parameter top be returned
  ;                   by this function (case - insensitive). Allowed values: 'float', 'int', 'string', 'char'.
  ;                   Optional, the default value is 'float'.
  ;
  ; OUTPUTS:
  ;     parameter   - the value of the returned parameter, if vartype is recognized;
  ;                   !values.F_NAN otherwise.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     print, EXTRACT_PARAMETER('MLSO_Nov_2014_parameters.txt', 'Xcenter')
  ;         513.000
  ;     (returns the X position of the Sun disk in MLSO images)
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------

  setkeyword, vartype, 'float'

  close,1 & openr, 1, fname_par

  s_arr = [''] & s='' & i=0

  while not eof(1) do begin
    readf,1,s
    ;print,s
    s_arr = [s_arr, strtrim(s,2)]
    i = i+1
  endwhile
  close,1
  s_arr = s_arr[1:*]


  w = where( (strmid(s_arr[*], 0,1) ne '#') and (strtrim(s_arr[*],2) ne '') )

  s_arr = s_arr[w]

  names =  strarr(n_elements(s_arr))
  values = strarr(n_elements(s_arr))

  for i=0, n_elements(s_arr)-1 do begin
    s = strsplit(s_arr[i],'=',/extract)

    names[i] = strupcase(strtrim(s[0],2))
    values[i] = strtrim(s[1],2)

  endfor


  ind = where(names eq strupcase(varname))

  if ind[0] ne -1 then begin

    v = strsplit(values[ind], ' ,', /extract)

    if n_elements(v) eq 1 then v = v[0]

    case strupcase(vartype) of
      'FLOAT': v = float(v)
      'INT': v = fix(v)
      'STRING': v = v
      'CHAR': v = v
    endcase

    endif else v = !values.F_NAN

  return, v

End
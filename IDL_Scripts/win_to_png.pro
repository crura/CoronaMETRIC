pro win_to_png, win_id, fname
 
  ; Saves the contents of a selected graphic window to a PNG file
  ;
  ; examples
  ;
  ; saving window #0 to file "test.png"
  ; win_to_png, 0, "test.png"
  ;  
  ; saving window #0 to an interactively selected file
  ; win_to_png, 0
  ;
 
  if n_elements(fname) eq 0 then fname=dialog_pickfile(/write)
 
  wset, win_id
  img = tvrd(/true)
  write_png, fname, img

End
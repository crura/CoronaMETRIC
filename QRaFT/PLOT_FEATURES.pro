 PRO PLOT_FEATURES, IMG, FEATURES, P, range=range, old_win=old_win, title=title, position=position, polar = polar

   ;+
   ; NAME:
   ;     PLOT_FEATURES
   ;
   ; PURPOSE:
   ;     Plots the set of the detected solar features (FEATURES)
   ;     overplotted with the coronal image (IMG) used to detect the features.
   ;
   ; CALLING SEQUENCE:
   ;     PLOT_FEATURES, IMG, FEATURES, P, range=, /old, title=
   ;
   ; INPUTS:
   ;     IMG      - The image used to detect the features, to be plotted on the background.
   ;                The image pixels must be in rectangular (Cartesian) coordinates.
   ;                To ensure proper alignment of the detected features with actual coronal structures,
   ;                used the enhanced coronal image for IMG, i.e. IMG=enhance_image_angular().
   ;                (see PROCESS_FILE for more details).
   ;	   FEATURES - array of structures containing detected features
   ;     P        - structure of processing parameters, retrieved from file using P=READ_PARAMETERS()
   ;     range    - dynamic range of the color scale of the IMG plot.
   ;                Optional; the deault value is the original dynamic range of IMG.
   ;     old_win  - boolean keyword. Set this keyword to use the current preexisting plotting window
   ;                when plotting in the DEVICE context.
   ;                Optional; the default value is 0 (plotting in a new window).
   ;     title    - text variable containing the plot title. Optional; the default vlaue is ''.
   ;
   ; OUTPUTS:
   ;     Plotted coronal features; no numercal outputs.
   ;
   ; RESTRICTIONS:
   ;    none
   ;
   ; EXAMPLE:
   ;
   ;    IMG_enh = enhance_image_angulart(IMG_orig, P.XYCenter, P.Rs)
   ;
   ;    PLOT_FEATURES, IMG_enh, FEATURES, P, range=[-0.025, 0.025], title='Detected coronal features'
   ;
   ;    Here, IMG_orig is the raw image opened from a file,
   ;          FEATURES is an array of structures represented individual detected features,
   ;          P is the structure of optimized processing parameters obtained using P=READ_PARAMETERS().
   ;
   ; MODIFICATION HISTORY:
   ;       V. Uritsky, 2019-2022
   ;-
   ;----------------------------------------------------------------
   ;

   ;if not keyword_set(old_win) then begin
   ;  window, xsize=700, ysize=700, title=title, /free
   ;  erase
   ;endif
  setkeyword, position, [0.13, 0.1, 0.86, 0.95  ]
  loadct,0, /silent & device, dec=0 & setbw

   if not keyword_set(polar) then begin

     image_plot_1, IMG, range=range, xtitle='X, pixels', ytitle='Y, pixels', plottitle=title, ctable=0, position=position
	 ;setcolors
	 loadct,13, /silent
     for i=0, n_elements(Features)-1 do begin
       plots, Features[i].xx_r[0:Features[i].n_nodes-1], Features[i].yy_r[0:Features[i].n_nodes-1], color=i*255.0/(1.0*n_elements(Features)), thick=3
       ;plots, Features[i].xx_r, Features[i].yy_r, color=3, psym=2
     endfor
     plots, [P.XYCenter[0], P.XYCenter[0]], [P.XYCenter[1]-30, P.XYCenter[1]+30]
     plots, [P.XYCenter[0]-30, P.XYCenter[0]+30], [P.XYCenter[1], P.XYCenter[1]]
	 loadct, 0, /silent
   endif else begin

     X = findgen(n_elements(img[*,0]))*P.d_phi
     Y = findgen(n_elements(img[0,*]))*P.d_rho/P.Rs

     image_plot_1, IMG, X,Y, range=range, xtitle='Position angle, radians', ytitle='Radial distance, Rs', plottitle=title, ctable=0, position=position
     setcolors
     loadct,13, /silent
     for i=0, n_elements(Features)-1 do begin
       plots, Features[i].xx_p[0:Features[i].n_nodes-1], Features[i].yy_p[0:Features[i].n_nodes-1]/P.Rs, color=i*255.0/n_elements(Features), thick=2
     endfor
     loadct, 0, /silent
   endelse

 END

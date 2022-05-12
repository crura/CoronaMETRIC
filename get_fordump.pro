
;restore, 'model_parameters.sav'
function get_2D_coord,R_occult,range ;, Nxy, dx, dy, R_occult
;
; V. Uritsky 2021
;
  ;R_occult = OCCLT ; this is what OCCULT is set to in generate_forward_model.pro
  im_range = range * 2
  dx = RANGE/256.0 & dy = dx ; 12.16 is rsun range which is abs(rsun xmax) + abs(rsun xmin) of forward model
  Nxy = 256

  X0 = 0  & Y0 = 0; Sun's disk center

  X = findgen(Nxy)*((im_range)/Nxy) - RANGE + dx/2  ; grid node positions, centered, 6.08 is rsun of forward model
  Y = X

  i_arr = [0.0] & j_arr = i_arr
  N = 0LL

  for i =0, Nxy-1 do for j=0, Nxy-1 do $
    if ((X[i] - X0)^2 + (Y[j] - Y0)^2) gt R_occult^2 then begin
      N = N+1
      i_arr = [i_arr,i]
      j_arr = [j_arr,j]
    endif
  print,N

  return, {i:i_arr[1:*], j:j_arr[1:*], Nxy:Nxy, N:N}

End

;------------------------------------------------------------

function convert_psi_array, arr_1d, OCCLT, RANGE
;
; V. Uritsky 2021
;


 map = get_2D_coord(OCCLT,RANGE)

 arr_2d = dblarr(map.Nxy, map.Nxy)

 for k = 0LL, map.N-1 do arr_2d[map.i[k], map.j[k]] = arr_1d[k]

 return, transpose(arr_2d)

End

;------------------------------------------------------------


;-------------------------------------------------------------------------------
; Quick pro to take R, Theta, Phi and give xyz coords
;-------------------------------------------------------------------------------

function rtp2xyz, Rtp_D

r_=0
t_=1
p_=2

x_=0
y_=1
z_=2

Xyz_D = dblarr(3)

Xyz_D[x_] = Rtp_D[r_]*cos(Rtp_D[p_])*sin(Rtp_D[t_])
Xyz_D[y_] = Rtp_D[r_]*sin(Rtp_D[p_])*sin(Rtp_D[t_])
Xyz_D[z_] = Rtp_D[r_]*cos(Rtp_D[t_])

return, xyz_D

end



function get_fordump


  ;spawn, 'cp /Users/crura/SSW/packages/forward/datadump /Users/crura/Desktop/Research/github/Image-Coalignment/Data'
  restore, '/Users/crura/SSW/packages/forward/datadump',/v
  restore, 'model_parameters.sav'
  BX = BROBS*sin(THETA3DUSE)*cos(PHI3DUSE) + BTHOBS*cos(THETA3DUSE)*cos(PHI3DUSE) - BPHOBS*sin(PHI3DUSE)
  BY = BROBS*sin(THETA3DUSE)*sin(PHI3DUSE) + BTHOBS*cos(THETA3DUSE)*sin(PHI3DUSE) + BPHOBS*cos(PHI3DUSE)
  BZ = BROBS*cos(THETA3DUSE) - BTHOBS*sin(THETA3DUSE)

  X =R3DUSE*sin(THETA3DUSE)*cos(PHI3DUSE) + THETA3DUSE*cos(THETA3DUSE)*cos(PHI3DUSE) - PHI3DUSE*sin(PHI3DUSE)
  Y =R3DUSE*sin(THETA3DUSE)*sin(PHI3DUSE) + THETA3DUSE*cos(THETA3DUSE)*sin(PHI3DUSE) + PHI3DUSE*cos(PHI3DUSE)
  Z =R3DUSE*cos(THETA3DUSE) - THETA3DUSE*sin(THETA3DUSE)

  help,BX[78,*]

  BX_2d =  convert_psi_array(BX[39,*],OCCLT, RANGE)
  BY_2d = convert_psi_array(BY[39,*],OCCLT, RANGE)
  BZ_2d = convert_psi_array(BZ[39,*],OCCLT, RANGE)

  X_2d =  convert_psi_array(X[39,*],OCCLT, RANGE)
  Y_2d = convert_psi_array(Y[39,*],OCCLT, RANGE)
  Z_2d = convert_psi_array(Z[39,*],OCCLT, RANGE)

  Dens_2d_center = convert_psi_array(DENSOBS[39,*],OCCLT, RANGE)

  save,Dens_2d_center,filename='/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Electron_Density_Center.sav'

  save,BX_2d,filename='/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Bx_2d_Center.sav'
  save,BY_2d,filename='/Users/crura/Desktop/Research/github/Image-Coalignment/Data/By_2d_Center.sav'
  save,BZ_2d,filename='/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Bz_2d_Center.sav'

  write_csv,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Central_Parameters/rotated_Bx_2d.csv',BX_2d
  write_csv,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Central_Parameters/rotated_By_2d.csv',BY_2d
  write_csv,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Central_Parameters/rotated_Bz_2d.csv',BZ_2d
  write_csv,'/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Central_Parameters/rotated_Dens_2d.csv',Dens_2d_center

  ;write_csv,'rotated_x_2dtest.csv',X_2d
  ;write_csv,'rotated_y_2dtest.csv',Y_2d
  ;write_csv,'rotated_z_2dtest.csv',Z_2d



  ; STUFF THAT WORKS IS ABOVE

  Nr = n_elements(R3DUSE[0,*])
  Nt = n_elements(THETA3DUSE[0,*])
  Np = n_elements(PHI3DUSE[0,*])
  rho_xyz = fltarr(256,256)

  ;for i=0, Nr-1 do begin

  ;if i mod 10 eq 0 then begin
  ; print, (100*i)/Nr,'%'
  ;wait,0.1
  ;endif


  rho_new = convert_psi_array(DENSOBS,OCCLT, RANGE)
  thetanew = convert_psi_array(THETA3DUSE,OCCLT, RANGE)
  phinew = convert_psi_array(PHI3DUSE,OCCLT, RANGE)
  BXnew =  convert_psi_array(BX,OCCLT, RANGE)
  BYnew = convert_psi_array(BY,OCCLT, RANGE)
  BZnew = convert_psi_array(BZ,OCCLT, RANGE)


  for j=0, 255 do $
    for k = 0, 255 do begin
    ;j = i
    ;k = i
      X = sin(thetanew[j])*cos(phinew[k]) + cos(thetanew[j])*cos(phinew[k]) - sin(phinew[k])
      Y = sin(thetanew[j])*sin(phinew[k]) + cos(thetanew[j])*sin(phinew[k]) + cos(phinew[k])
      Z = cos(thetanew[j]) - sin(thetanew[k])

      rho_xyz[x,y] = rho_xyz[x,y] + rho_new[j,k]




   endfor

   densarr = fltarr(256,256)

   for i =0,78 do begin
    jstring = STRTRIM(i,2)
    rho_xyzproj = convert_psi_array(DENSOBS[i,*],OCCLT, RANGE)
    BX_2d =  convert_psi_array(BX[i,*],OCCLT, RANGE)
    BY_2d = convert_psi_array(BY[i,*],OCCLT, RANGE)
    BZ_2d = convert_psi_array(BZ[i,*],OCCLT, RANGE)
    spath = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Rotated_Density_LOS/Frame_' + jstring + '.csv'
    spath1 = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Bx_Rotated/Frame_' + jstring + '.csv'
    spath2 = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/By_Rotated/Frame_' + jstring + '.csv'
    spath3 = '/Users/crura/Desktop/Research/github/Image-Coalignment/Data/Bz_Rotated/Frame_' + jstring + '.csv'
    write_csv,spath,rho_xyzproj
    write_csv,spath1,BX_2d
    write_csv,spath2,BY_2d
    write_csv,spath3,BZ_2d

   endfor

 ;write_csv,'DENSROTATED.csv',rho_xyzproj
 return,X_2d

End

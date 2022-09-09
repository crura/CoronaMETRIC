
function get_2D_coord ;, Nxy, dx, dy, R_occult
;
; V. Uritsky 2021
;
  spawn, 'git rev-parse --show-toplevel', git_repo
  restore, git_repo + '/Data/model_parameters.sav'
  R_occult = occlt ; this is what OCCULT is set to in generate_forward_model.pro
  rsun_abs = range + range
  dx = rsun_abs/256.0 & dy = dx ; 12.16 is rsun range which is abs(rsun xmax) + abs(rsun xmin) of forward model
  Nxy = 256

  X0 = 0  & Y0 = 0; Sun's disk center

  X = findgen(Nxy)*(rsun_abs/Nxy) - range + dx/2  ; grid node positions, centered, 6.08 is rsun of forward model
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

function convert_psi_array, arr_1d
;
; V. Uritsky 2021
;


 map = get_2D_coord()

 arr_2d = dblarr(map.Nxy, map.Nxy)

 for k = 0LL, map.N-1 do arr_2d[map.i[k], map.j[k]] = arr_1d[k]

 return, transpose(arr_2d)

End

;------------------------------------------------------------


function get_fordump


  ;spawn, 'cp /Users/crura/SSW/packages/forward/datadump /Users/crura/Desktop/Research/github/Image-Coalignment/Data'
  restore, '/Users/crura/SSW/packages/forward/datadump',/v
  spawn, 'git rev-parse --show-toplevel', git_repo
  print,git_repo
  BX = BROBS*sin(THETA3DUSE)*cos(PHI3DUSE) + BTHOBS*cos(THETA3DUSE)*cos(PHI3DUSE) - BPHOBS*sin(PHI3DUSE)
  BY = BROBS*sin(THETA3DUSE)*sin(PHI3DUSE) + BTHOBS*cos(THETA3DUSE)*sin(PHI3DUSE) + BPHOBS*cos(PHI3DUSE)
  BZ = BROBS*cos(THETA3DUSE) - BTHOBS*sin(THETA3DUSE)

  X =R3DUSE*sin(THETA3DUSE)*cos(PHI3DUSE) + THETA3DUSE*cos(THETA3DUSE)*cos(PHI3DUSE) - PHI3DUSE*sin(PHI3DUSE)
  Y =R3DUSE*sin(THETA3DUSE)*sin(PHI3DUSE) + THETA3DUSE*cos(THETA3DUSE)*sin(PHI3DUSE) + PHI3DUSE*cos(PHI3DUSE)
  Z =R3DUSE*cos(THETA3DUSE) - THETA3DUSE*sin(THETA3DUSE)

  ;help,BX[78,*]

  BX_2d =  convert_psi_array(BX[39,*])
  BY_2d = convert_psi_array(BY[39,*])
  BZ_2d = convert_psi_array(BZ[39,*])

  X_2d =  convert_psi_array(X[39,*])
  Y_2d = convert_psi_array(Y[39,*])
  Z_2d = convert_psi_array(Z[39,*])

  Dens_2d_center = convert_psi_array(DENSOBS[39,*])

  save,Dens_2d_center,filename= git_repo + '/Data/Electron_Density_Center.sav'

  save,BX_2d,filename= git_repo + '/Data/Bx_2d_Center.sav'
  save,BY_2d,filename= git_repo + '/Data/By_2d_Center.sav'
  save,BZ_2d,filename= git_repo + '/Data/Bz_2d_Center.sav'

  write_csv, git_repo + '/Data/Central_Parameters/rotated_Bx_2d.csv',BX_2d
  write_csv, git_repo + '/Data/Central_Parameters/rotated_By_2d.csv',BY_2d
  write_csv, git_repo + '/Data/Central_Parameters/rotated_Bz_2d.csv',BZ_2d
  write_csv, git_repo + '/Data/Central_Parameters/rotated_Dens_2d.csv',Dens_2d_center

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


  rho_new = convert_psi_array(DENSOBS)
  thetanew = convert_psi_array(THETA3DUSE)
  phinew = convert_psi_array(PHI3DUSE)
  BXnew =  convert_psi_array(BX)
  BYnew = convert_psi_array(BY)
  BZnew = convert_psi_array(BZ)


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

   spawn, 'rm -r ' + git_repo + '/Data/Rotated_Density_LOS; mkdir ' + git_repo + '/Data/Rotated_Density_LOS'
   spawn, 'rm -r ' + git_repo + '/Data/Bx_Rotated; mkdir ' + git_repo + '/Data/Bx_Rotated'
   spawn, 'rm -r ' + git_repo + '/Data/By_Rotated; mkdir ' + git_repo + '/Data/By_Rotated'
   spawn, 'rm -r ' + git_repo + '/Data/Bz_Rotated; mkdir ' + git_repo + '/Data/Bz_Rotated'
   for i =0,78 do begin
    jstring = STRTRIM(i,2)
    rho_xyzproj = convert_psi_array(DENSOBS[i,*])
    BX_2d =  convert_psi_array(BX[i,*])
    BY_2d = convert_psi_array(BY[i,*])
    BZ_2d = convert_psi_array(BZ[i,*])
    spath =  git_repo + '/Data/Rotated_Density_LOS/Frame_' + jstring + '.csv'
    spath1 =  git_repo + '/Data/Bx_Rotated/Frame_' + jstring + '.csv'
    spath2 =  git_repo + '/Data/By_Rotated/Frame_' + jstring + '.csv'
    spath3 =  git_repo + '/Data/Bz_Rotated/Frame_' + jstring + '.csv'
    write_csv,spath,rho_xyzproj
    write_csv,spath1,BX_2d
    write_csv,spath2,BY_2d
    write_csv,spath3,BZ_2d

   endfor

 ;write_csv,'DENSROTATED.csv',rho_xyzproj
 return,0

End

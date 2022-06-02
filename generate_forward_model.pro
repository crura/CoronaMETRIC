function generate_forward_model, directory

; set path variable for rest of idl code defined by git
spawn, 'git rev-parse --show-toplevel', git_repo

; for_drive parameters are set here

crln_obs = 303.44300
crlt_obs = 7.1530000
occlt = 1.0600000
range = 6.0799999

crlt_obs_print = strcompress(string(CRLT_OBS),/remove_all)
crln_obs_print = strcompress(string(CRLN_OBS),/remove_all)
save,crln_obs,crlt_obs,occlt,range,crlt_obs_print,crln_obs_print,filename=git_repo + '/model_parameters.sav'
; read, crlt_obs, 'enter carrington latitude (B angle): '
; read, crln_obs, 'enter carrington longitude (CMER) : '
; read, occlt, 'enter occulting disk radius (R_Sun): '
; read, range, 'enter model axes range (R_Sun): '


;for_drive,'PSIMAS',instrument='WL',line='PB',gridtype='PLANEOFSKY',pos=0.0000000,CMER=183.44300,BANG=7.1530000,OCCULT=1.0600000,XXMIN=-6.0799999,XXMAX=6.0799999,YYMIN=-6.0799999,YYMAX=6.0799999,CUBENAME='/Users/crura/Desktop/Research/Magnetic_Field/eclipse2017_mhd_final_copy/2194_WTD_local_cube_MAS.dat',DATE='2017-08-30T02:23:16.968',/verbose
for_drive,'PSIMAS',instrument='WL',line='PB',gridtype='PLANEOFSKY',pos=0.0000000,CMER=crln_obs,BANG=crlt_obs,OCCULT=occlt,XXMIN=-range,XXMAX=range,YYMIN=-range,YYMAX=range,CUBENAME='/Users/crura/Desktop/Research/Magnetic_Field/eclipse2017_mhd_final_copy/2194_WTD_local_cube_MAS.dat',DATE='2017-08-30T02:23:16.968',SAVEPRAMS='output',SAVEMAP=1,MAPNAME='output',/verbose
restore,'/Users/crura/SSW/packages/forward/output.sav',/v
write_csv, git_repo + '/Data/Forward_PB_data.csv',quantmap.data
return, 0
END

function generate_forward_model, directory

; for_drive parameters are set here

crln_obs = 183.44300
crlt_obs = 7.1530000
occlt = 1.0600000
range = 6.0799999

save,crln_obs,crlt_obs,occlt,range,filename='model_parameters.sav'
; read, crlt_obs, 'enter carrington latitude (B angle): '
; read, crln_obs, 'enter carrington longitude (CMER) : '
; read, occlt, 'enter occulting disk radius (R_Sun): '
; read, range, 'enter model axes range (R_Sun): '


;for_drive,'PSIMAS',instrument='WL',line='PB',gridtype='PLANEOFSKY',pos=0.0000000,CMER=183.44300,BANG=7.1530000,OCCULT=1.0600000,XXMIN=-6.0799999,XXMAX=6.0799999,YYMIN=-6.0799999,YYMAX=6.0799999,CUBENAME='/Users/crura/Desktop/Research/Magnetic_Field/eclipse2017_mhd_final_copy/2194_WTD_local_cube_MAS.dat',DATE='2017-08-30T02:23:16.968',/verbose
for_drive,'PSIMAS',instrument='WL',line='PB',gridtype='PLANEOFSKY',pos=0.0000000,CMER=crln_obs,BANG=crlt_obs,OCCULT=occlt,XXMIN=-range,XXMAX=range,YYMIN=-range,YYMAX=range,CUBENAME='/Users/crura/Desktop/Research/Magnetic_Field/eclipse2017_mhd_final_copy/2194_WTD_local_cube_MAS.dat',DATE='2017-08-30T02:23:16.968',/verbose

return, 0
END

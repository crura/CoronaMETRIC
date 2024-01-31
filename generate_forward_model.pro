function generate_forward_model, directory

; set path variable for rest of idl code defined by git
spawn, 'git rev-parse --show-toplevel', git_repo

; for_drive parameters are set here

fits_directory = git_repo + '/' + directory; '/Data/MLSO/20170911_202927_kcor_l2_avg.fts'
head = headfits(fits_directory)
head_struct = fitshead2struct(head)

telescope = head_struct.telescop
if (telescope eq 'COSMO K-Coronagraph') then begin
  SXADDPAR, head, 'detector', 'KCor'
  head_struct = fitshead2struct(head)
  telescope = head_struct.telescop
endif


wcs = fitshead2wcs(head)
position = wcs.position
rsun = head_struct.rsun / wcs.cdelt[0] ;number of pixels in radius of sun
shape = fix(wcs.NAXIS[0])
crln_obs = position.crln_obs
crlt_obs = position.crlt_obs
detector = head_struct.detector

if (detector eq 'COR1') then begin
  ; rad_occlt_pix = sxpar(head,'RCAM_DCR')
  print,'COR1'
  occlt = 1.45; citing https://cor1.gsfc.nasa.gov/docs/spie_paper.pdf page 3
endif else begin
  rad_occlt_pix = sxpar(head,'RCAM_DCR')
  occlt = rad_occlt_pix/rsun;1.0600000
endelse

range = shape/rsun; 6.0799999
time = wcs.time
date_obs = time.observ_date
date_split = strsplit(date_obs,'T',/extract)
date_print = repstr(date_split[0],'-','_')


crlt_obs_print = strcompress(string(CRLT_OBS),/remove_all)
crln_obs_print = strcompress(string(CRLN_OBS),/remove_all)
; read, crlt_obs, 'enter carrington latitude (B angle): '
; read, crln_obs, 'enter carrington longitude (CMER) : '
; read, occlt, 'enter occulting disk radius (R_Sun): '
; read, range, 'enter model axes range (R_Sun): '


;for_drive,'PSIMAS',instrument='WL',line='PB',gridtype='PLANEOFSKY',pos=0.0000000,CMER=183.44300,BANG=7.1530000,OCCULT=1.0600000,XXMIN=-6.0799999,XXMAX=6.0799999,YYMIN=-6.0799999,YYMAX=6.0799999,CUBENAME='/Users/crura/Desktop/Research/Magnetic_Field/eclipse2017_mhd_final_copy/2194_WTD_local_cube_MAS.dat',DATE='2017-08-30T02:23:16.968',/verbose
for_drive,'PSIMAS',instrument='WL',line='PB',gridtype='PLANEOFSKY',pos=0.0000000,CMER=crln_obs,BANG=crlt_obs,OCCULT=occlt,XXMIN=-range,XXMAX=range,YYMIN=-range,YYMAX=range,CUBENAME='/Users/crura/Desktop/Research/Magnetic_Field/eclipse2017_mhd_final_copy/2194_WTD_local_cube_MAS.dat',DATE=date_obs,SAVEPRAMS='output',SAVEMAP=1,MAPNAME='output',/verbose
restore,'/Users/crura/SSW/packages/forward/output.sav',/v
write_csv, git_repo + '/Data/Forward_PB_data.csv',quantmap.data
forward_pb_image = quantmap.data
save,crln_obs,crlt_obs,occlt,range,crlt_obs_print,crln_obs_print,forward_pb_image,date_obs,fits_directory, shape, rsun, date_print, detector, filename=git_repo + '/Data/model_parameters.sav'
restore, '/Users/crura/SSW/packages/forward/datadump',/v
spawn, 'cp /Users/crura/SSW/packages/forward/datadump /Users/crura/Desktop/Research/Data/datadump_' + crlt_obs_print + '_' + crln_obs_print
spawn, 'python Python_Scripts/forward_model_db_save.py'
hi = get_fordump()
spawn, 'python Python_Scripts/integrate.py'
hi2 = image_coalignment()
hi3 = save_parameters()
spawn, 'python Python_Scripts/plot.py'
spawn, 'python -m unittest discover Python_Scripts/'
return, 0
END

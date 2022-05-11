;----------------------------------------------------------------------
; write_psi_image_as_fits
;----------------------------------------------------------------------
;
; Write a PSI image made with getpb or geteit as a fits file.
;
; - The FITS metadata that is written is designed to be compatible with
;   sunpy.map for inferring image and observer coordinate information.
;
; - The standard assumption is that you computed the image with the
;   plane-parallel assumption --> [x,y] are in solar radii.
;
; - if ObsDistanceAU is set (like -r in getpb) then it assumes that
;   [x,y] are in degrees.
;
; Inputs:
;
;   OutFile: Path of the output file name.
;
;   Im: A 2D array containing the image values.
;
;    x: 1D array of pixel coordinates in x [Rs] or [Degrees].
;
;    y: 1D array of pixel coordinates in y [Rs] or [Degrees].
;
;    ObsDate: UT time string in the this format: '2014-04-13T18:00:00'
;
;    Lon: Carrington longitude of the viewpoint [0-360]
;
;    B0: Carrington latitude of the viewpoint [-90,90].
;
; Keywords: (should be self explanatory).
;
; Compatibility Note:
;   - sunpy.map parses the Wavelength and WaveUnit fields in a specific way.
;     - Wavelength must be a number.
;     - WaveUnit must be a as a valid AstroPy unit ('' defaults to "one").
;       sunpy.map will crash if you put something weird in those fields.
;
; HISTORY:
;  v1.0, 2020ish, Cooper Downs, Predictive Science Inc. (cdowns@predsci.com)
;     - First version I've shared externally.
;     - This is based on several previous iterations since 2017 of CD's
;       SSW/IDL stuff for converting with PSI/MAS forward modeled data
;       products from HDF to FITS.
;
;----------------------------------------------------------------------

pro write_psi_image_as_fits, OutFile, Im, x, y, ObsDate, Lon, B0, $
   pAngle=pAngle, $
   ObsDistanceAU=ObsDistanceAU, $
   Observatory=Observatory, $
   Telescope=Telescope, $
   Instrument=Instrument, $
   Detector=Detector, $
   Wavelength=Wavelength, $
   WaveUnit=WaveUnit, $
   bUnit=BUnit, $
   Model=Model, $
   SimName=SimName, $
   SimTime=SimTime, $
   Sequence=Sequence, $
   ViewName=ViewName, $
   Author=Author, $
   Comment=Comment, $
   HdrOut=HdrOut, $
   GetCoords=GetCoords, $
   ForceXYRs=ForceXYRs

   if not keyword_set(Observatory) then Observatory = 'PSI-MAS Forward Model'
   if not keyword_set(Instrument) then Instrument = 'not_specified'
   if not keyword_set(Telescope) then Telescope = 'not_specified'
   if not keyword_set(Detector) then Detector = 'not_specified'
   if n_elements(Wavelength) eq 0 then Wavelength = -1
   if n_elements(WaveUnit) eq 0 then WaveUnit = ''
   if not keyword_set(Model) then Model = 'PSI-MAS'
   if not keyword_set(SimName) then SimName = 'not_specified'
   if not keyword_set(SimTime) then SimTime = -777d0
   if not keyword_set(Sequence) then Sequence = -777
   if not keyword_set(ViewName) then ViewName = 'not_specified'
   if not keyword_set(bUnit) then bUnit = 'not_specified'
   if not keyword_set(Author) then Author = 'cdowns'
   if n_elements(Comment) eq 0 then Comment = strarr(15)
   if n_elements(pAngle) eq 0 then pAngle = -0.0d0

   nPixX = n_elements(x)
   nPixY = n_elements(y)

   ; these are the standard lengths in meters (AIA convention)
   DSUN_REF = 1.49597870691d11
   RSUN_REF = 6.96d8

   ; if Observer Distance is set, assume your x, y are in degrees.
   if n_elements(ObsDistanceAU) ne 0 then begin
     DSUN_OBS = DSUN_REF*ObsDistanceAU
     ArcSecX = x*3600d0
     ArcSecY = y*3600d0

     ; override if you computed plane parallel but observer isn't at 1AU
     if keyword_set(ForceXyRs) then begin
       ArcSecX = atan(x*RSUN_REF,DSUN_OBS)/!dtor*3600d0
       ArcSecY = atan(y*RSUN_REF,DSUN_OBS)/!dtor*3600d0
     endif

   ; if plane-parallel then convert x,y in Rs to arcsec (good approximation in low corona)
   endif else begin
     DSUN_OBS = DSUN_REF
     ArcSecX = atan(x*RSUN_REF,DSUN_REF)/!dtor*3600d0
     ArcSecY = atan(y*RSUN_REF,DSUN_REF)/!dtor*3600d0
   endelse

   ; determine the plate scale / centering WCS/fits Keywords
   CDELT1 = mean((double(shift(ArcSecX, -1)) - double(ArcSecX))[0:nPixX-2])
   CDELT2 = mean((double(shift(ArcSecY, -1)) - double(ArcSecY))[0:nPixY-2])
   CRPIX1 = interpol(dindgen(nPixX)+1, double(ArcSecX), 0.0d0)
   CRPIX2 = interpol(dindgen(nPixY)+1, double(ArcSecY), 0.0d0)
   Rs_arcsec = atan(RSUN_REF,DSUN_OBS)/!dtor*3600d0
   print,CDELT1,CDELT2,CRPIX1,CRPIX2

   ; this R_SUN is in pixels (consistent with AIA documentation)
   R_SUN = Rs_arcsec/CDELT1


   ; Compute HCI and HAE corrdinates for reference
   ; i'm pretty sure I don't need to supply KM unless transformations are non-linear
   if keyword_set(GetCoords) then begin
     r = DSUN_OBS
     t = (90-B0)*!dtor
     p = Lon*!dtor
     Rtp_D = [r,t,p]
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

     Carr_Vec = xyz_D

     Coord = Carr_Vec
     convert_sunspice_coord, ObsDate, Coord, 'Carrington', 'HCI'
     HCIX_Obs = Coord[0]
     HCIY_Obs = Coord[1]
     HCIZ_Obs = Coord[2]

     Coord = Carr_Vec
     convert_sunspice_coord, ObsDate, Coord, 'Carrington', 'HAE'
     HAEX_Obs = Coord[0]
     HAEY_Obs = Coord[1]
     HAEZ_Obs = Coord[2]
   endif

   ;-- create the header
   HdrOut = create_struct($
      'SIMPLE', 1, $
      'NAXIS', 2L, $
      'NAXIS1', long(nPixX), $
      'NAXIS2', long(nPixY), $
      'EXTEND', 1, $
      'DATE_OBS', ObsDate, $
      'T_REC', ObsDate, $
      'T_OBS', ObsDate, $
      'OBSRVTRY', Observatory, $
      'INSTRUME', Instrument, $
      'TELESCOP', Telescope, $
      'DETECTOR', Detector, $
      'IMG_TYPE', 'model', $
      'EXPTIME', 1d0, $
      'WAVELNTH', double(Wavelength), $
      'WAVEUNIT', string(WaveUnit), $
      'BUNIT', string(Bunit), $
      'CTYPE1', 'HPLN-TAN', $
      'CUNIT1', 'arcsec', $
      'CRVAL1', 0d0, $
      'CDELT1', CDELT1, $
      'CRPIX1', 128, $;128
      'CTYPE2', 'HPLT-TAN', $
      'CUNIT2', 'arcsec', $
      'CRVAL2', 0d0, $
      'CDELT2', CDELT2, $
      'CRPIX2', 128, $;128
      'CROTA2', double(-pAngle), $
      'R_SUN', R_SUN, $
      'DSUN_REF', DSUN_REF, $
      'DSUN_OBS', DSUN_OBS, $
      'RSUN_REF', RSUN_REF, $
      'RSUN_OBS', Rs_arcsec, $
      'CRLN_OBS', Lon, $
      'CRLT_OBS', B0, $
      'L0', Lon, $
      'B0', B0, $
      'RSUN', Rs_arcsec, $
      'HCIX_OBS', HCIX_Obs, $
      'HCIY_OBS', HCIY_Obs, $
      'HCIZ_OBS', HCIZ_Obs, $
      'HAEX_OBS', HAEX_Obs, $
      'HAEY_OBS', HAEY_Obs, $
      'HAEZ_OBS', HAEZ_Obs, $
      'SIM_MODL', string(Model), $
      'SIM_NAME', string(SimName), $
      'SIM_DUMP', long(fix(Sequence)), $
      'SIM_TIME', SimTime, $
      'VIEWNAME', string(ViewName), $
      'AUTHOR', 'cdowns', $
      'COMMENT', Comment)

   mwritefits, HdrOut, Im, OutFile=OutFile

end

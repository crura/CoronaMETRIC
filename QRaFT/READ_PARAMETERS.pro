
function READ_PARAMETERS, fname_par

  ;+
  ; NAME:
  ;     READ_PARAMETERS
  ;
  ; PURPOSE:
  ;     Reads a text file containing optimized vlaues of processing
  ;     parameters and returns these values as an IDL structure
  ;     for subsequent calls by image-processing subroutines.
  ;
  ; CALLING SEQUENCE:
  ;     P = READ_PARAMETERS(fname_par)
  ;
  ; INPUTS:
  ;     fname_par - name of ASCII file containing values of the processing parameters
  ;                   optimized for a particular image type
  ;
  ; OUTPUTS:
  ;     P         - IDL structure containing the all returned parameter stored in file fname_par.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     help, READ_PARAMETERS('MLSO_Nov_2014_parameters.txt')
  ;
  ;   DATA_SOURCE     STRING    ''MLSO2014''  ; type
  ;   XYCENTER        FLOAT     Array[2]
  ;   ROT_ANGLE       FLOAT          0.000000
  ;   RS              FLOAT           195.000
  ;   D_ROI           FLOAT          0.000000
  ;   D_PHI           FLOAT        0.00872665
  ;   D_RHO           FLOAT           1.00000
  ;   PHI_SHIFT       FLOAT           2.00000
  ;   PNG_CHANNEL     INT              0
  ;   BLOB_FILT_THRESH
  ;   FLOAT          0.500000
  ;   BLOB_FILT_MODE  INT              1
  ;   BLOB_FILT_MIN   INT             20
  ;   BLOB_DETECT_MIN INT             20
  ;   BLOB_DETECT_N_SEGMENTS
  ;   INT              5
  ;   BLOB_DETECT_THRESHOLDS
  ;   FLOAT     Array[2, 2]
  ;   L_RANGE         FLOAT     Array[2]
  ;   PHI_RANGE       FLOAT     Array[2]
  ;   ALPHA_RANGE     FLOAT     Array[2]
  ;   FLUX_RHO_TREND_RANGE
  ;   FLOAT     Array[2]
  ;
  ; LEGEND:
  ;
  ; IMAGE ENHANCEMENT PARAMETERS
  ; data_source: type of image to be processed (MLSO14, MLSO2016, ECL1)
  ; Xcenter, Ycenter: position of the Sun disk center, pixels
  ; Rot_angle: initial rotation angle (used for inital image plane derotation)
  ; Rs: radius of the Sun disk to be excluded fom processing, pixels
  ; d_roi: distance (in pixels) in x and y directions from the disk center defining outer edges of the ROI
  ; png_channel: PNG channel to be used if image data is in PNG format
  ;
  ; BLOB FILTERING PARAMETERS:
  ; blob_filt_min, blob_detect_min: minimum size of image blobs to be included during noise filtering and detection
  ; blob_detect_thresh_pairs: number of pairs of min amd max pixel values (in the processed image) used to detect features
  ; blob_detect_thresholds: detection thresholds, sorted by [min, max] pairs. Total number of thresholds must be 2 x  blob_detect_thresh_pairs.
  ;
  ; FEATURE FILTERING parameters:
  ; L_min, L_max: allowed range of feature length (in discrete rectangular coordinates)
  ; Phi_min, Phi_max: excluded range of polar angles (in radians), used to remove persistent MLSO artifacts
  ; Alpha_min, Alpha_max: allowed range of average relative angles (in radians) measured wrt local radial direciton
  ; Flux_rho_trend_min, Flux_rho_trend_max: allowed range of average fluxes along the feature computed using averaged radial trend
  ;
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------




  data_source = extract_parameter(fname_par, 'data_source', 'char')   ;reform(values[where(names eq 'DATA_SOURCE')])

  XYcenter = [extract_parameter(fname_par, 'Xcenter'), extract_parameter(fname_par, 'Ycenter')]

  Rot_angle = extract_parameter(fname_par, 'Rot_angle')

  Rs = extract_parameter(fname_par, 'Rs')

  d_roi = extract_parameter(fname_par, 'd_roi')

  png_channel = extract_parameter(fname_par, 'png_channel', 'int')

  blob_filt_min = extract_parameter(fname_par, 'blob_filt_min', 'int')

  blob_detect_min = extract_parameter(fname_par, 'blob_detect_min','int')

  blob_detect_thresh_pairs = extract_parameter(fname_par, 'blob_detect_thresh_pairs', 'int')

  blob_detect_thresholds = reform(extract_parameter(fname_par, 'blob_detect_thresholds'), 2, blob_detect_thresh_pairs)

  L_range = [ extract_parameter(fname_par, 'L_min') , extract_parameter(fname_par, 'L_max') ]

  Phi_range = [ extract_parameter(fname_par, 'Phi_min') , extract_parameter(fname_par, 'Phi_max') ]

  Alpha_range = [ extract_parameter(fname_par, 'Alpha_min') , extract_parameter(fname_par, 'Alpha_max') ]

  Flux_rho_trend_range = [ extract_parameter(fname_par, 'Flux_rho_trend_min') , extract_parameter(fname_par, 'Flux_rho_trend_max') ]

  blob_detect_n_segments = extract_parameter(fname_par, 'blob_detect_n_segments', 'int')

  P ={ Data_source:Data_source, $
    XYCenter:XYCenter, $
    ; Solar disk center in pixel coordinates, MLSO images
    Rot_angle:Rot_angle, $ ; rotation angle (image will be derotated)
    Rs: Rs, $ ; solar radius in image pixels
    ;Rho_range:[195., 400.], $
    ; range of polar radial distances used for feature detection
    d_roi: d_roi, $ ;distance (in pxl) in x and y directions from the disk center defining edges of ROI
    d_phi: !Pi/360.0, $
    ; angular resolution used for polar coordinate transform, degrees
    d_rho: 1.0, $;2.0 ONLY takes 1.0 -- CORRECT THIS LATER
    ; radial distance resolution used for polar coordinate transform, pixels
    phi_shift: 2.0, $
    ; diiscrete angle shift (in phi pixels) used for the second angular differencing in polar coordinates;
    ; must be EVEN for an accurate derotation
   png_channel:png_channel, $
   blob_filt_thresh: 0.5, $
    ; threshold for blob filtering; interpretation depends on blob_filt_mode (see below)
    blob_filt_mode: 1, $
    ; mode for applying threshold in blob filtering,
    ;      0: threshold = blob_filt_thresh,
    ;      1  threshold = mean + blob_filt_thresh*SD
    blob_filt_min: blob_filt_min, $
    ; size of the smallest blob (in polar coordinates pixels) to be kept; smaller blobs removed.
    blob_detect_min: blob_detect_min, $
    ; size of the smallest blob (in polar coordinates pixels) to be interpolated using ellipses; smaller blobs discarded.
    blob_detect_n_segments: blob_detect_n_segments, $;5, $
    ; number of major axis segments of each interpolated ellipse used to compute orientaiton angles
    ;blob_detect_dyn_range: [0,25] $
    ; dynamic ranges used to identify blobs
    ;blob_detect_thresholds: [-40, -30, -20, -10, 0, 10, 20, 30, 40]  $
    blob_detect_thresh_pairs: blob_detect_thresh_pairs, $
    blob_detect_thresholds: blob_detect_thresholds, $
    L_range:L_range, Phi_range: Phi_range, Alpha_range:Alpha_range, Flux_rho_trend_range:Flux_rho_trend_range $
  }




  return, p

End


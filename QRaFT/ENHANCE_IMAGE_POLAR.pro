FUNCTION ENHANCE_IMAGE_POLAR, IMG, P, X_p=X_p, Y_p=Y_p, Phi_p=Phi_p, Rho_p=Rho_p, img_p_filt=img_p_filt

  ;+
  ; NAME:
  ;     ENHANCE_IMAGE_POLAR
  ;
  ; PURPOSE:
  ;     Enhances quasi-radial structures  in a solar image "IMG"
  ;     using 2nd-order azimuthal differencing
  ;
  ; CALLING SEQUENCE:
  ;     IMG_p_enh = ENHANCE_IMAGE_POLAR(IMG, P, X_p=X_p, Y_p=Y_p, Phi_p=Phi_p, Rho_p=Rho_p, img_p_filt=img_p_filt)
  ;
  ; INPUTS:
  ;     IMG - 2D array representing the processed image, in Cartesian cordinates
  ;     P   - structure of processing parameters, retrieved from file using P=READ_PARAMETERS()
  ;
  ; OUTPUTS:
  ;     IMG_p_enh - 2D array containing the enhanced image in plane polar coordinates (Phi_p, Rho_p), obtained using an adaptive 2nd-order azimuthal differencing
  ;     X_p       - 2D array of cartesian X-positions of IMG_p_enh pixels
  ;     Y_p       - 2D array of cartesian Y-positions of IMG_p_enh pixels
  ;     Phi_p     - 2D array of angular coordinates of IMG_p_enh pixels
  ;     Rho_p     - 2D array of radial coordinates of IMG_p_enh pixels
  ;     img_p_filt - enhanced image in plane polar coordinates (Phi_p, Rho_p) obtained using adaptive 1st-order azimuthal differencing
  ;                  and image despiking
  ;     Comments:
  ;       1) Array positions X_p, Y_p, and Rho_p are in the coordinate system of the array indices.
  ;       2) Either IMG_p_enh or img_p_filt can be used for subsequent processing.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;
  ;     fname = "20140404_172405_kcor_l1.fts"
  ;             ; FITS file containing raw image
  ;     fname_par = "MLSO_Nov_2016_parameters.txt"
  ;             ; file with processing parameters optimized for the particular image type
  ;     P = READ_PARAMETERS(fname_par)
  ;             ; reading processing parameters
  ;     IMG_orig = OPEN_FILE(fname, P)
  ;             ; opening raw image
  ;     IMG_p_enh = ENHANCE_IMAGE_POLAR(IMG, P)
  ;     tvscl, IMG_p_enh
  ;             ; obtaining and displaying the enhanced image transformed to polar coordinates
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------

  img_1_detr = radial_detrending(ENHANCE_IMAGE_ANGULAR(img, P.XYCenter, P.Rs), P.XYCenter)
  ;img_1_detr = radial_detrending( img)
  ; subtracts average radial dependence,
  ; enhances radial structure using 1st order angular differencing,
  ; subtracts average radial dependence again

  Rect_to_Polar, img_1_detr, P.XYcenter, P.d_phi, P.d_rho, Img_p, X_p=X_p, Y_p=Y_p, Phi_p=Phi_p, Rho_p=Rho_p ;Rho_range=P.Rho_range,
  ; computes polar coordinates with given accuracy;
  ; computes matching rectangular coordinates;
  ; converts image into polar coordinates.
  ;

  ; img_p =  img_p - smooth(img_p, [100,1], /edge)


  img_p_filt = blob_filtering(img_p, P.blob_filt_thresh, P.blob_filt_mode, P.blob_filt_min)
  ; function Blob_filtering, img, thresh, thresh_mode, min_blob, blobs=blobs
  ; removes spiky noise by removing small pixel clusters ("blobs") above a given threshold

  img_p_2 = shift( img_p_filt - shift(img_p_filt, [P.phi_shift, 0]), [-P.phi_shift/2, 0] )
  ; angular differencing in polar coordinates,
  ; angular correction (rotating back half the angle)

  RETURN, img_p_2

 END

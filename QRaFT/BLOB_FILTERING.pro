function BLOB_FILTERING, IMG, thresh, thresh_mode, min_blob, blobs=blobs

  ;+
  ; NAME:
  ;     BLOB_FILTERING
  ;
  ; PURPOSE:
  ;     Identifies contiguous image regions (blobs) simultaneously satisfying two thresholding condition:
  ;     1) Pixel value is greater than the specified intensity threshold "thresh" (interpreted based on "thresh_mode", see below)
  ;     2) Blob size is greater than the specified blob size threshold "min_blob"
  ;
  ;     As a result, the algorithm keeps only sufficiently large and  intense image regions
  ;     and eliminates (sets to 0) the smallest and weakest ones. It can be used for removing
  ;     spicky noise and other small-scale contaminations from solar images.
  ;
  ;     The blob decomposition step is implemented using the IDL's LABEL_REGION() funciton.
  ;
  ; CALLING SEQUENCE:
  ;     IMG_filt = BLOB_FILTERING(IMG, thresh, thresh_mode, blobs=blobs)
  ;
  ; INPUTS:
  ;     IMG         - 2D data array representing studied solar image
  ;     thresh      - the value of the applied threshold. Its interpretation depends on the parameter "thresh_mode"
  ;                   as explained below.
  ;     thresh_mode - integer-valued parameter controling the thresholding mode as follows:
  ;
  ;                   thresh_mode = 0: absolute thresholding satisfying the condition IMG > thresh.
  ;                   thresh_mode = 1: relative thresholding based on a given number of standard deviaitons above the mean,
  ;                                    IMG > mean(IMG) + thresh*STDEV(IMG)
  ;                   thresh_mode = 2: extreme thresholding based on a given fraction of the maximum value,
  ;                                    IMG > thresh*max(IMG)
  ;
  ;                   [see funciton THRESH_IMAGE() for more details on the thresholding techniques]
  ;
  ;     min_blob    - minimum blob size to be included in the resulting filtered image
  ;
  ;
  ; OUTPUTS:
  ;     IMG_filt    - data array of the same size as IMG, containing the original IMG array values
  ;                   at the position fullfilling the filtering criteria (based on the intensity and
  ;                   blob-size thresholds), and 0s at the remaining positions.
  ;
  ; RESTRICTIONS:
  ;     none
  ;
  ; EXAMPLE:
  ;     IMG_filt = BLOB_FILTERING(IMG, 2.0, 1, 30, blobs=blobs)
  ;
  ;     Plotting the portions of IMG satisfying filtering requirements using 2 methods:
  ;
  ;     1) plotted pixel values within filtered blobs equal the IMG values at the same positions:
  ;
  ;       tvscl, img_filt
  ;
  ;     2) values within each blobs are set to the blob index:
  ;
  ;       tvscl, blobs.lbl*(img_filt gt 0)
  ;
  ;     In either plotting esample, the image regions not fulfilling the filtering requirements are set to 0.
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------

  img_th = thresh_image(img, thresh, thresh_mode)

  lbl = label_region(img_th, /ulong)
  hst_lbl = histogram(lbl, reverse_indices = hst_r)

  img_filt = img;_th
  for i=0, n_elements(hst_lbl)-1 do begin
    w = hst_r[hst_r[i]:hst_r[i+1]-1]
    if n_elements(w) lt min_blob then img_filt[w] = 0
  endfor

  blobs = {lbl:lbl, hst_lbl:hst_lbl, hst_r:hst_r}

  return, img_filt*img_th

End

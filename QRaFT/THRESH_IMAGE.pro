function THRESH_IMAGE,  IMG, thresh, thresh_mode

  ;+
  ; NAME:
  ;     THRESH_IMAGE
  ;
  ; PURPOSE:
  ;     Determines which elements of an input daya array are abouve a specified threshold,
  ;     based on several different thresholding methods. 
  ;
  ; CALLING SEQUENCE:
  ;     Mask = THRESH_IMAGE(IMG, thresh, thresh_mode)
  ;
  ; INPUTS:
  ;     IMG         - data array representing studied image
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
  ; OUTPUTS:
  ;     Mask        - binary array of the same type as IMG, with 1s feeling the positions satisfying 
  ;                   the applied thresholding condition and 0s filling the remaining positions.    
  ;
  ; RESTRICTIONS:
  ;
  ;
  ; EXAMPLES:
  ;     Absolute thresholding:  
  ;     Mask = THRESH_IMAGE(IMG, 0.025, 0)
  ;     (Mask =1 where IMG > 0.025)

  ;     Relative thresholding:
  ;     Mask = THRESH_IMAGE(IMG, 2.0, 1)
  ;     (Mask =1 where IMG > 2 standard deviations above the mean)
  ;
  ; MODIFICATION HISTORY:
  ;       V. Uritsky, 2019-2022
  ;-
  ;----------------------------------------------------------------
  
  case thresh_mode of
  
    0: begin  ; absolute threshold
         img_th = img gt thresh
       end
    1: begin  ; relative threshould = number of SD above mean
         w = where(img ne 0)
         img_th = img gt (mean(img[w]) +  thresh * stdev(img[w]))
       end
       
    2: begin  ; relative threshould = fraction of max 
         img_th = img gt thresh*max(img)
       end
   
  endcase
  
  return, img_th
End
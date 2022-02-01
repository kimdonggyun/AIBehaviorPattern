# definitions of image analysis

import cv2
import numpy as np

def property(roi, cnt):
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    solidity = float(area)/hull_area # calculate solidity
    eq = np.sqrt(4*area/np.pi) # calculate equivaletn diameter


    # calculate gray mean value within contour
    #mask_array = np.zeros(roi.shape,  np.uint8) # create 0 array having same shape of image
    #masked_img = np.ma.masked_array(roi, mask= (mask_array != 255)) # careful!! 1 is True, 0 is False. in this mask_array, background is False and object is True
    #gray_mean = np.mean(masked_img) # unmasked(False=background) will be ignored and masked(Ture=object) will only be considered


    return area, solidity, eq
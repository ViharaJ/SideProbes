# -*- coding: utf-8 -*-

"""
Created on Fri Oct 20 11:05:03 2023

@author: v.jayaweera
"""

import numpy as np
import cv2 
import os 

def longestContour(contours):
    maxIndx = 0
    maxLen = len(contours[0])
    
    for i in range(1, len(contours)):
        if (len(contours[i]) > maxLen):
            maxIndx = i
            maxLen = len(contours[i])
    
    return contours[maxIndx]



sourcePath = "C:/Users/v.jayaweera/Documents/Anne/Side Probes/Roughness_Routine_Output_Downskin/Hantel16"


for file in os.listdir(sourcePath):
    
    # Extract contour
    img = cv2.imread(sourcePath + '/' + file, cv2.IMREAD_GRAYSCALE)
    cont, hier = cv2.findContours(img, cv2.RETR_LIST , cv2.CHAIN_APPROX_NONE)
    
    if (cont):
        #Get main contour of interest, ignore pores
        k = longestContour(cont)
        
        k = np.squeeze(k, axis=1)
        
        k = np.unique(k)
        
        print(file, " :", len(k))

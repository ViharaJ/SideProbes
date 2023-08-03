# -*- coding: utf-8 -*-
"""
Input: Binary image(s)
Scale: adjust to fit image resolution
"""
import numpy as np
import shapely
import matplotlib.pyplot as plt
from scipy import signal
import scipy
import cv2
import Module.Functions as fb
import os 
import sys
import math
import time

def longestContour(contours):
    maxIndx = 0
    maxLen = len(contours[0])
    
    for i in range(1, len(contours)):
        if (len(contours[i]) > maxLen):
            maxIndx = i
            maxLen = len(contours[i])
    
    return contours[maxIndx]


start = time.time()
#"C:/Users/v.jayaweera/Pictures/FindingEdgesCutContour/OneFileContours"
sourcePath = "C:/Users/v.jayaweera/Documents/Side Probes/Roughness_Routine_Output/HantelTest"
csvPath = '/Users/v.jayaweera/Documents/SRAvg-ContourDiv-NoInvert.csv'
acceptedFileTypes = ["jpg", "png", "bmp", "tif"]
dirPictures = os.listdir(sourcePath)
imageID = []
scale = 6.249
averageSR = []

if(len(dirPictures)  <= 0):
    print('The specified folder is empty!')
    sys.exit()
else:
    
    for path in dirPictures:
        if( '.' in path and path.split('.')[-1].lower() in acceptedFileTypes):
            
            #Reset plots to default figure size
            plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
            plt.gca().invert_yaxis()
            
            
            # Extract contour
            img = cv2.imread(sourcePath + '/' + path, cv2.IMREAD_GRAYSCALE)
            height, width = img.shape
            
            
            x = []
            y = []
            
            count = 0
            for i in range(width):
                col = np.transpose(img[:,i])
                indices = np.where(col==255)[0]
                 
                
                for j in range(len(indices)):
                    x.append(count)
                    y.append(indices[j])
                    count = count+1
                
            
            
            print(x,y)
            
            plt.plot(x,y)
            plt.show()
                
                
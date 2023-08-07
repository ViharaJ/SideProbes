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

def checkFeature(img, row, col):
    prevCol = img[:, col-1]
    
    whitePx = np.where()


start = time.time()
#"C:/Users/v.jayaweera/Pictures/FindingEdgesCutContour/OneFileContours"
sourcePath = "C:/Users/v.jayaweera/Documents/Side Probes/Roughness_Routine_Output/Hantel01_Outline"
csvPath = '/Users/v.jayaweera/Documents/SRAvg-ContourDiv-NoInvert.csv'
acceptedFileTypes = ["jpg", "png", "bmp", "tif"]
dirPictures = os.listdir(sourcePath)
imageID = []
scale = 6.249
averageSR = []
doubleBack = 0

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
            cont, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            
            if (cont):
                #Get main contour of interest, ignore pores
                k = longestContour(cont)
                #REMOVE properly
                # _, idx = np.unique(k, axis=0,  return_index=True)
                # print(idx.shape)
                # k = k[np.sort(idx)]
                
                # # sig is sigma of Gauss, size is kernel's full length
                # sig = 60
                # size = 20
                # distanceE = []   
                # saveIndex = []
                            
                x = np.array(k[:,0,0])*scale
                y = np.array(k[:,0,1])*scale
                
                pairs = []
                
                for i in range(len(x)):
                    if(doubleBack == 0 and [x[i], y[i]] in pairs):
                        pairs = []
                        pairs.append([x[i], y[i]])
                        doubleBack = doubleBack + 1
                    elif (doubleBack == 1 and [x[i], y[i]] in pairs):
                        pairs = []
                        pairs.append([x[i], y[i]])
                        doubleBack = doubleBack + 1
                
                    pairs.append([x[i], y[i]])
                    
                
                pairs = np.array(pairs)
                plt.title(path)
                plt.plot(pairs[:,0], pairs[:,1],'r.-')
                plt.show()
                